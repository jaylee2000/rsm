import math
import torch

from typing import Optional, Tuple, Union
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from accelerate.logging import get_logger

from utils.consts import PCPO

logger = get_logger(__name__, log_level="INFO")


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(timestep, prev_timestep, alphas_cumprod, final_alpha_cumprod):
    """ DDPM variance schedule """
    alpha_prod_t = torch.gather(alphas_cumprod, 0, timestep.cpu()).to(
        timestep.device
    )
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        alphas_cumprod.gather(0, prev_timestep.cpu()),
        final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

    return variance


def _get_std(timestep, prev_timestep, alphas_cumprod, final_alpha_cumprod, weight):
    """ Adjusted variance schedule for PCPO """
    alpha_prod_t = torch.gather(alphas_cumprod, 0, timestep.cpu()).to(
        timestep.device
    )
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        alphas_cumprod.gather(0, prev_timestep.cpu()),
        final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    c1_t_squared = beta_prod_t / (alpha_prod_t / alpha_prod_t_prev)
    sigma = (weight * c1_t_squared  ** 0.5 + (beta_prod_t_prev * (weight ** 2 + 1) - c1_t_squared) ** 0.5) / (weight ** 2 + 1)
    return sigma


def get_alpha_prod_t(noise_scheduler, timestep, sample):
    alpha_prod_t = noise_scheduler.alphas_cumprod.gather(0, timestep.cpu())  # torch scalar
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    return alpha_prod_t

def get_alpha_t(noise_scheduler, timestep, sample, num_train_timesteps, num_inference_steps):
    prev_timestep = (timestep - num_train_timesteps // num_inference_steps)
    prev_timestep = torch.clamp(prev_timestep, 0, num_train_timesteps - 1)

    alpha_prod_t = noise_scheduler.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        noise_scheduler.alphas_cumprod.gather(0, prev_timestep.cpu()),
        noise_scheduler.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)
    alpha_t = alpha_prod_t / alpha_prod_t_prev
    alpha_t = _left_broadcast(alpha_t, sample.shape).to(sample.device)
    return alpha_t


def ddim_step_with_logprob(
    noise_scheduler: DDIMScheduler,
    model_output: torch.FloatTensor,
    variance_noise: Optional[torch.FloatTensor],
    timestep: int,
    sample: torch.FloatTensor,
    num_train_timesteps: int = 1000,
    num_inference_steps: int = 50,
    clip_sample_range: float = 1.0,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    thresholding: bool = False,
    clip_sample: bool = False,
    generator=None,
    reverse_grad: bool = False,
    retain_graph: bool = False,
    allow_2nd: bool = False,
    strength: float = 1.0,
    prev_sample: Optional[torch.FloatTensor] = None,
    use_pcpo: Optional[bool] = False,
    calculate_pb: Optional[bool] = False,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Predict the sample at the previous timestep by reversing the SDE.
    Core function to propagate the diffusion process from
    learned model outputs (predicted noise).

    Args:
        model_output_* (`torch.FloatTensor`): predicted epsilon from the model.
        variance_noise (`torch.FloatTensor`): instead of generating noise
            for the variance using `generator`, we can directly provide
            the noise for the variance itself. This is useful for methods such as
            CycleDiffusion. (https://arxiv.org/abs/2210.05559)
        timestep (`int`): current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            current instance of sample being created by diffusion process.
        eta (`float`): weight of noise for added noise in diffusion step.
        use_clipped_model_output (`bool`): if `True`, compute "corrected"
            `model_output` from the clipped predicted original sample.
            Necessary because predicted original sample is clipped to [-1, 1] when
            `self.config.clip_sample` is `True`.
            If no clipping has happened, "corrected" `model_output` would
            coincide with the one provided as input and `use_clipped_model_output`
            will have no effect.
        return_dict (`bool`): option for returning tuple rather than DDIMSchedulerOutput class

    Returns:
        (sample tensor, log probability of this step, used variance noise tensor)
    """
    # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
    # Ideally, read DDIM paper in-detail understanding

    # Notation (<variable name> -> <name in paper>
    # - pred_noise_t -> e_theta(x_t, t)
    # - pred_original_sample -> f_theta(x_t, t) or x_0
    # - std_dev_t -> sigma_t
    # - eta -> η
    # - pred_sample_direction -> "direction pointing to x_t"
    # - pred_prev_sample -> "x_t-1"

    # 1. get previous step value (=t-1)
    prev_timestep = (timestep - num_train_timesteps // num_inference_steps)
    # prevent OOB on gather
    prev_timestep = torch.clamp(prev_timestep, 0, num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = noise_scheduler.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        noise_scheduler.alphas_cumprod.gather(0, prev_timestep.cpu()),
        noise_scheduler.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(
        sample.device
    )
    alpha_t = alpha_prod_t / alpha_prod_t_prev
    alpha_t = _left_broadcast(alpha_t, sample.shape).to(sample.device)
    beta_prod_t = 1 - alpha_prod_t

    # 3. Sample noise if `variance_noise` is not provided
    if variance_noise is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )

    # 4. Compute variance: "sigma_t(η)" -> see formula (16)
    if use_pcpo:
        # σ_t = (w * sqrt(β_t/α_t) + sqrt( (1+w^2) * β_t-1 - (β_t/α_t))) / (1+w^2)
        std_dev_t = _get_std(
            timestep,
            prev_timestep,
            noise_scheduler.alphas_cumprod,
            noise_scheduler.final_alpha_cumprod,
            weight=PCPO['const_weight']
        ) * eta
    else:
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = _get_variance(
            timestep,
            prev_timestep,
            noise_scheduler.alphas_cumprod,
            noise_scheduler.final_alpha_cumprod
        )
        std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    # 5. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_original_sample = (
        sample - beta_prod_t ** (0.5) * model_output
    ) / alpha_prod_t ** (0.5)
    pred_epsilon = model_output

    # 6. Clip or threshold "predicted x_0"
    if thresholding:
        pred_original_sample = noise_scheduler._threshold_sample(pred_original_sample)
    elif clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -clip_sample_range, clip_sample_range
        )

    if use_clipped_model_output:
        # the pred_epsilon is always re-derived from the clipped x_0 in Glide
        pred_epsilon = (
            sample - alpha_prod_t ** (0.5) * pred_original_sample
        ) / beta_prod_t ** (0.5)

    # 7. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
        0.5
    ) * pred_epsilon

    # 8. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
    prev_sample_mean = (
        alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
    )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    if prev_sample is None:
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # log prob of prev_sample given prev_sample_mean and std_dev_t, w/o meaningless constants
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if reverse_grad:
        target = ((prev_sample.detach() - prev_sample_mean) ** 2).sum()
        grad = torch.autograd.grad(
            outputs=target,                # The value whose gradient we want
            inputs=sample,                 # The intermediate node we want the gradient with respect to
            retain_graph=retain_graph,     # Retain graph for further gradient computations
            create_graph=allow_2nd         # If higher-order gradients are needed
        )[0]
        score = -grad / (2 * (std_dev_t ** 2)) * strength
    else:
        score = -(prev_sample.detach() - prev_sample_mean) / (std_dev_t ** 2) * strength

    if calculate_pb:
        assert prev_sample is not None
        pb_mean = alpha_t ** (0.5) * prev_sample
        pb_std = (1 - alpha_t) ** (0.5)
        log_pb = (
            -((sample.detach() - pb_mean.detach()) ** 2) / (2 * (pb_std**2))
            - torch.log(pb_std)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_pb = log_pb.mean(dim=tuple(range(1, log_pb.ndim)))
        return prev_sample.type(sample.dtype), log_prob, variance_noise, score, log_pb

    return prev_sample.type(sample.dtype), log_prob, variance_noise, score


def pred_orig_latent(
    noise_scheduler: DDIMScheduler,
    model_output: torch.FloatTensor,
    sample: torch.FloatTensor,
    timestep: int
):
    alpha_prod_t = noise_scheduler.alphas_cumprod.gather(0, timestep.cpu())  # torch scalar
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t = alpha_prod_t.to(sample.dtype) # float32 -> bf16
    beta_prod_t = 1 - alpha_prod_t

    beta_prod_t[timestep == 0] = 0
    alpha_prod_t[timestep == 0] = 1

    pred_original_sample = (
        sample - beta_prod_t ** (0.5) * model_output
    ) / alpha_prod_t ** (0.5)

    return pred_original_sample


def ddim_step_with_logprob_rg(
    noise_scheduler: DDIMScheduler,
    model_output: torch.FloatTensor,
    model_output_ref: Optional[torch.FloatTensor],
    flow_correction: Optional[torch.FloatTensor],
    variance_noise: Optional[torch.FloatTensor],
    timestep: int,
    sample: torch.FloatTensor,
    rg_scale: Optional[float] = 1.0,
    cg_scale: Optional[float] = 1.0,
    num_train_timesteps: int = 1000,
    num_inference_steps: int = 50,
    clip_sample_range: float = 1.0,
    eta: float = 0.0,
    use_clipped_model_output: bool = False,
    thresholding: bool = False,
    clip_sample: bool = False,
    generator=None,
    prev_sample: Optional[torch.FloatTensor] = None,
    const_weight: Optional[float] = None,
    algorithm: str = "uw-sigma",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DDIM step with optional RG-style policy/reference interpolation.

    This matches the ddgrpo-rg-new implementation: if `"lambda" in algorithm`,
    we mix policy vs reference predictions; otherwise we behave like vanilla DDIM.

    Returns: (prev_sample, log_prob, variance_noise)
    """

    # 1. get previous step value (=t-1)
    prev_timestep = timestep - num_train_timesteps // num_inference_steps
    prev_timestep = torch.clamp(prev_timestep, 0, num_train_timesteps - 1)

    # 2. compute alphas, betas
    alpha_prod_t = noise_scheduler.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        noise_scheduler.alphas_cumprod.gather(0, prev_timestep.cpu()),
        noise_scheduler.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)
    alpha_t = alpha_prod_t / alpha_prod_t_prev
    alpha_t = _left_broadcast(alpha_t, sample.shape).to(sample.device)
    beta_prod_t = 1 - alpha_prod_t

    # 3. Sample noise if `variance_noise` is not provided
    if variance_noise is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
    assert variance_noise is not None

    # 4. Compute variance / std dev
    if "sigma" in algorithm:
        std_dev_t = _get_std(
            timestep,
            prev_timestep,
            noise_scheduler.alphas_cumprod,
            noise_scheduler.final_alpha_cumprod,
            weight=const_weight,
        ) * eta
    else:
        variance = _get_variance(
            timestep,
            prev_timestep,
            noise_scheduler.alphas_cumprod,
            noise_scheduler.final_alpha_cumprod,
        )
        std_dev_t = eta * variance ** 0.5

    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    # 5. Compute possibly-mixed epsilon (policy/reference)
    if "lambda" in algorithm:
        if model_output_ref is None:
            raise ValueError("model_output_ref must be provided when using a 'lambda' algorithm")

        if const_weight is not None:
            c1_t = torch.sqrt(1 - alpha_prod_t) / torch.sqrt(alpha_t)
            c2_t = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2)
            lambda_t = const_weight * std_dev_t / (c1_t - c2_t)
        else:
            rg_scale_value = 1.0 if rg_scale is None else float(rg_scale)
            lambda_t = rg_scale_value * torch.ones_like(alpha_prod_t)

        model_output_mine = model_output_ref + lambda_t * (model_output - model_output_ref)

        if flow_correction is not None:
            cg_scale_value = 1.0 if cg_scale is None else float(cg_scale)
            # lambda2_t = -cg_scale_value * std_dev_t 
            lambda2_t = cg_scale_value * torch.ones_like(alpha_prod_t)
            model_output_mine = model_output_mine + lambda2_t * flow_correction
    else:
        model_output_mine = model_output

    # 6. predicted x_0
    pred_original_sample = (sample - beta_prod_t**0.5 * model_output_mine) / alpha_prod_t**0.5
    pred_epsilon = model_output_mine

    # 7. Clip/threshold
    if thresholding:
        pred_original_sample = noise_scheduler._threshold_sample(pred_original_sample)
    elif clip_sample:
        pred_original_sample = pred_original_sample.clamp(-clip_sample_range, clip_sample_range)

    if use_clipped_model_output:
        pred_epsilon = (sample - alpha_prod_t**0.5 * pred_original_sample) / beta_prod_t**0.5

    # 8. direction
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** 0.5 * pred_epsilon

    # 9. prev mean
    prev_sample_mean = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction

    if prev_sample is not None and generator is not None:
        raise ValueError("Cannot pass both generator and prev_sample")

    if prev_sample is None:
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t**2))
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob, variance_noise
