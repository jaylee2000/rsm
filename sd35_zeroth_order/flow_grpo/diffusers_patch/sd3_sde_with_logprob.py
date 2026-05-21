# Copied from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# We adapt it from flow to flow matching.

import math
from typing import Optional, Union

import torch

from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor


def _normalize_timestep_batch(
    timestep: Union[float, torch.FloatTensor],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(timestep):
        timestep = timestep.to(device=device, dtype=dtype)
    else:
        timestep = torch.as_tensor(timestep, device=device, dtype=dtype)

    if timestep.ndim == 0:
        timestep = timestep.unsqueeze(0)
    else:
        timestep = timestep.reshape(-1)

    if timestep.shape[0] == 1 and batch_size > 1:
        timestep = timestep.expand(batch_size)
    elif timestep.shape[0] != batch_size:
        raise ValueError(
            "timestep batch size must be 1 or match sample batch size, got "
            f"{timestep.shape[0]} and {batch_size}."
        )
    return timestep


def sde_step_with_logprob(
    self: FlowMatchEulerDiscreteScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    noise_level: float = 0.7,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
    sde_type: Optional[str] = "sde",
    return_sqrt_dt: Optional[bool] = False,
    return_noise: Optional[bool] = False,
):
    """
    Predict the sample from the previous timestep by reversing the SDE. This function propagates the flow
    process from the learned model outputs (most often the predicted velocity).

    Args:
        model_output (`torch.FloatTensor`):
            The direct output from learned flow model.
        timestep (`float`):
            The current discrete timestep in the diffusion chain.
        sample (`torch.FloatTensor`):
            A current instance of a sample created by the diffusion process.
        generator (`torch.Generator`, *optional*):
            A random number generator.
    """
    # bf16 can overflow here when compute prev_sample_mean, we must convert all variable to fp32
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    if model_output.shape[0] != sample.shape[0]:
        raise ValueError(
            "model_output and sample must have the same batch size, got "
            f"{model_output.shape[0]} and {sample.shape[0]}."
        )
    if prev_sample is not None and prev_sample.shape[0] != sample.shape[0]:
        raise ValueError(
            "prev_sample and sample must have the same batch size, got "
            f"{prev_sample.shape[0]} and {sample.shape[0]}."
        )

    if prev_sample is not None and generator is not None:
        raise ValueError(
            "Cannot pass both generator and prev_sample. Please make sure that either `generator` or"
            " `prev_sample` stays `None`."
        )

    timestep = _normalize_timestep_batch(
        timestep,
        batch_size=sample.shape[0],
        device=sample.device,
        dtype=self.timesteps.dtype,
    )

    step_index = [self.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]

    t = self.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    t_prev = self.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    t_max = self.sigmas[1].item()
    minus_dt = t_prev - t
    sqrt_dt = torch.sqrt(-1 * minus_dt)

    random_noise = None

    if sde_type == "sde": # Flow-SDE
        tilde_sigma_t = torch.sqrt(t / (1 - torch.where(t == 1, t_max, t))) * noise_level
        prev_sample_mean = sample * (1 + tilde_sigma_t**2 / (2 * t) * minus_dt) + model_output * (
            1 + tilde_sigma_t**2 * (1 - t) / (2 * t)
        ) * minus_dt
        sigma_t = tilde_sigma_t * sqrt_dt

        if prev_sample is None:
            random_noise = randn_tensor(sample.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
            prev_sample = prev_sample_mean + sigma_t * random_noise

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (sigma_t**2))
            - torch.log(sigma_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

    elif sde_type == "cps":
        sigma_t = sigma_prev * math.sin(noise_level * math.pi / 2)  # sigma_t in paper
        pred_x_0 = sample - sigma * model_output  # predicted x_0 in paper
        pred_x_1 = sample + model_output * (1 - sigma)  # predicted x_1 in paper
        prev_sample_mean = pred_x_0 * (1 - sigma_prev) + pred_x_1 * torch.sqrt(
            sigma_prev**2 - sigma_t**2
        )

        if prev_sample is None:
            random_noise = randn_tensor(model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype)
            prev_sample = prev_sample_mean + sigma_t * random_noise

        # remove all constants
        # reweight by 2 * (sigma_t ** 2)
        log_prob = (
            -(prev_sample.detach() - prev_sample_mean) ** 2
        )

    elif sde_type == "deterministic":
        tilde_sigma_t = torch.sqrt(t / (1 - torch.where(t == 1, t_max, t))) * noise_level
        prev_sample_mean = sample * (1 + tilde_sigma_t**2 / (2 * t) * minus_dt) + model_output * (
            1 + tilde_sigma_t**2 * (1 - t) / (2 * t)
        ) * minus_dt
        sigma_t = tilde_sigma_t * sqrt_dt
        prev_sample = sample + minus_dt * model_output

        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (sigma_t**2))
            - torch.log(sigma_t)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )

    else:
        raise ValueError(f"Unsupported sde_type: {sde_type}. Expected one of ['sde', 'cps', 'deterministic'].")

    # mean along all but batch dimension
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    if return_sqrt_dt:
        if return_noise:
            return prev_sample, log_prob, prev_sample_mean, sigma_t, sqrt_dt, random_noise
        return prev_sample, log_prob, prev_sample_mean, sigma_t, sqrt_dt

    if return_noise:
        return prev_sample, log_prob, prev_sample_mean, sigma_t, random_noise
    return prev_sample, log_prob, prev_sample_mean, sigma_t
