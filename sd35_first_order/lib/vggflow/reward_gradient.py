"""Reward gradient computation with consistent clipping."""

import torch


def latent_to_image(pipeline, latents, clamp=True):
    """
    Convert latents to images using VAE decoder.

    Args:
        pipeline: Diffusion pipeline with VAE
        latents: Latent tensors (B, C, H, W)
        clamp: Whether to clamp output to [0, 1]

    Returns:
        images: Decoded images (B, 3, H, W) in [0, 1]
    """
    latents = latents.to(torch.float32)

    if hasattr(pipeline.vae.config, "shift_factor"):
        latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    else:  # AutoencoderTiny
        latents = latents / pipeline.vae.config.scaling_factor

    image = pipeline.vae.decode(latents, return_dict=False)[0]

    if clamp:
        image = torch.clamp((image + 1) / 2, 0., 1.)
    else:
        # Soft clamp using softplus
        from torch.nn.functional import softplus
        beta = 10.0
        low, high = 0.0, 1.0
        image = (image + 1) / 2
        image = low + softplus(beta * (image - low), beta=1.0) / beta \
                    - softplus(beta * (image - high), beta=1.0) / beta

    return image


class RewardGradientComputer:
    """
    Computes reward gradients ∇_x r(decode(x)) with consistent clipping.

    This class handles:
    1. Latent to image decoding
    2. Reward computation
    3. Gradient computation via autograd
    4. Gradient clipping using quantile thresholds
    """

    def __init__(self, reward_fn, pipeline, config):
        """
        Initialize reward gradient computer.

        Args:
            reward_fn: Reward function that takes (images, prompts, metadata)
            pipeline: Diffusion pipeline (for VAE decoding)
            config: Training configuration
        """
        self.reward_fn = reward_fn
        self.pipeline = pipeline
        self.quantile_clipping = config.training.quantile_clipping
        self.n_jitter = getattr(config.training, 'n_jitter', 1)
        self.std_jitter = getattr(config.training, 'std_jitter', 0.0)

    def compute_from_latent(
        self,
        latent_input,
        latent_requires_grad,
        prompts,
        prompt_metadata,
        rgrad_threshold,
        retain_graph=True,
    ):
        """
        Compute ∇_x r(decode(x)).

        Args:
            latent_input: Input latent for decoding (B, C, H, W)
            latent_requires_grad: Latent variable to compute gradient w.r.t.
            prompts: Text prompts (list of strings)
            prompt_metadata: Prompt metadata (list of dicts)
            rgrad_threshold: Clipping threshold for gradient norm
            retain_graph: Whether to retain computation graph after gradient

        Returns:
            rgrad: Reward gradient (B, C, H, W)
            reward_mask: Mask for positive rewards (B,)
        """
        latent_input = latent_input.to(torch.float32)
        bs, c, h, w = latent_input.size()

        # Add jitter if enabled (for variance reduction)
        if self.n_jitter > 1 or self.std_jitter > 0:
            latent_jittered = (
                latent_input.unsqueeze(1) +
                self.std_jitter * torch.randn(
                    bs, self.n_jitter, c, h, w,
                    device=latent_input.device
                )
            ).view(-1, c, h, w)
        else:
            latent_jittered = latent_input

        with torch.autocast(dtype=torch.float32, device_type="cuda"):
            # Decode to image
            images = latent_to_image(self.pipeline, latent_jittered)

            # Compute reward
            rewards, _ = self.reward_fn(images.float(), prompts, prompt_metadata)

            # Release image memory immediately after reward computation
            images = None

            # Compute gradient
            rgrad = torch.autograd.grad(
                rewards.sum(),
                latent_requires_grad,
                retain_graph=retain_graph
            )[0].detach().float()

            # Create mask for positive rewards before releasing rewards
            reward_mask = (rewards >= 0).float()

            # Release reward memory immediately after gradient computation
            rewards = None
            del rewards

            # Average over jitter samples
            if self.n_jitter > 1:
                rgrad = rgrad / self.n_jitter

            # Compute norm before clipping (for adaptive threshold)
            rgrad_norm = torch.linalg.norm(
                rgrad.view(rgrad.size(0), -1), dim=1
            )

            # Clip gradient norm
            rgrad = self.clip_gradient(rgrad, rgrad_threshold)

        # Release jittered latent memory
        latent_jittered = None

        return rgrad, reward_mask, rgrad_norm

    def clip_gradient(self, rgrad, threshold):
        """
        Clip gradient norm using quantile threshold.

        Args:
            rgrad: Reward gradient (B, C, H, W)
            threshold: Clipping threshold

        Returns:
            rgrad_clipped: Clipped gradient (B, C, H, W)
        """
        if not self.quantile_clipping:
            return rgrad

        # Compute gradient norm per sample
        rgrad_norm = torch.linalg.norm(
            rgrad.view(rgrad.size(0), -1), dim=1
        )

        # Clip to threshold
        rgrad_clipped = (
            rgrad / (rgrad_norm.view(-1, 1, 1, 1) + 1e-8) *
            rgrad_norm.view(-1, 1, 1, 1).clamp(max=threshold)
        )

        return rgrad_clipped

    def _clip_gradient(self, rgrad, threshold):
        """Backward-compatible alias for reward gradient clipping."""
        return self.clip_gradient(rgrad, threshold)
