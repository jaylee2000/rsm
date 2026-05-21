"""Classifier-Free Guidance (CFG) wrapper for consistent application."""

import torch


class CFGWrapper:
    """
    Wrapper for applying classifier-free guidance consistently.

    CFG computes: output = uncond + guidance_scale * (cond - uncond)
    """

    def __init__(self, guidance_scale, do_cfg):
        """
        Initialize CFG wrapper.

        Args:
            guidance_scale: Guidance scale factor
            do_cfg: Whether to apply CFG (True if guidance_scale > 1.0)
        """
        self.guidance_scale = guidance_scale
        self.do_cfg = do_cfg

    def prepare_inputs(self, latent, timestep, prompt_embeds, pooled_prompt_embeds):
        """
        Prepare inputs for CFG forward pass.

        Args:
            latent: Latent tensor (B, C, H, W)
            timestep: Timestep tensor (B,)
            prompt_embeds: Text embeddings (2*B, seq_len, dim) or (B, seq_len, dim)
            pooled_prompt_embeds: Pooled embeddings (2*B, dim) or (B, dim)

        Returns:
            latent_input: (2*B, C, H, W) if CFG else (B, C, H, W)
            timestep_input: (2*B,) if CFG else (B,)
            prompt_embeds: Unchanged
            pooled_prompt_embeds: Unchanged
        """
        if not self.do_cfg:
            return latent, timestep, prompt_embeds, pooled_prompt_embeds

        latent_input = torch.cat([latent] * 2)
        timestep_input = timestep.repeat(2) if timestep.dim() == 1 else torch.cat([timestep] * 2)

        return latent_input, timestep_input, prompt_embeds, pooled_prompt_embeds

    def apply(self, model_output, detach_uncond=True):
        """
        Apply CFG to model output.

        Args:
            model_output: Model output, shape (2*B, ...) if CFG else (B, ...)
            detach_uncond: Whether to detach unconditional branch gradient

        Returns:
            guided_output: CFG-applied output, shape (B, ...)
        """
        if not self.do_cfg:
            return model_output

        uncond_output, cond_output = model_output.chunk(2)

        if detach_uncond:
            uncond_output = uncond_output.detach()

        return uncond_output + self.guidance_scale * (cond_output - uncond_output)
