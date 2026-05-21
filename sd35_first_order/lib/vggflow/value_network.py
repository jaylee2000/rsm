"""Value network wrapper with unified interface."""

import torch


class ValueNetworkWrapper:
    """
    Wrapper for value network with consistent interface.

    Handles different value network architectures (LoRA, small UNet, etc.)
    with a single forward interface.
    """

    def __init__(self, value_net, value_net_param, pipeline):
        """
        Initialize value network wrapper.

        Args:
            value_net: The value network module
            value_net_param: Type of value network ('lora', 'copy', 'small')
            pipeline: Diffusion pipeline (for joint_attention_kwargs)
        """
        self.value_net = value_net
        self.value_net_param = value_net_param
        self.pipeline = pipeline

    def forward(self, xt, timestep, prompt_embeds, pooled_prompt_embeds=None):
        """
        Unified forward pass for value network.

        Args:
            xt: Latent state (B, C, H, W)
            timestep: Timestep tensor (B,) or scalar
            prompt_embeds: Text embeddings (B, seq_len, dim) or (2*B, seq_len, dim)
            pooled_prompt_embeds: Pooled text embeddings (B, dim) or (2*B, dim)

        Returns:
            value_correction: Value function output (B, C, H, W)
        """
        batch_size = xt.size(0)

        # Ensure prompt embeddings match batch size
        if prompt_embeds.size(0) > batch_size:
            prompt_embeds = prompt_embeds[:batch_size]
        if pooled_prompt_embeds is not None and pooled_prompt_embeds.size(0) > batch_size:
            pooled_prompt_embeds = pooled_prompt_embeds[:batch_size]

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            if self.value_net_param == 'small':
                # Small UNet architecture
                value_correction = self.value_net(
                    sample=xt.float(),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds.float()
                ).sample
            else:
                # LoRA or copy architecture (transformer-based)
                value_correction = self.value_net(
                    hidden_states=xt.float(),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds.float(),
                    pooled_projections=pooled_prompt_embeds.float(),
                    joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
                    return_dict=False,
                )[0]

        return value_correction

    def forward_with_jvp(self, xt, timestep_var, prompt_embeds, pooled_prompt_embeds=None):
        """
        Forward pass with timestep requiring gradients (for JVP computation).

        Args:
            xt: Latent state (B, C, H, W)
            timestep_var: Timestep tensor requiring gradients (B,)
            prompt_embeds: Text embeddings
            pooled_prompt_embeds: Pooled text embeddings

        Returns:
            value_correction: Value function output (B, C, H, W)
        """
        batch_size = xt.size(0)

        # Ensure prompt embeddings match batch size
        if prompt_embeds.size(0) > batch_size:
            prompt_embeds = prompt_embeds[:batch_size]
        if pooled_prompt_embeds is not None and pooled_prompt_embeds.size(0) > batch_size:
            pooled_prompt_embeds = pooled_prompt_embeds[:batch_size]

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            if self.value_net_param == 'small':
                value_correction = self.value_net(
                    sample=xt.float(),
                    timestep=timestep_var,
                    encoder_hidden_states=prompt_embeds.float()
                ).sample
            else:
                value_correction = self.value_net(
                    hidden_states=xt.float(),
                    timestep=timestep_var,
                    encoder_hidden_states=prompt_embeds.float(),
                    pooled_projections=pooled_prompt_embeds.float(),
                    joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
                    return_dict=False,
                )[0]

        return value_correction
