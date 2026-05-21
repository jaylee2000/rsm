"""Core VGG-Flow algorithm implementation."""

import torch


class VGGFlowAlgorithm:
    """Core VGG-Flow algorithm with behavior aligned to vggflow-main."""

    def __init__(
        self,
        transformer,
        value_net_wrapper,
        cfg_wrapper,
        reward_grad_computer,
        config,
        pipeline,
    ):
        self.transformer = transformer
        self.value_net = value_net_wrapper
        self.cfg = cfg_wrapper
        self.reward_grad = reward_grad_computer
        self.config = config
        self.pipeline = pipeline

        self.guidance_scale = config.model.reward_scale
        self.eta_mode = config.model.eta_mode
        self.use_value_net = config.model.use_value_net
        self.detach_dir = config.training.detach_dir
        self.eps_step = 1e-4
        self.eps_time = 1e-4

    def _transformer_model(self):
        """Return the underlying transformer, with or without DDP wrapping."""
        return self.transformer.module if hasattr(self.transformer, "module") else self.transformer

    def compute_velocity_target(
        self,
        xt,
        timestep,
        sigma,
        prompt_embeds,
        pooled_prompt_embeds,
        prompts,
        prompt_metadata,
        rgrad_threshold,
        saved_rgrad_x1=None,
        rewards=None,
    ):
        """
        Compute target velocity for velocity matching.

        Returns:
            velocity_target: target velocity field
            loss_components: tensors required by downstream losses/logging
        """
        if saved_rgrad_x1 is None:
            # Reward gradient branch: x1 projection with current LoRA-enabled model.
            xt_for_grad = xt.detach()
            xt_for_grad.requires_grad_(True)

            with torch.enable_grad():
                velocity_current = self._compute_velocity(
                    xt_for_grad,
                    timestep,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    detach_uncond=True,
                )
                sigma_last = self.pipeline.scheduler.sigmas[-1]
                x1_pred = xt_for_grad + (sigma_last - sigma) * velocity_current.detach()

            rgrad, reward_mask, rgrad_norm = self.reward_grad.compute_from_latent(
                latent_input=x1_pred,
                latent_requires_grad=xt_for_grad,
                prompts=prompts,
                prompt_metadata=prompt_metadata,
                rgrad_threshold=rgrad_threshold,
            )
        else:
            if rewards is None:
                raise ValueError(
                    "rewards must be provided when using saved terminal rgrad_x1."
                )

            velocity_current = self._compute_velocity(
                xt,
                timestep,
                prompt_embeds,
                pooled_prompt_embeds,
                detach_uncond=True,
            )
            rgrad = saved_rgrad_x1.detach().float()
            rgrad_norm = torch.linalg.norm(rgrad.view(rgrad.size(0), -1), dim=1)
            rgrad = self.reward_grad.clip_gradient(rgrad, rgrad_threshold)
            reward_mask = (rewards >= 0).float()

        # Reference velocity branch: adapter-disabled model.
        velocity_target_raw = self.compute_reference_velocity(
            xt,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
        )

        velocity_target = (
            velocity_target_raw
            - self._eta(sigma) * self.guidance_scale * rgrad.float()
        )

        if self.use_value_net:
            with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
                value_correction = self.value_net.forward(
                    xt, timestep, prompt_embeds, pooled_prompt_embeds
                )
            nabla_v = self._compute_value_gradient(value_correction, rgrad, sigma)
            velocity_target = velocity_target + value_correction
        else:
            value_correction = None
            nabla_v = None

        return velocity_target, {
            "velocity_target_raw": velocity_target_raw,
            "velocity_current": velocity_current,
            "value_correction": value_correction,
            "nabla_V": nabla_v,
            "rgrad": rgrad,
            "rgrad_norm": rgrad_norm,
            "reward_mask": reward_mask,
            "sigma": sigma,
        }

    def compute_reference_velocity(
        self,
        xt,
        timestep,
        prompt_embeds,
        pooled_prompt_embeds,
    ):
        """Compute reference velocity with adapters disabled."""
        transformer = self._transformer_model()
        with torch.inference_mode():
            transformer.disable_adapters()
            velocity_reference = self._compute_velocity(
                xt,
                timestep,
                prompt_embeds,
                pooled_prompt_embeds,
                detach_uncond=False,
            )
            transformer.enable_adapters()
            transformer.set_adapter("default")

        return velocity_reference

    def compute_transition_drift(
        self,
        sigma,
        step_index,
        velocity_current,
        velocity_reference,
        reduce=True,
    ):
        """Compute deterministic transition drift between finetuned and reference flows."""
        next_step_index = torch.clamp(
            step_index + 1, max=self.pipeline.scheduler.sigmas.shape[0] - 1
        )
        sigma_next = self.pipeline.scheduler.sigmas.gather(0, next_step_index).view(
            -1, 1, 1, 1
        )
        dt = sigma_next - sigma
        drift = (
            dt.float() * (velocity_current.float() - velocity_reference.float())
        ).pow(2).mean(dim=(1, 2, 3))

        if reduce:
            return drift.mean()
        return drift

    def compute_value_consistency_loss(
        self,
        xt,
        timestep,
        sigma,
        value_correction,
        nabla_V,
        velocity_target_raw,
        prompt_embeds,
        pooled_prompt_embeds,
        prompts,
        prompt_metadata,
        rgrad_threshold,
    ):
        """
        Compute consistency losses used by value-net training.

        Aligned with finite-difference terms used in vggflow-main:
            del_time = (nabla_V(t-eps) - nabla_V(t)) / eps
            fd1      = (v_ref(x + eps*nabla_V) - v_ref(x)) / eps
            fd2      = (scaled_nabla_V(x + eps*g_dir) - scaled_nabla_V(x)) / eps
        """
        temporal_derivative = self._compute_temporal_derivative(
            xt=xt,
            timestep=timestep,
            current_nabla_v=nabla_V,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompts=prompts,
            prompt_metadata=prompt_metadata,
            rgrad_threshold=rgrad_threshold,
        )

        velocity_derivative = self._compute_velocity_derivative(
            xt=xt,
            timestep=timestep,
            current_nabla_v=nabla_V,
            velocity_target_raw=velocity_target_raw,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        spatial_derivative = self._compute_spatial_derivative(
            xt=xt,
            timestep=timestep,
            sigma=sigma,
            current_nabla_v=nabla_V,
            velocity_target_raw=velocity_target_raw,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            prompts=prompts,
            prompt_metadata=prompt_metadata,
            rgrad_threshold=rgrad_threshold,
        )

        with torch.autocast(dtype=torch.float32, device_type="cuda"):
            residual = temporal_derivative - velocity_derivative - spatial_derivative
            loss_consistency = (residual / self.guidance_scale).float().pow(2).mean()

        is_terminal = timestep == self.pipeline.scheduler.timesteps[-1]
        with torch.autocast(dtype=torch.float32, device_type="cuda"):
            loss_terminal = (
                value_correction.float().pow(2).mean(dim=[1, 2, 3]) * is_terminal.float()
            ).sum() / (is_terminal.float().sum() + 1e-6)

        return loss_consistency, loss_terminal

    def _compute_velocity(
        self,
        xt,
        timestep,
        prompt_embeds,
        pooled_prompt_embeds,
        detach_uncond=True,
    ):
        """Compute CFG velocity from the transformer."""
        latent_input, timestep_input, _, _ = self.cfg.prepare_inputs(
            xt, timestep, prompt_embeds, pooled_prompt_embeds
        )

        velocity = self.transformer(
            hidden_states=latent_input,
            timestep=timestep_input,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=self.pipeline.joint_attention_kwargs,
            return_dict=False,
        )[0]

        return self.cfg.apply(velocity, detach_uncond=detach_uncond)

    def _compute_value_gradient(self, value_correction, rgrad, sigma):
        """Compute nabla_V = V/guidance - eta(sigma) * reward_grad."""
        return (
            value_correction / self.guidance_scale
            - self._eta(sigma) * rgrad.float()
        )

    def _eta(self, sigma):
        if self.eta_mode == "linear":
            return 1 - sigma
        if self.eta_mode == "constant":
            return 1.0
        return (1 - sigma).pow(2)

    def _compute_temporal_derivative(
        self,
        xt,
        timestep,
        current_nabla_v,
        prompt_embeds,
        pooled_prompt_embeds,
        prompts,
        prompt_metadata,
        rgrad_threshold,
    ):
        """Finite difference for d(nabla_V)/dt using t-eps."""
        timestep_prev = timestep - self.eps_time * torch.ones_like(timestep)
        sigma_prev = timestep_prev[: xt.size(0)].view(-1, 1, 1, 1) / 1000.0

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            value_correction_prev = self.value_net.forward(
                xt, timestep_prev, prompt_embeds, pooled_prompt_embeds
            )

        xt_copy = xt.detach()
        xt_copy.requires_grad_(True)

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            velocity_prev = self._compute_velocity(
                xt_copy,
                timestep_prev,
                prompt_embeds,
                pooled_prompt_embeds,
                detach_uncond=True,
            )
        x1_prev = xt - sigma_prev * velocity_prev
        rgrad_prev, _, _ = self.reward_grad.compute_from_latent(
            latent_input=x1_prev,
            latent_requires_grad=xt_copy,
            prompts=prompts,
            prompt_metadata=prompt_metadata,
            rgrad_threshold=rgrad_threshold,
        )

        nabla_v_prev = self._compute_value_gradient(
            value_correction_prev, rgrad_prev, sigma_prev
        )
        return (nabla_v_prev - current_nabla_v) / self.eps_time

    def _compute_spatial_derivative(
        self,
        xt,
        timestep,
        sigma,
        current_nabla_v,
        velocity_target_raw,
        prompt_embeds,
        pooled_prompt_embeds,
        prompts,
        prompt_metadata,
        rgrad_threshold,
    ):
        """Finite difference for scaled nabla_V along g_dir direction."""
        if self.detach_dir:
            g_dir = (velocity_target_raw / self.guidance_scale + current_nabla_v).detach()
        else:
            g_dir = velocity_target_raw / self.guidance_scale + current_nabla_v

        x_perturbed = (xt + g_dir * self.eps_step).detach()
        x_perturbed.requires_grad_(True)

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            value_correction_perturbed = self.value_net.forward(
                x_perturbed, timestep, prompt_embeds, pooled_prompt_embeds
            )

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            velocity_perturbed = self._compute_velocity(
                x_perturbed,
                timestep,
                prompt_embeds,
                pooled_prompt_embeds,
                detach_uncond=True,
            )
        timestep_scale = timestep[: x_perturbed.size(0)].view(-1, 1, 1, 1) / 1000.0
        x1_perturbed = x_perturbed - timestep_scale * velocity_perturbed

        rgrad_perturbed, _, _ = self.reward_grad.compute_from_latent(
            latent_input=x1_perturbed,
            latent_requires_grad=x_perturbed,
            prompts=prompts,
            prompt_metadata=prompt_metadata,
            rgrad_threshold=rgrad_threshold,
            retain_graph=False,
        )

        with torch.autocast(dtype=torch.float32, device_type="cuda"):
            nabla_v_perturbed_scaled = (
                value_correction_perturbed
                - self._eta(sigma) * self.guidance_scale * rgrad_perturbed
            )
            current_scaled = current_nabla_v * self.guidance_scale
            return (nabla_v_perturbed_scaled - current_scaled) / self.eps_step

    def _compute_velocity_derivative(
        self,
        xt,
        timestep,
        current_nabla_v,
        velocity_target_raw,
        prompt_embeds,
        pooled_prompt_embeds,
    ):
        """Finite difference for reference velocity along nabla_V direction."""
        transformer = self._transformer_model()
        transformer.disable_adapters()
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            velocity_inc = self._compute_velocity(
                xt + current_nabla_v * self.eps_step,
                timestep,
                prompt_embeds,
                pooled_prompt_embeds,
                detach_uncond=False,
            )
        transformer.enable_adapters()
        transformer.set_adapter("default")

        with torch.autocast(dtype=torch.float32, device_type="cuda"):
            return (velocity_inc.float() - velocity_target_raw.float()) / self.eps_step
