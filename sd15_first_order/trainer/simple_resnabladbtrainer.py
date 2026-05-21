import os
import time
import torch
import wandb
import numpy as np

import torch.distributed as dist

from collections import defaultdict
from packaging import version
from peft import get_peft_model_state_dict, LoraConfig, set_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.utils.import_utils import is_xformers_available
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import cast_training_params

from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors.torch import load_file

from utils.diffusers_patch.ddim_step import ddim_step_with_logprob, pred_orig_latent, get_alpha_prod_t
from trainer.ddptrainer import unwrap_model, DDPTrainer


class SimpleResNablaDBTrainer(DDPTrainer):
    def __init__(self, config_path):
        super().__init__(config_path)
        self.config['use_pcpo'] = False
        self.run_name_prefix = "resnabladb-db"

    def _setup_pipeline(self):
        config = self.config

        # Load scheduler, tokenizer and models.
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config['model']['pretrained'], subfolder="scheduler", revision=config['model']['revision']
        )
        self.noise_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config) # switch to DDIM
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config['model']['pretrained'], subfolder="tokenizer", revision=config['model']['revision']
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            config['model']['pretrained'], subfolder="unet", revision=config['model']['revision'], variant=config['model']['variant']
        )
        self.vae = AutoencoderKL.from_pretrained(
            config['model']['pretrained'], subfolder="vae", revision=config['model']['revision'], variant=config['model']['variant']
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            config['model']['pretrained'], subfolder="text_encoder", revision=config['model']['revision']
        )

        # Freeze parameters of models to save more memory
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)

        for name, param in self.unet.named_parameters():
            param.requires_grad_(False)
        self.unet.to(self.device, dtype=self.weight_dtype)
        unet_lora_config = LoraConfig(
            r=config['model']['lora_rank'],
            lora_alpha=config['model']['lora_alpha'],
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        self.unet.add_adapter(unet_lora_config, adapter_name="pf") ## LoRA

        use_xformers = bool(config.get("model", {}).get("use_xformers", True))
        if use_xformers:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    if self.is_main_process:
                        self.logger.warning(
                            "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                        )
                try:
                    self.unet.enable_xformers_memory_efficient_attention()
                    if self.is_main_process:
                        self.logger.info("xformers is enabled for memory efficient attention")
                except Exception as exc:
                    if self.is_main_process:
                        self.logger.warning(
                            "xformers is installed but could not be enabled; falling back to default attention. "
                            f"error={type(exc).__name__}: {exc}"
                        )
            else:
                if self.is_main_process:
                    self.logger.warning(
                        "xformers is not available; falling back to default attention."
                    )
        elif self.is_main_process:
            self.logger.info("xformers is disabled by config (model.use_xformers=false).")

        self.unet.set_adapter("pf")

        # upcast trainable params into fp32
        if config['mixed_precision'] in ['fp16', 'bf16']:
            cast_training_params(self.unet, dtype=torch.float32)

        if config['train']['gradient_checkpointing']:
            self.unet.enable_gradient_checkpointing()

        self.unet.to(self.device)
        self.unet = DDP(self.unet, device_ids=[self.local_rank])

    def _setup_optimizer(self):
        """
        ResNablaDB: Setup GradScaler too
        """
        config = self.config
        unet = self.unet

        pf_params = [param for name, param in unet.named_parameters() if '.pf.' in name]
        params = [
            {"params": pf_params, "lr": config['train']['lr']},
        ]

        if config['train']['adam']['use_8bit']:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("Run `pip install bitsandbytes` to use 8-bit Adam.")
            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            params,
            betas=(config['train']['adam']['beta1'], config['train']['adam']['beta2']),
            weight_decay=config['train']['adam']['weight_decay'],
            eps=config['train']['adam']['epsilon'],
        )

        if self.config['mixed_precision'] in ['fp16', 'bf16']:
            self.scaler = torch.amp.GradScaler(
                'cuda',
                growth_interval=self.config['train']['gradscaler_growth_interval']
            )

    def load_from_checkpoint(self):
        config = self.config
        resume_from = config['model']['resume_from_checkpoint']

        if resume_from is None:
            return 0

        if self.is_main_process:
            self.logger.info(f"Resuming from checkpoint: {resume_from}")

        dirname = os.path.basename(os.path.normpath(resume_from))
        if "checkpoint_epoch" in dirname:
            epoch = int(dirname.replace("checkpoint_epoch", ""))
        else:
            raise ValueError(f"Cannot parse epoch from {dirname}")

        unwrapped_unet = unwrap_model(self.unet)
        lora_path = os.path.join(resume_from, "unet_lora_weights.safetensors")
        state_dict = load_file(lora_path)
        # The checkpoint has 'unet.' prefix, but the model does not. We must strip it.
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("unet."):
                new_state_dict[k.replace("unet.", "")] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        peft_state_dict = convert_unet_state_dict_to_peft(state_dict)

        incompatible_keys = set_peft_model_state_dict(unwrapped_unet, peft_state_dict, adapter_name="pf")
        if self.is_main_process:
            if incompatible_keys is not None:
                # Handle varying return types (tuple vs object)
                if isinstance(incompatible_keys, tuple):
                    missing, unexpected = incompatible_keys
                else:
                    missing = incompatible_keys.missing_keys
                    unexpected = incompatible_keys.unexpected_keys
                
                if len(unexpected) > 0:
                    self.logger.warning(f"⚠️ LoRA UNEXPECTED KEYS ({len(unexpected)}): {unexpected[:5]}...")
                lora_missing = [k for k in missing if 'lora' in k or '.pf.' in k]
                
                if len(lora_missing) > 0:
                     self.logger.warning(f"⚠️ LoRA MISSING KEYS ({len(lora_missing)}): {lora_missing[:5]}...")
                else:
                     # If the only missing keys were base model weights (conv_in, etc), that's success.
                     self.logger.info(f"Loaded UNet LoRA weights from {resume_from} (Base weights skipped as expected).")
            else:
                self.logger.info(f"Loaded UNet LoRA weights from {resume_from} (Return value was None).")

        conv_out_path = os.path.join(resume_from, f"conv_out_weights_epoch{epoch}.pt")
        conv_out_weights = torch.load(conv_out_path, map_location=self.device)
        conv_load_result = unwrapped_unet.conv_out.load_state_dict(conv_out_weights, strict=False)
        if self.is_main_process:
            if len(conv_load_result.missing_keys) > 0:
                self.logger.warning(f"⚠️ conv_out MISSING keys: {conv_load_result.missing_keys}")
            elif len(conv_load_result.unexpected_keys) > 0:
                self.logger.warning(f"⚠️ conv_out UNEXPECTED keys: {conv_load_result.unexpected_keys}")
            else:
                self.logger.info(f"Loaded conv_out weights from {conv_out_path}.")

        return epoch

    def save_checkpoint(self, epoch, global_step):
        config = self.config

        if epoch % config['save_freq'] == 0 or epoch == config['train']['max_epochs'] - 1:
            if self.is_main_process:
                save_path = os.path.join(self.output_dir, f"checkpoint_epoch{epoch}")

                # unet (lora)
                unwrapped_unet = unwrap_model(self.unet)
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_unet, adapter_name="pf")
                )
                StableDiffusionPipeline.save_lora_weights(
                    save_directory=save_path,
                    unet_lora_layers=unet_lora_state_dict,
                    is_main_process=True,
                    safe_serialization=True,
                    weight_name="unet_lora_weights.safetensors"
                )
                self.logger.info(f"Saved UNet LoRA weights to {save_path}")

                # conv_out
                conv_out_weights = unwrapped_unet.conv_out.state_dict()  # Extract only the weights of the conv_out layer
                torch.save(
                    conv_out_weights,
                    os.path.join(save_path, f"conv_out_weights_epoch{epoch}.pt")
                )
                self.logger.info(f"Saved state to {save_path}")

    def policy_update(self, samples, epoch, epoch_start_time, global_step):
        config = self.config
        train_cfg = config['train']
        use_z0 = train_cfg.get('use_z0', False)
        normalize_unet_reg_by_one_minus_alpha_bar_t = train_cfg.get(
            'normalize_unet_reg_by_one_minus_alpha_bar_t', False
        )
        pretrained_strength_mode = train_cfg['pretrained_strength']
        if isinstance(pretrained_strength_mode, str):
            pretrained_strength_mode = pretrained_strength_mode.strip().lower()

        reward_adaptive_mode = train_cfg['reward_adaptive_mode']
        if isinstance(reward_adaptive_mode, str):
            reward_adaptive_mode = reward_adaptive_mode.strip().lower()

        if pretrained_strength_mode == '3sigma':
            raise ValueError(
                "train.pretrained_strength='3sigma' is no longer supported in SimpleResNablaDBTrainer. "
                "Use train.pretrained_strength='sigma' and train.pretrained_strength_sigma_scale=3.0 instead."
            )
        if reward_adaptive_mode == '3sigma':
            raise ValueError(
                "train.reward_adaptive_mode='3sigma' is no longer supported in SimpleResNablaDBTrainer. "
                "Use train.reward_adaptive_mode='sigma' and train.reward_adaptive_sigma_scale=3.0 instead."
            )
        supported_reward_adaptive_modes = (
            'squared',
            'sigma',
            'sigma_times_oldgamma',
            'constant',
        )
        if not isinstance(reward_adaptive_mode, str):
            raise ValueError(
                "train.reward_adaptive_mode must be one of "
                + ", ".join([f"'{mode}'" for mode in supported_reward_adaptive_modes])
                + "."
            )
        if reward_adaptive_mode not in supported_reward_adaptive_modes:
            raise ValueError(
                "Unsupported train.reward_adaptive_mode "
                f"'{reward_adaptive_mode}'. Supported values are "
                + ", ".join([f"'{mode}'" for mode in supported_reward_adaptive_modes])
                + "."
            )

        def _parse_positive_finite_scale(name, value):
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a positive finite float.")
            try:
                scale = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{name} must be a positive finite float.")
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"{name} must be a positive finite float.")
            return scale

        pretrained_strength_sigma_scale = _parse_positive_finite_scale(
            'train.pretrained_strength_sigma_scale',
            train_cfg.get('pretrained_strength_sigma_scale', 1.0),
        )
        reward_adaptive_sigma_scale = _parse_positive_finite_scale(
            'train.reward_adaptive_sigma_scale',
            train_cfg.get('reward_adaptive_sigma_scale', 1.0),
        )
        unet = self.unet
        noise_scheduler = self.noise_scheduler
        optimizer = self.optimizer
        vae = self.vae

        def decode(latents, clamp=True):
            image = vae.decode(
                latents / vae.config.scaling_factor, return_dict=False
            )[0]
            image = image / 2.0 + 0.5
            if clamp:
                image = image.clamp(0, 1)
            return image

        def _apply_reward_adaptive_scaling(score_r_next, alpha_prod_next, reward_sigma):
            if reward_adaptive_mode == 'squared':
                return score_r_next * alpha_prod_next
            if reward_adaptive_mode in ('sigma', 'sigma_times_oldgamma'):
                if reward_sigma is None:
                    raise ValueError(
                        "`reward_sigma` is required when `reward_adaptive_mode` is "
                        "'sigma' or 'sigma_times_oldgamma'."
                    )
                sigma_scaled = reward_adaptive_sigma_scale * reward_sigma
                if reward_adaptive_mode == 'sigma':
                    return score_r_next * sigma_scaled
                return score_r_next * alpha_prod_next * sigma_scaled
            if reward_adaptive_mode == 'constant':
                return score_r_next
            raise ValueError(
                "Unsupported train.reward_adaptive_mode "
                f"'{reward_adaptive_mode}'. Supported values are "
                + ", ".join([f"'{mode}'" for mode in supported_reward_adaptive_modes])
                + "."
            )

        def _compute_score_r_next(
            latent_next_tmp,
            timestep_next,
            embeds,
            prompts,
            prompt_metadata,
            sample_z0=None,
            reward_sigma=None,
        ):
            if use_z0:
                if sample_z0 is None:
                    raise ValueError("`use_z0=True` requires `sample['z0']` for reward gradient computation.")

                z0 = sample_z0.detach().clone()
                z0.requires_grad_()
                pred_xdata_next = decode(z0).float()
                with torch.amp.autocast('cuda', enabled=False):
                    logr_next_tmp = self.reward_fn(pred_xdata_next, prompts, prompt_metadata)[0]
                    nabla_r = torch.autograd.grad(
                        outputs=logr_next_tmp.sum(),
                        inputs=z0,
                        retain_graph=False,
                        create_graph=False,
                    )[0].detach()
                    score_r_next = config['train']['reward_scale'] * nabla_r
                    alpha_prod_next = get_alpha_prod_t(noise_scheduler, timestep_next, z0)
                    z0 = None
                    score_r_next = _apply_reward_adaptive_scaling(
                        score_r_next=score_r_next,
                        alpha_prod_next=alpha_prod_next,
                        reward_sigma=reward_sigma,
                    )
                nabla_r = None
                return score_r_next

            if latent_next_tmp is None:
                raise ValueError("`use_z0=False` requires `latent_next_tmp` for reward gradient computation.")

            noise_pred_next_tmp = unet(
                torch.cat([latent_next_tmp] * 2),
                torch.cat([timestep_next] * 2),
                embeds,
            ).sample
            noise_pred_uncond_next_tmp, noise_pred_next_text_tmp = noise_pred_next_tmp.chunk(2)
            noise_pred_next_tmp = (
                    noise_pred_uncond_next_tmp
                    + config['sample']['cfg_scale']
                    * (noise_pred_next_text_tmp - noise_pred_uncond_next_tmp)
            )
            noise_pred_uncond_next_tmp = noise_pred_next_text_tmp = None

            pred_z0_next = pred_orig_latent(
                noise_scheduler,
                noise_pred_next_tmp,
                latent_next_tmp,
                timestep_next,
            )
            pred_xdata_next = decode(pred_z0_next).float()
            with torch.amp.autocast('cuda', enabled=False):
                logr_next_tmp = self.reward_fn(pred_xdata_next, prompts, prompt_metadata)[0]
                nabla_r = torch.autograd.grad(
                    outputs=logr_next_tmp.sum(),    # The value whose gradient we want
                    inputs=latent_next_tmp,         # The intermediate node we want the gradient with respect to
                    retain_graph=False,             # Retain graph for further gradient computations
                    create_graph=False              # If higher-order gradients are needed
                )[0].detach()
                score_r_next = config['train']['reward_scale'] * nabla_r
                alpha_prod_next = get_alpha_prod_t(noise_scheduler, timestep_next, latent_next_tmp)
                latent_next_tmp = None
                score_r_next = _apply_reward_adaptive_scaling(
                    score_r_next=score_r_next,
                    alpha_prod_next=alpha_prod_next,
                    reward_sigma=reward_sigma,
                )
            noise_pred_next_tmp = None
            nabla_r = None
            return score_r_next

        for inner_epoch in range(config['train']['num_inner_epochs']):
            samples_batched = self.shuffle_and_batch_samples(samples)

            unet.train()
            info = defaultdict(list)

            grad_stats = {'mean': [], 'var': []}

            # Define gradient hook to capture statistics
            def grad_hook(grad):
                if grad is not None:
                    grad_flat = grad.detach().flatten().to(torch.float32)
                    grad_abs = torch.abs(grad_flat)  # Take absolute value
                    grad_stats['mean'].append(grad_abs.mean().item())
                    grad_stats['var'].append(grad_abs.var().item())
                return None

            # Register grad hook on first LoRA layer with gradients
            hook_handle = None
            for name, param in unet.named_parameters():
                if param.requires_grad and 'lora' in name.lower():
                    hook_handle = param.register_hook(grad_hook)
                    self.logger.info(f"Registered gradient hook on: {name}")
                    break

            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}, inner epoch {inner_epoch}: training",
                position=0,
                disable=not self.is_main_process,
            ):
                if config['train']['cfg']:
                    embeds = torch.cat(
                        [self.train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                ddpm_weights = self.get_ddpm_weights(sample)
                ddpm_Omegas = self.get_ddpm_Omegas(sample)
                ddpm_sigmas = self.get_ddpm_sigmas(sample)
                if use_z0 and "z0" not in sample:
                    raise KeyError("`train.use_z0=True` requires `samples['z0']`, but it is missing.")

                timestep_pbar = tqdm(
                    range(self.num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not self.is_main_process,
                )
                for j in timestep_pbar:
                    if pretrained_strength_mode == 'sigma':
                        strength_t = pretrained_strength_sigma_scale * ddpm_sigmas[:, j]
                    else:
                        strength_t = self.resolve_pretrained_strength(
                            pretrained_strength_config=pretrained_strength_mode,
                            sigma_t=ddpm_sigmas[:, j],
                        )
                    latent_tmp = sample["latents"][:, j].detach().clone()
                    latent_tmp.requires_grad_()

                    # Before inference, disable the LoRA adapters
                    unet.module.disable_adapters()  # This should deactivate any applied LoRA adapter

                    noise_pred_ref = unet(
                        torch.cat([latent_tmp] * 2),
                        torch.cat([sample["timesteps"][:, j]] * 2),
                        embeds,
                    ).sample
                    noise_pred_uncond_ref, noise_pred_text_ref = noise_pred_ref.chunk(2)
                    noise_pred_ref = (
                            noise_pred_uncond_ref
                            + config['sample']['cfg_scale']
                            * (noise_pred_text_ref - noise_pred_uncond_ref)
                    )
                    noise_pred_uncond_ref = noise_pred_text_ref = None

                    with torch.inference_mode():
                        _, _, _, Omega_div_sigmaSquared_times_score_ref = ddim_step_with_logprob(
                            noise_scheduler,
                            noise_pred_ref,
                            None,
                            sample["timesteps"][:, j],
                            sample["latents"][:, j],
                            num_inference_steps=self.num_inference_steps,
                            eta=config['sample']['eta'],
                            prev_sample=sample["next_latents"][:, j],
                            strength=strength_t,
                        )

                    noise_pred_ref_saved = noise_pred_ref.detach()
                    noise_pred_ref = None

                    # Re-enable LoRA configuration
                    unet.module.enable_adapters()
                    unet.module.set_adapter("pf")

                    noise_pred = unet(
                        torch.cat([latent_tmp] * 2),
                        torch.cat([sample["timesteps"][:, j]] * 2),
                        embeds,
                    ).sample
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = (
                            noise_pred_uncond
                            + config['sample']['cfg_scale']
                            * (noise_pred_text - noise_pred_uncond)
                    )
                    noise_pred_uncond = noise_pred_text = None

                    unetdiff = (noise_pred - sample["noise_preds"][:, j]).pow(2)
                    unetreg = torch.mean(unetdiff, dim=(1, 2, 3))
                    if normalize_unet_reg_by_one_minus_alpha_bar_t:
                        eps = 1e-20
                        alpha_bar_t = noise_scheduler.alphas_cumprod.gather(
                            0, sample["timesteps"][:, j].cpu()
                        ).to(sample["timesteps"][:, j].device)
                        denom_t = 1 - alpha_bar_t

                        scheduler_dt = noise_scheduler.timesteps[0] - noise_scheduler.timesteps[1]
                        timestep_prev = torch.clamp(
                            sample["timesteps"][:, j] - scheduler_dt, min=0
                        )
                        alpha_bar_prev = noise_scheduler.alphas_cumprod.gather(
                            0, timestep_prev.cpu()
                        ).to(timestep_prev.device)
                        denom_prev = 1 - alpha_bar_prev

                        denom = torch.where(denom_t > eps, denom_t, denom_prev)
                        denom = torch.clamp(denom, min=eps)
                        unetreg = unetreg / denom
                    unetdiffnorm = unetdiff.sum(dim=(1, 2, 3)).sqrt()

                    _, _, _, Omega_div_sigmaSquared_times_score_theta = ddim_step_with_logprob(
                        noise_scheduler,
                        noise_pred,
                        None,
                        sample["timesteps"][:, j],
                        latent_tmp,
                        num_inference_steps=self.num_inference_steps,
                        eta=config['sample']['eta'],
                        prev_sample=sample["next_latents"][:, j],
                        strength=strength_t,
                    )
                    _ = None

                    scheduler_dt = noise_scheduler.timesteps[0] - noise_scheduler.timesteps[1]
                    timestep_next = torch.clamp(
                        sample["timesteps"][:, j] - scheduler_dt, min=0
                    )

                    if use_z0:
                        latent_next_tmp = None
                        sample_z0 = sample["z0"]
                    else:
                        latent_next_tmp = sample["next_latents"][:, j].detach().clone()
                        latent_next_tmp.requires_grad_()
                        sample_z0 = None
                    unet.module.set_adapter("pf")

                    prompts = sample["prompts"]
                    prompt_metadata = sample["prompt_metadata"]
                    score_r_next = _compute_score_r_next(
                        latent_next_tmp,
                        timestep_next,
                        embeds,
                        prompts,
                        prompt_metadata,
                        sample_z0=sample_z0,
                        reward_sigma=ddpm_sigmas[:, j],
                    )

                    with torch.inference_mode():
                        grad_norm_score_ref = Omega_div_sigmaSquared_times_score_ref.pow(2).sum(dim=[1,2,3]).sqrt()
                        grad_norm_res_score = (Omega_div_sigmaSquared_times_score_theta- Omega_div_sigmaSquared_times_score_ref).pow(2).sum(dim=[1,2,3]).sqrt()
                        grad_norm_score_r = score_r_next.pow(2).sum(dim=[1,2,3]).sqrt()
                        ddpm_weight = ddpm_weights[:, j]
                        batch_approx_kl = 0.5 * torch.mean((ddpm_weight * (noise_pred.detach() - noise_pred_ref_saved)) ** 2, dim=[1, 2, 3])
                    
                    noise_pred = noise_pred_ref_saved = None

                    losses_forward = (Omega_div_sigmaSquared_times_score_theta - Omega_div_sigmaSquared_times_score_ref.float() - score_r_next.float()).pow(2)
                    Omega_div_sigmaSquared_times_score_theta = Omega_div_sigmaSquared_times_score_ref = score_r_next = None
                    loss_forward_mean = losses_forward.mean()

                    losses = (config['train']['forward_loss_scale'] * losses_forward).mean()
                    losses = losses + config['train']['unet_reg_scale'] * unetreg.mean()
                    loss = torch.mean(losses)
                    accumulation_steps = config['train']['gradient_accumulation_steps'] * self.num_train_timesteps
                    loss = loss / accumulation_steps
                    if self.scaler:
                        # Backward passes under autocast are not recommended
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    #### Log
                    info["loss_forward"].append(loss_forward_mean.detach())
                    info["approx_kl"].append(batch_approx_kl.mean())

                    with torch.inference_mode():
                        info["norm_score_ref_mean"].append(grad_norm_score_ref.mean())
                        info["norm_score_ref_min"].append(grad_norm_score_ref.min())
                        info["norm_score_ref_max"].append(grad_norm_score_ref.max())
                        info["norm_score_residual_mean"].append(grad_norm_res_score.mean())
                        info["norm_score_residual_min"].append(grad_norm_res_score.min())
                        info["norm_score_residual_max"].append(grad_norm_res_score.max())
                        info["norm_score_r_mean"].append(grad_norm_score_r.mean())
                        info["norm_score_r_min"].append(grad_norm_score_r.min())
                        info["norm_score_r_max"].append(grad_norm_score_r.max())
                        info["norm_unet_diff_mean"].append(unetdiffnorm.mean())
                        info["norm_unet_diff_min"].append(unetdiffnorm.min())
                        info["norm_unet_diff_max"].append(unetdiffnorm.max())

                    info["losses_forward_max"].append(losses_forward.max())
                    info["unetreg"].append(unetreg.mean().detach())


                    # prevent OOM
                    noise_pred_uncond = noise_pred_text = noise_pred = None
                    logr_next_tmp = None
                    unetreg = losses =  None
                    score_pf_ref = score_pf = None
                    score_r_next = score_r_next_tmp = None
                    noise_pred_uncond_ref = noise_pred_text_ref = noise_pred_ref = None
                    score_pf_target = None

                if ((j == self.num_train_timesteps - 1) and
                        (i + 1) % config['train']['gradient_accumulation_steps'] == 0):
                    if self.scaler:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for name, p in unet.named_parameters() if '.pf.' in name],
                            config['train']['max_grad_norm']
                        )
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            [p for name, p in unet.named_parameters() if '.pf.' in name],
                            config['train']['max_grad_norm']
                        )
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    ### avoid memory leak
                    for param in unet.parameters():
                        param.grad = None

                    old_info = info
                    info = {}
                    for k, v in old_info.items():
                        if '_min' in k:
                            info[k] = torch.min(torch.stack(v))
                        elif '_max' in k:
                            info[k] = torch.max(torch.stack(v))
                        else:
                            info[k] = torch.mean(torch.stack(v))

                    dist.barrier()
                    for k, v in info.items():
                        if '_min' in k:
                            dist.all_reduce(v, op=dist.ReduceOp.MIN)
                        elif '_max' in k:
                            dist.all_reduce(v, op=dist.ReduceOp.MAX)
                        else:
                            dist.all_reduce(v, op=dist.ReduceOp.SUM)
                    info = {
                        k: v / self.num_processes
                        if ('_min' not in k and '_max' not in k)
                        else v for k, v in info.items()
                    }

                    info.update({"epoch": epoch})
                    info.update({"global_step": global_step})

                    if self.is_main_process:
                        if self.scaler:
                            info.update({"grad_scale": self.scaler.get_scale()})
                        if len(grad_stats['mean']) > 0:
                            info.update({
                                "grad_mean": np.mean(grad_stats['mean']),
                                "grad_var": np.mean(grad_stats['var']),
                            })
                            # Reset
                            grad_stats = {'mean': [], 'var': []}
                        wandb.log(info, step=global_step)
                        self.logger.info(
                            f"global_step={global_step}  " +
                            " ".join([f"{k}={v:.6f}" for k, v in info.items()])
                        )
                    info = defaultdict(list) # reset info dict
                    global_step += 1
            
            if hook_handle is not None:
                hook_handle.remove()
                self.logger.info("Removed gradient hook.")

        dist.barrier()
        total_epoch_time = time.time() - epoch_start_time
        if self.is_main_process:
            wandb.log(
                {
                    "total_time_per_epoch": total_epoch_time,
                },
                step=global_step,
            )
        return global_step
