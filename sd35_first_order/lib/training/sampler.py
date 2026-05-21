"""Trajectory sampling for VGG-Flow training."""

import torch
import tempfile
import os
import numpy as np
from PIL import Image
from functools import partial
import tqdm
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

from lib.diffusion.sample_trajectory_sd3 import sample_trajectory


class TrajectorySampler:
    """Handles trajectory sampling for VGG-Flow."""

    def __init__(self, pipeline, prompt_fn, reward_fn, reward_grad_computer, config):
        """
        Initialize trajectory sampler.

        Args:
            pipeline: Diffusion pipeline
            prompt_fn: Function to generate prompts
            reward_fn: Reward function
            reward_grad_computer: Reward gradient computer
            config: Configuration object
        """
        self.pipeline = pipeline
        self.prompt_fn = prompt_fn
        self.reward_fn = reward_fn
        self.reward_grad = reward_grad_computer
        self.config = config
        self._last_log_images = None
        self._last_log_prompts = None
        self._last_log_rewards = None

    def sample_epoch(self, transformer, epoch, device, is_main_process):
        """
        Sample trajectories for one epoch.

        Args:
            transformer: Current transformer model
            epoch: Current epoch number
            device: Device to use
            is_main_process: Whether this is the main process

        Returns:
            samples: Dict containing sampled trajectories
        """
        samples_list = []

        for batch_idx in tqdm(
            range(self.config.sampling.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process,
            position=0,
        ):
            batch_samples = self._sample_batch(transformer)
            samples_list.append(batch_samples)

        # Wait for all rewards to be computed
        for sample in tqdm(
            samples_list,
            desc="Waiting for rewards",
            disable=not is_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"]
            sample["rewards"] = torch.as_tensor(rewards, device=device)

        # Collate all samples
        samples = self._collate_samples(samples_list)

        return samples

    def _sample_batch(self, transformer):
        """
        Sample a single batch of trajectories.

        Args:
            transformer: Transformer model to use for sampling

        Returns:
            batch_samples: Dict containing batch samples
        """
        # Generate prompts
        prompts_and_metadata = [
            self.prompt_fn(**self.config.experiment.prompt_fn_kwargs)
            for _ in range(self.config.sampling.batch_size)
        ]
        prompts, prompt_metadata = zip(*prompts_and_metadata)
        prompts = list(prompts)

        # Sample trajectory
        outputs = sample_trajectory(
            self.pipeline,
            transformer,
            prompt=prompts,
            negative_prompt=None,
            num_inference_steps=self.config.sampling.num_steps,
            guidance_scale=self.config.sampling.guidance_scale,
            output_type="image",
            return_output=self.config.model.unet_reg_scale > 0.0,
        )

        (images, latents, timesteps, prompt_embeds, pooled_prompt_embeds,
         negative_prompt_embeds, negative_pooled_prompt_embeds, unet_outputs) = outputs

        # Stack latents along time dimension
        latents = torch.stack(latents, dim=1)  # (B, T+1, C, H, W)

        if self.config.model.unet_reg_scale > 0.0:
            unet_outputs = torch.stack(unet_outputs, dim=1)  # (B, T, C, H, W)

        # Prepare timesteps
        timesteps = self.pipeline.scheduler.timesteps.repeat(
            self.config.sampling.batch_size, 1
        )  # (B, T)

        # Create step indices
        step_index = torch.arange(
            timesteps.size(1),
            device=timesteps.device,
            dtype=torch.int64
        ).view(1, -1).expand(timesteps.size(0), -1)

        # Compute rewards
        rewards = self.reward_fn(images.float(), prompts, prompt_metadata)
        rewards_value = rewards[0] if isinstance(rewards, (tuple, list)) else rewards
        if torch.is_tensor(rewards_value):
            rewards_value = rewards_value.detach().float().cpu().tolist()

        self._last_log_images = images.detach().cpu()
        self._last_log_prompts = list(prompts)
        self._last_log_rewards = rewards_value

        # Package samples
        batch_samples = {
            "prompts": prompts,
            "prompt_metadata": prompt_metadata,
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "timesteps": timesteps,
            "latents": latents,
            "rewards": rewards,
            "step_index": step_index,
        }

        if self.config.training.use_saved_rgrad_x1:
            x1_denoised = latents[:, -1].detach()
            x1_denoised.requires_grad_(True)
            rgrad_x1, _, _ = self.reward_grad.compute_from_latent(
                latent_input=x1_denoised,
                latent_requires_grad=x1_denoised,
                prompts=prompts,
                prompt_metadata=prompt_metadata,
                rgrad_threshold=float("inf"),
                retain_graph=False,
            )
            batch_samples["rgrad_x1"] = rgrad_x1.detach().float()

        if self.config.model.unet_reg_scale > 0.0:
            batch_samples["unet_outputs"] = unet_outputs

        return batch_samples

    def get_last_log_data(self):
        """Return last sampled images/prompts/rewards for logging."""
        if self._last_log_images is None:
            return None
        return self._last_log_images, self._last_log_prompts, self._last_log_rewards

    def _collate_samples(self, samples_list):
        """
        Collate list of sample dicts into single dict.

        Args:
            samples_list: List of sample dicts

        Returns:
            collated_samples: Single dict with concatenated tensors
        """
        collated = {}

        for key in samples_list[0].keys():
            if key in ["prompts", "prompt_metadata"]:
                # Flatten list of tuples into single list
                collated[key] = [
                    item for sample in samples_list for item in sample[key]
                ]
            else:
                # Concatenate tensors
                collated[key] = torch.cat([s[key] for s in samples_list])

        return collated

    def log_samples(self, images, prompts, rewards, global_step, wandb, is_main_process):
        """
        Log sample images to wandb.

        Args:
            images: Generated images (B, 3, H, W)
            prompts: Text prompts
            rewards: Reward values
            global_step: Current global step
            wandb: Wandb module
            is_main_process: Whether this is the main process
        """
        if not self.config.logging.use_wandb or not is_main_process:
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray(
                    (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))

            wandb.log(
                {
                    "epoch": global_step,
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt} | {reward:.2f}",
                        )
                        for i, (prompt, reward) in enumerate(zip(prompts, rewards))
                    ],
                },
                step=global_step,
            )
