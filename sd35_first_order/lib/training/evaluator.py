"""Periodic evaluation utilities for VGG-Flow training."""

from functools import partial

import torch
import torch.distributed as dist
import tqdm

import lib.reward_func.prompts_eval
from lib.diffusion.sample_trajectory_sd3 import sample_trajectory

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


class PeriodicEvaluator:
    """Runs deterministic prompt-set evaluation during training."""

    def __init__(self, pipeline, reward_fn, vggflow_algorithm, config):
        self.pipeline = pipeline
        self.reward_fn = reward_fn
        self.algorithm = vggflow_algorithm
        self.config = config
        self.eval_items = self._load_eval_items()

    def evaluate(self, transformer, device, is_main_process):
        """Run evaluation across all ranks and return reduced metrics."""
        transformer.eval()
        local_items = self._get_local_items()

        reward_sum = torch.zeros(1, device=device, dtype=torch.float64)
        reward_count = torch.zeros(1, device=device, dtype=torch.float64)
        drift_sum = torch.zeros(1, device=device, dtype=torch.float64)
        drift_count = torch.zeros(1, device=device, dtype=torch.float64)

        for start in tqdm(
            range(0, len(local_items), self.config.sampling.batch_size),
            desc="Periodic eval",
            disable=not is_main_process,
            position=0,
        ):
            batch_items = local_items[start:start + self.config.sampling.batch_size]
            if not batch_items:
                continue

            prompts = [prompt for prompt, _ in batch_items]
            prompt_metadata = [metadata for _, metadata in batch_items]

            outputs = sample_trajectory(
                self.pipeline,
                transformer,
                prompt=prompts,
                negative_prompt=None,
                num_inference_steps=self.config.sampling.num_steps,
                guidance_scale=self.config.sampling.guidance_scale,
                output_type="image",
                return_output=False,
            )

            (
                images,
                latents,
                timesteps,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                _,
            ) = outputs

            rewards = self._extract_rewards(
                self.reward_fn(images.float(), prompts, prompt_metadata),
                device,
            )
            reward_sum += rewards.double().sum()
            reward_count += rewards.numel()
            images = None

            batch_drift_sum, batch_drift_count = self._compute_transition_drift(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
            )
            drift_sum += batch_drift_sum
            drift_count += batch_drift_count

        stats = torch.cat([reward_sum, reward_count, drift_sum, drift_count])
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        reward_mean = stats[0] / stats[1].clamp_min(1.0)
        transition_drift = stats[2] / stats[3].clamp_min(1.0)

        return {
            "eval_reward_mean": reward_mean.item(),
            "eval_transition_drift": transition_drift.item(),
        }

    def _load_eval_items(self):
        prompt_fn = getattr(lib.reward_func.prompts_eval, self.config.eval.prompt_fn)
        raw_items = prompt_fn(**self.config.eval.prompt_fn_kwargs)
        return self._normalize_prompt_items(raw_items)

    def _normalize_prompt_items(self, raw_items):
        if isinstance(raw_items, tuple) and len(raw_items) == 2:
            prompt_items, shared_metadata = raw_items
        else:
            prompt_items, shared_metadata = raw_items, {}

        normalized = []
        for item in prompt_items:
            if isinstance(item, tuple) and len(item) == 2:
                prompt, metadata = item
            else:
                prompt, metadata = item, shared_metadata
            normalized.append((prompt, dict(metadata)))

        return normalized

    def _get_local_items(self):
        if not dist.is_initialized():
            return self.eval_items

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return self.eval_items[rank::world_size]

    def _extract_rewards(self, reward_output, device):
        rewards = reward_output[0] if isinstance(reward_output, (tuple, list)) else reward_output
        return torch.as_tensor(rewards, device=device, dtype=torch.float32).reshape(-1)

    def _compute_transition_drift(
        self,
        latents,
        timesteps,
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        device,
    ):
        timesteps = timesteps.to(device=device)
        batch_size = latents[0].size(0)
        step_index = torch.arange(
            timesteps.size(0), device=device, dtype=torch.int64
        )

        if self.algorithm.cfg.do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        drift_sum = torch.zeros(1, device=device, dtype=torch.float64)
        drift_count = torch.zeros(1, device=device, dtype=torch.float64)

        with torch.inference_mode():
            for step_idx in range(timesteps.size(0)):
                xt = latents[step_idx]
                timestep = timesteps[step_idx].expand(batch_size)
                step_ids = step_index[step_idx].expand(batch_size)
                sigma = self.pipeline.scheduler.sigmas.gather(0, step_ids).view(-1, 1, 1, 1)

                velocity_current = self.algorithm._compute_velocity(
                    xt=xt,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    detach_uncond=True,
                )
                velocity_reference = self.algorithm.compute_reference_velocity(
                    xt=xt,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )
                transition_drift = self.algorithm.compute_transition_drift(
                    sigma=sigma,
                    step_index=step_ids,
                    velocity_current=velocity_current,
                    velocity_reference=velocity_reference,
                    reduce=False,
                )
                drift_sum += transition_drift.double().sum()
                drift_count += transition_drift.numel()

        return drift_sum, drift_count
