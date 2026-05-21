from collections import defaultdict
import contextlib
import os
import datetime
import hashlib
from numbers import Integral
from concurrent import futures
import time
import h5py
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob_perstep import pipeline_with_logprob as pipeline_with_logprob_perstep
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.sd3_train_shared import (
    VALID_LOSS_TYPES,
    VALID_REWEIGHT_TYPES,
    combine_matching_terms_by_reweight,
    combine_ppo_terms_by_reweight,
    calculate_zero_std_ratio,
    compute_matching_weight_by_reweight,
    compute_ppo_ratio_by_reweight,
    save_ckpt,
)
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from dataset.sd35_dataset import PrecomputedEmbeddingDataset

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class KRepeatSampler(Sampler):
    # adopted to tempflow-sampling, from sampler in scripts/train_sd3.py
    def __init__(self, dataset, batch_size, image_per_prompt, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_per_prompt = image_per_prompt
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        if self.total_samples % self.image_per_prompt != 0:
            raise ValueError(
                f"image_per_prompt must divide num_replicas * batch_size, got "
                f"{self.image_per_prompt} vs {self.num_replicas} * {self.batch_size}"
            )
        self.prompt_per_step = self.total_samples // self.image_per_prompt
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g)[: self.prompt_per_step].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.image_per_prompt)]
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_rank_batches = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_rank_batches.append(shuffled_samples[start:end])
            yield per_rank_batches[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def create_seed_ids(prompts, base_seed, rank, device):
    seed_ids = []
    for idx, prompt in enumerate(prompts):
        seed_key = f"{prompt}|{base_seed}|{rank}|{idx}"
        hash_digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
        seed_id = int.from_bytes(hash_digest[:8], "big", signed=False) % (2**63 - 1)
        seed_ids.append(seed_id)
    return torch.as_tensor(seed_ids, device=device, dtype=torch.long)


def normalize_exploration_schedule(exploration_k, num_inference_steps):
    num_inference_steps = int(num_inference_steps)
    if num_inference_steps < 2:
        raise ValueError(
            "num_inference_steps must be >= 2 so at least one transition exists, got "
            f"{num_inference_steps}."
        )

    num_transitions = num_inference_steps - 1
    if isinstance(exploration_k, Integral):
        k = int(exploration_k)
        if k < 1:
            raise ValueError(f"exploration_k must be >= 1, got {k}.")
        return [k] * num_transitions

    if isinstance(exploration_k, (list, tuple)):
        if len(exploration_k) != num_inference_steps:
            raise ValueError(
                "When exploration_k is a list, its length must equal sample.num_steps "
                f"({num_inference_steps}), got {len(exploration_k)}."
            )
        schedule = []
        for i, value in enumerate(exploration_k[:-1]):
            if not isinstance(value, Integral):
                raise ValueError(
                    f"exploration_k[{i}] must be an integer, got {type(value).__name__}."
                )
            k = int(value)
            if k < 0:
                raise ValueError(f"exploration_k[{i}] must be >= 0, got {k}.")
            schedule.append(k)
        return schedule

    raise ValueError(
        "exploration_k must be either an int or a list/tuple of ints, got "
        f"{type(exploration_k).__name__}."
    )


def extract_train_timestep_values_for_logging(timesteps, expected_num_timesteps):
    timestep_values = sorted(
        {int(t) for t in timesteps.to(dtype=torch.long).detach().cpu().tolist()},
        reverse=True,
    )
    if len(timestep_values) != int(expected_num_timesteps):
        raise ValueError(
            "Unexpected number of unique training timesteps for clipfrac logging, got "
            f"{len(timestep_values)} and {expected_num_timesteps}."
        )
    return timestep_values


def compute_exploration_k_avg_for_active_steps(exploration_schedule):
    active_schedule = [int(k) for k in exploration_schedule if int(k) > 0]
    if len(active_schedule) == 0:
        raise ValueError(
            "Effective training exploration schedule contains no active timesteps (k > 0). "
            "At least one training transition must have exploration_k > 0."
        )
    return float(sum(active_schedule)) / float(len(active_schedule))


def compute_clipfrac_per_timestep(
    clip_event,
    timesteps,
    row_mask,
    tracked_timestep_values,
):
    clipfrac_per_timestep = {}
    clip_event_float = clip_event.float()
    row_mask_float = row_mask.to(dtype=torch.float32)
    timesteps_long = timesteps.to(dtype=torch.long)
    for timestep_value in tracked_timestep_values:
        timestep_mask = row_mask_float * (
            timesteps_long == int(timestep_value)
        ).to(dtype=row_mask_float.dtype)
        timestep_valid_count = torch.clamp(timestep_mask.sum(), min=1.0)
        clipfrac_per_timestep[f"clipfrac_timestep_{int(timestep_value)}"] = (
            clip_event_float * timestep_mask
        ).sum() / timestep_valid_count
    return clipfrac_per_timestep


def reward_value_to_tensor(value, device):
    if isinstance(value, list):
        if len(value) == 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        if torch.is_tensor(value[0]):
            value = torch.stack(
                [v.to(dtype=torch.float32) if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32) for v in value],
                dim=0,
            )
        else:
            value = torch.as_tensor(value, dtype=torch.float32)
    elif not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=torch.float32)

    value = value.to(device=device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.unsqueeze(0)
    if value.ndim > 1:
        value = value.reshape(value.shape[0], -1)
        if value.shape[1] != 1:
            raise ValueError(
                "Reward values must be scalar per transition row after flattening, got "
                f"shape {tuple(value.shape)}."
            )
        value = value.squeeze(1)
    return value


def compute_log_prob_rows(transformer, pipeline, sample, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"]] * 2),
            timestep=torch.cat([sample["timesteps"]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"],
            timestep=sample["timesteps"],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, sigma_t, sqrt_dt = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"],
        sample["latents"].float(),
        prev_sample=sample["next_latents"].float(),
        noise_level=config.sample.noise_level,
        return_sqrt_dt=True,
    )
    tilde_sigma_t = sigma_t / torch.clamp(sqrt_dt, min=1e-12)

    return prev_sample, log_prob, prev_sample_mean, tilde_sigma_t


def compute_noise_pred_rows(transformer, pipeline, sample, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"]] * 2),
            timestep=torch.cat([sample["timesteps"]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"],
            timestep=sample["timesteps"],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    timestep = sample["timesteps"]
    sample_latents = sample["latents"].float()
    noise_level = config.sample.noise_level

    scheduler = pipeline.scheduler
    model_output = noise_pred.float()
    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]

    t = scheduler.sigmas[step_index].view(-1, *([1] * (len(sample_latents.shape) - 1)))
    t_prev = scheduler.sigmas[prev_step_index].view(-1, *([1] * (len(sample_latents.shape) - 1)))
    t_max = scheduler.sigmas[1].item()
    minus_dt = t_prev - t

    tilde_sigma_t = torch.sqrt(t / (1 - torch.where(t == 1, t_max, t))) * noise_level

    prev_sample_mean = sample_latents * (1 + tilde_sigma_t**2 / (2 * t) * minus_dt) + model_output * (
        1 + tilde_sigma_t**2 * (1 - t) / (2 * t)
    ) * minus_dt

    return prev_sample_mean, tilde_sigma_t, noise_pred

def eval(
    pipeline,
    test_dataloader,
    tokenizer,
    neg_prompt_embed,
    neg_pooled_prompt_embed,
    config,
    accelerator,
    eval_epoch,
    global_step,
    reward_fn,
    executor,
    autocast,
    num_train_timesteps,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    all_rewards = defaultdict(list)
    eval_kl_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    eval_kl_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)

    train_exploration_schedule = normalize_exploration_schedule(
        config.sample.exploration_k,
        int(config.sample.num_steps),
    )[:num_train_timesteps]
    exploration_k_avg_for_loss = compute_exploration_k_avg_for_active_steps(
        train_exploration_schedule
    )

    def _disable_adapter_context(transformer_module):
        module_for_ctx = (
            transformer_module.module
            if hasattr(transformer_module, "module")
            else transformer_module
        )
        disable_adapter = getattr(module_for_ctx, "disable_adapter", None)
        if callable(disable_adapter):
            return disable_adapter()
        return contextlib.nullcontext()

    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts = test_batch["prompts"]
        prompt_embeds = test_batch["prompt_embeds"]
        pooled_prompt_embeds = test_batch["pooled_prompt_embeds"]
        prompt_metadata = test_batch["metadatas"]
        neg_prompt_embeds = neg_prompt_embed.repeat(len(prompts), 1, 1)
        neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(len(prompts), 1)

        if accelerator.mixed_precision == "fp16":
            prompt_embeds = prompt_embeds.half()
            pooled_prompt_embeds = pooled_prompt_embeds.half()
            neg_prompt_embeds = neg_prompt_embeds.half()
            neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.half()
        with autocast():
            with torch.no_grad():
                images, latents, log_probs  = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embeds,
                    negative_pooled_prompt_embeds=neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    return_dict=False,
                    height=config.resolution,
                    width=config.resolution, 
                    determistic=True,
                    noise_level=config.sample.noise_level,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)

        latents = torch.stack(latents, dim=1)
        timesteps = pipeline.scheduler.timesteps.repeat(latents.shape[0], 1)
        eval_steps = min(num_train_timesteps, latents.shape[1] - 1, len(train_exploration_schedule))
        if eval_steps > 0:
            if config.train.cfg:
                embeds = torch.cat([neg_prompt_embeds, prompt_embeds])
                pooled_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds])
            else:
                embeds = prompt_embeds
                pooled_embeds = pooled_prompt_embeds

            with autocast():
                with torch.no_grad():
                    for j in range(eval_steps):
                        if int(train_exploration_schedule[j]) == 0:
                            continue
                        step_sample = {
                            "latents": latents[:, j],
                            "next_latents": latents[:, j + 1],
                            "timesteps": timesteps[:, j],
                        }
                        timestep = timesteps[:, j].to(dtype=torch.float32)
                        next_timestep = timesteps[:, j + 1].to(dtype=torch.float32)
                        delta_t = (timestep - next_timestep) / 1000.0
                        exploration_loss_weight = (
                            exploration_k_avg_for_loss
                            / float(train_exploration_schedule[j])
                        )

                        if config.train.loss_type == "ppo":
                            _, _, prev_sample_mean, tilde_sigma_t = compute_log_prob_rows(
                                pipeline.transformer, pipeline, step_sample, embeds, pooled_embeds, config
                            )
                            with _disable_adapter_context(pipeline.transformer):
                                _, _, prev_sample_mean_ref, _ = compute_log_prob_rows(
                                    pipeline.transformer, pipeline, step_sample, embeds, pooled_embeds, config
                                )
                        else:
                            prev_sample_mean, tilde_sigma_t, _ = compute_noise_pred_rows(
                                pipeline.transformer, pipeline, step_sample, embeds, pooled_embeds, config
                            )
                            with _disable_adapter_context(pipeline.transformer):
                                prev_sample_mean_ref, _, _ = compute_noise_pred_rows(
                                    pipeline.transformer, pipeline, step_sample, embeds, pooled_embeds, config
                                )

                        tilde_sigma_row = tilde_sigma_t.reshape(tilde_sigma_t.shape[0], -1)[:, 0]
                        kl_denom = tilde_sigma_row.pow(2) * delta_t
                        kl_per_row = (
                            (prev_sample_mean - prev_sample_mean_ref) ** 2
                        ).mean(dim=(1, 2, 3)) / (2 * kl_denom)
                        kl_per_row = kl_per_row * exploration_loss_weight
                        kl_loss = kl_per_row.mean()

                        eval_kl_sum = eval_kl_sum + kl_loss.detach().to(dtype=torch.float64)
                        eval_kl_count = eval_kl_count + 1

        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizer(
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    kl_stats = accelerator.reduce(
        torch.stack([eval_kl_sum, eval_kl_count]),
        reduction="sum",
    )
    if kl_stats[1].item() > 0:
        eval_kl_divergence = (kl_stats[0] / kl_stats[1]).item()
    else:
        eval_kl_divergence = float("nan")
    accelerator.print(f"eval_kl_divergence: {eval_kl_divergence}")
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # use new index
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_epoch": int(eval_epoch),
                    "eval_global_step": int(global_step),
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    "eval_kl_divergence": eval_kl_divergence,
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=int(global_step),
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def main(_): 
    # basic Accelerate and logging setup
    config = FLAGS.config
    config.train.loss_type = getattr(config.train, "loss_type", "ppo")
    if config.train.loss_type not in VALID_LOSS_TYPES:
        raise NotImplementedError(
            f"Unknown loss type {config.train.loss_type}. Expected one of {sorted(VALID_LOSS_TYPES)}."
        )

    config.train.reweight_type = "base" if config.train.reweight_type is None else config.train.reweight_type
    if config.train.reweight_type not in VALID_REWEIGHT_TYPES:
        raise NotImplementedError(
            "Unknown reweight type "
            f"{config.train.reweight_type}. Expected one of {sorted(VALID_REWEIGHT_TYPES)}."
        )

    if "sampling_mode" not in config.sample:
        raise ValueError(
            "Missing required key sample.sampling_mode. "
            "scripts/train_sd3_pr.py only supports sample.sampling_mode='sde_branching'."
        )
    use_ode_kl_anchor = bool(getattr(config.train, "use_ode_kl_anchor", False))
    sampling_mode = str(config.sample.sampling_mode).lower()
    if sampling_mode != "sde_branching":
        raise ValueError(
            "scripts/train_sd3_pr.py only supports sample.sampling_mode='sde_branching'. "
            f"Received: {sampling_mode!r}."
        )

    missing_branching_keys = [
        key
        for key in ("group_strategy", "exploration_k", "collection_batch_size")
        if key not in config.sample
    ]
    if "latent_chunk_size" not in config:
        missing_branching_keys.append("latent_chunk_size")
    if missing_branching_keys:
        raise ValueError(
            "sde_branching mode requires keys: sample.group_strategy, sample.exploration_k, "
            "sample.collection_batch_size, latent_chunk_size. Missing: "
            + ", ".join(missing_branching_keys)
        )

    group_strategy = str(config.sample.group_strategy).lower()
    if group_strategy not in {"seed", "prompt", "batch"}:
        raise ValueError(
            f"Unsupported group_strategy: {group_strategy}. Expected one of ['seed', 'prompt', 'batch']."
        )
    config.sample.global_std = group_strategy == "batch"
    num_inference_steps = int(config.sample.num_steps)
    exploration_schedule = normalize_exploration_schedule(
        config.sample.exploration_k,
        num_inference_steps,
    )
    # Keep normalized list form local to avoid ml_collections type conflicts
    # when `config.sample.exploration_k` is declared as an int in configs.
    normalized_exploration_schedule = exploration_schedule

    latent_chunk_size = int(config.latent_chunk_size)

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    if num_train_timesteps < 1:
        raise ValueError(
            "num_train_timesteps must be >= 1. Check sample.num_steps and train.timestep_fraction."
        )
    if num_train_timesteps > len(normalized_exploration_schedule):
        raise ValueError(
            "num_train_timesteps exceeds available transition schedule length, got "
            f"{num_train_timesteps} and {len(normalized_exploration_schedule)}."
        )
    train_exploration_schedule = normalized_exploration_schedule[:num_train_timesteps]
    if sum(train_exploration_schedule) <= 0:
        raise ValueError(
            "Effective training exploration schedule contains no active timesteps (k > 0). "
            "Increase sample.exploration_k within the train timestep window."
        )
    num_active_train_timesteps = sum(1 for k in train_exploration_schedule if int(k) > 0)
    num_zero_train_timesteps = sum(1 for k in train_exploration_schedule if int(k) == 0)
    policy_chunk_timestep_factor = num_train_timesteps
    if policy_chunk_timestep_factor < 1:
        raise ValueError(
            "policy_chunk_timestep_factor must be >= 1, got "
            f"{policy_chunk_timestep_factor}."
        )
    policy_update_batch_size = latent_chunk_size * policy_chunk_timestep_factor
    # Normalize per-row losses so larger exploration_k timesteps do not dominate solely due
    # to having more sampled rows.
    exploration_k_avg_for_loss = compute_exploration_k_avg_for_active_steps(
        train_exploration_schedule
    )

    collection_batch_size = int(config.sample.collection_batch_size)
    if collection_batch_size < 1:
        raise ValueError(
            f"collection_batch_size must be >= 1, got {collection_batch_size}."
        )
    num_rollout_batches_per_epoch = int(config.sample.num_batches_per_epoch)
    if num_rollout_batches_per_epoch < 1:
        raise ValueError(
            "sample.num_batches_per_epoch must be >= 1, got "
            f"{num_rollout_batches_per_epoch}."
        )
    expected_transition_rows_per_rollout = (
        collection_batch_size * sum(train_exploration_schedule)
    )
    expected_transition_rows_per_epoch = (
        expected_transition_rows_per_rollout * num_rollout_batches_per_epoch
    )
    expected_kl_anchor_rows_per_rollout = (
        collection_batch_size * num_zero_train_timesteps
        if use_ode_kl_anchor
        else 0
    )
    expected_kl_anchor_rows_per_epoch = (
        expected_kl_anchor_rows_per_rollout * num_rollout_batches_per_epoch
    )
    # Policy-update units are sampled transition rows; we process
    # `policy_update_batch_size = latent_chunk_size * num_train_timesteps`
    # rows per forward, with padding only on the final batch.
    expected_policy_units_per_rollout = expected_transition_rows_per_rollout
    expected_policy_units_per_epoch = (
        expected_policy_units_per_rollout * num_rollout_batches_per_epoch
    )
    expected_padded_policy_units_per_epoch = (
        (expected_policy_units_per_epoch + policy_update_batch_size - 1)
        // policy_update_batch_size
    ) * policy_update_batch_size
    expected_microbatches_per_epoch = (
        expected_padded_policy_units_per_epoch // policy_update_batch_size
    )

    updates_per_epoch = getattr(config.train, "updates_per_epoch", None)
    if updates_per_epoch is None:
        raise ValueError(
            "Missing required key train.updates_per_epoch for sde_branching mode. "
            "Set it in config (e.g., train.updates_per_epoch = 2)."
        )
    updates_per_epoch = int(updates_per_epoch)
    if updates_per_epoch < 1:
        raise ValueError(
            f"train.updates_per_epoch must be >= 1, got {updates_per_epoch}."
        )
    if expected_microbatches_per_epoch % updates_per_epoch != 0:
        raise ValueError(
            "No-carry accumulation requires expected microbatches per epoch to be divisible "
            "by train.updates_per_epoch, got "
            f"{expected_microbatches_per_epoch} and {updates_per_epoch}. "
            "Adjust train.updates_per_epoch, latent_chunk_size, or sample schedule/batch settings."
        )
    effective_grad_accum_steps = expected_microbatches_per_epoch // updates_per_epoch

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # Gradient accumulation is derived from expected per-epoch microbatches and updates_per_epoch.
        gradient_accumulation_steps=effective_grad_accum_steps,
    )
    if accelerator.is_main_process:
        wandb.init(
            project="tempflow_grpo",
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    pipeline.text_encoder_2.to("cpu")
    pipeline.text_encoder_3.to("cpu")
    pipeline.register_modules(
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        tokenizer_2=None,
        tokenizer_3=None,
    )


    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # Move vae and transformer to device
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            # After loading with PeftModel.from_pretrained, all parameters have requires_grad set to False. You need to call set_adapter to enable gradients for the adapter parameters.
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)
    
    transformer = pipeline.transformer
    if config.activation_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 x 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    if config.prompt_fn in ("general_ocr", "geneval"):
        if not config.train_hdf5_path or not config.test_hdf5_path:
            raise ValueError(
                "Precomputed embedding mode requires `train_hdf5_path` and `test_hdf5_path` in config."
            )
        if not config.train_prompt_file_path or not config.test_prompt_file_path:
            raise ValueError(
                "Precomputed embedding mode requires `train_prompt_file_path` and `test_prompt_file_path` in config."
            )

        train_dataset = PrecomputedEmbeddingDataset(config.train_hdf5_path, config.train_prompt_file_path)
        test_dataset = PrecomputedEmbeddingDataset(config.test_hdf5_path, config.test_prompt_file_path)

        train_sampler = KRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.collection_batch_size,
            image_per_prompt=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            # KRepeatSampler updates `epoch` every step; worker prefetch can consume stale epoch values.
            # Keep this at 0 to avoid repeated early batches (e.g., epoch 0 prompt collapse).
            num_workers=0,
            collate_fn=PrecomputedEmbeddingDataset.collate_fn,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=PrecomputedEmbeddingDataset.collate_fn,
            shuffle=False,
            num_workers=0,
        )
    else:
        raise NotImplementedError("Only general_ocr and geneval are supported with precomputed embeddings.")

    with h5py.File(config.train_hdf5_path, "r") as hf:
        neg_group = hf["negative"]
        neg_prompt_embed = torch.from_numpy(neg_group["prompt_embeds"][:]).to(accelerator.device)
        neg_pooled_prompt_embed = torch.from_numpy(neg_group["pooled_prompt_embeds"][:]).to(accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.collection_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.collection_batch_size, 1)

    # initialize stat tracker for key-based grouping (prompt / seed)
    stat_tracker = None
    if group_strategy in {"prompt", "seed"}:
        stat_tracker = PerPromptStatTracker(global_std=False)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    transition_rows_per_epoch = (
        expected_transition_rows_per_epoch
        * accelerator.num_processes
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * effective_grad_accum_steps
    )
    num_gradient_updates_per_inner_epoch = updates_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Sampling mode = {sampling_mode}")
    logger.info(f"  Loss type = {config.train.loss_type}")
    logger.info(f"  Reweight type = {config.train.reweight_type}")
    logger.info(f"  Collection batch size per device = {config.sample.collection_batch_size}")
    logger.info(f"  Exploration (config/raw) = {config.sample.exploration_k}")
    logger.info(f"  Exploration (normalized transitions) = {normalized_exploration_schedule}")
    logger.info(f"  Training exploration schedule (used steps) = {train_exploration_schedule}")
    logger.info(f"  Active training timesteps (k > 0) = {num_active_train_timesteps}")
    logger.info(f"  Zero-k training timesteps (k == 0) = {num_zero_train_timesteps}")
    logger.info(f"  Use ODE KL anchors = {use_ode_kl_anchor}")
    logger.info(
        "  Exploration loss normalization factor (avg) = "
        f"{exploration_k_avg_for_loss:.6f}"
    )
    logger.info(f"  Expected transition rows per collected rollout = {expected_transition_rows_per_rollout}")
    logger.info(f"  Expected transition rows per epoch (unpadded, per device) = {expected_transition_rows_per_epoch}")
    logger.info(f"  Expected KL-anchor rows per collected rollout = {expected_kl_anchor_rows_per_rollout}")
    logger.info(f"  Expected KL-anchor rows per epoch (per device) = {expected_kl_anchor_rows_per_epoch}")
    logger.info(
        "  Expected policy units per collected rollout = "
        f"{expected_policy_units_per_rollout}"
    )
    logger.info(
        "  Expected policy units per epoch (unpadded, per device) = "
        f"{expected_policy_units_per_epoch}"
    )
    logger.info(
        "  Expected policy units per epoch (padded, per device) = "
        f"{expected_padded_policy_units_per_epoch}"
    )
    logger.info(f"  Expected microbatches per epoch (per device) = {expected_microbatches_per_epoch}")
    logger.info(f"  Training chunk size = {latent_chunk_size}")
    logger.info(f"  Policy update batch size (row-steps) = {policy_update_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Target optimizer updates per inner epoch = {updates_per_epoch}")
    logger.info(
        f"  Effective Gradient Accumulation steps = {effective_grad_accum_steps}"
    )
    logger.info(
        "  Accumulation plan (microbatches -> accum -> updates) = "
        f"{expected_microbatches_per_epoch} -> {effective_grad_accum_steps} -> {updates_per_epoch}"
    )
    logger.info("")
    logger.info(f"  Total number of transition rows per epoch = {transition_rows_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {num_gradient_updates_per_inner_epoch}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)
    trained_prompt_history = set()

    while True:
        if epoch >= config.num_epochs:
            break
        #################### EVAL ####################
        pipeline.transformer.eval()
        completed_epoch = int(epoch)
        if completed_epoch > 0 and completed_epoch % config.eval_freq == 0:
            eval(
                pipeline,
                test_dataloader,
                pipeline.tokenizer,
                neg_prompt_embed,
                neg_pooled_prompt_embed,
                config,
                accelerator,
                completed_epoch,
                global_step,
                eval_reward_fn,
                executor,
                autocast,
                num_train_timesteps,
                ema,
                transformer_trainable_parameters,
            )
        if completed_epoch > 0 and completed_epoch % config.save_freq == 0 and accelerator.is_main_process:
            save_ckpt(
                config.save_dir,
                transformer,
                completed_epoch,
                accelerator,
                ema,
                transformer_trainable_parameters,
                config,
            )
            wandb.log(
                {
                    "checkpoint_epoch": completed_epoch,
                    "checkpoint_global_step": global_step,
                },
                step=int(global_step),
            )
        epoch_start_time = time.time()
        sample_time_per_epoch = None

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        kl_anchor_samples = []
        vis_payload = None
        pbar = tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        )
        need_prev_sample_mean = (
            config.train.loss_type == "matching"
            or config.train.reweight_type == "grpo_guard"
        )
        collect_matching_aux = config.train.loss_type == "matching"
        epoch_useful_row_steps_2b = 0
        epoch_processed_row_steps_2b = 0
        epoch_padded_row_steps_2b = 0
        for i in pbar:
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            batch = next(train_iter)
            prompts = batch["prompts"]
            prompt_metadata = batch["metadatas"]
            prompt_embeds = batch["prompt_embeds"]
            pooled_prompt_embeds = batch["pooled_prompt_embeds"]
            prompt_ids = batch["prompt_ids"]
            seed_ids = create_seed_ids(
                prompts,
                base_seed=epoch * config.sample.num_batches_per_epoch + i,
                rank=accelerator.process_index,
                device=accelerator.device,
            )
            # sample
            with autocast():
                with torch.no_grad():
                    rollout = pipeline_with_logprob_perstep(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pil",
                        return_dict=False,
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        return_prev_sample_mean=need_prev_sample_mean,
                        collect_matching_aux=collect_matching_aux,
                        exploration_k=normalized_exploration_schedule,
                        collect_kl_anchor_rows=use_ode_kl_anchor,
                        latent_chunk_size=latent_chunk_size,
                    )
            required_rollout_keys = (
                "row_step_idx",
                "row_sample_idx",
                "row_exploration_k",
                "row_timesteps",
                "row_next_timesteps",
                "row_latents",
                "row_next_latents",
                "row_log_probs",
                "kl_row_step_idx",
                "kl_row_sample_idx",
                "kl_row_timesteps",
                "kl_row_next_timesteps",
                "kl_row_latents",
                "row_images",
                "useful_row_steps_2b",
                "processed_row_steps_2b",
                "padded_row_steps_2b",
            )
            missing_rollout_keys = [k for k in required_rollout_keys if k not in rollout]
            if missing_rollout_keys:
                raise ValueError(
                    "Packed rollout contract violation. Missing keys: "
                    + ", ".join(missing_rollout_keys)
                )

            row_step_idx_all = rollout["row_step_idx"]
            if row_step_idx_all.ndim != 1:
                raise ValueError(
                    "Packed rollout row_step_idx must be rank-1, got "
                    f"shape {tuple(row_step_idx_all.shape)}."
                )
            train_row_indices = torch.nonzero(
                row_step_idx_all < num_train_timesteps,
                as_tuple=False,
            ).squeeze(1)
            if train_row_indices.numel() == 0:
                raise ValueError("Packed rollout produced zero training rows.")

            expected_rows_per_rollout = int(prompt_embeds.shape[0]) * int(
                sum(train_exploration_schedule)
            )
            if int(train_row_indices.numel()) != expected_rows_per_rollout:
                raise ValueError(
                    "Packed rollout row count mismatch after train-step filtering, got "
                    f"{int(train_row_indices.numel())} and {expected_rows_per_rollout}."
                )
            kl_row_step_idx_all = rollout["kl_row_step_idx"]
            if kl_row_step_idx_all.ndim != 1:
                raise ValueError(
                    "Packed rollout kl_row_step_idx must be rank-1, got "
                    f"shape {tuple(kl_row_step_idx_all.shape)}."
                )
            train_kl_row_indices = torch.nonzero(
                kl_row_step_idx_all < num_train_timesteps,
                as_tuple=False,
            ).squeeze(1)
            expected_kl_rows_per_rollout = int(prompt_embeds.shape[0]) * int(
                num_zero_train_timesteps if use_ode_kl_anchor else 0
            )
            if int(train_kl_row_indices.numel()) != expected_kl_rows_per_rollout:
                raise ValueError(
                    "Packed rollout KL-anchor row count mismatch after train-step filtering, got "
                    f"{int(train_kl_row_indices.numel())} and {expected_kl_rows_per_rollout}."
                )

            row_sample_idx = rollout["row_sample_idx"].index_select(0, train_row_indices)
            row_step_idx = rollout["row_step_idx"].index_select(0, train_row_indices)
            row_exploration_k = rollout["row_exploration_k"].index_select(0, train_row_indices)
            row_timesteps = rollout["row_timesteps"].index_select(0, train_row_indices)
            row_next_timesteps = rollout["row_next_timesteps"].index_select(0, train_row_indices)
            row_latents = rollout["row_latents"].index_select(0, train_row_indices)
            row_next_latents = rollout["row_next_latents"].index_select(0, train_row_indices)
            row_log_probs = rollout["row_log_probs"].index_select(0, train_row_indices)
            kl_row_sample_idx = rollout["kl_row_sample_idx"].index_select(0, train_kl_row_indices)
            kl_row_step_idx = rollout["kl_row_step_idx"].index_select(0, train_kl_row_indices)
            kl_row_timesteps = rollout["kl_row_timesteps"].index_select(0, train_kl_row_indices)
            kl_row_next_timesteps = rollout["kl_row_next_timesteps"].index_select(
                0, train_kl_row_indices
            )
            kl_row_latents = rollout["kl_row_latents"].index_select(0, train_kl_row_indices)

            train_row_indices_cpu = train_row_indices.detach().cpu().tolist()
            packed_images = rollout["row_images"]
            if torch.is_tensor(packed_images):
                row_images = packed_images.index_select(
                    0, train_row_indices.to(packed_images.device)
                )
            else:
                row_images = [packed_images[idx] for idx in train_row_indices_cpu]

            if need_prev_sample_mean:
                if "row_prev_sample_mean" not in rollout:
                    raise ValueError(
                        "Packed rollout missing row_prev_sample_mean while return_prev_sample_mean is required."
                    )
                row_prev_sample_mean = rollout["row_prev_sample_mean"].index_select(0, train_row_indices)
            else:
                row_prev_sample_mean = None
            if collect_matching_aux:
                if "row_noises" not in rollout or "row_noise_preds" not in rollout:
                    raise ValueError(
                        "Packed rollout missing row_noises/row_noise_preds while matching aux is required."
                    )
                row_noises = rollout["row_noises"].index_select(0, train_row_indices)
                row_noise_preds = rollout["row_noise_preds"].index_select(0, train_row_indices)
            else:
                row_noises = None
                row_noise_preds = None

            train_row_count = int(train_row_indices.numel())
            for name, tensor in (
                ("row_step_idx", row_step_idx),
                ("row_exploration_k", row_exploration_k),
                ("row_timesteps", row_timesteps),
                ("row_next_timesteps", row_next_timesteps),
                ("row_latents", row_latents),
                ("row_next_latents", row_next_latents),
                ("row_log_probs", row_log_probs),
            ):
                if tensor.shape[0] != train_row_count:
                    raise ValueError(
                        f"Packed rollout {name} row count mismatch, got {tensor.shape[0]} and {train_row_count}."
                    )
            if need_prev_sample_mean and row_prev_sample_mean.shape[0] != train_row_count:
                raise ValueError(
                    "Packed rollout row_prev_sample_mean row count mismatch, got "
                    f"{row_prev_sample_mean.shape[0]} and {train_row_count}."
                )
            if collect_matching_aux and row_noises.shape[0] != train_row_count:
                raise ValueError(
                    "Packed rollout row_noises row count mismatch, got "
                    f"{row_noises.shape[0]} and {train_row_count}."
                )
            if collect_matching_aux and row_noise_preds.shape[0] != train_row_count:
                raise ValueError(
                    "Packed rollout row_noise_preds row count mismatch, got "
                    f"{row_noise_preds.shape[0]} and {train_row_count}."
                )
            train_kl_row_count = int(train_kl_row_indices.numel())
            for name, tensor in (
                ("kl_row_step_idx", kl_row_step_idx),
                ("kl_row_sample_idx", kl_row_sample_idx),
                ("kl_row_timesteps", kl_row_timesteps),
                ("kl_row_next_timesteps", kl_row_next_timesteps),
                ("kl_row_latents", kl_row_latents),
            ):
                if tensor.shape[0] != train_kl_row_count:
                    raise ValueError(
                        f"Packed rollout {name} row count mismatch, got {tensor.shape[0]} and {train_kl_row_count}."
                    )

            row_sample_indices_cpu = row_sample_idx.detach().cpu().tolist()
            row_prompts = [prompts[idx] for idx in row_sample_indices_cpu]
            row_prompt_metadata = [prompt_metadata[idx] for idx in row_sample_indices_cpu]
            reward_future = executor.submit(
                reward_fn,
                row_images,
                row_prompts,
                row_prompt_metadata,
                only_strict=True,
            )
            # yield so asynchronous reward execution begins immediately
            time.sleep(0)

            if vis_payload is None:
                vis_payload = {
                    "images": row_images,
                    "prompts": row_prompts,
                    "reward_future": reward_future,
                }

            samples.append(
                {
                    "prompt_ids": prompt_ids.index_select(0, row_sample_idx),
                    "seed_ids": seed_ids.index_select(0, row_sample_idx),
                    "prompt_embeds": prompt_embeds.index_select(0, row_sample_idx),
                    "pooled_prompt_embeds": pooled_prompt_embeds.index_select(0, row_sample_idx),
                    "row_step_idx": row_step_idx,
                    "exploration_k": row_exploration_k,
                    "timesteps": row_timesteps,
                    "next_timesteps": row_next_timesteps,
                    "latents": row_latents,
                    "next_latents": row_next_latents,
                    "log_probs": row_log_probs,
                    "rewards": reward_future,
                }
            )
            if need_prev_sample_mean:
                samples[-1]["prev_sample_mean"] = row_prev_sample_mean
            if collect_matching_aux:
                samples[-1]["noises"] = row_noises
                samples[-1]["noise_preds"] = row_noise_preds
            if train_kl_row_count > 0:
                kl_anchor_samples.append(
                    {
                        "row_step_idx": kl_row_step_idx,
                        "timesteps": kl_row_timesteps,
                        "next_timesteps": kl_row_next_timesteps,
                        "latents": kl_row_latents,
                        "prompt_embeds": prompt_embeds.index_select(0, kl_row_sample_idx),
                        "pooled_prompt_embeds": pooled_prompt_embeds.index_select(0, kl_row_sample_idx),
                    }
                )

            epoch_useful_row_steps_2b += int(rollout["useful_row_steps_2b"])
            epoch_processed_row_steps_2b += int(rollout["processed_row_steps_2b"])
            epoch_padded_row_steps_2b += int(rollout["padded_row_steps_2b"])

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample_batch_size = sample["timesteps"].shape[0]
            sample["rewards"] = {
                key: reward_value_to_tensor(value, accelerator.device)
                for key, value in rewards.items()
            }
            for key, value in sample["rewards"].items():
                if value.shape[0] != sample_batch_size:
                    raise ValueError(
                        "Reward batch size must match transition row count, got "
                        f"reward[{key}]={value.shape[0]} vs transitions={sample_batch_size}."
                    )

        if accelerator.is_main_process:
            padding_ratio_2b = (
                float(epoch_processed_row_steps_2b - epoch_useful_row_steps_2b)
                / float(epoch_processed_row_steps_2b)
                if epoch_processed_row_steps_2b > 0
                else 0.0
            )
            wandb.log(
                {
                    "2b_useful_row_steps": epoch_useful_row_steps_2b,
                    "2b_processed_row_steps": epoch_processed_row_steps_2b,
                    "2b_padded_row_steps": epoch_padded_row_steps_2b,
                    "2b_padding_ratio": padding_ratio_2b,
                },
                step=global_step,
            )

        # collate samples into dict where each entry has shape
        # (num_batches_per_epoch * collection_batch_size * sum(train_exploration_schedule), ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }
        if len(kl_anchor_samples) > 0:
            kl_anchor_rows = {
                k: torch.cat([s[k] for s in kl_anchor_samples], dim=0)
                for k in kl_anchor_samples[0].keys()
            }
        else:
            kl_anchor_rows = {
                "row_step_idx": torch.empty(0, device=accelerator.device, dtype=torch.long),
                "timesteps": torch.empty(
                    0,
                    device=samples["timesteps"].device,
                    dtype=samples["timesteps"].dtype,
                ),
                "next_timesteps": torch.empty(
                    0,
                    device=samples["next_timesteps"].device,
                    dtype=samples["next_timesteps"].dtype,
                ),
                "latents": torch.empty(
                    (0, *samples["latents"].shape[1:]),
                    device=samples["latents"].device,
                    dtype=samples["latents"].dtype,
                ),
                "prompt_embeds": torch.empty(
                    (0, *samples["prompt_embeds"].shape[1:]),
                    device=samples["prompt_embeds"].device,
                    dtype=samples["prompt_embeds"].dtype,
                ),
                "pooled_prompt_embeds": torch.empty(
                    (0, *samples["pooled_prompt_embeds"].shape[1:]),
                    device=samples["pooled_prompt_embeds"].device,
                    dtype=samples["pooled_prompt_embeds"].dtype,
                ),
            }
        if int(kl_anchor_rows["timesteps"].shape[0]) != expected_kl_anchor_rows_per_epoch:
            raise ValueError(
                "KL-anchor row count mismatch per epoch, got "
                f"{int(kl_anchor_rows['timesteps'].shape[0])} and {expected_kl_anchor_rows_per_epoch}."
            )
        
        if epoch % 10 == 0 and accelerator.is_main_process and vis_payload is not None:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                vis_images = vis_payload["images"]
                vis_prompts = vis_payload["prompts"]
                vis_rewards_raw, _ = vis_payload["reward_future"].result()
                vis_avg = reward_value_to_tensor(
                    vis_rewards_raw.get("avg", torch.zeros(len(vis_images))),
                    accelerator.device,
                ).detach().cpu().numpy()

                num_samples = min(15, len(vis_images))
                sample_indices = random.sample(range(len(vis_images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = vis_images[i]
                    # output_type="pil" returns PIL images (wrapped in a list for batch dimension)
                    pil = image[0] if isinstance(image, list) else image
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [vis_prompts[i] for i in sample_indices]
                sampled_rewards = [float(vis_avg[i]) for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"].clone()
        samples["rewards"]["avg"] = samples["rewards"]["avg"]
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        # log rewards and images
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-group mean/std tracking
        if group_strategy == "batch":
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)
            if accelerator.is_main_process:
                wandb.log({"group_strategy": group_strategy}, step=global_step)
        else:
            gathered_prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompt_labels = pipeline.tokenizer.batch_decode(
                gathered_prompt_ids, skip_special_tokens=True
            )
            gathered_timesteps = accelerator.gather(samples["timesteps"]).cpu().numpy().reshape(-1)
            if len(prompt_labels) != gathered_timesteps.shape[0]:
                raise ValueError(
                    "Prompt/timestep row alignment mismatch during stat tracking, got "
                    f"{len(prompt_labels)} and {gathered_timesteps.shape[0]}."
                )
            timestep_labels = [str(t) for t in gathered_timesteps.tolist()]
            trained_prompt_history.update(prompt_labels)
            trained_prompt_num = len(trained_prompt_history)

            if group_strategy == "prompt":
                group_labels = [
                    f"{prompt}|t={timestep}"
                    for prompt, timestep in zip(prompt_labels, timestep_labels)
                ]
            else:
                gathered_seed_ids = accelerator.gather(samples["seed_ids"]).cpu().numpy().tolist()
                if len(gathered_seed_ids) != len(timestep_labels):
                    raise ValueError(
                        "Seed/timestep row alignment mismatch during stat tracking, got "
                        f"{len(gathered_seed_ids)} and {len(timestep_labels)}."
                    )
                group_labels = [
                    f"{seed_id}|t={timestep}"
                    for seed_id, timestep in zip(gathered_seed_ids, timestep_labels)
                ]

            advantages = stat_tracker.update(group_labels, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(groups)", len(group_labels))
                print("len unique groups", len(set(group_labels)))

            group_size, _ = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(group_labels, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_strategy": group_strategy,
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()

        # ungather advantages; keep only the entries corresponding to this process
        advantages = torch.as_tensor(advantages, device=accelerator.device, dtype=torch.float32).reshape(-1)
        if advantages.numel() % accelerator.num_processes != 0:
            raise ValueError(
                "Gathered advantage count must be divisible by number of processes, got "
                f"{advantages.numel()} and {accelerator.num_processes}."
            )
        samples["advantages"] = advantages.reshape(accelerator.num_processes, -1)[
            accelerator.process_index
        ]
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]
        del samples["seed_ids"]

        # Keep all rows for fixed update cadence; use row_mask to ignore zero-advantage rows in loss.
        mask = samples["advantages"].abs() != 0
        if int(mask.sum().item()) == 0:
            mask = torch.ones_like(mask, dtype=torch.bool)

        total_transition_rows = samples["timesteps"].shape[0]
        selected_transition_rows = int(mask.sum().item())
        clipfrac_logging_timesteps = extract_train_timestep_values_for_logging(
            samples["timesteps"],
            num_active_train_timesteps,
        )
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": latent_chunk_size,
                    "policy_update_batch_size": policy_update_batch_size,
                    "actual_num_rollout_batches": (
                        (total_transition_rows + latent_chunk_size - 1) // latent_chunk_size
                    ),
                    "selected_transition_rows": selected_transition_rows,
                },
                step=global_step,
            )
        # Do not drop rows; keep alignment and mask invalid rows in loss.
        transition_rows = {
            "row_step_idx": samples["row_step_idx"],
            "exploration_k": samples["exploration_k"],
            "timesteps": samples["timesteps"],
            "next_timesteps": samples["next_timesteps"],
            "latents": samples["latents"],
            "next_latents": samples["next_latents"],
            "log_probs": samples["log_probs"],
            "advantages": samples["advantages"],
            "prompt_embeds": samples["prompt_embeds"],
            "pooled_prompt_embeds": samples["pooled_prompt_embeds"],
            "row_mask": mask,
        }
        if "prev_sample_mean" in samples:
            transition_rows["prev_sample_mean"] = samples["prev_sample_mean"]
        if "noises" in samples:
            transition_rows["noises"] = samples["noises"]
        if "noise_preds" in samples:
            transition_rows["noise_preds"] = samples["noise_preds"]

        # Build policy-update units directly from sampled transition rows.
        base_row_indices = torch.arange(
            total_transition_rows,
            device=accelerator.device,
            dtype=torch.long,
        )
        policy_unit_row_indices = base_row_indices
        if policy_unit_row_indices.numel() == 0:
            raise ValueError(
                "Policy-update unit construction produced zero rows. "
                "Check timestep_fraction and branching schedule."
            )
        policy_unit_mask = torch.ones(
            policy_unit_row_indices.shape[0],
            device=accelerator.device,
            dtype=torch.bool,
        )
        pad_policy_row_steps = (-policy_unit_row_indices.shape[0]) % policy_update_batch_size
        if pad_policy_row_steps > 0:
            policy_unit_row_indices = torch.cat(
                [
                    policy_unit_row_indices,
                    policy_unit_row_indices[:1].repeat(pad_policy_row_steps),
                ],
                dim=0,
            )
            policy_unit_mask = torch.cat(
                [
                    policy_unit_mask,
                    torch.zeros(
                        pad_policy_row_steps,
                        device=accelerator.device,
                        dtype=torch.bool,
                    ),
                ],
                dim=0,
            )

        padded_policy_row_steps = int(policy_unit_row_indices.shape[0])
        actual_microbatches_per_epoch = (
            padded_policy_row_steps // policy_update_batch_size
        )
        if actual_microbatches_per_epoch != expected_microbatches_per_epoch:
            raise ValueError(
                "No-carry invariant failed: actual microbatches per epoch must equal expected, got "
                f"{actual_microbatches_per_epoch} and {expected_microbatches_per_epoch}. "
                "This indicates rollout row construction drifted from initialization assumptions."
            )
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_transition_rows": total_transition_rows,
                    "actual_kl_anchor_rows": int(kl_anchor_rows["timesteps"].shape[0]),
                    "actual_policy_units": int(policy_unit_mask.sum().item()),
                    "actual_num_policy_batches": (
                        padded_policy_row_steps // policy_update_batch_size
                    ),
                    "padded_policy_units": pad_policy_row_steps,
                    "expected_microbatches_per_epoch": expected_microbatches_per_epoch,
                    "actual_microbatches_per_epoch": actual_microbatches_per_epoch,
                },
                step=global_step,
            )

        training_start_time = time.time()
        sample_time_per_epoch = training_start_time - epoch_start_time
        #################### TRAINING ####################
        optimizer_updates_this_epoch = 0
        expected_optimizer_updates_this_epoch = (
            updates_per_epoch * int(config.train.num_inner_epochs)
        )
        for inner_epoch in range(config.train.num_inner_epochs):
            # Shuffle policy-update units (row indices), then gather each batch on demand.
            perm = torch.randperm(padded_policy_row_steps, device=accelerator.device)
            policy_unit_row_indices_epoch = policy_unit_row_indices.index_select(0, perm)
            policy_unit_mask_epoch = policy_unit_mask.index_select(0, perm)
            num_policy_batches = padded_policy_row_steps // policy_update_batch_size
            kl_anchor_row_count = int(kl_anchor_rows["timesteps"].shape[0])
            if kl_anchor_row_count > 0:
                kl_anchor_perm = torch.randperm(
                    kl_anchor_row_count, device=accelerator.device
                )
                kl_anchor_row_indices_epoch_batches = torch.tensor_split(
                    kl_anchor_perm,
                    num_policy_batches,
                )
            else:
                empty_kl_indices = torch.empty(
                    0, device=accelerator.device, dtype=torch.long
                )
                kl_anchor_row_indices_epoch_batches = [
                    empty_kl_indices for _ in range(num_policy_batches)
                ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            optimizer_updates_this_inner_epoch = 0
            for i in tqdm(
                range(num_policy_batches),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                batch_start = i * policy_update_batch_size
                batch_end = batch_start + policy_update_batch_size
                sample_row_indices = policy_unit_row_indices_epoch[batch_start:batch_end]
                sample_unit_mask = policy_unit_mask_epoch[batch_start:batch_end]
                kl_anchor_row_indices = kl_anchor_row_indices_epoch_batches[i]
                sample = {
                    key: value.index_select(0, sample_row_indices)
                    for key, value in transition_rows.items()
                    if key not in {"row_step_idx", "row_mask"}
                }
                sample["row_mask"] = transition_rows["row_mask"].index_select(
                    0, sample_row_indices
                ) & sample_unit_mask

                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    neg_prompt_batch = neg_prompt_embed.repeat(len(sample["prompt_embeds"]), 1, 1)
                    neg_pooled_batch = neg_pooled_prompt_embed.repeat(len(sample["pooled_prompt_embeds"]), 1)
                    embeds = torch.cat([neg_prompt_batch, sample["prompt_embeds"]])
                    pooled_embeds = torch.cat([neg_pooled_batch, sample["pooled_prompt_embeds"]])
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                with accelerator.accumulate(transformer):
                    with autocast():
                        timestep = sample["timesteps"].to(dtype=torch.float32)              # [B]
                        next_timestep = sample["next_timesteps"].to(dtype=torch.float32)    # [B]
                        delta_t = (timestep - next_timestep) / 1000.0                       # [B]

                        if config.train.loss_type == "ppo":
                            _, log_prob, prev_sample_mean, tilde_sigma_t = compute_log_prob_rows(
                                transformer, pipeline, sample, embeds, pooled_embeds, config
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob_rows(
                                            transformer, pipeline, sample, embeds, pooled_embeds, config
                                        )

                            advantages = torch.clamp(
                                sample["advantages"],
                                -config.train.adv_clip_max,
                                config.train.adv_clip_max,
                            )

                            row_mask = sample["row_mask"].float()
                            valid_count = torch.clamp(row_mask.sum(), min=1.0)
                            exploration_k = torch.clamp(
                                sample["exploration_k"].to(dtype=torch.float32), min=1.0
                            )
                            exploration_loss_weight = (
                                exploration_k_avg_for_loss / exploration_k
                            ).to(dtype=advantages.dtype)

                            sqrt_dt = torch.sqrt(delta_t)
                            tilde_sigma_row = tilde_sigma_t.reshape(tilde_sigma_t.shape[0], -1)[:, 0]
                            grpo_guard_scale = sqrt_dt * tilde_sigma_row
                            ratio = compute_ppo_ratio_by_reweight(
                                config.train.reweight_type,
                                log_prob,
                                sample["log_probs"],
                                delta_t=delta_t,
                                timestep=timestep,
                                tilde_sigma=tilde_sigma_row,
                                prev_sample_mean=prev_sample_mean,
                                old_prev_sample_mean=sample["prev_sample_mean"]
                                if "prev_sample_mean" in sample
                                else None,
                                grpo_guard_bias_scale=grpo_guard_scale,
                                grpo_guard_logprob_scale=grpo_guard_scale,
                            )

                            unclipped_loss = -advantages * ratio
                            clipped_loss = -advantages * torch.clamp(
                                ratio,
                                1.0 - config.train.clip_range,
                                1.0 + config.train.clip_range,
                            )
                            policy_per_row = torch.maximum(unclipped_loss, clipped_loss)
                            policy_per_row = combine_ppo_terms_by_reweight(
                                config.train.reweight_type,
                                policy_per_row,
                                sqrt_dt=sqrt_dt,
                                tilde_sigma=tilde_sigma_row,
                            )
                            policy_per_row = policy_per_row * exploration_loss_weight
                            policy_loss = (policy_per_row * row_mask).sum() / valid_count

                            if config.train.beta > 0:
                                if config.train.heuristic_kldenom_trick:
                                    kl_denom = tilde_sigma_row.pow(2)
                                else:
                                    kl_denom = tilde_sigma_row.pow(2) * delta_t
                                kl_per_row = (
                                    (prev_sample_mean - prev_sample_mean_ref) ** 2
                                ).mean(dim=(1, 2, 3)) / (2 * kl_denom)
                                kl_per_row = kl_per_row * exploration_loss_weight
                                kl_sde_loss = (kl_per_row * row_mask).sum() / valid_count
                            else:
                                kl_sde_loss = torch.zeros(
                                    (),
                                    device=policy_loss.device,
                                    dtype=policy_loss.dtype,
                                )

                            approx_kl_per_row = 0.5 * ((log_prob - sample["log_probs"]) ** 2)
                            clip_event = torch.abs(ratio - 1.0) > config.train.clip_range
                            clip_event_gt = ratio - 1.0 > config.train.clip_range
                            clip_event_lt = 1.0 - ratio > config.train.clip_range
                        else:
                            prev_sample_mean, tilde_sigma_t, noise_pred = compute_noise_pred_rows(
                                transformer, pipeline, sample, embeds, pooled_embeds, config
                            )
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        prev_sample_mean_ref, _, _ = compute_noise_pred_rows(
                                            transformer, pipeline, sample, embeds, pooled_embeds, config
                                        )

                            advantages = torch.clamp(
                                sample["advantages"],
                                -config.train.adv_clip_max,
                                config.train.adv_clip_max,
                            )

                            row_mask = sample["row_mask"].float()
                            valid_count = torch.clamp(row_mask.sum(), min=1.0)
                            exploration_k = torch.clamp(
                                sample["exploration_k"].to(dtype=torch.float32), min=1.0
                            )
                            exploration_loss_weight = (
                                exploration_k_avg_for_loss / exploration_k
                            ).to(dtype=advantages.dtype)

                            tilde_sigma_row = tilde_sigma_t.reshape(tilde_sigma_t.shape[0], -1)[:, 0]
                            weight = compute_matching_weight_by_reweight(
                                config.train.reweight_type,
                                delta_t=delta_t,
                                timestep=timestep,
                                tilde_sigma=tilde_sigma_row,
                            ).to(dtype=noise_pred.dtype)

                            clip_range = config.train.clip_range

                            pred_diff = noise_pred - sample["noise_preds"]
                            noise = sample["noises"]

                            pred_matching_term_1 = 0.5 * (pred_diff * weight).pow(2).mean(dim=(1, 2, 3), dtype=torch.float32)
                            pred_matching_term_2 = ((weight * pred_diff) * noise).mean(dim=(1, 2, 3), dtype=torch.float32)
                            pred_matching_term = combine_matching_terms_by_reweight(
                                config.train.reweight_type,
                                pred_matching_term_1,
                                pred_matching_term_2,
                                tilde_sigma_row,
                                delta_t,
                            )

                            unclipped_loss = (
                                clip_range * torch.abs(advantages)
                                + advantages * pred_matching_term
                            )
                            clipped_loss = torch.zeros_like(advantages)
                            policy_per_row = torch.maximum(unclipped_loss, clipped_loss)

                            policy_per_row = policy_per_row * exploration_loss_weight

                            if config.train.reweight_type == "grpo_guard":
                                policy_per_row = policy_per_row / delta_t
                            elif config.train.reweight_type == "fair_clip":
                                policy_per_row = policy_per_row / (tilde_sigma_row**2 * delta_t)
                            elif config.train.reweight_type == "fair_clip2":
                                # Follow fair_clip scaling, then map to the same end scale as grpo_guard.
                                policy_per_row = (
                                    policy_per_row / (tilde_sigma_row**2 * delta_t)
                                ) * (tilde_sigma_row / torch.sqrt(delta_t))
                            policy_loss = (policy_per_row * row_mask).sum() / valid_count

                            if config.train.beta > 0:
                                if config.train.heuristic_kldenom_trick:
                                    kl_denom = tilde_sigma_row ** 2
                                else:
                                    kl_denom = (tilde_sigma_row ** 2) * delta_t
                                kl_per_row = (
                                    (prev_sample_mean - prev_sample_mean_ref) ** 2
                                ).mean(dim=(1, 2, 3)) / (2 * kl_denom)
                                kl_per_row = kl_per_row * exploration_loss_weight
                                kl_sde_loss = (kl_per_row * row_mask).sum() / valid_count
                            else:
                                kl_sde_loss = torch.zeros(
                                    (),
                                    device=policy_loss.device,
                                    dtype=policy_loss.dtype,
                                )

                            approx_kl_per_row = 0.5 * pred_matching_term.pow(2)
                            clip_event = (
                                clip_range * torch.abs(advantages)
                                + advantages * pred_matching_term
                                < 0
                            )
                            clip_event_gt = clip_event & (advantages > 0)
                            clip_event_lt = clip_event & (advantages < 0)
                        if (
                            config.train.beta > 0
                            and use_ode_kl_anchor
                            and kl_anchor_row_indices.numel() > 0
                        ):
                            kl_anchor_sample = {
                                key: value.index_select(0, kl_anchor_row_indices)
                                for key, value in kl_anchor_rows.items()
                                if key != "row_step_idx"
                            }

                            if config.train.cfg:
                                neg_prompt_batch = neg_prompt_embed.repeat(
                                    len(kl_anchor_sample["prompt_embeds"]), 1, 1
                                )
                                neg_pooled_batch = neg_pooled_prompt_embed.repeat(
                                    len(kl_anchor_sample["pooled_prompt_embeds"]), 1
                                )
                                kl_embeds = torch.cat(
                                    [neg_prompt_batch, kl_anchor_sample["prompt_embeds"]]
                                )
                                kl_pooled_embeds = torch.cat(
                                    [neg_pooled_batch, kl_anchor_sample["pooled_prompt_embeds"]]
                                )
                            else:
                                kl_embeds = kl_anchor_sample["prompt_embeds"]
                                kl_pooled_embeds = kl_anchor_sample["pooled_prompt_embeds"]

                            (
                                kl_prev_sample_mean,
                                kl_tilde_sigma_t,
                                _,
                            ) = compute_noise_pred_rows(
                                transformer,
                                pipeline,
                                kl_anchor_sample,
                                kl_embeds,
                                kl_pooled_embeds,
                                config,
                            )
                            with torch.no_grad():
                                with transformer.module.disable_adapter():
                                    (
                                        kl_prev_sample_mean_ref,
                                        _,
                                        _,
                                    ) = compute_noise_pred_rows(
                                        transformer,
                                        pipeline,
                                        kl_anchor_sample,
                                        kl_embeds,
                                        kl_pooled_embeds,
                                        config,
                                    )

                            kl_timestep = kl_anchor_sample["timesteps"].to(dtype=torch.float32)
                            kl_next_timestep = kl_anchor_sample["next_timesteps"].to(
                                dtype=torch.float32
                            )
                            kl_delta_t = (kl_timestep - kl_next_timestep) / 1000.0
                            kl_tilde_sigma_row = kl_tilde_sigma_t.reshape(
                                kl_tilde_sigma_t.shape[0], -1
                            )[:, 0]
                            if config.train.heuristic_kldenom_trick:
                                kl_anchor_denom = kl_tilde_sigma_row.pow(2)
                            else:
                                kl_anchor_denom = kl_tilde_sigma_row.pow(2) * kl_delta_t
                            kl_anchor_per_row = (
                                (kl_prev_sample_mean - kl_prev_sample_mean_ref) ** 2
                            ).mean(dim=(1, 2, 3)) / (2 * kl_anchor_denom)
                            kl_anchor_weight = torch.as_tensor(
                                exploration_k_avg_for_loss,
                                device=kl_anchor_per_row.device,
                                dtype=kl_anchor_per_row.dtype,
                            )
                            kl_anchor_per_row = kl_anchor_per_row * kl_anchor_weight
                            kl_anchor_loss = kl_anchor_per_row.mean()
                        else:
                            kl_anchor_loss = torch.zeros(
                                (),
                                device=policy_loss.device,
                                dtype=policy_loss.dtype,
                            )

                        if config.train.beta > 0:
                            kl_loss = kl_sde_loss + kl_anchor_loss
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            kl_loss = torch.zeros(
                                (),
                                device=policy_loss.device,
                                dtype=policy_loss.dtype,
                            )
                            loss = policy_loss

                    approx_kl_value = (approx_kl_per_row * row_mask).sum() / valid_count
                    clipfrac_value = (clip_event.float() * row_mask).sum() / valid_count
                    clipfrac_gt_one_value = (clip_event_gt.float() * row_mask).sum() / valid_count
                    clipfrac_lt_one_value = (clip_event_lt.float() * row_mask).sum() / valid_count

                    info["approx_kl"].append(approx_kl_value)
                    info["clipfrac"].append(clipfrac_value)
                    info["clipfrac_gt_one"].append(clipfrac_gt_one_value)
                    info["clipfrac_lt_one"].append(clipfrac_lt_one_value)
                    clipfrac_per_timestep = compute_clipfrac_per_timestep(
                        clip_event=clip_event,
                        timesteps=sample["timesteps"],
                        row_mask=row_mask,
                        tracked_timestep_values=clipfrac_logging_timesteps,
                    )
                    for metric_name, metric_value in clipfrac_per_timestep.items():
                        info[metric_name].append(metric_value)
                    info["policy_loss"].append(policy_loss)
                    if config.train.beta > 0:
                        info["kl_loss"].append(kl_loss)
                        info["kl_anchor_loss"].append(kl_anchor_loss)

                    info["loss"].append(loss)

                    # backward pass
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            transformer.parameters(), config.train.max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    # log training-related stuff
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch + 1, "inner_epoch": inner_epoch})
                    if accelerator.is_main_process:
                        wandb.log(info, step=global_step)
                    global_step += 1
                    optimizer_updates_this_inner_epoch += 1
                    optimizer_updates_this_epoch += 1
                    info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            if optimizer_updates_this_inner_epoch != updates_per_epoch:
                raise ValueError(
                    "No-carry invariant failed: optimizer updates per inner epoch mismatch, got "
                    f"{optimizer_updates_this_inner_epoch} and {updates_per_epoch}. "
                    "Check train.updates_per_epoch and accumulation settings."
                )
        if optimizer_updates_this_epoch != expected_optimizer_updates_this_epoch:
            raise ValueError(
                "No-carry invariant failed: optimizer updates per epoch mismatch, got "
                f"{optimizer_updates_this_epoch} and {expected_optimizer_updates_this_epoch}."
            )
        epoch_end_time = time.time()
        policy_update_time_per_epoch = epoch_end_time - training_start_time
        time_per_epoch = epoch_end_time - epoch_start_time
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "time_per_epoch": time_per_epoch,
                    "sample_time_per_epoch": sample_time_per_epoch,
                    "policy_update_time_per_epoch": policy_update_time_per_epoch,
                    "optimizer_updates_this_epoch": optimizer_updates_this_epoch,
                    "optimizer_updates_target_per_epoch": expected_optimizer_updates_this_epoch,
                },
                step=global_step,
            )
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)
