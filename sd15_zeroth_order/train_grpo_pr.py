#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import json
import logging
import os
import datetime
import hashlib
import random
from numbers import Integral
from dataclasses import asdict
from collections import defaultdict, OrderedDict
from copy import deepcopy
from contextlib import nullcontext
from concurrent import futures
from peft import get_peft_model_state_dict, LoraConfig
import time

import datasets
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from diffusers.utils import is_wandb_available
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

import utils.prompts
import utils.rewards
from utils.args import parse_args
from diffusion.ddim_step import ddim_step_with_logprob
from diffusion.reverse_pipeline import pipeline_with_logprob
from diffusion.reverse_pipeline_perstep import pipeline_with_logprob as pipeline_with_logprob_perstep

import tempfile
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

if is_wandb_available():
    import wandb

logger = get_logger(__name__, log_level="INFO")


def compute_inference_weights(noise_scheduler, timesteps, device):
    timesteps = timesteps.to(device=device, dtype=torch.long)
    alphas_cumprod = noise_scheduler.alphas_cumprod.clone().detach().to(device)
    num_steps = timesteps.shape[-1]

    # a_t
    alphas_cumprod_inference = alphas_cumprod[timesteps]
    # a_{t-1}
    prev_ts = (timesteps - 1000 // num_steps).clamp(min=0)
    alphas_cumprod_prev_inference = alphas_cumprod[prev_ts]
    # sigma_t^2
    sigmas_squared_inference = (
        (1 - alphas_cumprod_prev_inference)
        * (1 - alphas_cumprod_inference / alphas_cumprod_prev_inference)
        / (1 - alphas_cumprod_inference)
    )
    # sigma_t
    sigmas_inference = torch.sqrt(sigmas_squared_inference)
    # c2(t)
    c2s_inference = torch.sqrt(
        1 - alphas_cumprod_prev_inference - sigmas_squared_inference
    )
    # c1(t)
    c1s_inference = torch.sqrt(1 - alphas_cumprod_inference) / torch.sqrt(
        alphas_cumprod_inference / alphas_cumprod_prev_inference
    )
    # w(t) = (c1(t) - c2(t)) / sigma_t
    return (c1s_inference - c2s_inference) / sigmas_inference


def compute_inference_weights_for_rows(
    noise_scheduler,
    timesteps,
    device,
    num_inference_steps,
):
    timesteps = timesteps.to(device=device, dtype=torch.long).reshape(-1)
    alphas_cumprod = noise_scheduler.alphas_cumprod.clone().detach().to(device)
    num_steps = int(num_inference_steps)
    if num_steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {num_steps}.")

    alphas_cumprod_inference = alphas_cumprod[timesteps]
    prev_ts = (timesteps - 1000 // num_steps).clamp(min=0)
    alphas_cumprod_prev_inference = alphas_cumprod[prev_ts]
    sigmas_squared_inference = (
        (1 - alphas_cumprod_prev_inference)
        * (1 - alphas_cumprod_inference / alphas_cumprod_prev_inference)
        / (1 - alphas_cumprod_inference)
    )
    sigmas_inference = torch.sqrt(sigmas_squared_inference)
    c2s_inference = torch.sqrt(
        1 - alphas_cumprod_prev_inference - sigmas_squared_inference
    )
    c1s_inference = torch.sqrt(1 - alphas_cumprod_inference) / torch.sqrt(
        alphas_cumprod_inference / alphas_cumprod_prev_inference
    )
    return (c1s_inference - c2s_inference) / sigmas_inference


def get_step_weight(args, selected_weights, batch_size, step_index):
    if args.const_weight is not None:
        return torch.full(
            (batch_size, 1, 1, 1),
            args.const_weight,
            device=selected_weights.device,
            dtype=selected_weights.dtype,
        )
    return selected_weights[:, step_index].view(batch_size, 1, 1, 1)


def build_fixed_eval_prompts(
    args,
    accelerator,
    eval_num_batches,
    eval_batch_size,
    prompt_fn=None,
    prompt_pool=None,
):
    total_eval_samples = eval_num_batches * eval_batch_size
    if total_eval_samples <= 0:
        raise ValueError("eval_num_batches * eval_batch_size must be > 0 for evaluation.")

    eval_seed = getattr(args, "eval_seed", args.seed if args.seed is not None else 0)
    eval_prompt_seed = getattr(args, "eval_prompt_seed", eval_seed)
    process_seed = int(eval_prompt_seed) + accelerator.process_index

    if prompt_pool is not None:
        if len(prompt_pool) == 0:
            raise ValueError("prompt_pool is empty.")
        rng = np.random.default_rng(process_seed)
        prompts = []
        prompt_metadata = []
        for _ in range(eval_num_batches):
            if len(prompt_pool) < eval_batch_size:
                batch_prompts = [
                    prompt_pool[j % len(prompt_pool)] for j in range(eval_batch_size)
                ]
            else:
                chosen = rng.choice(
                    len(prompt_pool), size=eval_batch_size, replace=False
                )
                batch_prompts = [prompt_pool[int(idx)] for idx in chosen]
            prompts.extend(batch_prompts)
            prompt_metadata.extend([None for _ in range(eval_batch_size)])
        return prompts, prompt_metadata

    if prompt_fn is None:
        raise ValueError("prompt_fn must be provided when prompt_pool is None.")

    prompt_fn_kwargs = getattr(args, "prompt_fn_kwargs", {}) or {}
    py_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    random.seed(process_seed)
    np.random.seed(process_seed % (2**32))
    try:
        generated = [prompt_fn(**prompt_fn_kwargs) for _ in range(total_eval_samples)]
    finally:
        random.setstate(py_rng_state)
        np.random.set_state(np_rng_state)

    prompts, prompt_metadata = zip(*generated)
    return list(prompts), list(prompt_metadata)


def build_fixed_eval_batch_seeds(args, accelerator, eval_num_batches):
    eval_seed = getattr(args, "eval_seed", args.seed if args.seed is not None else 0)
    eval_noise_seed = getattr(args, "eval_noise_seed", eval_seed)
    process_seed_base = int(eval_noise_seed) + accelerator.process_index * 100000
    return [process_seed_base + batch_idx for batch_idx in range(eval_num_batches)]


class PromptDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


def sample_prompt_batch(dataset, sampler, batch_size):
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    local_indices = list(iter(sampler))
    if len(local_indices) == 0:
        raise ValueError(
            "No prompt indices were assigned to this process. "
            "Ensure prompt_file has at least one non-empty prompt."
        )
    # When local shard is smaller than batch_size (common at higher world sizes),
    # wrap around the same shuffled local order to keep a fixed-size batch.
    selected = [local_indices[i % len(local_indices)] for i in range(batch_size)]
    return [dataset[idx] for idx in selected]


def eval(unet, vae, text_encoder, tokenizer, noise_scheduler,
         sample_neg_prompt_embeds, args, accelerator, global_step, reward_fn,
         executor, autocast_ctx, fixed_eval_prompts, fixed_eval_prompt_metadata,
         fixed_eval_batch_seeds, unet_ref=None):
    unet.eval()
    all_rewards = []
    all_images = []
    all_prompts = []
    all_approx_kl = []

    # Determine number of eval batches
    eval_num_batches = getattr(args, 'eval_num_batches', 4)
    eval_batch_size = getattr(
        args,
        'eval_batch_size',
        getattr(args, "collection_batch_size", 1),
    )
    eval_num_steps = getattr(args, 'eval_num_steps', args.sample_num_steps)

    for i in tqdm(
        range(eval_num_batches),
        desc="Eval: sampling",
        disable=not accelerator.is_local_main_process,
        position=0,
    ):
        start_idx = i * eval_batch_size
        end_idx = (i + 1) * eval_batch_size
        prompts = fixed_eval_prompts[start_idx:end_idx]
        prompt_metadata = fixed_eval_prompt_metadata[start_idx:end_idx]

        # Encode prompts
        prompt_ids = tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = text_encoder(prompt_ids)[0]

        # Adjust negative prompt embeds to match batch size
        neg_prompt_embeds = sample_neg_prompt_embeds
        if len(prompt_embeds) != len(neg_prompt_embeds):
            neg_prompt_embeds = neg_prompt_embeds[:1].repeat(len(prompt_embeds), 1, 1)

        # Sample images
        with autocast_ctx:
            with torch.no_grad():
                eval_generator = torch.Generator(device=accelerator.device).manual_seed(
                    int(fixed_eval_batch_seeds[i])
                )
                images, _, _, _, _, _, noise_pred_refs, _, noise_preds = pipeline_with_logprob(
                    unet,
                    vae,
                    noise_scheduler,
                    tokenizer,
                    text_encoder,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=neg_prompt_embeds,
                    num_inference_steps=eval_num_steps,
                    rg_scale=args.sample_rg_scale,
                    guidance_scale=args.sample_cfg_scale,
                    eta=args.sample_eta,
                    output_type="pt",
                    unet_ref=unet_ref,
                    disable_progress_bar=not accelerator.is_local_main_process,
                    const_weight=args.const_weight,
                    algorithm=args.algorithm,
                    generator=eval_generator,
                    eval=True,
                )

        # Stack tensors for batch processing
        noise_preds = torch.stack(noise_preds, dim=1)  # (batch_size, num_steps, 4, 64, 64)
        noise_pred_refs = torch.stack(noise_pred_refs, dim=1) if noise_pred_refs[0] is not None else None

        # Compute approx_kl if we have reference predictions
        if noise_pred_refs is not None:
            if args.const_weight is not None:
                kl_weights = torch.full(
                    (
                        noise_preds.shape[0],
                        noise_preds.shape[1],
                        1,
                        1,
                        1,
                    ),
                    args.const_weight,
                    device=noise_preds.device,
                    dtype=noise_preds.dtype,
                )
            else:
                eval_timesteps = noise_scheduler.timesteps.unsqueeze(0).repeat(
                    noise_preds.shape[0], 1
                )
                kl_weights = compute_inference_weights(
                    noise_scheduler,
                    eval_timesteps,
                    noise_preds.device,
                ).view(noise_preds.shape[0], noise_preds.shape[1], 1, 1, 1)
                kl_weights = kl_weights.to(dtype=noise_preds.dtype)

            # Compute KL divergence approximation: 0.5 * mean(w^2 * (noise_pred - noise_pred_ref)^2)
            approx_kl = 0.5 * torch.mean(
                kl_weights.pow(2) * (noise_preds - noise_pred_refs).pow(2),
                dim=[1, 2, 3, 4],
            )
            all_approx_kl.append(accelerator.gather(approx_kl).cpu().numpy())

        # Compute rewards asynchronously
        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata)
        # Yield to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()
        # Gather rewards across devices
        rewards_gather = accelerator.gather(
            torch.as_tensor(rewards, device=accelerator.device)
        ).cpu().numpy()

        all_rewards.append(rewards_gather)
        # Store last batch for visualization
        all_images.append(images)
        all_prompts.extend(prompts)

    # Concatenate all rewards
    all_rewards_concat = np.concatenate(all_rewards)

    # Log eval metrics
    to_log = {"eval_reward_mean": np.mean(all_rewards_concat), "eval_reward_std": np.std(all_rewards_concat)}
    if len(all_approx_kl) > 0:
        to_log["eval_approx_kl"] = np.mean(np.concatenate(all_approx_kl))
    accelerator.log(to_log, step=global_step)

    # Log images and rewards from the last batch
    if accelerator.is_main_process:
        last_batch_images = all_images[-1]
        last_batch_prompts = all_prompts[-eval_batch_size:]
        last_batch_rewards = all_rewards[-1][-eval_batch_size:]

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(8, len(last_batch_images))
            sample_indices = range(num_samples)

            for idx in sample_indices:
                image = last_batch_images[idx].cpu().numpy()
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts = [last_batch_prompts[idx] for idx in sample_indices]
            sampled_rewards = [last_batch_rewards[idx] for idx in sample_indices]

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.100} | {reward:.2f}",
                        )
                        for idx, (prompt, reward) in enumerate(
                            zip(sampled_prompts, sampled_rewards)
                        )
                    ],
                    "eval_reward_mean": np.mean(all_rewards_concat),
                    "eval_reward_std": np.std(all_rewards_concat),
                },
                step=global_step,
            )


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
    if num_inference_steps < 1:
        raise ValueError(
            "sample_num_steps must be >= 1 so at least one diffusion step exists, got "
            f"{num_inference_steps}."
        )

    if isinstance(exploration_k, Integral):
        k = int(exploration_k)
        if k < 0:
            raise ValueError(f"exploration_k must be >= 0, got {k}.")
        return [k] * num_inference_steps

    if isinstance(exploration_k, (list, tuple)):
        if len(exploration_k) != num_inference_steps:
            raise ValueError(
                "When exploration_k is a list, its length must be exactly sample_num_steps "
                f"({num_inference_steps}), got {len(exploration_k)}."
            )

        schedule = []
        for i, value in enumerate(exploration_k):
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


def compute_exploration_k_avg_for_active_steps(exploration_schedule):
    active_schedule = [int(k) for k in exploration_schedule if int(k) > 0]
    if len(active_schedule) == 0:
        raise ValueError(
            "Effective training exploration schedule contains no active timesteps (k > 0). "
            "At least one training step must have exploration_k > 0."
        )
    return float(sum(active_schedule)) / float(len(active_schedule))


def assert_finite_tensor(name: str, tensor: torch.Tensor):
    if not torch.is_tensor(tensor):
        return
    if tensor.numel() == 0:
        return
    finite_mask = torch.isfinite(tensor)
    if bool(finite_mask.all().item()):
        return
    non_finite_count = int((~finite_mask).sum().item())
    total_count = int(tensor.numel())
    raise ValueError(
        f"Non-finite values detected in {name}: {non_finite_count}/{total_count}. "
        "Check sample_eta, exploration_k schedule, and optimization hyperparameters."
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
                [
                    v.to(dtype=torch.float32)
                    if torch.is_tensor(v)
                    else torch.as_tensor(v, dtype=torch.float32)
                    for v in value
                ],
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


def compute_group_advantages(group_labels, rewards):
    labels = np.asarray(group_labels)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if labels.shape[0] != rewards.shape[0]:
        raise ValueError(
            "Group labels and rewards must be row-aligned, got "
            f"{labels.shape[0]} and {rewards.shape[0]}."
        )

    unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
    if unique_labels.size == 0:
        return (
            np.zeros_like(rewards, dtype=np.float64),
            0.0,
            0,
            0.0,
            0.0,
        )

    advantages = np.zeros_like(rewards, dtype=np.float64)
    group_stds = []
    zero_std_count = 0
    for group_idx in range(unique_labels.size):
        mask = inverse_indices == group_idx
        group_rewards = rewards[mask]
        group_mean = float(np.mean(group_rewards))
        group_std = float(np.std(group_rewards))
        group_stds.append(group_std)
        if group_std < 1e-12:
            zero_std_count += 1
            advantages[mask] = 0.0
        else:
            advantages[mask] = (group_rewards - group_mean) / (group_std + 1e-4)

    avg_group_size = float(labels.shape[0]) / float(unique_labels.size)
    zero_std_ratio = float(zero_std_count) / float(unique_labels.size)
    reward_std_mean = float(np.mean(group_stds))
    return (
        advantages,
        avg_group_size,
        int(unique_labels.size),
        zero_std_ratio,
        reward_std_mean,
    )


def main():
    args = parse_args()
    args.use_lora = getattr(args, "use_lora", False)
    args.prompt_file = getattr(args, "prompt_file", "./prompts_dancegrpo.txt")
    args.sampling_mode = str(getattr(args, "sampling_mode", "")).lower()
    if args.sampling_mode != "ddim_branching":
        raise ValueError(
            "train_grpo_pr.py only supports sampling_mode='ddim_branching'. "
            f"Received: {args.sampling_mode!r}."
        )

    missing_branching_keys = []
    for key in (
        "group_strategy",
        "exploration_k",
        "collection_batch_size",
        "latent_chunk_size",
        "updates_per_epoch",
    ):
        if not hasattr(args, key):
            missing_branching_keys.append(key)
    if missing_branching_keys:
        raise ValueError(
            "ddim_branching mode requires keys: group_strategy, exploration_k, "
            "collection_batch_size, latent_chunk_size, updates_per_epoch. Missing: "
            + ", ".join(missing_branching_keys)
        )

    group_strategy = str(args.group_strategy).lower()
    if group_strategy not in {"seed", "prompt", "batch"}:
        raise ValueError(
            f"Unsupported group_strategy: {group_strategy}. Expected one of ['seed', 'prompt', 'batch']."
        )
    use_ode_kl_anchor = bool(getattr(args, "use_ode_kl_anchor", False))
    if use_ode_kl_anchor:
        if not hasattr(args, "ode_kl_anchor_beta"):
            raise ValueError(
                "use_ode_kl_anchor=true requires explicit ode_kl_anchor_beta in config."
            )
        if not hasattr(args, "anchor_row_max_ratio"):
            raise ValueError(
                "use_ode_kl_anchor=true requires explicit anchor_row_max_ratio in config."
            )
        ode_kl_anchor_beta = float(args.ode_kl_anchor_beta)
        if ode_kl_anchor_beta < 0:
            raise ValueError(
                f"ode_kl_anchor_beta must be >= 0, got {ode_kl_anchor_beta}."
            )
        anchor_row_max_ratio = float(args.anchor_row_max_ratio)
        if anchor_row_max_ratio < 0.0 or anchor_row_max_ratio > 1.0:
            raise ValueError(
                "anchor_row_max_ratio must be in [0.0, 1.0], got "
                f"{anchor_row_max_ratio}."
            )
    else:
        ode_kl_anchor_beta = 0.0
        anchor_row_max_ratio = 0.0

    num_inference_steps = int(args.sample_num_steps)
    normalized_exploration_schedule = normalize_exploration_schedule(
        args.exploration_k,
        num_inference_steps,
    )

    num_train_timesteps = int(args.sample_num_steps * args.train_timestep_fraction)
    if num_train_timesteps < 1:
        raise ValueError(
            "num_train_timesteps must be >= 1. Check sample_num_steps and train_timestep_fraction."
        )
    if num_train_timesteps > len(normalized_exploration_schedule):
        raise ValueError(
            "num_train_timesteps exceeds available step schedule length, got "
            f"{num_train_timesteps} and {len(normalized_exploration_schedule)}."
        )
    train_exploration_schedule = normalized_exploration_schedule[:num_train_timesteps]
    if sum(train_exploration_schedule) <= 0:
        raise ValueError(
            "Effective training exploration schedule contains no active timesteps (k > 0). "
            "Increase exploration_k within the train timestep window."
        )
    num_active_train_timesteps = sum(1 for k in train_exploration_schedule if int(k) > 0)
    num_zero_train_timesteps = sum(1 for k in train_exploration_schedule if int(k) == 0)
    num_zero_nonterminal_train_timesteps = sum(
        1
        for step_idx, k in enumerate(train_exploration_schedule)
        if int(k) == 0 and step_idx < (num_inference_steps - 1)
    )
    exploration_k_avg_for_loss = compute_exploration_k_avg_for_active_steps(
        train_exploration_schedule
    )

    latent_chunk_size = int(args.latent_chunk_size)
    if latent_chunk_size < 1:
        raise ValueError(f"latent_chunk_size must be >= 1, got {latent_chunk_size}.")
    default_policy_update_batch_size = latent_chunk_size * num_train_timesteps
    policy_update_batch_size = int(
        getattr(args, "policy_update_row_batch_size", default_policy_update_batch_size)
    )
    if policy_update_batch_size < 1:
        raise ValueError(
            "policy_update_batch_size must be >= 1, got "
            f"{policy_update_batch_size}."
        )
    policy_batch_size_source = (
        "policy_update_row_batch_size"
        if hasattr(args, "policy_update_row_batch_size")
        else "latent_chunk_size * num_train_timesteps"
    )

    collection_batch_size = int(args.collection_batch_size)
    if collection_batch_size < 1:
        raise ValueError(
            f"collection_batch_size must be >= 1, got {collection_batch_size}."
        )

    total_root_samples_per_epoch = (
        int(args.num_prompts_per_epoch) * int(args.num_generations_per_prompt)
    )
    if total_root_samples_per_epoch % collection_batch_size != 0:
        raise ValueError(
            "collection_batch_size must divide num_prompts_per_epoch * num_generations_per_prompt."
        )
    num_rollout_batches_per_epoch = total_root_samples_per_epoch // collection_batch_size
    args.sample_num_batches_per_epoch = num_rollout_batches_per_epoch

    expected_transition_rows_per_rollout = (
        collection_batch_size * sum(train_exploration_schedule)
    )
    expected_transition_rows_per_epoch = (
        expected_transition_rows_per_rollout * num_rollout_batches_per_epoch
    )
    expected_kl_anchor_rows_per_rollout = (
        collection_batch_size * num_zero_nonterminal_train_timesteps
        if use_ode_kl_anchor
        else 0
    )
    expected_kl_anchor_rows_per_epoch = (
        expected_kl_anchor_rows_per_rollout * num_rollout_batches_per_epoch
    )
    expected_policy_units_per_epoch = expected_transition_rows_per_epoch
    expected_padded_policy_units_per_epoch = (
        (expected_policy_units_per_epoch + policy_update_batch_size - 1)
        // policy_update_batch_size
    ) * policy_update_batch_size
    expected_microbatches_per_epoch = (
        expected_padded_policy_units_per_epoch // policy_update_batch_size
    )

    updates_per_epoch = int(args.updates_per_epoch)
    if updates_per_epoch < 1:
        raise ValueError(f"updates_per_epoch must be >= 1, got {updates_per_epoch}.")
    if expected_microbatches_per_epoch % updates_per_epoch != 0:
        raise ValueError(
            "No-carry accumulation requires expected microbatches per epoch to be divisible "
            "by updates_per_epoch, got "
            f"{expected_microbatches_per_epoch} and {updates_per_epoch}. "
            "Adjust updates_per_epoch, policy_update_row_batch_size (or latent_chunk_size), "
            "or sample schedule/batch settings."
        )
    effective_grad_accum_steps = expected_microbatches_per_epoch // updates_per_epoch

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    unique_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if not args.run_name:
        args.run_name = unique_id

    logging_dir = os.path.join(args.logging_dir, args.run_name)

    accelerator_config = ProjectConfiguration(
        project_dir=logging_dir,
        automatic_checkpoint_naming=False
    )
    accelerator = Accelerator(
        log_with=args.report_to,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=effective_grad_accum_steps,
    )
    if accelerator.is_main_process:
        # init accelerator
        accelerator.init_trackers(
            project_name="ddgrpo-rg-new",
            config=vars(args),
            init_kwargs={"wandb": {"name": args.run_name}},
        )
        # create local folder
        os.makedirs(logging_dir, exist_ok=True)
        # save training args
        with open(os.path.join(logging_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)

    logger.info(f"\n{args}")

    # Set seed (device_specific gets different prompts for each GPU)
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler", revision=args.revision
    )
    noise_scheduler = DDIMScheduler.from_config(noise_scheduler.config) # switch to DDIM
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model, subfolder="tokenizer", revision=args.revision
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model, subfolder="unet", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model, subfolder="vae", revision=args.revision, variant=args.variant
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model, subfolder="text_encoder", revision=args.revision
    )

    # Freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if not args.use_lora:
        unet.requires_grad_(True)

    # For mixed precision training, cast all non-trainable weights
    # (vae, non-lora text_encoder and non-lora unet) to half-precision.
    # These weights are only used for inference, so full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet_lora_config = None
    lora_layers = None
    trainable_params = None

    if args.use_lora:
        unet_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    else:
        trainable_params = [p for p in unet.parameters() if p.requires_grad]

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    if args.use_lora:
        unet.to(accelerator.device, dtype=weight_dtype)
        # Add adapter and make sure trainable params are in float32.
        unet.add_adapter(unet_lora_config)
        if args.mixed_precision == "fp16":
            # Upcast only trainable parameters (LoRA) into fp32
            cast_training_params(unet, dtype=torch.float32)
        lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    else:
        unet.to(accelerator.device)
        trainable_params = [p for p in unet.parameters() if p.requires_grad]

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Create a memory-efficient copy of the current UNet model (unet_ref) for reference,
    # disabling gradient tracking and avoiding unnecessary memory usage.
    unet_ref = deepcopy(unet)
    unet_ref.requires_grad_(False)
    with torch.no_grad():
        for pA, pB in zip(unet.parameters(), unet_ref.parameters()):
            if not pA.requires_grad:
                pB.data = pA.data
            pB.requires_grad = False
    torch.cuda.empty_cache()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize optimizer
    if args.scale_lr:
        gradient_accumulation_steps = effective_grad_accum_steps
        args.learning_rate = (
            args.learning_rate
            * gradient_accumulation_steps
            * policy_update_batch_size
            * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("Run `pip install bitsandbytes` to use 8-bit Adam.")
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if args.use_lora:
        optimizer_params = lora_layers
    else:
        optimizer_params = trainable_params

    optimizer = optimizer_cls(
        optimizer_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare prompt and reward function
    dataset = PromptDataset(args.prompt_file)
    if len(dataset) == 0:
        raise ValueError(
            f"Prompt file {args.prompt_file!r} contains no non-empty lines."
        )
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=args.seed,
        shuffle=True,
    )
    reward_fn = getattr(utils.rewards, args.reward_fn)()
    eval_num_batches = getattr(args, "eval_num_batches", 4)
    eval_batch_size = getattr(
        args,
        "eval_batch_size",
        collection_batch_size,
    )
    fixed_eval_prompts, fixed_eval_prompt_metadata = build_fixed_eval_prompts(
        args=args,
        accelerator=accelerator,
        eval_num_batches=eval_num_batches,
        eval_batch_size=eval_batch_size,
        prompt_pool=dataset.prompts,
    )
    fixed_eval_batch_seeds = build_fixed_eval_batch_seeds(
        args=args,
        accelerator=accelerator,
        eval_num_batches=eval_num_batches,
    )
    logger.info(
        "Prepared fixed eval set with %d prompts and %d batch seeds (process %d).",
        len(fixed_eval_prompts),
        len(fixed_eval_batch_seeds),
        accelerator.process_index,
    )

    # Generate negative prompt embeddings
    neg_prompt_embed = text_encoder(
        tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(eval_batch_size, 1, 1)
    collection_neg_prompt_embeds = neg_prompt_embed.repeat(collection_batch_size, 1, 1)

    autocast = accelerator.autocast
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    # Prepare everything with accelerator
    unet, unet_ref, optimizer = accelerator.prepare(unet, unet_ref, optimizer)

    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Initialize step and epoch
    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint is not None:
        ckpt = torch.load(args.resume_from_checkpoint, weights_only=False)

        def add_adapter_suffix(state_dict, adapter="default"):
            """Convert 0.7 style keys → 0.8+ style keys."""
            new = OrderedDict()
            for k, v in state_dict.items():
                if (".lora_A." in k) and (f".lora_A.{adapter}." not in k):
                    k = k.replace(".lora_A.", f".lora_A.{adapter}.")
                if (".lora_B." in k) and (f".lora_B.{adapter}." not in k):
                    k = k.replace(".lora_B.", f".lora_B.{adapter}.")
                new[k] = v
            return new

        peft_unet = accelerator.unwrap_model(unet)
        if args.use_lora:
            if "lora_weights" not in ckpt:
                raise ValueError("Expected LoRA checkpoint with key 'lora_weights'.")
            patched = add_adapter_suffix(ckpt["lora_weights"])
            missing, unexpected = peft_unet.load_state_dict(patched, strict=False)
            assert not unexpected, "unexpected keys in state dict"
        else:
            if "unet_state_dict" not in ckpt:
                raise ValueError("Expected full fine-tuning checkpoint with key 'unet_state_dict'.")
            missing, unexpected = peft_unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
            if unexpected:
                raise ValueError(f"unexpected keys in UNet state dict: {unexpected}")
            if missing:
                logger.warning(f"Missing keys when loading UNet state dict: {len(missing)}")

        optimizer.load_state_dict(ckpt["optimizer_state"]["optimizer"])
        first_epoch = ckpt["optimizer_state"]["epoch"] + 1
        global_step = ckpt["optimizer_state"]["global_step"]
        logger.info(f"Resume training from checkpoint {args.resume_from_checkpoint}")
        logger.info(f"  epoch {first_epoch}, global step {global_step}")

    # Train!
    transition_rows_per_epoch = (
        expected_transition_rows_per_epoch * accelerator.num_processes
    )
    total_policy_units_per_optimizer_step = (
        policy_update_batch_size * accelerator.num_processes * effective_grad_accum_steps
    )
    num_gradient_updates_per_inner_epoch = updates_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Sampling mode = {args.sampling_mode}")
    logger.info(f"  Group strategy = {group_strategy}")
    logger.info(f"  Collection batch size per device = {collection_batch_size}")
    logger.info(f"  Exploration (config/raw) = {args.exploration_k}")
    logger.info(f"  Exploration (normalized steps) = {normalized_exploration_schedule}")
    logger.info(f"  Training exploration schedule (used steps) = {train_exploration_schedule}")
    logger.info(f"  Active training timesteps (k > 0) = {num_active_train_timesteps}")
    logger.info(f"  Zero-k training timesteps (k == 0) = {num_zero_train_timesteps}")
    logger.info(
        "  Zero-k non-terminal train timesteps (eligible for ODE KL anchor) = "
        f"{num_zero_nonterminal_train_timesteps}"
    )
    logger.info(f"  Use ODE KL anchors = {use_ode_kl_anchor}")
    logger.info(f"  ODE KL anchor beta = {ode_kl_anchor_beta}")
    logger.info(f"  Anchor row max ratio = {anchor_row_max_ratio}")
    logger.info(
        "  Exploration loss normalization factor (avg) = "
        f"{exploration_k_avg_for_loss:.6f}"
    )
    logger.info(f"  Expected transition rows per collected rollout = {expected_transition_rows_per_rollout}")
    logger.info(f"  Expected transition rows per epoch (unpadded, per device) = {expected_transition_rows_per_epoch}")
    logger.info(
        f"  Expected KL-anchor rows per collected rollout = {expected_kl_anchor_rows_per_rollout}"
    )
    logger.info(
        f"  Expected KL-anchor rows per epoch (per device) = {expected_kl_anchor_rows_per_epoch}"
    )
    logger.info(f"  Expected policy units per epoch (unpadded, per device) = {expected_policy_units_per_epoch}")
    logger.info(f"  Expected policy units per epoch (padded, per device) = {expected_padded_policy_units_per_epoch}")
    logger.info(f"  Expected microbatches per epoch (per device) = {expected_microbatches_per_epoch}")
    logger.info(f"  Training chunk size = {latent_chunk_size}")
    logger.info(f"  Policy update batch size (row-steps) = {policy_update_batch_size}")
    logger.info(f"  Policy update batch source = {policy_batch_size_source}")
    logger.info(f"  Target optimizer updates per inner epoch = {updates_per_epoch}")
    logger.info(f"  Effective Gradient Accumulation steps = {effective_grad_accum_steps}")
    logger.info(
        "  Accumulation plan (microbatches -> accum -> updates) = "
        f"{expected_microbatches_per_epoch} -> {effective_grad_accum_steps} -> {updates_per_epoch}"
    )
    logger.info(f"  Total number of transition rows per epoch = {transition_rows_per_epoch}")
    logger.info(
        "  Total policy units per optimizer step "
        f"(parallel + accumulation) = {total_policy_units_per_optimizer_step}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {num_gradient_updates_per_inner_epoch}"
    )
    logger.info(f"  Number of inner epochs = {args.train_num_inner_epochs}")

    trained_prompt_history = set()

    for epoch in range(first_epoch, args.num_train_epochs):
        # Start epoch timer (from sampling to end of training)
        epoch_start_time = time.time()
        sampler.set_epoch(epoch)
        prompt_batch = sample_prompt_batch(
            dataset=dataset,
            sampler=sampler,
            batch_size=int(args.num_prompts_per_epoch),
        )

        expanded_prompts = []
        for prompt in prompt_batch:
            expanded_prompts.extend([prompt] * args.num_generations_per_prompt)
        if len(expanded_prompts) != collection_batch_size * num_rollout_batches_per_epoch:
            raise ValueError(
                "Expanded prompt count does not match collection_batch_size * num_rollout_batches_per_epoch."
            )

        #################### EVALUATION ####################
        if epoch % getattr(args, 'eval_freq', 10) == 0 and epoch > 0:
            eval(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                noise_scheduler=noise_scheduler,
                sample_neg_prompt_embeds=sample_neg_prompt_embeds,
                args=args,
                accelerator=accelerator,
                global_step=global_step,
                reward_fn=reward_fn,
                executor=executor,
                autocast_ctx=autocast_ctx,
                fixed_eval_prompts=fixed_eval_prompts,
                fixed_eval_prompt_metadata=fixed_eval_prompt_metadata,
                fixed_eval_batch_seeds=fixed_eval_batch_seeds,
                unet_ref=unet_ref,
            )

        #################### SAMPLING ####################
        unet.eval()
        samples = []
        kl_anchor_samples = []
        vis_payload = None

        global_input_latents = torch.randn(
            (1, 4, 64, 64),
            device=accelerator.device,
            dtype=weight_dtype,
        )
        sample_input_latents = global_input_latents.repeat(collection_batch_size, 1, 1, 1).clone()
        global_input_latents_2 = torch.randn(
            (1, 4, 64, 64),
            device=accelerator.device,
            dtype=weight_dtype,
        )
        sample_input_latents_2 = global_input_latents_2.repeat(collection_batch_size, 1, 1, 1).clone()
        max_iters_in_loop = max(1, num_rollout_batches_per_epoch // 2)
        cnt_iter_in_loop = 0
        epoch_useful_row_steps_2b = 0
        epoch_processed_row_steps_2b = 0
        epoch_padded_row_steps_2b = 0

        for i in tqdm(
            range(num_rollout_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            start_idx = i * collection_batch_size
            end_idx = (i + 1) * collection_batch_size
            prompts = expanded_prompts[start_idx:end_idx]
            prompt_metadata = [None for _ in range(len(prompts))]

            # encode prompts
            prompt_ids = tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).input_ids.to(accelerator.device)
            prompt_embeds = text_encoder(prompt_ids)[0]
            seed_ids = create_seed_ids(
                prompts,
                base_seed=epoch * num_rollout_batches_per_epoch + i,
                rank=accelerator.process_index,
                device=accelerator.device,
            )

            # sample
            if cnt_iter_in_loop < max_iters_in_loop:
                mylatent = sample_input_latents
            else:
                mylatent = sample_input_latents_2
            cnt_iter_in_loop += 1

            with autocast_ctx:
                rollout = pipeline_with_logprob_perstep(
                    unet=unet,
                    vae=vae,
                    scheduler=noise_scheduler,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=collection_neg_prompt_embeds[:len(prompt_embeds)],
                    num_inference_steps=args.sample_num_steps,
                    rg_scale=args.sample_rg_scale,
                    guidance_scale=args.sample_cfg_scale,
                    eta=args.sample_eta,
                    latents=mylatent,
                    output_type="pt",
                    return_dict=False,
                    unet_ref=unet_ref,
                    disable_progress_bar=not accelerator.is_local_main_process,
                    const_weight=args.const_weight,
                    algorithm=args.algorithm,
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
                "row_noises",
                "row_noise_preds",
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
                num_zero_nonterminal_train_timesteps if use_ode_kl_anchor else 0
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
            row_noises = rollout["row_noises"].index_select(0, train_row_indices)
            row_noise_preds = rollout["row_noise_preds"].index_select(0, train_row_indices)
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

            train_row_count = int(train_row_indices.numel())
            for name, tensor in (
                ("row_step_idx", row_step_idx),
                ("row_exploration_k", row_exploration_k),
                ("row_timesteps", row_timesteps),
                ("row_next_timesteps", row_next_timesteps),
                ("row_latents", row_latents),
                ("row_next_latents", row_next_latents),
                ("row_log_probs", row_log_probs),
                ("row_noises", row_noises),
                ("row_noise_preds", row_noise_preds),
            ):
                if tensor.shape[0] != train_row_count:
                    raise ValueError(
                        f"Packed rollout {name} row count mismatch, got {tensor.shape[0]} and {train_row_count}."
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

            assert_finite_tensor("rollout.row_latents", row_latents)
            assert_finite_tensor("rollout.row_next_latents", row_next_latents)
            assert_finite_tensor("rollout.kl_row_latents", kl_row_latents)
            if args.algorithm in ["ddpo", "logrho"]:
                assert_finite_tensor("rollout.row_log_probs", row_log_probs)
            else:
                assert_finite_tensor("rollout.row_noises", row_noises)
                assert_finite_tensor("rollout.row_noise_preds", row_noise_preds)

            row_sample_indices_cpu = row_sample_idx.detach().cpu().tolist()
            row_prompts = [prompts[idx] for idx in row_sample_indices_cpu]
            row_prompt_metadata = [prompt_metadata[idx] for idx in row_sample_indices_cpu]
            reward_future = executor.submit(
                reward_fn,
                row_images,
                row_prompts,
                row_prompt_metadata,
            )
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
                    "row_step_idx": row_step_idx,
                    "exploration_k": row_exploration_k,
                    "timesteps": row_timesteps,
                    "next_timesteps": row_next_timesteps,
                    "latents": row_latents,
                    "next_latents": row_next_latents,
                    "log_probs": row_log_probs,
                    "noises": row_noises,
                    "noise_preds": row_noise_preds,
                    "rewards": reward_future,
                }
            )
            if train_kl_row_count > 0:
                kl_anchor_samples.append(
                    {
                        "row_step_idx": kl_row_step_idx,
                        "timesteps": kl_row_timesteps,
                        "next_timesteps": kl_row_next_timesteps,
                        "latents": kl_row_latents,
                        "prompt_embeds": prompt_embeds.index_select(0, kl_row_sample_idx),
                    }
                )

            epoch_useful_row_steps_2b += int(rollout["useful_row_steps_2b"])
            epoch_processed_row_steps_2b += int(rollout["processed_row_steps_2b"])
            epoch_padded_row_steps_2b += int(rollout["padded_row_steps_2b"])

        # Wait for all rewards
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards_raw, reward_metadata = sample["rewards"].result()
            sample_batch_size = sample["timesteps"].shape[0]

            if isinstance(rewards_raw, dict):
                rewards_dict = {
                    key: reward_value_to_tensor(value, accelerator.device)
                    for key, value in rewards_raw.items()
                }
                if "avg" not in rewards_dict:
                    if len(rewards_dict) == 1:
                        rewards_dict["avg"] = next(iter(rewards_dict.values()))
                    else:
                        raise ValueError(
                            "Reward dict must include 'avg' or contain only one reward key."
                        )
            else:
                rewards_dict = {
                    "avg": reward_value_to_tensor(rewards_raw, accelerator.device)
                }

            for key, value in rewards_dict.items():
                if value.shape[0] != sample_batch_size:
                    raise ValueError(
                        "Reward batch size must match transition row count, got "
                        f"reward[{key}]={value.shape[0]} vs transitions={sample_batch_size}."
                    )

            sample["rewards"] = rewards_dict

        # Collate samples into dict where each entry has shape
        # (num_batches_per_epoch * collection_batch_size * sum(train_exploration_schedule), ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                reward_key: torch.cat([s[k][reward_key] for s in samples], dim=0)
                for reward_key in samples[0][k].keys()
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
            }
        if int(kl_anchor_rows["timesteps"].shape[0]) != expected_kl_anchor_rows_per_epoch:
            raise ValueError(
                "KL-anchor row count mismatch per epoch, got "
                f"{int(kl_anchor_rows['timesteps'].shape[0])} and {expected_kl_anchor_rows_per_epoch}."
            )

        if accelerator.is_main_process:
            padding_ratio_2b = (
                float(epoch_processed_row_steps_2b - epoch_useful_row_steps_2b)
                / float(epoch_processed_row_steps_2b)
                if epoch_processed_row_steps_2b > 0
                else 0.0
            )
            accelerator.log(
                {
                    "2b_useful_row_steps": epoch_useful_row_steps_2b,
                    "2b_processed_row_steps": epoch_processed_row_steps_2b,
                    "2b_padded_row_steps": epoch_padded_row_steps_2b,
                    "2b_padding_ratio": padding_ratio_2b,
                },
                step=global_step,
            )

        if epoch % 1 == 0 and accelerator.is_main_process and vis_payload is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                vis_images = vis_payload["images"]
                vis_prompts = vis_payload["prompts"]
                vis_rewards_raw, _ = vis_payload["reward_future"].result()
                if isinstance(vis_rewards_raw, dict):
                    vis_avg_raw = vis_rewards_raw.get("avg", next(iter(vis_rewards_raw.values())))
                else:
                    vis_avg_raw = vis_rewards_raw
                vis_avg = reward_value_to_tensor(
                    vis_avg_raw,
                    accelerator.device,
                ).detach().cpu().numpy()

                num_samples = min(15, len(vis_images))
                sample_indices = random.sample(range(len(vis_images)), num_samples)

                for idx, j in enumerate(sample_indices):
                    if torch.is_tensor(vis_images):
                        image = vis_images[j]
                    else:
                        image = vis_images[j]
                    if torch.is_tensor(image):
                        pil = Image.fromarray(
                            (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        )
                    else:
                        pil = image
                    pil = pil.resize((256, 256))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [vis_prompts[j] for j in sample_indices]
                sampled_rewards = [float(vis_avg[j]) for j in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.25} | {reward:.2f}",
                            )
                            for idx, (prompt, reward) in enumerate(
                                zip(sampled_prompts, sampled_rewards)
                            )
                        ],
                    },
                    step=global_step,
                )

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"].clone()
        # gather rewards across processes
        gathered_rewards = {
            key: accelerator.gather(value).cpu().numpy()
            for key, value in samples["rewards"].items()
        }

        # log rewards and images
        accelerator.log(
            {
                **{f"reward_{key}_mean": value.mean() for key, value in gathered_rewards.items()},
                **{f"reward_{key}_std": value.std() for key, value in gathered_rewards.items()},
                "epoch": epoch,
                "reward": gathered_rewards["avg"],
                "reward_mean": gathered_rewards["avg"].mean(),
                "reward_std": gathered_rewards["avg"].std(),
            },
            step=global_step,
        )

        # Normalize Reward => Advantage
        if group_strategy == "batch":
            advantages = (
                gathered_rewards["avg"] - gathered_rewards["avg"].mean()
            ) / (gathered_rewards["avg"].std() + 1e-4)
            if accelerator.is_main_process:
                accelerator.log({"group_strategy": group_strategy}, step=global_step)
        else:
            gathered_prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompt_labels = tokenizer.batch_decode(
                gathered_prompt_ids,
                skip_special_tokens=True,
            )
            gathered_timesteps = accelerator.gather(samples["timesteps"]).cpu().numpy().reshape(-1)
            if len(prompt_labels) != gathered_timesteps.shape[0]:
                raise ValueError(
                    "Prompt/timestep row alignment mismatch during stat tracking, got "
                    f"{len(prompt_labels)} and {gathered_timesteps.shape[0]}."
                )
            timestep_labels = [str(int(t)) for t in gathered_timesteps.tolist()]
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

            (
                advantages,
                group_size,
                group_count,
                zero_std_ratio,
                reward_std_mean,
            ) = compute_group_advantages(group_labels, gathered_rewards["avg"])
            if accelerator.is_main_process:
                accelerator.log(
                    {
                        "group_strategy": group_strategy,
                        "group_size": group_size,
                        "group_count": group_count,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )

        advantages = torch.as_tensor(
            advantages,
            device=accelerator.device,
            dtype=torch.float32,
        ).reshape(-1)
        if advantages.numel() % accelerator.num_processes != 0:
            raise ValueError(
                "Gathered advantage count must be divisible by number of processes, got "
                f"{advantages.numel()} and {accelerator.num_processes}."
            )
        samples["advantages"] = advantages.reshape(accelerator.num_processes, -1)[
            accelerator.process_index
        ]
        assert_finite_tensor("samples.advantages", samples["advantages"])

        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]
        del samples["seed_ids"]

        # Keep all rows for fixed update cadence; use row_mask to ignore zero-advantage rows in loss.
        row_mask = samples["advantages"].abs() != 0
        if int(row_mask.sum().item()) == 0:
            row_mask = torch.ones_like(row_mask, dtype=torch.bool)
        total_transition_rows = samples["timesteps"].shape[0]
        selected_transition_rows = int(row_mask.sum().item())
        raw_kl_anchor_rows = int(kl_anchor_rows["timesteps"].shape[0])
        if use_ode_kl_anchor:
            anchor_row_target = int(total_transition_rows * anchor_row_max_ratio)
            if (
                anchor_row_max_ratio > 0.0
                and raw_kl_anchor_rows > 0
                and anchor_row_target < 1
            ):
                anchor_row_target = 1
            effective_kl_anchor_rows = min(raw_kl_anchor_rows, anchor_row_target)
            if effective_kl_anchor_rows < raw_kl_anchor_rows:
                keep_indices = torch.randperm(
                    raw_kl_anchor_rows, device=accelerator.device
                )[:effective_kl_anchor_rows]
                kl_anchor_rows = {
                    key: value.index_select(0, keep_indices)
                    for key, value in kl_anchor_rows.items()
                }
        else:
            anchor_row_target = 0
            effective_kl_anchor_rows = raw_kl_anchor_rows
        effective_anchor_to_transition_ratio = (
            float(effective_kl_anchor_rows) / float(total_transition_rows)
            if total_transition_rows > 0
            else 0.0
        )
        clipfrac_logging_timesteps = extract_train_timestep_values_for_logging(
            samples["timesteps"],
            num_active_train_timesteps,
        )
        if accelerator.is_main_process:
            accelerator.log(
                {
                    "rollout_latent_chunk_size": latent_chunk_size,
                    "policy_update_batch_size": policy_update_batch_size,
                    "actual_num_rollout_chunks_by_latent_chunk": (
                        (total_transition_rows + latent_chunk_size - 1) // latent_chunk_size
                    ),
                    "actual_num_policy_chunks_by_policy_batch": (
                        (total_transition_rows + policy_update_batch_size - 1) // policy_update_batch_size
                    ),
                    "selected_transition_rows": selected_transition_rows,
                    "raw_kl_anchor_rows": raw_kl_anchor_rows,
                    "effective_kl_anchor_rows": effective_kl_anchor_rows,
                    "anchor_row_target": anchor_row_target,
                    "effective_anchor_to_transition_ratio": effective_anchor_to_transition_ratio,
                },
                step=global_step,
            )

        transition_rows = {
            "row_step_idx": samples["row_step_idx"],
            "exploration_k": samples["exploration_k"],
            "timesteps": samples["timesteps"],
            "next_timesteps": samples["next_timesteps"],
            "latents": samples["latents"],
            "next_latents": samples["next_latents"],
            "log_probs": samples["log_probs"],
            "noises": samples["noises"],
            "noise_preds": samples["noise_preds"],
            "advantages": samples["advantages"],
            "prompt_embeds": samples["prompt_embeds"],
            "row_mask": row_mask,
        }
        assert_finite_tensor("transition_rows.latents", transition_rows["latents"])
        assert_finite_tensor("transition_rows.next_latents", transition_rows["next_latents"])
        assert_finite_tensor("transition_rows.advantages", transition_rows["advantages"])
        if args.algorithm in ["ddpo", "logrho"]:
            assert_finite_tensor("transition_rows.log_probs", transition_rows["log_probs"])
        else:
            assert_finite_tensor("transition_rows.noises", transition_rows["noises"])
            assert_finite_tensor("transition_rows.noise_preds", transition_rows["noise_preds"])

        base_row_indices = torch.arange(
            total_transition_rows,
            device=accelerator.device,
            dtype=torch.long,
        )
        policy_unit_row_indices = base_row_indices
        if policy_unit_row_indices.numel() == 0:
            raise ValueError(
                "Policy-update unit construction produced zero rows. "
                "Check train_timestep_fraction and branching schedule."
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
            accelerator.log(
                {
                    "actual_transition_rows": total_transition_rows,
                    "actual_kl_anchor_rows": effective_kl_anchor_rows,
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
        #################### TRAINING ####################
        optimizer_updates_this_epoch = 0
        expected_optimizer_updates_this_epoch = (
            updates_per_epoch * int(args.train_num_inner_epochs)
        )
        for inner_epoch in range(args.train_num_inner_epochs):
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
            unet.train()
            info = defaultdict(list)
            optimizer_updates_this_inner_epoch = 0

            # Storage for ratio and gradient statistics
            all_ratios = []
            grad_stats = {'mean': [], 'var': []}

            # Define gradient hook to capture statistics
            def grad_hook(grad):
                if grad is not None:
                    grad_flat = grad.detach().flatten().to(torch.float32)
                    grad_abs = torch.abs(grad_flat)  # Take absolute value
                    grad_stats['mean'].append(grad_abs.mean().item())
                    grad_stats['var'].append(grad_abs.var().item())
                return None

            # Register hook on the first LoRA layer with gradients
            hook_handle = None
            for name, param in unet.named_parameters():
                if param.requires_grad and (not args.use_lora or 'lora' in name.lower()):
                    hook_handle = param.register_hook(grad_hook)
                    logger.info(f"Registered gradient hook on: {name}")
                    break
            if hook_handle is None:
                for name, param in unet.named_parameters():
                    if param.requires_grad:
                        hook_handle = param.register_hook(grad_hook)
                        logger.info(f"Registered fallback gradient hook on: {name}")
                        break

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
                    if key != "row_mask"
                }
                sample["row_mask"] = transition_rows["row_mask"].index_select(
                    0, sample_row_indices
                ) & sample_unit_mask

                if args.train_cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    neg_prompt_batch = neg_prompt_embed.repeat(len(sample["prompt_embeds"]), 1, 1)
                    embeds = torch.cat([neg_prompt_batch, sample["prompt_embeds"]])
                else:
                    embeds = sample["prompt_embeds"]

                with accelerator.accumulate(unet):
                    with autocast():
                        if args.train_cfg:
                            noise_pred = unet(
                                torch.cat([sample["latents"]] * 2),
                                torch.cat([sample["timesteps"]] * 2),
                                embeds,
                            ).sample
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = (
                                noise_pred_uncond
                                + args.sample_cfg_scale
                                * (noise_pred_text - noise_pred_uncond)
                            )
                            noise_pred_ref = unet_ref(
                                torch.cat([sample["latents"]] * 2),
                                torch.cat([sample["timesteps"]] * 2),
                                embeds,
                            ).sample
                            noise_pred_ref_uncond, noise_pred_ref_text = noise_pred_ref.chunk(2)
                            noise_pred_ref = (
                                noise_pred_ref_uncond
                                + args.sample_cfg_scale
                                * (noise_pred_ref_text - noise_pred_ref_uncond)
                            )
                        else:
                            noise_pred = unet(
                                sample["latents"],
                                sample["timesteps"],
                                embeds,
                            ).sample
                            noise_pred_text = noise_pred
                            noise_pred_ref = unet_ref(
                                sample["latents"],
                                sample["timesteps"],
                                embeds,
                            ).sample

                        assert_finite_tensor("train.noise_pred", noise_pred)
                        assert_finite_tensor("train.noise_pred_ref", noise_pred_ref)

                        if args.algorithm in ["ddpo", "logrho"]:
                            _, log_prob, _ = ddim_step_with_logprob(
                                noise_scheduler,
                                noise_pred,
                                None,
                                noise_pred_text,
                                None,
                                sample["noises"],
                                sample["timesteps"],
                                sample["latents"],
                                rg_scale=args.sample_rg_scale,
                                eta=args.sample_eta,
                                prev_sample=sample["next_latents"],
                                const_weight=args.const_weight,
                                algorithm=args.algorithm,
                                num_inference_steps=args.sample_num_steps,
                            )
                            assert_finite_tensor("train.recomputed_log_prob", log_prob)

                        if args.const_weight is not None:
                            weight_t = torch.full(
                                (sample["timesteps"].shape[0], 1, 1, 1),
                                args.const_weight,
                                device=sample["timesteps"].device,
                                dtype=noise_pred.dtype,
                            )
                        else:
                            weights_inference = compute_inference_weights_for_rows(
                                noise_scheduler,
                                sample["timesteps"],
                                accelerator.device,
                                num_inference_steps=args.sample_num_steps,
                            ).view(-1, 1, 1, 1)
                            weight_t = weights_inference.to(dtype=noise_pred.dtype)
                        assert_finite_tensor("train.weight_t", weight_t)

                    row_mask_float = sample["row_mask"].to(dtype=torch.float32)
                    valid_count = torch.clamp(row_mask_float.sum(), min=1.0)
                    exploration_k = torch.clamp(
                        sample["exploration_k"].to(dtype=torch.float32), min=1.0
                    )
                    exploration_loss_weight = (
                        exploration_k_avg_for_loss / exploration_k
                    ).to(dtype=torch.float32)
                    advantages = torch.clamp(
                        sample["advantages"],
                        -args.train_adv_clip_max,
                        args.train_adv_clip_max,
                    )

                    if args.algorithm in ["ddpo", "logrho"]:
                        assert_finite_tensor("train.sample_log_probs", sample["log_probs"])
                        log_rho = log_prob - sample["log_probs"]
                        assert_finite_tensor("train.log_rho", log_rho)
                        rho = torch.exp(log_rho)
                        assert_finite_tensor("train.rho", rho)

                        if args.algorithm == "ddpo":
                            all_ratios.append(rho.detach())
                            unclipped_loss = -advantages * rho
                            clipped_loss = -advantages * torch.clamp(
                                rho,
                                1 - args.train_clip_range,
                                1 + args.train_clip_range
                            )
                            clip_event = (torch.abs(rho - 1) > args.train_clip_range)
                        else:
                            unclipped_loss = -advantages * (1 + log_rho)
                            clipped_loss = -advantages * torch.clamp(
                                1 + log_rho,
                                1 - args.train_clip_range,
                                1 + args.train_clip_range
                            )
                            clip_event = (torch.abs(log_rho) > args.train_clip_range)

                        policy_per_row = torch.maximum(unclipped_loss, clipped_loss)
                        policy_per_row = policy_per_row * exploration_loss_weight
                        assert_finite_tensor("train.policy_per_row", policy_per_row)
                        loss = (policy_per_row * row_mask_float).sum() / valid_count

                        eps = 1e-10
                        near_mask = (rho - 1).abs() < eps
                        log_approx_error = torch.where(
                            near_mask,
                            torch.zeros_like(log_rho),
                            (log_rho - (rho - 1)) / (rho - 1)
                        )
                    else:
                        noise_diff = noise_pred - sample["noise_preds"]
                        noise = sample["noises"]

                        noise_matching_term0 = 0.5 * noise.pow(2).mean(dim=[1, 2, 3], dtype=torch.float32)
                        noise_matching_term1 = 0.5 * (weight_t * noise_diff / args.weight_scale_factor).pow(2).mean(dim=[1, 2, 3], dtype=torch.float32)
                        noise_matching_term2 = ((weight_t * noise_diff) * noise / args.weight_scale_factor).mean(dim=[1, 2, 3], dtype=torch.float32)
                        if args.noise_matching_precision == "best":
                            noise_matching_term = noise_matching_term2
                        elif args.noise_matching_precision == "bad":
                            noise_matching_term = noise_matching_term1 + noise_matching_term2
                        elif args.noise_matching_precision == "worst":
                            tmp = noise_matching_term0 + noise_matching_term1 + noise_matching_term2
                            noise_matching_term = tmp - noise_matching_term0
                        else:
                            raise ValueError(f"Unknown noise matching precision {args.noise_matching_precision}")
                        assert_finite_tensor("train.noise_matching_term", noise_matching_term)

                        if args.train_do_clip:
                            policy_per_row = torch.maximum(
                                torch.zeros_like(advantages),
                                args.train_clip_range * torch.abs(advantages)
                                + advantages * noise_matching_term,
                            )
                            clip_event = (
                                args.train_clip_range * torch.abs(advantages)
                                + advantages * noise_matching_term < 0
                            )
                        else:
                            policy_per_row = (
                                args.train_clip_range * torch.abs(advantages)
                                + advantages * noise_matching_term
                            )
                            clip_event = torch.zeros_like(advantages, dtype=torch.bool)

                        policy_per_row = policy_per_row * exploration_loss_weight
                        assert_finite_tensor("train.policy_per_row", policy_per_row)
                        loss = (policy_per_row * row_mask_float).sum() / valid_count

                    if use_ode_kl_anchor and kl_anchor_row_indices.numel() > 0:
                        kl_anchor_sample = {
                            key: value.index_select(0, kl_anchor_row_indices)
                            for key, value in kl_anchor_rows.items()
                            if key != "row_step_idx"
                        }

                        if args.train_cfg:
                            kl_neg_prompt_batch = neg_prompt_embed.repeat(
                                len(kl_anchor_sample["prompt_embeds"]), 1, 1
                            )
                            kl_embeds = torch.cat(
                                [kl_neg_prompt_batch, kl_anchor_sample["prompt_embeds"]]
                            )
                            kl_noise_pred = unet(
                                torch.cat([kl_anchor_sample["latents"]] * 2),
                                torch.cat([kl_anchor_sample["timesteps"]] * 2),
                                kl_embeds,
                            ).sample
                            kl_noise_pred_uncond, kl_noise_pred_text = kl_noise_pred.chunk(2)
                            kl_noise_pred = (
                                kl_noise_pred_uncond
                                + args.sample_cfg_scale
                                * (kl_noise_pred_text - kl_noise_pred_uncond)
                            )
                            kl_noise_pred_ref = unet_ref(
                                torch.cat([kl_anchor_sample["latents"]] * 2),
                                torch.cat([kl_anchor_sample["timesteps"]] * 2),
                                kl_embeds,
                            ).sample
                            (
                                kl_noise_pred_ref_uncond,
                                kl_noise_pred_ref_text,
                            ) = kl_noise_pred_ref.chunk(2)
                            kl_noise_pred_ref = (
                                kl_noise_pred_ref_uncond
                                + args.sample_cfg_scale
                                * (kl_noise_pred_ref_text - kl_noise_pred_ref_uncond)
                            )
                        else:
                            kl_noise_pred = unet(
                                kl_anchor_sample["latents"],
                                kl_anchor_sample["timesteps"],
                                kl_anchor_sample["prompt_embeds"],
                            ).sample
                            kl_noise_pred_ref = unet_ref(
                                kl_anchor_sample["latents"],
                                kl_anchor_sample["timesteps"],
                                kl_anchor_sample["prompt_embeds"],
                            ).sample
                        assert_finite_tensor("train.kl_noise_pred", kl_noise_pred)
                        assert_finite_tensor("train.kl_noise_pred_ref", kl_noise_pred_ref)

                        if args.const_weight is not None:
                            kl_weight_t = torch.full(
                                (kl_anchor_sample["timesteps"].shape[0], 1, 1, 1),
                                args.const_weight,
                                device=kl_anchor_sample["timesteps"].device,
                                dtype=kl_noise_pred.dtype,
                            )
                        else:
                            kl_weights_inference = compute_inference_weights_for_rows(
                                noise_scheduler,
                                kl_anchor_sample["timesteps"],
                                accelerator.device,
                                num_inference_steps=args.sample_num_steps,
                            ).view(-1, 1, 1, 1)
                            kl_weight_t = kl_weights_inference.to(dtype=kl_noise_pred.dtype)
                        assert_finite_tensor("train.kl_weight_t", kl_weight_t)
                        kl_anchor_per_row = 0.5 * torch.mean(
                            kl_weight_t.pow(2)
                            * (kl_noise_pred - kl_noise_pred_ref).pow(2),
                            dim=[1, 2, 3],
                        )
                        assert_finite_tensor("train.kl_anchor_per_row", kl_anchor_per_row)
                        kl_anchor_loss = kl_anchor_per_row.mean()
                    else:
                        kl_anchor_loss = torch.zeros(
                            (),
                            device=loss.device,
                            dtype=loss.dtype,
                        )

                    if use_ode_kl_anchor:
                        loss = loss + (ode_kl_anchor_beta * kl_anchor_loss)

                    assert_finite_tensor("train.loss", loss)
                    assert_finite_tensor("train.kl_anchor_loss", kl_anchor_loss)
                    clipfrac = (clip_event.float() * row_mask_float).sum() / valid_count
                    info["clipfrac"].append(clipfrac)
                    info["loss"].append(loss)
                    if use_ode_kl_anchor:
                        info["kl_anchor_loss"].append(kl_anchor_loss)

                    batch_approx_kl = 0.5 * torch.mean(
                        weight_t.pow(2) * (noise_pred - noise_pred_ref).pow(2),
                        dim=[1, 2, 3]
                    )
                    assert_finite_tensor("train.batch_approx_kl", batch_approx_kl)
                    approx_kl_value = (batch_approx_kl * row_mask_float).sum() / valid_count
                    assert_finite_tensor("train.approx_kl_value", approx_kl_value)
                    info["approx_kl"].append(approx_kl_value)

                    clipfrac_per_timestep = compute_clipfrac_per_timestep(
                        clip_event=clip_event,
                        timesteps=sample["timesteps"],
                        row_mask=sample["row_mask"],
                        tracked_timestep_values=clipfrac_logging_timesteps,
                    )
                    for metric_name, metric_value in clipfrac_per_timestep.items():
                        info[metric_name].append(metric_value)

                    if args.algorithm in ["ddpo", "logrho"]:
                        log_approx_error_value = (
                            log_approx_error.abs() * row_mask_float
                        ).sum() / valid_count
                        info["log_approx_error"].append(log_approx_error_value)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            unet.parameters(), args.train_max_grad_norm
                        )
                    optimizer.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch})

                    if args.algorithm == "ddpo" and len(all_ratios) > 0:
                        all_ratios_tensor = torch.cat(all_ratios, dim=0)
                        info.update({
                            "ratio_mean": all_ratios_tensor.mean().item(),
                            "ratio_std": all_ratios_tensor.std().item(),
                            "ratio_min": all_ratios_tensor.min().item(),
                            "ratio_max": all_ratios_tensor.max().item(),
                        })

                    if len(grad_stats["mean"]) > 0:
                        info.update({
                            "grad_mean": np.mean(grad_stats["mean"]),
                            "grad_var": np.mean(grad_stats["var"]),
                        })

                    accelerator.log(info, step=global_step)
                    global_step += 1
                    optimizer_updates_this_inner_epoch += 1
                    optimizer_updates_this_epoch += 1
                    info = defaultdict(list)

            # Remove gradient hook at the end of inner epoch
            if hook_handle is not None:
                hook_handle.remove()

            if optimizer_updates_this_inner_epoch != updates_per_epoch:
                raise ValueError(
                    "No-carry invariant failed: optimizer updates per inner epoch mismatch, got "
                    f"{optimizer_updates_this_inner_epoch} and {updates_per_epoch}. "
                    "Check updates_per_epoch and accumulation settings."
                )

        if optimizer_updates_this_epoch != expected_optimizer_updates_this_epoch:
            raise ValueError(
                "No-carry invariant failed: optimizer updates per epoch mismatch, got "
                f"{optimizer_updates_this_epoch} and {expected_optimizer_updates_this_epoch}."
            )

        # Calculate total epoch time (sampling + training) and log it
        total_epoch_time = time.time() - epoch_start_time
        sample_time_per_epoch = training_start_time - epoch_start_time
        policy_update_time_per_epoch = total_epoch_time - sample_time_per_epoch
        if accelerator.is_main_process:
            accelerator.log(
                {
                    "total_time_per_epoch": total_epoch_time,
                    "sample_time_per_epoch": sample_time_per_epoch,
                    "policy_update_time_per_epoch": policy_update_time_per_epoch,
                    "optimizer_updates_this_epoch": optimizer_updates_this_epoch,
                    "optimizer_updates_target_per_epoch": expected_optimizer_updates_this_epoch,
                },
                step=global_step,
            )

        if (epoch % args.save_freq == args.save_freq - 1) and accelerator.is_main_process:
            ckpt_dir = os.path.join("logs", args.run_name)
            os.makedirs(ckpt_dir, exist_ok=True)

            state = {
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "stat_tracker": None
            }

            if args.use_lora:
                peft_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(unet))
                torch.save({
                    "lora_config": asdict(unet_lora_config),
                    "lora_weights": peft_state_dict,
                    "optimizer_state": state,
                }, os.path.join(ckpt_dir, f"checkpoint_{epoch}.pt"))
            else:
                torch.save({
                    "unet_state_dict": accelerator.unwrap_model(unet).state_dict(),
                    "optimizer_state": state,
                }, os.path.join(ckpt_dir, f"checkpoint_{epoch}.pt"))


if __name__ == "__main__":
    main()
