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
import random
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
from utils.stat_tracking import PerPromptStatTracker
from diffusion.ddim_step import ddim_step_with_logprob
from diffusion.reverse_pipeline import pipeline_with_logprob

import tempfile
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
    eval_batch_size = getattr(args, 'eval_batch_size', args.sample_batch_size)
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


def main():
    args = parse_args()
    args.use_lora = getattr(args, "use_lora", False)
    args.prompt_file = getattr(args, "prompt_file", "./prompts_dancegrpo.txt")
    total_samples_per_epoch = args.num_prompts_per_epoch * args.num_generations_per_prompt
    if total_samples_per_epoch % args.sample_batch_size != 0:
        raise ValueError(
            "sample_batch_size must divide num_prompts_per_epoch * num_generations_per_prompt."
        )
    args.sample_num_batches_per_epoch = total_samples_per_epoch // args.sample_batch_size

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    unique_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if not args.run_name:
        args.run_name = unique_id

    num_train_timesteps = int((args.sample_num_steps - 1) * args.train_timestep_fraction)

    logging_dir = os.path.join(args.logging_dir, args.run_name)

    accelerator_config = ProjectConfiguration(
        project_dir=logging_dir,
        automatic_checkpoint_naming=False
    )
    accelerator = Accelerator(
        log_with=args.report_to,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=args.train_gradient_accumulation_steps * num_train_timesteps,
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
        gradient_accumulation_steps = getattr(
            args, "gradient_accumulation_steps", args.train_gradient_accumulation_steps
        )
        args.learning_rate = (
            args.learning_rate
            * gradient_accumulation_steps
            * args.train_batch_size
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
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=args.seed,
        shuffle=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.num_prompts_per_epoch,
        sampler=sampler,
        pin_memory=True,
        drop_last=True,
    )
    reward_fn = getattr(utils.rewards, args.reward_fn)()
    eval_num_batches = getattr(args, "eval_num_batches", 4)
    eval_batch_size = getattr(args, "eval_batch_size", args.sample_batch_size)
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
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(args.sample_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(args.train_batch_size, 1, 1)

    # Initialize stat tracker
    stat_tracker = None

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
    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Sample batch size per device = {args.sample_batch_size}")
    logger.info(f"  Train batch size per device = {args.train_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.train_gradient_accumulation_steps}")
    logger.info(f"  Effective sample batch size (w. parallel, distributed & accumulation) ="
                f" {args.sample_batch_size * accelerator.num_processes * args.sample_num_batches_per_epoch}")
    logger.info(f"  Effective train batch size (w. parallel, distributed & accumulation) = "
                f"{args.train_batch_size * accelerator.num_processes * args.train_gradient_accumulation_steps}")
    logger.info(f"  Model updates per epoch = "
                f"{args.sample_batch_size * args.sample_num_batches_per_epoch \
                    // (args.train_batch_size * args.train_gradient_accumulation_steps)}")

    # Main loop
    for epoch in range(first_epoch, args.num_train_epochs):
        # Start epoch timer (from sampling to end of training)
        epoch_start_time = time.time()
        sampler.set_epoch(epoch)
        prompt_batch = next(iter(loader))

        expanded_prompts = []
        for prompt in prompt_batch:
            expanded_prompts.extend([prompt] * args.num_generations_per_prompt)
        if len(expanded_prompts) != args.sample_batch_size * args.sample_num_batches_per_epoch:
            raise ValueError(
                "Expanded prompt count does not match sample_batch_size * sample_num_batches_per_epoch."
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
        prompts = []

        global_input_latents = torch.randn(
            (1, 4, 64, 64),
            device=accelerator.device,
            dtype=weight_dtype,
        )
        sample_input_latents = global_input_latents.repeat(args.sample_batch_size, 1, 1, 1).clone()
        global_input_latents_2 = torch.randn(
            (1, 4, 64, 64),
            device=accelerator.device,
            dtype=weight_dtype,
        )
        sample_input_latents_2 = global_input_latents_2.repeat(args.sample_batch_size, 1, 1, 1).clone()
        max_iters_in_loop = max(1, args.sample_num_batches_per_epoch // 2)
        cnt_iter_in_loop = 0

        for i in tqdm(
            range(args.sample_num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            start_idx = i * args.sample_batch_size
            end_idx = (i + 1) * args.sample_batch_size
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

            # sample
            if cnt_iter_in_loop < max_iters_in_loop:
                mylatent = sample_input_latents
            else:
                mylatent = sample_input_latents_2
            cnt_iter_in_loop += 1

            with autocast_ctx:
                images, _, latents, log_probs, noises, noise_pred_texts, \
                noise_pred_refs, noise_pred_ref_texts, \
                noise_preds = pipeline_with_logprob(
                    unet,
                    vae,
                    noise_scheduler,
                    tokenizer,
                    text_encoder,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=args.sample_num_steps,
                    rg_scale=args.sample_rg_scale,
                    guidance_scale=args.sample_cfg_scale,
                    eta=args.sample_eta,
                    latents=mylatent,
                    output_type="pt",
                    unet_ref=unet_ref,
                    disable_progress_bar=not accelerator.is_local_main_process,
                    const_weight=args.const_weight,
                    algorithm=args.algorithm,
                )

            latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
            noises = torch.stack(noises, dim=1)  # (batch_size, num_steps, 4, 64, 64)
            noise_pred_texts = torch.stack(noise_pred_texts, dim=1)
            noise_preds = torch.stack(noise_preds, dim=1)
            if args.algorithm in ["ddpo", "logrho"]:
                log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)

            timesteps = noise_scheduler.timesteps.repeat(args.sample_batch_size, 1)  # (batch_size, num_steps)

            # Compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            # Yield to make sure reward computation starts
            time.sleep(0)

            if args.algorithm in ["ddpo", "logrho"]:
                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "timesteps": timesteps[:, :-1],
                        "latents": latents[:, :-1][:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:][:, :-1],
                        "noises": noises[:, :-1],
                        "rewards": rewards,

                        "noise_preds": noise_preds[:, :-1],
                        "log_probs": log_probs[:, :-1],
                    }
                )
            else:
                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "timesteps": timesteps[:, :-1],
                        "latents": latents[:, :-1][:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:][:, :-1],
                        "noises": noises[:, :-1],
                        "rewards": rewards,

                        "noise_preds": noise_preds[:, :-1],
                        "noise_pred_texts": noise_pred_texts[:, :-1],
                    }
                )

        # Wait for all rewards
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # Collate samples into dict where each entry has shape
        # (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # Hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray(
                    (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerator.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt:.25} | {reward:.2f}",
                        )
                        for i, (prompt, reward) in enumerate(
                            zip(prompts, rewards)
                        )  # only log rewards from process 0
                    ],
                },
                step=global_step,
            )

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # log rewards and images
        accelerator.log(
            {
                "reward": rewards,
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

        # Normalize Reward => Advantage (grouped by prompt for GRPO)
        group_size = args.num_generations_per_prompt
        if len(samples["rewards"]) % group_size != 0:
            raise ValueError("Number of samples is not divisible by num_generations_per_prompt.")
        advantages = torch.zeros_like(samples["rewards"], dtype=torch.float32)
        for start_idx in range(0, len(samples["rewards"]), group_size):
            end_idx = start_idx + group_size
            group_rewards = samples["rewards"][start_idx:end_idx].to(torch.float64)
            group_mean = group_rewards.mean(dtype=torch.float64)
            group_std = group_rewards.std(unbiased=False) + 1e-8
            advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
        samples["advantages"] = advantages

        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == args.sample_batch_size * args.sample_num_batches_per_epoch
        )
        assert num_timesteps == args.sample_num_steps - 1

        #################### TRAINING ####################
        for inner_epoch in range(args.train_num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=accelerator.device)
                    for _ in range(total_batch_size)
                ]
            )
            if args.algorithm in ["ddpo", "logrho"]:
                for key in ["timesteps", "latents", "next_latents",
                            "noises", "log_probs", "noise_preds",
                            ]:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=accelerator.device)[:, None],
                        perms,
                    ]
            else:
                for key in ["timesteps", "latents", "next_latents",
                            "noises", "noise_preds",
                            "noise_pred_texts",
                            ]:
                    samples[key] = samples[key][
                        torch.arange(total_batch_size, device=accelerator.device)[:, None],
                        perms,
                    ]

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, args.train_batch_size, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            unet.train()
            info = defaultdict(list)

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

            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if args.train_cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds, sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                batch_size, num_steps = sample['timesteps'].shape
                weights_inference = compute_inference_weights(
                    noise_scheduler,
                    sample["timesteps"],
                    accelerator.device,
                )
                selected_weights = weights_inference.view(batch_size, num_steps, 1, 1, 1)

                for j in tqdm(
                    range(num_train_timesteps),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(unet):
                        with autocast():
                            if args.train_cfg:
                                noise_pred = unet(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
                                    embeds,
                                ).sample
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                                noise_pred = (
                                    noise_pred_uncond
                                    + args.sample_cfg_scale
                                    * (noise_pred_text - noise_pred_uncond)
                                )
                                noise_pred_ref = unet_ref(
                                    torch.cat([sample["latents"][:, j]] * 2),
                                    torch.cat([sample["timesteps"][:, j]] * 2),
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
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                                noise_pred_text = noise_pred
                                noise_pred_ref = unet_ref(
                                    sample["latents"][:, j],
                                    sample["timesteps"][:, j],
                                    embeds,
                                ).sample
                            if args.algorithm in ["ddpo", "logrho"]:
                                _, log_prob, _ = ddim_step_with_logprob(
                                    noise_scheduler,
                                    noise_pred,
                                    None, # sample['noise_pred_refs'][:, j],
                                    noise_pred_text,
                                    None, # sample['noise_pred_ref_texts'][:, j],
                                    sample['noises'][:, j],
                                    sample['timesteps'][:, j],
                                    sample['latents'][:, j],
                                    rg_scale=args.sample_rg_scale,
                                    eta=args.sample_eta,
                                    prev_sample=sample['next_latents'][:, j],
                                    const_weight=args.const_weight,
                                    algorithm=args.algorithm,
                                )
                        weight_t = get_step_weight(
                            args,
                            selected_weights,
                            batch_size,
                            j,
                        )
                        # PG logic
                        advantages = torch.clamp(
                            sample["advantages"],
                            -args.train_adv_clip_max,
                            args.train_adv_clip_max,
                        )
                        if args.algorithm in ["ddpo", "logrho"]:
                            log_rho = log_prob - sample['log_probs'][:, j]
                            rho = torch.exp(log_rho)

                            # Track ratio distribution for logging
                            if args.algorithm == "ddpo":
                                all_ratios.append(rho.detach())

                            if args.algorithm == "ddpo":
                                unclipped_loss = -advantages * rho
                                clipped_loss = -advantages * torch.clamp(
                                    rho,
                                    1 - args.train_clip_range,
                                    1 + args.train_clip_range
                                )
                                clipfrac = torch.mean(
                                    (torch.abs(rho - 1) > args.train_clip_range).float()
                                )
                            else:
                                unclipped_loss = -advantages * (1 + log_rho)
                                clipped_loss = -advantages * torch.clamp(
                                    1 + log_rho,
                                    1 - args.train_clip_range,
                                    1 + args.train_clip_range
                                )
                                clipfrac = torch.mean(
                                    (torch.abs(log_rho) > args.train_clip_range).float()
                                )
                            loss = torch.mean(
                                torch.maximum(
                                    unclipped_loss,
                                    clipped_loss
                                )
                            )
                            eps = 1e-10
                            mask = (rho - 1).abs() < eps
                            log_approx_error = torch.where(
                                mask,
                                torch.zeros_like(log_rho),
                                (log_rho - (rho - 1)) / (rho - 1)
                            )
                        else:
                            noise_diff = (noise_pred - sample['noise_preds'][:, j])
                            noise = sample['noises'][:, j]

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

                            if args.train_do_clip:
                                loss = torch.mean(torch.maximum(
                                    torch.zeros_like(advantages),
                                    args.train_clip_range * torch.abs(advantages)
                                        + advantages * noise_matching_term
                                ))
                                clipfrac = torch.mean(
                                    (
                                        args.train_clip_range * torch.abs(advantages)
                                        + advantages * noise_matching_term < 0
                                    ).float()
                                )
                            else:
                                loss = torch.mean(
                                    args.train_clip_range * torch.abs(advantages)
                                        + advantages * noise_matching_term
                                )
                                clipfrac = torch.tensor(
                                    0.0,
                                    device=accelerator.device,
                                    dtype=torch.float32
                                )

                        info["clipfrac"].append(clipfrac)
                        info["loss"].append(loss)
                        # Compute approx_kl: compute per-sample KL first, then average
                        batch_approx_kl = 0.5 * torch.mean(
                            weight_t.pow(2) * (noise_pred - noise_pred_ref).pow(2),
                            dim=[1, 2, 3]
                        )
                        info["approx_kl"].append(batch_approx_kl.mean())
                        if args.algorithm in ["ddpo", "logrho"]:
                            info["log_approx_error"].append(torch.mean(log_approx_error.abs()))

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                unet.parameters(), args.train_max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        assert (j == num_train_timesteps - 1) and (
                            i + 1
                        ) % args.train_gradient_accumulation_steps == 0
                        # Log training-related stuff to accelerator
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})

                        # Log ratio distribution statistics if DDPO
                        if args.algorithm == "ddpo" and len(all_ratios) > 0:
                            all_ratios_tensor = torch.cat(all_ratios, dim=0)
                            info.update({
                                "ratio_mean": all_ratios_tensor.mean().item(),
                                "ratio_std": all_ratios_tensor.std().item(),
                                "ratio_min": all_ratios_tensor.min().item(),
                                "ratio_max": all_ratios_tensor.max().item(),
                            })

                        # Log gradient statistics
                        if len(grad_stats['mean']) > 0:
                            info.update({
                                "grad_mean": np.mean(grad_stats['mean']),
                                "grad_var": np.mean(grad_stats['var']),
                            })

                        accelerator.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)

            # make sure we did an optimization step at the end of the inner epoch
            assert accelerator.sync_gradients

            # Remove gradient hook at the end of inner epoch
            if hook_handle is not None:
                hook_handle.remove()

        # Calculate total epoch time (sampling + training) and log it
        total_epoch_time = time.time() - epoch_start_time
        if accelerator.is_main_process:
            accelerator.log(
                {
                    "total_time_per_epoch": total_epoch_time,
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
            if getattr(args, "per_prompt_stat_tracking", None):
                state["stat_tracker"] = stat_tracker.state_dict()

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
