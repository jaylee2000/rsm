from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import h5py
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
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
from torch.utils.data import DataLoader, BatchSampler
from flow_grpo.ema import EMAModuleWrapper
from dataset.sd35_dataset import PrecomputedEmbeddingDataset

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class KRepeatSampler(BatchSampler):
    def __init__(self, dataset, batch_size, image_per_prompt, prompt_per_epoch, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.image_per_prompt = image_per_prompt  # Number of images per prompt
        self.prompt_per_epoch = prompt_per_epoch  # Number of prompts per epoch
        self.seed = seed                          # Random seed
        
        self.total_samples = self.image_per_prompt * self.prompt_per_epoch
        self.epoch = 0

        self.num_batches = self.total_samples // self.batch_size

    def __iter__(self):
        # Generate a deterministic random sequence to ensure all replicas are synchronized
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Randomly select m unique prompts
        indices = torch.randperm(len(self.dataset), generator=g)[:self.prompt_per_epoch].tolist()
        
        # Repeat each prompt image_per_prompt times
        repeated_indices = [idx for idx in indices for _ in range(self.image_per_prompt)]

        # Shuffle to ensure uniform distribution
        shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
        shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

        # Yield micro-batches
        for i in range(0, len(shuffled_samples), self.batch_size):
            batch = shuffled_samples[i:i + self.batch_size]
            yield batch

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators


def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
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
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, sigma_t, sqrt_dt = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
        return_sqrt_dt=True,
    )
    tilde_sigma_t = sigma_t / torch.clamp(sqrt_dt, min=1e-12)

    return prev_sample, log_prob, prev_sample_mean, tilde_sigma_t


def compute_noise_pred(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
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
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    # prev_sample_mean calculation for KL regularization in matching loss.
    timestep = sample["timesteps"][:, j]
    sample_latents = sample["latents"][:, j].float()
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


def eval(pipeline, test_dataloader, tokenizer, neg_prompt_embed, neg_pooled_prompt_embed,
         config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps,
         ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    eval_kl_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    eval_kl_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)

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
        prompt_metadata = test_batch.get("metadatas", [{}] * len(prompts))
        neg_prompt_embeds = neg_prompt_embed.repeat(len(prompts), 1, 1)
        neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(len(prompts), 1)

        if accelerator.mixed_precision == "fp16":
            prompt_embeds = prompt_embeds.half()
            pooled_prompt_embeds = pooled_prompt_embeds.half()
            neg_prompt_embeds = neg_prompt_embeds.half()
            neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.half()

        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, latents, _ = pipeline_with_logprob(
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
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)

        latents = torch.stack(latents, dim=1)
        timesteps = pipeline.scheduler.timesteps.repeat(latents.shape[0], 1)
        eval_steps = min(num_train_timesteps, latents.shape[1] - 1)
        if eval_steps > 0:
            eval_sample = {
                "latents": latents[:, :-1],
                "next_latents": latents[:, 1:],
                "timesteps": timesteps,
            }
            if config.train.cfg:
                embeds = torch.cat([neg_prompt_embeds, prompt_embeds])
                pooled_embeds = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds])
            else:
                embeds = prompt_embeds
                pooled_embeds = pooled_prompt_embeds

            with autocast():
                with torch.no_grad():
                    for j in range(eval_steps):
                        timestep = eval_sample["timesteps"][:, j].to(dtype=torch.float32)
                        next_timestep = eval_sample["timesteps"][:, j + 1].to(dtype=torch.float32)
                        delta_t = (timestep - next_timestep) / 1000.0

                        if config.train.loss_type == "ppo":
                            _, _, prev_sample_mean, tilde_sigma_t = compute_log_prob(
                                pipeline.transformer, pipeline, eval_sample, j, embeds, pooled_embeds, config
                            )
                            with _disable_adapter_context(pipeline.transformer):
                                _, _, prev_sample_mean_ref, _ = compute_log_prob(
                                    pipeline.transformer, pipeline, eval_sample, j, embeds, pooled_embeds, config
                                )
                            kl_denom = (tilde_sigma_t ** 2) * delta_t[:, None, None, None]
                        else:
                            prev_sample_mean, tilde_sigma_t, _ = compute_noise_pred(
                                pipeline.transformer, pipeline, eval_sample, j, embeds, pooled_embeds, config
                            )
                            with _disable_adapter_context(pipeline.transformer):
                                prev_sample_mean_ref, _, _ = compute_noise_pred(
                                    pipeline.transformer, pipeline, eval_sample, j, embeds, pooled_embeds, config
                                )
                            tilde_sigma_t = tilde_sigma_t.view(-1)
                            kl_denom = (tilde_sigma_t ** 2) * delta_t

                        kl_loss = (
                            (prev_sample_mean - prev_sample_mean_ref) ** 2
                        ).mean(dim=(1, 2, 3), keepdim=True) / (2 * kl_denom)
                        kl_loss = torch.mean(kl_loss)
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
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
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
                step=global_step,
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
            "scripts/train_sd3.py only supports sample.sampling_mode='default'."
        )
    sampling_mode = str(config.sample.sampling_mode).lower()
    if sampling_mode != "default":
        raise ValueError(
            "scripts/train_sd3.py only supports sample.sampling_mode='default'. "
            f"Received: {sampling_mode!r}."
        )
    missing_default_keys = [
        key for key in ("train_batch_size", "num_batches_per_epoch") if key not in config.sample
    ]
    if missing_default_keys:
        raise ValueError(
            "default sampling mode requires keys: sample.train_batch_size and "
            "sample.num_batches_per_epoch. Missing: "
            + ", ".join(missing_default_keys)
        )

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
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
    
    # Enable gradient checkpointing to save memory
    if config.activation_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
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
            batch_size=config.sample.train_batch_size,
            image_per_prompt=config.sample.num_image_per_prompt,
            prompt_per_epoch=config.sample.num_prompt_per_epoch,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
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


    with h5py.File(config.train_hdf5_path, 'r') as hf:
        neg_group = hf['negative']
        neg_prompt_embed = torch.from_numpy(neg_group['prompt_embeds'][:]).to(accelerator.device)
        neg_pooled_prompt_embed = torch.from_numpy(neg_group['pooled_prompt_embeds'][:]).to(accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

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
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0

    while True:
        if epoch >= config.num_epochs:
            break
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0 and epoch > 0:
            eval(pipeline, test_dataloader, pipeline.tokenizer, neg_prompt_embed,
                 neg_pooled_prompt_embed, config, accelerator, global_step, eval_reward_fn,
                 executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema,
                      transformer_trainable_parameters, config)
        epoch_start_time = time.time()
        sample_time_per_epoch = None

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        train_sampler.set_epoch(epoch)
        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        )
        for i, batch in enumerate(pbar):
            prompts = batch['prompts']
            prompt_embeds = batch['prompt_embeds']
            pooled_prompt_embeds = batch['pooled_prompt_embeds']
            prompt_metadata = batch.get("metadatas", [{}] * len(prompts))
            prompt_ids = batch['prompt_ids']

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    if config.train.loss_type == "matching":
                        (
                            images,
                            latents,
                            log_probs,
                            noises,
                            noise_preds,
                            prev_sample_mean,
                        ) = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            return_dict=False,
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            generator=generator,
                            return_prev_sample_mean=True,
                            collect_matching_aux=True,
                        )
                    elif config.train.reweight_type == "grpo_guard":
                        images, latents, log_probs, prev_sample_mean = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            return_dict=False,
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            generator=generator,
                            return_prev_sample_mean=True,
                        )
                    else:
                        images, latents, log_probs = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            return_dict=False,
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            generator=generator,
                        )

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
            if config.train.loss_type == "matching":
                prev_sample_mean = torch.stack(prev_sample_mean, dim=1)
                noises = torch.stack(noises, dim=1)
                noise_preds = torch.stack(noise_preds, dim=1)
            elif config.train.reweight_type == "grpo_guard":
                prev_sample_mean = torch.stack(prev_sample_mean, dim=1)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            sample = {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "timesteps": timesteps,
                "latents": latents[
                    :, :-1
                ],  # each entry is the latent before timestep t
                "next_latents": latents[
                    :, 1:
                ],  # each entry is the latent after timestep t
                "log_probs": log_probs,
                "rewards": rewards,
            }
            if config.train.loss_type == "matching":
                sample["prev_sample_mean"] = prev_sample_mean
                sample["noises"] = noises
                sample["noise_preds"] = noise_preds
            elif config.train.reweight_type == "grpo_guard":
                sample["prev_sample_mean"] = prev_sample_mean
            samples.append(sample)

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

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
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
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

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)

        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count == 0:
            if accelerator.is_main_process:
                logger.warning("All sampled advantages are zero; re-sampling this epoch.")
            continue
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps
        training_start_time = time.time()
        sample_time_per_epoch = training_start_time - epoch_start_time

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat([train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]])
                    pooled_embeds = torch.cat([train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]])
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            timestep = sample["timesteps"][:, j]            # [B]
                            next_timestep = sample["timesteps"][:, j + 1]   # [B]
                            delta_t = (timestep - next_timestep) / 1000.0   # [B]

                            if config.train.loss_type == "ppo":
                                _, log_prob, prev_sample_mean, tilde_sigma_t = compute_log_prob(
                                    transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                )
                                if config.train.beta > 0:
                                    with torch.no_grad():
                                        with transformer.module.disable_adapter():
                                            _, _, prev_sample_mean_ref, _ = compute_log_prob(
                                                transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                            )

                                advantages = torch.clamp(
                                    sample["advantages"][:, j],
                                    -config.train.adv_clip_max,
                                    config.train.adv_clip_max,
                                )

                                sqrt_dt = torch.sqrt(delta_t)
                                tilde_sigma_row = tilde_sigma_t.reshape(tilde_sigma_t.shape[0], -1)[:, 0]
                                grpo_guard_scale = sqrt_dt * tilde_sigma_row
                                ratio = compute_ppo_ratio_by_reweight(
                                    config.train.reweight_type,
                                    log_prob,
                                    sample["log_probs"][:, j],
                                    delta_t=delta_t,
                                    timestep=timestep,
                                    tilde_sigma=tilde_sigma_row,
                                    prev_sample_mean=prev_sample_mean,
                                    old_prev_sample_mean=sample["prev_sample_mean"][:, j]
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
                                policy_per_sample = torch.maximum(unclipped_loss, clipped_loss)
                                policy_per_sample = combine_ppo_terms_by_reweight(
                                    config.train.reweight_type,
                                    policy_per_sample,
                                    sqrt_dt=sqrt_dt,
                                    tilde_sigma=tilde_sigma_row,
                                )
                                policy_loss = torch.mean(policy_per_sample)

                                if config.train.beta > 0:
                                    if config.train.heuristic_kldenom_trick:
                                        kl_denom = tilde_sigma_t ** 2
                                    else:
                                        kl_denom = (tilde_sigma_t ** 2) * delta_t[:, None, None, None]
                                    kl_loss = (
                                        (prev_sample_mean - prev_sample_mean_ref) ** 2
                                    ).mean(dim=(1, 2, 3), keepdim=True) / (2 * kl_denom)
                                    kl_loss = torch.mean(kl_loss)
                                    loss = policy_loss + config.train.beta * kl_loss
                                else:
                                    loss = policy_loss

                                approx_kl_per_row = 0.5 * (
                                    (log_prob - sample["log_probs"][:, j]) ** 2
                                )
                                clip_event = torch.abs(ratio - 1.0) > config.train.clip_range
                                clip_event_gt = ratio - 1.0 > config.train.clip_range
                                clip_event_lt = 1.0 - ratio > config.train.clip_range
                            else:
                                prev_sample_mean, tilde_sigma_t, noise_pred = compute_noise_pred(
                                    transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                )
                                if config.train.beta > 0:
                                    with torch.no_grad():
                                        with transformer.module.disable_adapter():
                                            prev_sample_mean_ref, _, _ = compute_noise_pred(
                                                transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                            )

                                advantages = torch.clamp(
                                    sample["advantages"][:, j],
                                    -config.train.adv_clip_max,
                                    config.train.adv_clip_max,
                                )

                                tilde_sigma_t = tilde_sigma_t.view(-1)  # [B]
                                weight = compute_matching_weight_by_reweight(
                                    config.train.reweight_type,
                                    delta_t=delta_t,
                                    timestep=timestep,
                                    tilde_sigma=tilde_sigma_t,
                                ).to(dtype=noise_pred.dtype)

                                clip_range = config.train.clip_range

                                pred_diff = noise_pred - sample["noise_preds"][:, j]
                                noise = sample["noises"][:, j]

                                pred_matching_term_1 = 0.5 * (pred_diff * weight).pow(2).mean(dim=(1, 2, 3), dtype=torch.float32)
                                pred_matching_term_2 = ((weight * pred_diff) * noise).mean(dim=(1, 2, 3), dtype=torch.float32)
                                pred_matching_term = combine_matching_terms_by_reweight(
                                    config.train.reweight_type,
                                    pred_matching_term_1,
                                    pred_matching_term_2,
                                    tilde_sigma_t,
                                    delta_t,
                                )

                                unclipped_loss = (
                                    clip_range * torch.abs(advantages)
                                    + advantages * pred_matching_term
                                )
                                clipped_loss = torch.zeros_like(advantages)
                                policy_per_sample = torch.maximum(unclipped_loss, clipped_loss)

                                if config.train.reweight_type == "grpo_guard":
                                    policy_per_sample = policy_per_sample / delta_t
                                elif config.train.reweight_type == "fair_clip":
                                    policy_per_sample = policy_per_sample / (tilde_sigma_t**2 * delta_t)
                                elif config.train.reweight_type == "fair_clip2":
                                    # Follow fair_clip scaling, then map to the same end scale as grpo_guard.
                                    policy_per_sample = (
                                        policy_per_sample / (tilde_sigma_t**2 * delta_t)
                                    ) * (tilde_sigma_t / torch.sqrt(delta_t))
                                policy_loss = torch.mean(policy_per_sample)

                                if config.train.beta > 0:
                                    if config.train.heuristic_kldenom_trick:
                                        kl_denom = tilde_sigma_t ** 2
                                    else:
                                        kl_denom = (tilde_sigma_t ** 2) * delta_t
                                    kl_loss = (
                                        (prev_sample_mean - prev_sample_mean_ref) ** 2
                                    ).mean(dim=(1, 2, 3), keepdim=True) / (2 * kl_denom)
                                    kl_loss = torch.mean(kl_loss)
                                    loss = policy_loss + config.train.beta * kl_loss
                                else:
                                    loss = policy_loss

                                approx_kl_per_row = 0.5 * pred_matching_term**2
                                clip_event = (
                                    clip_range * torch.abs(advantages)
                                    + advantages * pred_matching_term
                                    < 0
                                )
                                clip_event_gt = clip_event & (advantages > 0)
                                clip_event_lt = clip_event & (advantages < 0)

                            approx_kl_value = torch.mean(approx_kl_per_row)
                            clipfrac_value = torch.mean(clip_event.float())
                            clipfrac_gt_one_value = torch.mean(clip_event_gt.float())
                            clipfrac_lt_one_value = torch.mean(clip_event_lt.float())

                            info["approx_kl"].append(approx_kl_value)
                            info["clipfrac"].append(clipfrac_value)
                            info["clipfrac_gt_one"].append(clipfrac_gt_one_value)
                            info["clipfrac_lt_one"].append(clipfrac_lt_one_value)
                            info["policy_loss"].append(policy_loss)
                            if config.train.beta > 0:
                                info["kl_loss"].append(kl_loss)

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
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
        epoch_end_time = time.time()
        policy_update_time_per_epoch = epoch_end_time - training_start_time
        time_per_epoch = epoch_end_time - epoch_start_time
        if accelerator.is_main_process:
            wandb.log(
                {
                    "time_per_epoch": time_per_epoch,
                    "sample_time_per_epoch": sample_time_per_epoch,
                    "policy_update_time_per_epoch": policy_update_time_per_epoch,
                },
                step=global_step,
            )
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)
