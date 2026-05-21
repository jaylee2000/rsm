import contextlib
import datetime
import logging
import os
import random
import tempfile
import time
from numbers import Real
import torch
import wandb
import yaml
import numpy as np

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from abc import ABC, abstractmethod
from tqdm.auto import tqdm

from diffusers.utils.torch_utils import is_compiled_module

from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image

import utils.prompts
import utils.rewards
from utils.stat_tracking import PerPromptStatTracker
from utils.diffusers_patch.reverse_pipeline import pipeline_with_logprob


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def unwrap_model(model):
    model = model.module if isinstance(model, DDP) else model
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def get_noise_pred(config, unet, sample, embeds, j):
    if config['train']['cfg']:
        noise_pred = unet(
            torch.cat([sample["latents"][:, j]] * 2),
            torch.cat([sample["timesteps"][:, j]] * 2),
            embeds,
        ).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config['sample']['cfg_scale']
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = unet(
            sample["latents"][:, j],
            sample["timesteps"][:, j],
            embeds,
        ).sample
        noise_pred_text = None
    return noise_pred, noise_pred_text


@contextlib.contextmanager
def _temp_python_random_seed(seed: int):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def _seed_mix(*values: int) -> int:
    # Deterministic mixing; keep within torch's supported seed range.
    mod = 2**63 - 1
    mixed = 0
    for value in values:
        mixed = (mixed * 1000003 + int(value)) % mod
    return mixed


class _PromptDataset(Dataset):
    def __init__(
        self,
        prompt_fn,
        prompt_fn_kwargs: dict,
        *,
        epoch_key: int,
        size: int,
        base_seed: int,
    ):
        self._prompt_fn = prompt_fn
        self._prompt_fn_kwargs = prompt_fn_kwargs
        self._epoch_key = int(epoch_key)
        self._size = int(size)
        self._base_seed = int(base_seed)

    def __len__(self):
        return self._size

    def __getitem__(self, idx: int):
        seed = _seed_mix(self._base_seed, self._epoch_key, int(idx))
        with _temp_python_random_seed(seed):
            return self._prompt_fn(**self._prompt_fn_kwargs)  # (prompt: str, metadata: dict)


class DDPTrainer(ABC):
    def __init__(self, config_path: str):
        self.config = yaml.safe_load(open(config_path))
        assert self.config['model']['use_lora'] # TODO: Support full fine-tuning
        self._get_timesteps()
        self.run_name_prefix = None

    @property
    def is_main_process(self):
        """Shorthand for checking if this is the main process"""
        global_rank = getattr(self, "global_rank", None)
        if global_rank is not None:
            return global_rank == 0
        return getattr(self, "local_rank", 0) == 0

    @property
    def device(self):
        return torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')

    def _get_timesteps(self):
        self.num_inference_steps = self.config['sample']['num_steps']
        self.num_train_timesteps = int(self.num_inference_steps * self.config['train']['timestep_fraction'])
        if self.num_train_timesteps != self.num_inference_steps:
            self.num_train_timesteps += 1

    def _setup_distributed_backend(self, timeout=0):
        dist_url = "env://" # default

        # only works with torch.distributed.launch // torch.run
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = rank
        self.local_rank = local_rank
        self.num_processes = self.world_size = world_size

        if timeout == 0:
            timeout = dist.default_pg_timeout
        else:
            timeout = datetime.timedelta(seconds=timeout)
        if self.is_main_process:
            logging.info(f"Default timeout: {timeout}")

        # TODO: Add gloo support
        dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            timeout=timeout,
            rank=rank
        )

        # all .cuda() calls will use the correct device
        torch.cuda.set_device(local_rank)
        # synchronize all threads to reach this point before moving on
        dist.barrier()
        logging.info(f'setting up local_rank {local_rank} global_rank {rank} world size {world_size}')

        setup_for_distributed(rank == 0)

    def _get_run_name(self, prefix):
        unique_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        reward = self.config['reward_fn'].split('_')[0]
        self.run_name = f"{prefix}_{reward}_{unique_id}"

    def _set_output_dir(self):
        base_output_dir = self.config.get('logging_dir', './logs')
        self.output_dir = os.path.join(base_output_dir, self.run_name)
        os.makedirs(self.output_dir, exist_ok=True)

    def _setup_logging(self):
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self.logger = logging.getLogger(logger_name)

    def _setup_wandb(self):
        if self.is_main_process:
            wandb.init(
                project='sd-align',
                name=self.run_name,
                config=self.config,
                dir=self.output_dir,
                save_code=True,
            )

    def _set_seed(self):
        base_seed = self.config.get('seed', 42)
        rank_for_seed = getattr(self, "global_rank", getattr(self, "local_rank", 0))
        seed = int(base_seed) + 10001 * int(rank_for_seed)

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()

        if self.is_main_process:
            logging.info(f'Using seed: {seed}')

    def _determine_weight_dtype(self):
        mp = self.config['mixed_precision']
        if mp == 'fp16':
            self.weight_dtype = torch.float16
        elif mp == 'bf16':
            self.weight_dtype = torch.bfloat16
        else:
            self.weight_dtype = torch.float32

    def _enable_tf32(self):
        if self.config['allow_tf32']:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    @abstractmethod
    def _setup_pipeline(self):
        pass

    @abstractmethod
    def _setup_optimizer(self):
        pass

    def _setup_prompts(self):
        self.prompt_fn = getattr(utils.prompts, self.config['prompt_fn'])

        text_encoder = self.text_encoder
        tokenizer = self.tokenizer

        self.neg_prompt_embed = text_encoder(
            tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).input_ids.to(self.device)
        )[0]
        self.sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(
            self.config['sample']['batch_size'], 1, 1
        )
        self.train_neg_prompt_embeds = self.neg_prompt_embed.repeat(
            self.config['train']['batch_size'], 1, 1
        )

    def _setup_reward_fn(self):
        reward_fn_kwargs = dict(self.config.get("reward_fn_kwargs", {}) or {})
        if 'diff' in self.config['reward_fn']:
            self.reward_fn = getattr(utils.rewards, self.config['reward_fn'])(
                torch.float32,
                self.device,
                **reward_fn_kwargs,
            )
        else:
            self.reward_fn = getattr(utils.rewards, self.config['reward_fn'])(**reward_fn_kwargs)

    def _setup_stat_tracker(self):
        if self.config['per_prompt_stat_tracking']:
            self.stat_tracker = PerPromptStatTracker(
                self.config['per_prompt_stat_tracking']["buffer_size"],
                self.config['per_prompt_stat_tracking']["min_count"],
            )
        else:
            self.stat_tracker = None

    def _get_eval_prompt_fn_kwargs(self):
        config = self.config
        eval_prompt_fn_kwargs = config.get("eval_prompt_fn_kwargs")
        if eval_prompt_fn_kwargs is not None:
            return dict(eval_prompt_fn_kwargs)

        prompt_fn_kwargs = dict(config.get("prompt_fn_kwargs", {}))
        if config.get("prompt_fn") == "geneval":
            prompt_fn_kwargs.setdefault("split", "test")
        return prompt_fn_kwargs

    def _setup_boilerplate(self):
        self._setup_distributed_backend(timeout=30000)

        if self.run_name_prefix is not None:
            if self.is_main_process:
                self._get_run_name(self.run_name_prefix)
                objects = [self.run_name]
            else:
                objects = [None]
            dist.broadcast_object_list(objects, src=0)
            self.run_name = objects[0]

        self._set_output_dir()
        self._setup_logging()
        self._setup_wandb()
        self._set_seed()
        self._determine_weight_dtype()
        self._enable_tf32()
        self.num_processes = self.world_size

    def setup(self):
        self._setup_boilerplate()
        self._setup_pipeline()
        self._setup_optimizer()

        self._setup_prompts()
        self._setup_stat_tracker() # optional, for per-prompt stats
        self._setup_reward_fn()

    def validate_training_config(self):
        n_gpu = self.num_processes
        config = self.config
        logger = self.logger

        sample_batch_size = config['sample']['batch_size']
        n_batch_per_epoch = config['sample']['num_batches_per_epoch']
        train_batch_size = config['train']['batch_size']
        n_accum_steps = config['train']['gradient_accumulation_steps']

        samples_per_epoch = sample_batch_size * n_gpu * n_batch_per_epoch
        total_train_batch_size = train_batch_size * n_gpu * n_accum_steps

        if self.is_main_process:
            logger.info("***** Running training *****")
            logger.info(f"  Num Epochs = {config['train']['max_epochs']}")
            logger.info(f"  Sample batch size per device = {sample_batch_size}")
            logger.info(f"  Train batch size per device = {train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {n_accum_steps}")
            logger.info("")
            logger.info(f"  Total number of samples per epoch = sample_bs * num_batch_per_epoch * num_process = {samples_per_epoch}")
            logger.info(
                f"  Total train batch size (w. parallel, distributed & accumulation) = train_bs * grad_accumul * num_process = {total_train_batch_size}"
            )
            logger.info(
                f"  Number of gradient updates per inner epoch = samples_per_epoch // total_train_batch_size = {samples_per_epoch // total_train_batch_size}"
            )
            assert samples_per_epoch % total_train_batch_size == 0

    def evaluate(self, global_step):
        self.logger.info("Running evaluation...")
        config = self.config

        self.unet.eval()

        eval_batch_size = config['eval']['batch_size']
        num_eval_batches = config['eval']['num_batches']

        neg_prompt_embed = self.neg_prompt_embed

        base_seed = int(config.get("seed", 42))
        world_size = int(getattr(self, "world_size", 1))
        global_rank = int(getattr(self, "global_rank", 0))

        global_prompt_count = eval_batch_size * num_eval_batches * world_size
        eval_prompt_fn_kwargs = self._get_eval_prompt_fn_kwargs()
        prompt_dataset = _PromptDataset(
            self.prompt_fn,
            eval_prompt_fn_kwargs,
            epoch_key=int(global_step),
            size=global_prompt_count,
            base_seed=_seed_mix(base_seed, 31111),
        )
        prompt_sampler = DistributedSampler(
            prompt_dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=False,
            drop_last=False,
        )

        def _prompt_collate(batch):
            prompts, metas = zip(*batch)
            return list(prompts), list(metas)

        prompt_loader = DataLoader(
            prompt_dataset,
            batch_size=eval_batch_size,
            sampler=prompt_sampler,
            num_workers=0,
            collate_fn=_prompt_collate,
        )

        all_rewards = []
        all_images = []
        all_eval_kls = []

        with torch.inference_mode():
            local_last_batch_images = None
            local_last_batch_prompts = None
            local_last_batch_rewards = None

            for batch_idx, (prompts, prompts_metadata) in enumerate(prompt_loader):
                if batch_idx >= num_eval_batches:
                    break

                prompt_ids = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                ).input_ids.to(self.device)
                prompt_embeds = self.text_encoder(prompt_ids)[0]

                batch_neg_prompt_embeds = neg_prompt_embed.repeat(len(prompts), 1, 1)

                with torch.inference_mode():
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(_seed_mix(base_seed, 32222, int(global_step), int(global_rank), int(batch_idx)))
                    images, _, eval_latents, _, _, _, eval_noise_preds, _ = pipeline_with_logprob(
                        self.unet,
                        self.vae,
                        self.noise_scheduler,
                        self.tokenizer,
                        self.text_encoder,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=batch_neg_prompt_embeds,
                        num_inference_steps=config['eval']['num_steps'],
                        guidance_scale=config['sample']['cfg_scale'],
                        eta=config['sample']['eta'],
                        output_type="pt",
                        disable_progress_bar=not self.is_main_process,
                        use_pcpo=config['use_pcpo'],
                        generator=generator,
                    )

                    eval_timesteps = self.noise_scheduler.timesteps.repeat(len(prompts), 1)
                    eval_sample = {"timesteps": eval_timesteps}
                    eval_ddpm_weights = self.get_ddpm_weights(eval_sample)
                    do_classifier_free_guidance = config['sample']['cfg_scale'] > 1.0
                    if do_classifier_free_guidance:
                        eval_embeds = torch.cat([batch_neg_prompt_embeds, prompt_embeds])
                    else:
                        eval_embeds = prompt_embeds

                    raw_unet = self.unet.module if hasattr(self.unet, "module") else self.unet
                    if hasattr(raw_unet, "disable_adapters"):
                        raw_unet.disable_adapters()

                    batch_step_kls = []
                    for step_idx in range(eval_timesteps.shape[1]):
                        latent_step = eval_latents[step_idx]
                        timestep_step = eval_timesteps[:, step_idx]
                        timestep_step_scalar = timestep_step[0]

                        if do_classifier_free_guidance:
                            latent_model_input = torch.cat([latent_step] * 2)
                            timestep_model_input = torch.cat([timestep_step] * 2)
                        else:
                            latent_model_input = latent_step
                            timestep_model_input = timestep_step
                        latent_model_input = self.noise_scheduler.scale_model_input(
                            latent_model_input, timestep_step_scalar
                        )

                        noise_pred_ref = self.unet(
                            latent_model_input,
                            timestep_model_input,
                            eval_embeds,
                        ).sample
                        if do_classifier_free_guidance:
                            noise_pred_uncond_ref, noise_pred_text_ref = noise_pred_ref.chunk(2)
                            noise_pred_ref = (
                                noise_pred_uncond_ref
                                + config['sample']['cfg_scale']
                                * (noise_pred_text_ref - noise_pred_uncond_ref)
                            )

                        ddpm_weight_step = eval_ddpm_weights[:, step_idx]
                        noise_delta = eval_noise_preds[step_idx].detach().float() - noise_pred_ref.float()
                        batch_approx_kl = 0.5 * torch.mean(
                            (ddpm_weight_step.float() * noise_delta) ** 2,
                            dim=[1, 2, 3],
                        )
                        batch_step_kls.append(batch_approx_kl)

                    if hasattr(raw_unet, "enable_adapters"):
                        raw_unet.enable_adapters()
                    if hasattr(raw_unet, "set_adapter"):
                        raw_unet.set_adapter("pf")

                    local_eval_kl = torch.stack(batch_step_kls, dim=1).mean(dim=1)
                    eval_kl_gather = torch.zeros(
                        self.world_size * len(prompts), dtype=torch.float32, device=self.device
                    )
                    dist.all_gather_into_tensor(eval_kl_gather, local_eval_kl)
                    all_eval_kls.append(eval_kl_gather.detach().cpu().numpy())
                    
                # Compute rewards synchronously
                rewards, reward_metadata = self.reward_fn(images.float(), prompts, prompts_metadata)
                local_rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
                rewards_gather = torch.zeros(
                    self.world_size * len(rewards), dtype=torch.float32, device=self.device
                )
                dist.all_gather_into_tensor(rewards_gather, local_rewards)


                # Store last batch for visualization
                all_rewards.append(rewards_gather.detach().cpu().numpy())
                all_images.append(images)

                local_last_batch_images = images
                local_last_batch_prompts = prompts
                local_last_batch_rewards = local_rewards.detach().cpu().numpy()

            # Concatentae all rewards
            all_rewards_concat = np.concatenate(all_rewards)
            all_eval_kl_concat = np.concatenate(all_eval_kls)

            if self.is_main_process:
                last_batch_images = local_last_batch_images
                last_batch_prompts = local_last_batch_prompts
                last_batch_rewards = local_last_batch_rewards

                with tempfile.TemporaryDirectory() as tmpdir:
                    num_samples = min(8, len(last_batch_images))
                    sample_indices = range(num_samples)
                    
                    for idx in sample_indices:
                        image = last_batch_images[idx].float().cpu().numpy()
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
                            "eval_kl_divergence": np.mean(all_eval_kl_concat),
                        },
                        step=global_step,
                    )

    def sample(self, epoch):
        config = self.config
        use_z0 = config['train'].get('use_z0', False)

        base_seed = int(config.get("seed", 42))
        world_size = int(getattr(self, "world_size", 1))
        global_rank = int(getattr(self, "global_rank", 0))

        torch.cuda.empty_cache()
        self.unet.zero_grad()
        self.unet.eval()

        with torch.inference_mode():
            samples = []

            sample_bs = config['sample']['batch_size']
            num_batches = config['sample']['num_batches_per_epoch']
            global_prompt_count = sample_bs * num_batches * world_size

            prompt_dataset = _PromptDataset(
                self.prompt_fn,
                config.get("prompt_fn_kwargs", {}),
                epoch_key=int(epoch),
                size=global_prompt_count,
                base_seed=_seed_mix(base_seed, 21111),
            )
            prompt_sampler = DistributedSampler(
                prompt_dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=False,
                drop_last=False,
            )

            def _prompt_collate(batch):
                prompts, metas = zip(*batch)
                return list(prompts), list(metas)

            prompt_loader = DataLoader(
                prompt_dataset,
                batch_size=sample_bs,
                sampler=prompt_sampler,
                num_workers=0,
                collate_fn=_prompt_collate,
            )

            for i, (prompts, prompt_metadata) in enumerate(tqdm(
                prompt_loader,
                desc=f"Epoch {epoch}: sampling",
                disable=not self.is_main_process,
                position=0,
                total=num_batches,
            )):
                if i >= num_batches:
                    break
                prompt_ids = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.tokenizer.model_max_length,
                ).input_ids.to(self.device)
                prompt_embeds = self.text_encoder(prompt_ids)[0]

                with contextlib.nullcontext():
                    generator = torch.Generator(device=self.device)
                    generator.manual_seed(_seed_mix(base_seed, 22222, int(epoch), int(global_rank), int(i)))
                    images, _, latents, log_probs, noises, noise_pred_texts, \
                    noise_preds, scores = pipeline_with_logprob(
                        self.unet,
                        self.vae,
                        self.noise_scheduler,
                        self.tokenizer,
                        self.text_encoder,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=self.sample_neg_prompt_embeds,
                        num_inference_steps=self.num_inference_steps,
                        guidance_scale=config['sample']['cfg_scale'],
                        eta=config['sample']['eta'],
                        output_type="pt",
                        disable_progress_bar=not self.is_main_process,
                        use_pcpo=config['use_pcpo'],
                        generator=generator,
                    )

                    latents = torch.stack(latents, dim=1)  # (batch_size, num_steps + 1, 4, 64, 64)
                    noises = torch.stack(noises, dim=1)  # (batch_size, num_steps, 4, 64, 64)
                    noise_pred_texts = torch.stack(noise_pred_texts, dim=1)
                    noise_preds = torch.stack(noise_preds, dim=1)
                    log_probs = torch.stack(log_probs, dim=1)  # (batch_size, num_steps, 1)
                    scores = torch.stack(scores, dim=1)  # (batch_size, num_steps, 1)

                    timesteps = self.noise_scheduler.timesteps.repeat(len(prompts), 1)  # (batch_size, num_steps)
                    if use_z0:
                        z0 = latents[:, -1]

                    # Compute rewards synchronously
                    rewards, reward_metadata = self.reward_fn(images.float(), prompts, prompt_metadata)

                    sample_data = {
                        "prompts": prompts,
                        "prompt_metadata": prompt_metadata,
                        "prompt_ids": prompt_ids,
                        "prompt_embeds": prompt_embeds,
                        "timesteps": timesteps,
                        "latents": latents[:, :-1],  # each entry is the latent before timestep t
                        "next_latents": latents[:, 1:],
                        "noises": noises,
                        "rewards": rewards,
                        "scores": scores,
                        "noise_preds": noise_preds,
                        "noise_pred_texts": noise_pred_texts,
                        "log_probs": log_probs,
                    }
                    if use_z0:
                        sample_data["z0"] = z0

                    samples.append(sample_data)

            # Wait for all rewards (moved outside the sampling loop)
            for sample in tqdm(
                samples,
                desc="Waiting for rewards",
                disable=not self.is_main_process,
                position=0,
            ):
                rewards = sample["rewards"]
                sample["rewards"] = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)

            samples = self.collate_sample_to_dict(samples)

            samples_log = {
                "images": images,
                "prompts": prompts,
                "rewards": rewards,
            }

        return samples, samples_log

    def compute_advantages(self, samples, rewards):
        # Gather rewards from all processes
        local_rewards = rewards
        gathered_rewards = [torch.zeros_like(local_rewards) for _ in range(self.world_size)]
        dist.all_gather(gathered_rewards, local_rewards)
        all_rewards = torch.cat(gathered_rewards)
        all_rewards = all_rewards.cpu().numpy()

        if self.config['per_prompt_stat_tracking']:
            local_prompt_ids = samples["prompt_ids"]
            gathered_prompt_ids = [torch.zeros_like(local_prompt_ids) for _ in range(self.world_size)]
            dist.all_gather(gathered_prompt_ids, local_prompt_ids)
            prompt_ids = torch.cat(gathered_prompt_ids).cpu().numpy()

            prompts = self.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
            advantages = self.stat_tracker.update(prompts, all_rewards)
            # from self.stat_tracker, get the mean and std used for each advantage
            means = []
            stds = []
            for prompt in prompts:
                stats = self.stat_tracker.get_stats().get(prompt, None)
                if stats is not None:
                    means.append(stats['mean'])
                    stds.append(stats['std'])
                else:
                    means.append(np.mean(all_rewards))
                    stds.append(np.std(all_rewards) + 1e-8)
        else:
            advantages = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)
            means = [all_rewards.mean()] * len(all_rewards)
            stds = [all_rewards.std() + 1e-8] * len(all_rewards)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        rank_for_ungather = getattr(self, "global_rank", getattr(self, "local_rank", 0))
        # Keep these float32 to avoid unintended float64 upcasting in training.
        samples["advantages"] = (
            torch.as_tensor(advantages, dtype=torch.float32)
            .reshape(self.num_processes, -1)[rank_for_ungather]
            .to(self.device)
        )
        samples["means"] = (
            torch.as_tensor(means, dtype=torch.float32)
            .reshape(self.num_processes, -1)[rank_for_ungather]
            .to(self.device)
        )
        samples["stds"] = (
            torch.as_tensor(stds, dtype=torch.float32)
            .reshape(self.num_processes, -1)[rank_for_ungather]
            .to(self.device)
        )

        return samples

    def collate_sample_to_dict(self, samples):
        # Collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        new_samples = {}
        for k in samples[0].keys():
            if k in ["prompts", "prompt_metadata"]:
                # list of tuples [('cat', 'dog'), ('cat', 'tiger'), ...] -> list ['cat', 'dog', 'cat', 'tiger', ...]
                new_samples[k] = [item for s in samples for item in s[k]]
            else:
                new_samples[k] = torch.cat([s[k] for s in samples])
        return new_samples

    def wandb_log_samples(self, global_step, samples):
        if not self.is_main_process:
            return
        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(samples["images"]):
                # bf16 cannot be converted to numpy directly
                pil = Image.fromarray(
                    (image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            wandb.log(
                {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{i}.jpg"),
                            caption=f"{prompt} | {reward:.2f}",
                        )
                        for i, (prompt, reward) in enumerate(
                            zip(samples["prompts"], samples["rewards"])
                        )
                    ],
                },
                step=global_step,
            )

    def wandb_log_rewards(self, global_step, epoch, rewards):
        if not self.is_main_process:
            return
        wandb.log(
            {
                "reward": rewards, # TODO: fix this to log histogram
                "epoch": epoch,
                "reward_mean": rewards.mean(),
                "reward_std": rewards.std(),
            },
            step=global_step,
        )

    def get_ddpm_weights(self, sample):
        noise_scheduler = self.noise_scheduler

        batch_size, num_steps = sample['timesteps'].shape
        alphas_cumprod = noise_scheduler.alphas_cumprod.clone().detach().to(self.device)
        # ā_t
        alphas_cumprod_inference = alphas_cumprod[sample['timesteps']]
        # ā_{t-1}
        prev_ts = (sample['timesteps'] - 1000 // num_steps).clamp(min=0)
        alphas_cumprod_prev_inference = alphas_cumprod[prev_ts]
        # σ_t^2
        sigmas_squared_inference = (1 - alphas_cumprod_prev_inference) \
            * (1 - alphas_cumprod_inference / alphas_cumprod_prev_inference) \
            / (1 - alphas_cumprod_inference)
        # σ_t
        sigmas_inference = torch.sqrt(sigmas_squared_inference)
        # c2(t)
        c2s_inference = torch.sqrt(1 - alphas_cumprod_prev_inference - sigmas_squared_inference)
        # c1(t)
        c1s_inference = torch.sqrt(1 - alphas_cumprod_inference) \
            / torch.sqrt(alphas_cumprod_inference / alphas_cumprod_prev_inference)
        # w(t) = (c1(t) - c2(t)) / σ_t (DDPM Weights)
        weights_inference = (c1s_inference - c2s_inference) / sigmas_inference
        selected_weights = weights_inference.view(batch_size, num_steps, 1, 1, 1)
        return selected_weights

    def get_ddpm_Omegas(self, sample):
        noise_scheduler = self.noise_scheduler

        batch_size, num_steps = sample['timesteps'].shape
        alphas_cumprod = noise_scheduler.alphas_cumprod.clone().detach().to(self.device)
        # ā_t
        alphas_cumprod_inference = alphas_cumprod[sample['timesteps']]
        # ā_{t-1}
        prev_ts = (sample['timesteps'] - 1000 // num_steps).clamp(min=0)
        alphas_cumprod_prev_inference = alphas_cumprod[prev_ts]
        # σ_t^2
        sigmas_squared_inference = (1 - alphas_cumprod_prev_inference) \
            * (1 - alphas_cumprod_inference / alphas_cumprod_prev_inference) \
            / (1 - alphas_cumprod_inference)
        # c2(t)
        c2s_inference = torch.sqrt(1 - alphas_cumprod_prev_inference - sigmas_squared_inference)
        # c1(t)
        c1s_inference = torch.sqrt(1 - alphas_cumprod_inference) \
            / torch.sqrt(alphas_cumprod_inference / alphas_cumprod_prev_inference)
        # Omega(t) = (c1(t) - c2(t)) * sqrt(1 - \bar \alpha_t)
        weights_inference = (c1s_inference - c2s_inference) * torch.sqrt(1 - alphas_cumprod_inference)
        selected_weights = weights_inference.view(batch_size, num_steps, 1, 1, 1)
        return selected_weights

    def get_ddpm_sigmas(self, sample):
        noise_scheduler = self.noise_scheduler

        batch_size, num_steps = sample['timesteps'].shape
        alphas_cumprod = noise_scheduler.alphas_cumprod.clone().detach().to(self.device)
        # ā_t
        alphas_cumprod_inference = alphas_cumprod[sample['timesteps']]
        # ā_{t-1}
        prev_ts = (sample['timesteps'] - 1000 // num_steps).clamp(min=0)
        alphas_cumprod_prev_inference = alphas_cumprod[prev_ts]
        # σ_t^2
        sigmas_squared_inference = (1 - alphas_cumprod_prev_inference) \
            * (1 - alphas_cumprod_inference / alphas_cumprod_prev_inference) \
            / (1 - alphas_cumprod_inference)
        # σ_t
        sigmas_inference = torch.sqrt(sigmas_squared_inference)
        selected_sigmas = sigmas_inference.view(batch_size, num_steps, 1, 1, 1)
        return selected_sigmas

    def resolve_pretrained_strength(self, pretrained_strength_config, sigma_t=None):
        if isinstance(pretrained_strength_config, str):
            mode = pretrained_strength_config.strip().lower()
            if mode in {"sigma", "3sigma"}:
                if sigma_t is None:
                    raise ValueError(
                        f"train.pretrained_strength='{mode}' requires sigma_t for the current timestep."
                    )
                if mode == "3sigma":
                    return 3.0 * sigma_t
                return sigma_t
            raise ValueError(
                "Unsupported train.pretrained_strength string "
                f"'{pretrained_strength_config}'. Supported values are numeric constants, 'sigma', or '3sigma'."
            )

        if isinstance(pretrained_strength_config, bool) or not isinstance(pretrained_strength_config, Real):
            raise ValueError(
                "train.pretrained_strength must be a numeric constant, 'sigma', or '3sigma'."
            )

        return float(pretrained_strength_config)

    def shuffle_and_batch_samples(self, samples):
        config = self.config

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert (
            total_batch_size
            == config['sample']['batch_size'] * config['sample']['num_batches_per_epoch']
        )
        assert num_timesteps == config['sample']['num_steps']

        # shuffle samples along batch dimension
        perm = torch.randperm(total_batch_size, device=self.device)
        samples_shuffled = {}
        for k, v in samples.items():
            if k in ["prompts", "prompt_metadata"]:
                # For lists, use list comprehension with perm indices
                samples_shuffled[k] = [v[i] for i in perm]
            else:
                samples_shuffled[k] = v[perm]
        samples = samples_shuffled

        if config['train']['timestep_fraction'] < 1.0:
            if config['train']['low_var_subsampling']:
                n_trunks = int(self.num_inference_steps * config['train']['timestep_fraction'])
                assert n_trunks >= 1, "Must have at least one trunk"
                assert self.num_inference_steps % n_trunks == 0, \
                    "num_inference_steps must be divisible by number of trunks"

                trunk_size = self.num_inference_steps // n_trunks
                step_indices = torch.arange(self.num_inference_steps, device=self.device)
                trunks = step_indices.view(n_trunks, trunk_size) # (n_trunks, trunk_size)

                # Precompute trunk access pattern (reversed order, repeated)
                trunk_order = list(reversed(range(n_trunks))) * trunk_size # length = self.num_inference_steps

                perms_list = []
                for _ in range(total_batch_size):
                    tmp = []
                    for i in trunk_order:
                        trunk = trunks[i]
                        index = torch.randint(0, trunk_size, (1,))
                        tmp.append(trunk[index])
                    interleaved = torch.cat(tmp)
                    perms_list.append(torch.cat(
                        [torch.tensor([self.num_inference_steps - 1], device=self.device), interleaved]
                    ))

                perms = torch.stack(perms_list)
            else:
                perms = torch.stack(
                    [
                        torch.randperm(num_timesteps - 1, device=self.device)
                        for _ in range(total_batch_size)
                    ]
                )
                perms = torch.cat([num_timesteps - 1 + torch.zeros_like(perms[:, :1]), perms], dim=1)
        else:
            # shuffle along time dimension independently for each sample
            perms = torch.stack(
                [
                    torch.randperm(num_timesteps, device=self.device)
                    for _ in range(total_batch_size)
                ]
            ) # (total_batch_size, num_steps)

        timestep_keys = [
            "timesteps",
            "latents",
            "next_latents",
            "noises",
            "log_probs",
            "noise_preds",
            "noise_pred_texts",
            "scores",
        ]

        for key in timestep_keys:
            samples[key] = samples[key][
                torch.arange(total_batch_size, device=self.device)[:, None],
                perms,
            ]

        # rebatch for training
        samples_batched = {}
        for k, v in samples.items():
            if k in ["prompts", "prompt_metadata"]:
                # Slice lists into sublists
                samples_batched[k] = [
                    v[i:i + config['train']['batch_size']]
                    for i in range(0, len(v), config['train']['batch_size'])
                ]
            else:
                samples_batched[k] = v.reshape(-1, config['train']['batch_size'], *v.shape[1:])
        # dict of lists -> list of dicts for easier iteration
        samples_batched = [
            dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
        ]
        return samples_batched

    @abstractmethod
    def load_from_checkpoint(self):
        pass

    @abstractmethod
    def save_checkpoint(self, epoch, global_step):
        pass

    @abstractmethod
    def policy_update(self, samples, epoch, epoch_start_time, global_step):
        pass

    def train(self, first_epoch, global_step):
        for epoch in range(first_epoch, self.config['train']['max_epochs']):
            epoch_start_time = time.time()
            if epoch % self.config['eval']['freq'] == 0:
                self.evaluate(global_step)

            samples, samples_log = self.sample(epoch)

            self.wandb_log_samples(global_step, samples_log)
            self.wandb_log_rewards(global_step, epoch, samples['rewards'])
            # TODO: Replace rewards with advantages
            # self.compute_advantages(samples, samples['rewards'])
            # del samples["rewards"]
            del samples["prompt_ids"]

            global_step = self.policy_update(samples, epoch, epoch_start_time, global_step)
            self.save_checkpoint(epoch, global_step)
            dist.barrier()
        
        if self.is_main_process:
            wandb.finish()
        dist.destroy_process_group()
