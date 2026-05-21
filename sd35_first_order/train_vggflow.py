"""
VGG-Flow Training Script (Refactored)

This is the refactored version using modular components.
Clean separation of concerns and much easier to understand/modify.
"""

import os
from collections import defaultdict
import datetime
import time
import wandb
from functools import partial
import logging
import copy
import pickle
import gzip

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from diffusers import StableDiffusion3Pipeline, AutoencoderTiny
from diffusers.training_utils import cast_training_params
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from absl import app, flags
from ml_collections.config_flags import config_flags

from lib.distributed import init_distributed_singlenode, set_seed, setup_for_distributed
import lib.reward_func.prompts
import lib.reward_func.rewards

# Import refactored modules
from lib.models.cfg_wrapper import CFGWrapper
from lib.vggflow.value_network import ValueNetworkWrapper
from lib.vggflow.reward_gradient import RewardGradientComputer
from lib.vggflow.algorithm import VGGFlowAlgorithm
from lib.vggflow.trainer import VGGFlowTrainer
from lib.training.evaluator import PeriodicEvaluator
from lib.training.sampler import TrajectorySampler

try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    HAVE_SDPA_BACKEND = True
except Exception:
    HAVE_SDPA_BACKEND = False


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False
)
flags.DEFINE_string("exp_name", "", "Experiment name.")
flags.DEFINE_integer("seed", 0, "Seed.")


def unwrap_model(model):
    """Unwrap DDP and compiled models."""
    model = model.module if isinstance(model, DDP) else model
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def setup_logging_and_saving(config, is_local_main_process):
    """Setup logging and create save directory."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    # Set seed
    if FLAGS.seed is not None:
        config.seed = FLAGS.seed
    else:
        config.seed = 0
    set_seed(config.seed)

    # Setup wandb
    wandb_name = f"{config.experiment.reward_fn.split('_')[0]}_vggflow_{FLAGS.exp_name}_seed{config.seed}"

    if config.logging.use_wandb:
        wandb_key = config.logging.wandb_key
        wandb.login(key=wandb_key)
        wandb.init(
            project=config.logging.proj_name,
            name=wandb_name,
            config=config.to_dict(),
            dir=config.logging.wandb_dir,
            save_code=True,
            mode="online" if is_local_main_process else "disabled"
        )
        if is_local_main_process:
            wandb.define_metric("global_step")
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="global_step")
            wandb.define_metric("reward_*", step_metric="epoch")
            wandb.define_metric("eval_*", step_metric="epoch")
            wandb.define_metric("time_per_epoch", step_metric="epoch")
            wandb.define_metric("images", step_metric="epoch")

    # Create save directory
    save_dir = os.path.join(config.saving.output_dir, wandb_name)
    os.makedirs(save_dir, exist_ok=True)

    if is_local_main_process:
        logger.info(f"\n{config}")

    return logger, save_dir


def setup_models(config, device, local_rank, is_local_main_process):
    """Setup all models: pipeline, transformer, value network."""
    # Determine weight dtype
    weight_dtype = torch.float32
    if config.training.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.training.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load pipeline
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model,
        revision=config.pretrained.revision,
        torch_dtype=weight_dtype,
    )

    # Optionally use AutoencoderTiny for faster decoding
    if config.pretrained.autoencodertiny:
        if is_local_main_process:
            pipeline.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd3", torch_dtype=weight_dtype
            )
        dist.barrier()
        pipeline.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd3", torch_dtype=weight_dtype
        )

    # Freeze non-trainable components
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)

    # Move to device
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=weight_dtype)
    pipeline.text_encoder_2.to(device, dtype=weight_dtype)
    pipeline.text_encoder_3.to(device, dtype=weight_dtype)
    pipeline.to(device)
    pipeline.scheduler.set_timesteps(config.sampling.num_steps, device=device)

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # Setup transformer with LoRA
    transformer = pipeline.transformer
    transformer.requires_grad_(False)
    transformer.to(device, dtype=weight_dtype)

    transformer_lora_config = LoraConfig(
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    transformer.add_adapter(transformer_lora_config, adapter_name="default")
    transformer.set_adapter("default")

    if config.training.mixed_precision in ["fp16", "bf16"]:
        cast_training_params(transformer, dtype=torch.float32)

    transformer = DDP(transformer, device_ids=[local_rank])

    # Setup value network (if enabled)
    value_net = None
    value_net_params = None

    if config.model.use_value_net:
        from diffusers import UNet2DConditionModel

        if config.model.value_net_param == 'lora':
            value_net = copy.deepcopy(pipeline.transformer)
            value_net.train()
            value_net.to(device, dtype=weight_dtype)

            value_net_lora_config = LoraConfig(
                r=config.model.lora_rank,
                lora_alpha=config.model.lora_rank,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            value_net.add_adapter(value_net_lora_config, adapter_name="default_value")
            value_net.set_adapter("default_value")
            value_net.proj_out.weight.requires_grad_(True)
            value_net.proj_out.weight.data *= 1e-5

        elif config.model.value_net_param == 'copy':
            value_net = copy.deepcopy(pipeline.transformer)
            value_net.proj_out.weight.requires_grad_(True)
            value_net.proj_out.weight.data *= 1e-5
            value_net.to(device)

        elif config.model.value_net_param == 'small':
            value_net = UNet2DConditionModel(
                in_channels=16,
                out_channels=16,
                block_out_channels=config.model.value_channel_width,
                layers_per_block=config.model.value_layers_per_block,
                cross_attention_dim=4096,
            )
            value_net.conv_out.weight.requires_grad_(True)
            value_net.conv_out.weight.data *= 1e-3
            value_net.to(device, dtype=weight_dtype)

        else:
            raise NotImplementedError(
                f"value_net_param={config.model.value_net_param} not implemented"
            )

        value_net_params = value_net.parameters()

        if config.training.mixed_precision in ["fp16", "bf16"]:
            if config.model.value_net_param in ['lora', 'copy']:
                cast_training_params(value_net, dtype=torch.float32)

        value_net = DDP(value_net, device_ids=[local_rank])

    return pipeline, transformer, value_net, value_net_params, weight_dtype


def setup_optimizer(config, transformer, value_net_params):
    """Setup optimizer."""
    pf_params = [
        param for name, param in transformer.named_parameters()
        if '.default.' in name
    ]

    params = [{"params": pf_params, "lr": config.training.lr}]

    if value_net_params is not None:
        params.append({
            "params": value_net_params,
            "lr": config.training.lr,
        })

    optimizer = torch.optim.AdamW(
        params,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
        eps=config.training.adam_epsilon,
    )

    return optimizer


def train():
    """Main training function."""
    # Initialize distributed training
    local_rank, global_rank, world_size = init_distributed_singlenode(timeout=36000)
    num_processes = world_size
    is_local_main_process = local_rank == 0
    setup_for_distributed(is_local_main_process)

    config = FLAGS.config
    device = torch.device(local_rank)

    if config.training.use_saved_rgrad_x1 and config.model.use_value_net:
        raise ValueError(
            "training.use_saved_rgrad_x1 is not supported when model.use_value_net=True."
        )

    # Setup logging
    logger, save_dir = setup_logging_and_saving(config, is_local_main_process)

    # Setup models
    pipeline, transformer, value_net, value_net_params, weight_dtype = setup_models(
        config, device, local_rank, is_local_main_process
    )

    # Setup optimizer and scaler
    optimizer = setup_optimizer(config, transformer, value_net_params)

    scaler = None
    if config.training.mixed_precision in ["fp16", "bf16"]:
        scaler = torch.cuda.amp.GradScaler(
            growth_interval=config.training.gradscaler_growth_interval
        )

    # Enable TF32
    if config.training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Setup reward and prompt functions
    prompt_fn = getattr(lib.reward_func.prompts, config.experiment.prompt_fn)
    reward_fn = getattr(lib.reward_func.rewards, config.experiment.reward_fn)(
        torch.float32, device
    )

    # Create refactored components
    cfg_wrapper = CFGWrapper(
        guidance_scale=config.sampling.guidance_scale,
        do_cfg=config.sampling.guidance_scale > 1.0
    )

    value_net_wrapper = None
    if value_net is not None:
        value_net_wrapper = ValueNetworkWrapper(
            value_net=value_net,
            value_net_param=config.model.value_net_param,
            pipeline=pipeline
        )

    reward_grad_computer = RewardGradientComputer(
        reward_fn=reward_fn,
        pipeline=pipeline,
        config=config
    )

    vggflow_algorithm = VGGFlowAlgorithm(
        transformer=transformer,
        value_net_wrapper=value_net_wrapper,
        cfg_wrapper=cfg_wrapper,
        reward_grad_computer=reward_grad_computer,
        config=config,
        pipeline=pipeline
    )

    vggflow_trainer = VGGFlowTrainer(
        vggflow_algorithm=vggflow_algorithm,
        transformer=transformer,
        value_net=value_net,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        pipeline=pipeline,
        logger=logger,
        local_rank=local_rank
    )

    trajectory_sampler = TrajectorySampler(
        pipeline=pipeline,
        prompt_fn=prompt_fn,
        reward_fn=reward_fn,
        reward_grad_computer=reward_grad_computer,
        config=config
    )
    periodic_evaluator = None
    if config.eval.enabled:
        periodic_evaluator = PeriodicEvaluator(
            pipeline=pipeline,
            reward_fn=reward_fn,
            vggflow_algorithm=vggflow_algorithm,
            config=config,
        )

    # Training loop
    result = defaultdict(dict)
    result["config"] = config.to_dict()
    start_time = time.time()

    if is_local_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {config.training.num_epochs}")
        logger.info(f"  Sample batch size per device = {config.sampling.batch_size}")
        logger.info(f"  Train batch size per device = {config.training.batch_size}")

    global_step = 0
    first_epoch = 0

    for epoch in range(first_epoch, config.training.num_epochs):
        epoch_start_time = time.time()

        # Sampling phase
        torch.cuda.empty_cache()
        transformer.eval()
        transformer.zero_grad()

        # Reset rgrad threshold each epoch (matches main version behavior)
        rgrad_threshold = 1.0

        samples = trajectory_sampler.sample_epoch(
            transformer=transformer.module,
            epoch=epoch,
            device=device,
            is_main_process=is_local_main_process
        )

        # Log samples
        if epoch >= 0:
            # Compute and log reward statistics
            rewards = torch.zeros(
                world_size * len(samples["rewards"]),
                dtype=samples["rewards"].dtype,
                device=device
            )
            dist.all_gather_into_tensor(rewards, samples["rewards"])
            rewards = rewards.detach().cpu().float().numpy()
            result["reward_mean"][global_step] = rewards.mean()
            result["reward_std"][global_step] = rewards.std()

            if is_local_main_process:
                logger.info(f"global_step: {global_step}  rewards: {rewards.mean():.3f}")
                if config.logging.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "reward_mean": rewards.mean(),
                            "reward_std": rewards.std(),
                        },
                        step=global_step,
                    )

                log_data = trajectory_sampler.get_last_log_data()
                if log_data is not None:
                    images, prompts, rewards_value = log_data
                    trajectory_sampler.log_samples(
                        images, prompts, rewards_value, epoch, wandb, is_local_main_process
                    )

        # Training phase
        for inner_epoch in range(config.training.num_inner_epochs):
            global_step, rgrad_threshold = vggflow_trainer.train_inner_epoch(
                samples=samples,
                epoch=epoch,
                inner_epoch=inner_epoch,
                global_step=global_step,
                rgrad_threshold=rgrad_threshold,
                num_processes=num_processes
            )

        epoch_duration_sec = time.time() - epoch_start_time
        result["time_per_epoch"][epoch] = epoch_duration_sec
        if is_local_main_process:
            logger.info(
                f"epoch={epoch} time_per_epoch={epoch_duration_sec:.2f}s "
                f"({epoch_duration_sec / 60.0:.2f} min)"
            )
            if config.logging.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "time_per_epoch": epoch_duration_sec,
                    },
                    step=global_step,
                )

        should_run_eval = (
            periodic_evaluator is not None
            and (
                epoch % config.eval.every_epochs == 0
                or epoch == config.training.num_epochs - 1
            )
        )
        if should_run_eval:
            eval_metrics = periodic_evaluator.evaluate(
                transformer=transformer.module,
                device=device,
                is_main_process=is_local_main_process,
            )
            result["eval_reward_mean"][epoch] = eval_metrics["eval_reward_mean"]
            result["eval_transition_drift"][epoch] = eval_metrics["eval_transition_drift"]

            if is_local_main_process:
                logger.info(
                    f"epoch={epoch} eval_reward_mean={eval_metrics['eval_reward_mean']:.6f} "
                    f"eval_transition_drift={eval_metrics['eval_transition_drift']:.6f}"
                )
                if config.logging.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch,
                            **eval_metrics,
                        },
                        step=global_step,
                    )

        # Save results
        if is_local_main_process:
            result["epoch"][global_step] = epoch
            result["time"][global_step] = time.time() - start_time
            pickle.dump(result, gzip.open(os.path.join(save_dir, f"result.json"), 'wb'))

        dist.barrier()

        # Save checkpoint
        if epoch % config.logging.save_freq == 0 or epoch == config.training.num_epochs - 1:
            if is_local_main_process:
                save_path = os.path.join(save_dir, f"checkpoint_epoch{epoch}")
                unwrapped_transformer = unwrap_model(transformer)
                transformer_lora_layers = get_peft_model_state_dict(
                    unwrapped_transformer, adapter_name='default'
                )
                StableDiffusion3Pipeline.save_lora_weights(
                    save_directory=save_path,
                    transformer_lora_layers=transformer_lora_layers,
                    is_main_process=is_local_main_process,
                    safe_serialization=True,
                )
                logger.info(f"Saved state to {save_path}")

            dist.barrier()

    # Final save
    if is_local_main_process:
        save_path = os.path.join(save_dir, f"checkpoint_epoch{epoch}")
        unwrapped_transformer = unwrap_model(transformer)
        transformer_lora_layers = get_peft_model_state_dict(
            unwrapped_transformer, adapter_name='default'
        )
        StableDiffusion3Pipeline.save_lora_weights(
            save_directory=save_path,
            transformer_lora_layers=transformer_lora_layers,
            is_main_process=is_local_main_process,
            safe_serialization=True,
        )
        logger.info(f"Saved state to {save_path}")

    dist.barrier()

    if config.logging.use_wandb and is_local_main_process:
        wandb.finish()

    dist.destroy_process_group()


def main(args):
    """Entry point with optional JVP mode."""
    if FLAGS.config.training.use_jvp and HAVE_SDPA_BACKEND:
        with sdpa_kernel(SDPBackend.MATH):
            train()
    else:
        train()


if __name__ == '__main__':
    app.run(main)
