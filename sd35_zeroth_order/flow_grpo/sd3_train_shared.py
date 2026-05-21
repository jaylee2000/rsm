import os

import numpy as np
import torch
from diffusers.utils.torch_utils import is_compiled_module

ZETA = 4.3152134639895925
VALID_LOSS_TYPES = {"ppo", "matching"}
VALID_REWEIGHT_TYPES = {"base", "grpo_guard", "pcpo_reweight", "tempflow_reweight", "fair_clip", "fair_clip2"}


def calculate_zero_std_ratio(group_labels, gathered_rewards):
    """
    Calculate zero-std ratio and average std over groups.

    Args:
        group_labels: Labels that define groups (prompt strings or seed ids).
        gathered_rewards: Dictionary containing rewards, must include 'ori_avg'.

    Returns:
        zero_std_ratio: Fraction of groups whose reward std is zero.
        reward_std_mean: Mean reward std across groups.
    """
    labels = np.asarray(group_labels)
    rewards = np.asarray(gathered_rewards["ori_avg"], dtype=np.float64)

    if rewards.ndim == 1:
        rewards = rewards[:, None]
    if labels.shape[0] != rewards.shape[0]:
        raise ValueError(
            f"Group label count ({labels.shape[0]}) must match reward count ({rewards.shape[0]})."
        )

    unique_labels, inverse_indices = np.unique(labels, return_inverse=True)
    if unique_labels.size == 0:
        return 0.0, 0.0

    group_stds = []
    zero_std_count = 0
    for group_idx in range(unique_labels.size):
        group_rewards = rewards[inverse_indices == group_idx]
        std_per_dim = np.std(group_rewards, axis=0)
        group_std = float(np.mean(std_per_dim))
        group_stds.append(group_std)
        if np.all(std_per_dim < 1e-12):
            zero_std_count += 1

    zero_std_ratio = zero_std_count / unique_labels.size
    return zero_std_ratio, float(np.mean(group_stds))


def compute_base_reweight_factor(delta_t, timestep, tilde_sigma):
    return (
        torch.sqrt(delta_t)
        / tilde_sigma
        * (
            1
            + (1 - timestep / 1000.0)
            / (2 * timestep / 1000.0)
            * tilde_sigma.pow(2)
        )
    )


def compute_pcpo_weight(delta_t):
    return ZETA * delta_t


def compute_ppo_ratio_by_reweight(
    reweight_type,
    log_prob,
    old_log_prob,
    *,
    delta_t,
    timestep,
    tilde_sigma,
    prev_sample_mean=None,
    old_prev_sample_mean=None,
    grpo_guard_bias_scale=None,
    grpo_guard_logprob_scale=None,
):
    if reweight_type in {"base", "tempflow_reweight"}:
        return torch.exp(log_prob - old_log_prob)

    if reweight_type == "fair_clip":
        return torch.exp((log_prob - old_log_prob) * tilde_sigma**2 * delta_t)

    if reweight_type == "pcpo_reweight":
        weight_pcpo = compute_pcpo_weight(delta_t)
        weight_base = compute_base_reweight_factor(delta_t, timestep, tilde_sigma)
        reweight_factor = weight_pcpo / weight_base
        return torch.exp((log_prob - old_log_prob) * reweight_factor)

    if reweight_type == "grpo_guard":
        if prev_sample_mean is None or old_prev_sample_mean is None:
            raise ValueError(
                "grpo_guard requires prev_sample_mean and old_prev_sample_mean."
            )
        if grpo_guard_bias_scale is None or grpo_guard_logprob_scale is None:
            raise ValueError(
                "grpo_guard requires grpo_guard_bias_scale and grpo_guard_logprob_scale."
            )
        ratio_mean_bias = (
            prev_sample_mean - old_prev_sample_mean
        ).pow(2).mean(dim=tuple(range(1, prev_sample_mean.ndim)))
        ratio_mean_bias = ratio_mean_bias / (2 * (grpo_guard_bias_scale**2))
        return torch.exp(
            (log_prob - old_log_prob + ratio_mean_bias) * grpo_guard_logprob_scale
        )

    raise NotImplementedError(f"Unknown reweight type {reweight_type}")


def compute_matching_weight_by_reweight(
    reweight_type, *, delta_t, timestep, tilde_sigma
):
    delta_t = delta_t.reshape(delta_t.shape[0], -1)[:, 0]
    timestep = timestep.reshape(timestep.shape[0], -1)[:, 0]
    tilde_sigma = tilde_sigma.reshape(tilde_sigma.shape[0], -1)[:, 0]

    if reweight_type == "pcpo_reweight":
        weight_row = compute_pcpo_weight(delta_t)
        return weight_row[:, None, None, None]
    if reweight_type in {"base", "grpo_guard", "tempflow_reweight", "fair_clip", "fair_clip2"}:
        weight_row = compute_base_reweight_factor(delta_t, timestep, tilde_sigma)
        return weight_row[:, None, None, None]
    raise NotImplementedError(f"Unknown reweight type {reweight_type}")


def combine_matching_terms_by_reweight(
    reweight_type, pred_matching_term_1, pred_matching_term_2, tilde_sigma, delta_t
):
    if reweight_type == "grpo_guard":
        return pred_matching_term_2 * tilde_sigma * (delta_t ** 0.5)
    if reweight_type in {"fair_clip", "fair_clip2"}:
        return (pred_matching_term_1 + pred_matching_term_2) * (tilde_sigma ** 2 * delta_t)
    if reweight_type == "tempflow_reweight":
        return (pred_matching_term_1 + pred_matching_term_2) * (2.25 * tilde_sigma * (delta_t ** 0.5))
    if reweight_type in {"base", "pcpo_reweight"}:
        return pred_matching_term_1 + pred_matching_term_2
    raise NotImplementedError(f"Unknown reweight type {reweight_type}")


def combine_ppo_terms_by_reweight(
    reweight_type, policy_term, *, sqrt_dt, tilde_sigma
):
    if reweight_type == "grpo_guard":
        return policy_term / (sqrt_dt**2)
    if reweight_type == "fair_clip":
        return policy_term / (sqrt_dt**2 * tilde_sigma**2)
    if reweight_type == "fair_clip2":
        return policy_term / (sqrt_dt**2 * tilde_sigma**2) * tilde_sigma / sqrt_dt
    if reweight_type == "tempflow_reweight":
        return policy_term * (2.25 * tilde_sigma * sqrt_dt)
    if reweight_type in {"base", "pcpo_reweight"}:
        return policy_term
    raise NotImplementedError(f"Unknown reweight type {reweight_type}")


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(
    save_dir,
    transformer,
    global_step,
    accelerator,
    ema,
    transformer_trainable_parameters,
    config,
):
    run_name = None
    try:
        import wandb

        if wandb.run is not None and getattr(wandb.run, "name", None):
            run_name = str(wandb.run.name)
    except Exception:
        run_name = None

    if not run_name:
        run_name = str(getattr(config, "run_name", "run"))

    save_root = os.path.join(
        save_dir, run_name, "checkpoints", f"checkpoint-{global_step}"
    )
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)
