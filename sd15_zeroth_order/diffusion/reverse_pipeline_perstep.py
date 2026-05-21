import torch

from numbers import Integral
from typing import Any, Callable, Dict, List, Optional, Union
from accelerate.logging import get_logger
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusion.pipeline_helpers import encode_prompt, prepare_latents
from tqdm.auto import tqdm

from diffusion.ddim_step import ddim_step_with_logprob

logger = get_logger(__name__, log_level="INFO")


def _normalize_exploration_schedule(exploration_k, num_inference_steps: int) -> List[int]:
    num_inference_steps = int(num_inference_steps)
    if num_inference_steps < 1:
        raise ValueError(
            "num_inference_steps must be >= 1 so at least one diffusion step exists, got "
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
                "When exploration_k is a list, its length must be exactly num_inference_steps "
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


def _select_model_embeddings(
    embeds: torch.Tensor,
    sample_indices: torch.LongTensor,
    do_classifier_free_guidance: bool,
) -> torch.Tensor:
    if not do_classifier_free_guidance:
        return embeds.index_select(0, sample_indices)
    half = embeds.shape[0] // 2
    return torch.cat(
        [
            embeds.index_select(0, sample_indices),
            embeds.index_select(0, sample_indices + half),
        ],
        dim=0,
    )


@torch.no_grad()
def pipeline_with_logprob(
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    scheduler: DDIMScheduler,
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    num_train_timesteps: int = 1000,
    num_inference_steps: int = 50,
    rg_scale: Optional[float] = 1.0,
    guidance_scale: float = 7.5,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: int = 1,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    unet_ref: Optional[torch.nn.Module] = None,
    disable_progress_bar: bool = False,
    const_weight: Optional[float] = None,
    algorithm: str = "uw-sigma",
    generator: Union[torch.Generator, List[torch.Generator], None] = None,
    return_dict: bool = False,
    exploration_k: Union[int, List[int]] = 1,
    collect_kl_anchor_rows: bool = False,
    *,
    latent_chunk_size: int,
):
    # unwrap DDP if present, so we can read .config
    raw_unet = unet.module if hasattr(unet, "module") else unet
    raw_vae = vae.module if hasattr(vae, "module") else vae

    if latent_chunk_size < 1:
        raise ValueError(f"latent_chunk_size must be >= 1, got {latent_chunk_size}.")

    if "lambda" in algorithm and unet_ref is None:
        raise ValueError(
            "algorithm with 'lambda' requires unet_ref for reference predictions."
        )

    # 1. Default height and width to unet
    vae_scale_factor = 2 ** (len(raw_vae.config.block_out_channels) - 1)  # 8
    height = raw_unet.config.sample_size * vae_scale_factor
    width = raw_unet.config.sample_size * vae_scale_factor

    # 2. Define call parameters
    batch_size = prompt_embeds.shape[0]
    device = unet.device

    # `guidance_scale = 1` corresponds to no classifier-free guidance
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None)
        if cross_attention_kwargs is not None
        else None
    )
    prompt_embeds = encode_prompt(
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        tokenizer,
        text_encoder,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
    )

    # 4. Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    num_steps = int(timesteps.shape[0])
    if num_steps < 1:
        raise ValueError(
            "num_inference_steps must yield at least 1 scheduler timestep for branching rollout."
        )

    exploration_schedule = _normalize_exploration_schedule(exploration_k, num_steps)

    # 5. Prepare latent variables
    num_channels_latents = raw_unet.config.in_channels
    latents = prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        vae_scale_factor,
        scheduler,
        generator=generator,
        latents=latents,
    )

    def _predict_noise_rows(
        latent_rows: torch.FloatTensor,
        timestep_rows: torch.LongTensor,
        sample_indices: torch.LongTensor,
    ):
        model_prompt_embeds = _select_model_embeddings(
            prompt_embeds,
            sample_indices,
            do_classifier_free_guidance,
        )

        latent_model_input = (
            torch.cat([latent_rows] * 2) if do_classifier_free_guidance else latent_rows
        )
        timestep_model_input = (
            torch.cat([timestep_rows, timestep_rows], dim=0)
            if do_classifier_free_guidance
            else timestep_rows
        )

        noise_pred = unet(
            latent_model_input,
            timestep_model_input,
            encoder_hidden_states=model_prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            noise_pred_text = noise_pred

        noise_pred_ref = None
        noise_pred_ref_text = None
        if "lambda" in algorithm:
            noise_pred_ref = unet_ref(
                latent_model_input,
                timestep_model_input,
                encoder_hidden_states=model_prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]
            if do_classifier_free_guidance:
                noise_pred_ref_uncond, noise_pred_ref_text = noise_pred_ref.chunk(2)
                noise_pred_ref = noise_pred_ref_uncond + guidance_scale * (
                    noise_pred_ref_text - noise_pred_ref_uncond
                )

        return noise_pred, noise_pred_ref, noise_pred_text, noise_pred_ref_text

    # 6. Deterministic root trajectory (pass 1)
    all_latents = [latents]
    progress_bar = tqdm(
        desc="DDIM Deterministic Root",
        total=num_steps,
        dynamic_ncols=True,
        position=1,
        leave=False,
        disable=disable_progress_bar,
    )
    with progress_bar:
        for i, t in enumerate(timesteps):
            step_timestep = t.expand(latents.shape[0]).clone()

            noise_pred, noise_pred_ref, noise_pred_text, noise_pred_ref_text = _predict_noise_rows(
                latents,
                step_timestep,
                torch.arange(latents.shape[0], device=latents.device, dtype=torch.long),
            )

            latents, _, _ = ddim_step_with_logprob(
                scheduler,
                noise_pred,
                noise_pred_ref,
                noise_pred_text,
                noise_pred_ref_text,
                None,
                step_timestep,
                latents,
                rg_scale=rg_scale,
                num_train_timesteps=num_train_timesteps,
                num_inference_steps=num_steps,
                eta=0.0,
                const_weight=const_weight,
                algorithm=algorithm,
                generator=generator,
            )

            all_latents.append(latents)

            if i == len(timesteps) - 1 or (i + 1) % 1 == 0:
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    def _materialize_row_timestep(
        all_timesteps: torch.Tensor,
        step_positions: torch.LongTensor,
    ) -> torch.Tensor:
        if step_positions.numel() == 0:
            return torch.empty(0, device=all_timesteps.device, dtype=all_timesteps.dtype)
        return all_timesteps.index_select(0, step_positions)

    def _run_global_ragged_2b(
        row_step_idx: torch.LongTensor,
        row_sample_idx: torch.LongTensor,
        row_next_latents: torch.FloatTensor,
    ):
        row_count = int(row_next_latents.shape[0])
        if row_count == 0:
            return row_next_latents, 0, 0

        remaining_steps = torch.clamp(
            timesteps.shape[0] - (row_step_idx + 1),
            min=0,
        )

        useful_row_steps = int(remaining_steps.sum().item())
        processed_padded_row_steps = 0

        final_latents = row_next_latents.clone()
        expected_row_steps = remaining_steps.clone()
        executed_row_steps = torch.zeros_like(remaining_steps)
        finalized_rows = remaining_steps <= 0

        active_row_ids = torch.nonzero(~finalized_rows, as_tuple=False).squeeze(1)
        active_sample_idx = row_sample_idx.index_select(0, active_row_ids)
        active_next_step_pos = row_step_idx.index_select(0, active_row_ids) + 1
        active_remaining = remaining_steps.index_select(0, active_row_ids)

        while active_row_ids.numel() > 0:
            for branch_start in range(0, active_row_ids.shape[0], latent_chunk_size):
                branch_end = min(branch_start + latent_chunk_size, active_row_ids.shape[0])
                chunk_row_ids = active_row_ids[branch_start:branch_end]
                branch_indices = active_sample_idx[branch_start:branch_end]
                row_step_positions = active_next_step_pos[branch_start:branch_end]

                inner_latents = final_latents.index_select(0, chunk_row_ids)
                actual_branch_size = inner_latents.shape[0]
                model_rows = actual_branch_size

                if actual_branch_size < latent_chunk_size:
                    pad_count = latent_chunk_size - actual_branch_size
                    inner_latents = torch.cat(
                        [inner_latents, inner_latents[:1].repeat(pad_count, 1, 1, 1)],
                        dim=0,
                    )
                    branch_indices = torch.cat(
                        [branch_indices, branch_indices[:1].repeat(pad_count)],
                        dim=0,
                    )
                    row_step_positions = torch.cat(
                        [row_step_positions, row_step_positions[:1].repeat(pad_count)],
                        dim=0,
                    )
                    model_rows = latent_chunk_size

                processed_padded_row_steps += int(model_rows)
                executed_row_steps.index_add_(
                    0,
                    chunk_row_ids,
                    torch.ones(
                        actual_branch_size,
                        device=final_latents.device,
                        dtype=torch.long,
                    ),
                )

                row_timestep = _materialize_row_timestep(timesteps, row_step_positions)

                noise_pred, noise_pred_ref, noise_pred_text, noise_pred_ref_text = _predict_noise_rows(
                    inner_latents,
                    row_timestep,
                    branch_indices,
                )

                inner_latents, _, _ = ddim_step_with_logprob(
                    scheduler,
                    noise_pred,
                    noise_pred_ref,
                    noise_pred_text,
                    noise_pred_ref_text,
                    None,
                    row_timestep,
                    inner_latents,
                    rg_scale=rg_scale,
                    num_train_timesteps=num_train_timesteps,
                    num_inference_steps=num_steps,
                    eta=0.0,
                    const_weight=const_weight,
                    algorithm=algorithm,
                )

                if actual_branch_size < latent_chunk_size:
                    inner_latents = inner_latents[:actual_branch_size]

                final_latents.index_copy_(0, chunk_row_ids, inner_latents)

            active_next_step_pos = active_next_step_pos + 1
            active_remaining = active_remaining - 1
            finished_mask = active_remaining <= 0
            if finished_mask.any():
                finished_row_ids = active_row_ids[finished_mask]
                if finalized_rows[finished_row_ids].any():
                    raise ValueError(
                        "Pass-2b row-finalization invariant failed: a row was finalized more than once."
                    )
                finalized_rows[finished_row_ids] = True

            keep_mask = ~finished_mask
            active_sample_idx = active_sample_idx[keep_mask]
            active_next_step_pos = active_next_step_pos[keep_mask]
            active_remaining = active_remaining[keep_mask]
            active_row_ids = active_row_ids[keep_mask]

        if not torch.equal(executed_row_steps, expected_row_steps):
            raise ValueError(
                "Pass-2b step-count invariant failed: executed row-steps do not match expected remaining steps."
            )
        if not bool(torch.all(finalized_rows).item()):
            raise ValueError(
                "Pass-2b finalization invariant failed: some rows were never finalized."
            )

        return final_latents, useful_row_steps, processed_padded_row_steps

    def _decode_rows_in_chunks(latent_rows: torch.Tensor):
        if latent_rows.shape[0] == 0:
            return [] if output_type == "pil" else torch.empty(0)

        tensor_chunks = []
        list_chunks = []
        image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

        decode_rows = max(int(latent_chunk_size), 1)
        for start in range(0, latent_rows.shape[0], decode_rows):
            end = min(start + decode_rows, latent_rows.shape[0])
            chunk_latents = latent_rows[start:end]

            if output_type == "latent":
                decoded_chunk = chunk_latents
            else:
                image = vae.decode(
                    chunk_latents / raw_vae.config.scaling_factor,
                    return_dict=False,
                )[0]
                do_denormalize = [True] * image.shape[0]
                decoded_chunk = image_processor.postprocess(
                    image,
                    output_type=output_type,
                    do_denormalize=do_denormalize,
                )

            if torch.is_tensor(decoded_chunk):
                tensor_chunks.append(decoded_chunk)
            elif isinstance(decoded_chunk, list):
                list_chunks.extend(decoded_chunk)
            else:
                raise ValueError(
                    "Unexpected decoded chunk type: "
                    f"{type(decoded_chunk).__name__}."
                )

        if len(tensor_chunks) > 0:
            return torch.cat(tensor_chunks, dim=0)
        return list_chunks

    # 7. Branching transitions (pass 2a)
    packed_step_idx_chunks = []
    packed_sample_idx_chunks = []
    packed_exploration_k_chunks = []
    packed_timestep_chunks = []
    packed_next_timestep_chunks = []
    packed_latent_chunks = []
    packed_next_latent_chunks = []
    packed_log_prob_chunks = []
    packed_noise_chunks = []
    packed_pred_chunks = []
    packed_kl_step_idx_chunks = []
    packed_kl_sample_idx_chunks = []
    packed_kl_timestep_chunks = []
    packed_kl_next_timestep_chunks = []
    packed_kl_latent_chunks = []

    for step_index in range(num_steps):
        step_k = int(exploration_schedule[step_index])
        step_timestep = timesteps[step_index].expand(batch_size).clone()
        step_latents = all_latents[step_index]
        if step_k == 0:
            # Anchor only non-terminal zero-k rows (skip terminal t=0).
            if collect_kl_anchor_rows and step_index < (num_steps - 1):
                row_count = int(step_latents.shape[0])
                step_indices = torch.arange(
                    row_count,
                    device=step_latents.device,
                    dtype=torch.long,
                )
                packed_kl_step_idx_chunks.append(
                    torch.full(
                        (row_count,),
                        fill_value=int(step_index),
                        device=step_latents.device,
                        dtype=torch.long,
                    )
                )
                packed_kl_sample_idx_chunks.append(step_indices)
                packed_kl_timestep_chunks.append(step_timestep.clone())
                packed_kl_next_timestep_chunks.append(
                    timesteps[step_index + 1].expand(row_count).clone()
                )
                packed_kl_latent_chunks.append(step_latents.clone())
            continue

        expanded_step_indices = torch.arange(
            step_latents.shape[0],
            device=step_latents.device,
            dtype=torch.long,
        ).repeat_interleave(step_k)
        expanded_step_timestep = step_timestep.repeat_interleave(step_k)

        step_sde_latent_chunks = []
        step_log_prob_chunks = []
        step_noise_chunks = []
        step_pred_chunks = []
        step_branch_index_chunks = []

        for branch_start in range(0, expanded_step_indices.shape[0], latent_chunk_size):
            branch_end = min(branch_start + latent_chunk_size, expanded_step_indices.shape[0])
            step_indices = expanded_step_indices[branch_start:branch_end]
            step_timestep_chunk = expanded_step_timestep[branch_start:branch_end]
            actual_chunk_rows = step_indices.numel()

            if actual_chunk_rows < latent_chunk_size:
                pad_count = latent_chunk_size - actual_chunk_rows
                step_indices_model = torch.cat(
                    [step_indices, step_indices[:1].repeat(pad_count)],
                    dim=0,
                )
                step_timestep_model = torch.cat(
                    [step_timestep_chunk, step_timestep_chunk[:1].repeat(pad_count)],
                    dim=0,
                )
            else:
                step_indices_model = step_indices
                step_timestep_model = step_timestep_chunk

            step_latents_chunk = step_latents.index_select(0, step_indices_model)

            noise_pred, noise_pred_ref, noise_pred_text, noise_pred_ref_text = _predict_noise_rows(
                step_latents_chunk,
                step_timestep_model,
                step_indices_model,
            )

            sde_latents_chunk, log_prob_chunk, noise_chunk = ddim_step_with_logprob(
                scheduler,
                noise_pred,
                noise_pred_ref,
                noise_pred_text,
                noise_pred_ref_text,
                None,
                step_timestep_model,
                step_latents_chunk,
                rg_scale=rg_scale,
                num_train_timesteps=num_train_timesteps,
                num_inference_steps=num_steps,
                eta=eta,
                const_weight=const_weight,
                algorithm=algorithm,
                generator=generator,
            )

            if actual_chunk_rows < latent_chunk_size:
                sde_latents_chunk = sde_latents_chunk[:actual_chunk_rows]
                log_prob_chunk = log_prob_chunk[:actual_chunk_rows]
                noise_chunk = noise_chunk[:actual_chunk_rows]
                noise_pred = noise_pred[:actual_chunk_rows]

            step_sde_latent_chunks.append(sde_latents_chunk)
            step_log_prob_chunks.append(log_prob_chunk)
            step_noise_chunks.append(noise_chunk)
            step_pred_chunks.append(noise_pred)
            step_branch_index_chunks.append(step_indices)

        sde_latents = torch.cat(step_sde_latent_chunks, dim=0)
        log_prob = torch.cat(step_log_prob_chunks, dim=0)
        noise = torch.cat(step_noise_chunks, dim=0)
        noise_pred = torch.cat(step_pred_chunks, dim=0)
        step_branch_indices = torch.cat(step_branch_index_chunks, dim=0)

        expected_step_rows = expanded_step_indices.shape[0]
        if sde_latents.shape[0] != expected_step_rows:
            raise ValueError(
                "Unexpected SDE batch shape in pass 2a: "
                f"expected {expected_step_rows}, got {sde_latents.shape[0]}."
            )

        packed_step_idx_chunks.append(
            torch.full(
                (expected_step_rows,),
                fill_value=int(step_index),
                device=sde_latents.device,
                dtype=torch.long,
            )
        )
        packed_sample_idx_chunks.append(step_branch_indices)
        packed_exploration_k_chunks.append(
            torch.full(
                (expected_step_rows,),
                fill_value=float(step_k),
                device=sde_latents.device,
                dtype=torch.float32,
            )
        )
        packed_timestep_chunks.append(expanded_step_timestep)
        # For terminal-step rows, keep row_next_timesteps clamped to the final scheduler value
        # so packed-row metadata remains fixed-shape and explicit.
        next_step_index = min(step_index + 1, num_steps - 1)
        packed_next_timestep_chunks.append(
            timesteps[next_step_index].expand(expected_step_rows).clone()
        )
        packed_latent_chunks.append(step_latents.index_select(0, step_branch_indices))
        packed_next_latent_chunks.append(sde_latents)
        packed_log_prob_chunks.append(log_prob)
        packed_noise_chunks.append(noise)
        packed_pred_chunks.append(noise_pred)

    if len(packed_step_idx_chunks) == 0:
        raise ValueError(
            "No DDIM step rows were generated because exploration_k is zero for all sampled steps. "
            "Provide at least one step with exploration_k > 0."
        )

    packed_row_step_idx = torch.cat(packed_step_idx_chunks, dim=0)
    packed_row_sample_idx = torch.cat(packed_sample_idx_chunks, dim=0)
    packed_row_exploration_k = torch.cat(packed_exploration_k_chunks, dim=0)
    packed_row_timesteps = torch.cat(packed_timestep_chunks, dim=0)
    packed_row_next_timesteps = torch.cat(packed_next_timestep_chunks, dim=0)
    packed_row_latents = torch.cat(packed_latent_chunks, dim=0)
    packed_row_next_latents = torch.cat(packed_next_latent_chunks, dim=0)
    packed_row_log_probs = torch.cat(packed_log_prob_chunks, dim=0)
    packed_row_noises = torch.cat(packed_noise_chunks, dim=0)
    packed_row_preds = torch.cat(packed_pred_chunks, dim=0)
    if len(packed_kl_step_idx_chunks) > 0:
        packed_kl_row_step_idx = torch.cat(packed_kl_step_idx_chunks, dim=0)
        packed_kl_row_sample_idx = torch.cat(packed_kl_sample_idx_chunks, dim=0)
        packed_kl_row_timesteps = torch.cat(packed_kl_timestep_chunks, dim=0)
        packed_kl_row_next_timesteps = torch.cat(packed_kl_next_timestep_chunks, dim=0)
        packed_kl_row_latents = torch.cat(packed_kl_latent_chunks, dim=0)
    else:
        latent_shape_tail = all_latents[0].shape[1:]
        packed_kl_row_step_idx = torch.empty(
            0,
            device=all_latents[0].device,
            dtype=torch.long,
        )
        packed_kl_row_sample_idx = torch.empty(
            0,
            device=all_latents[0].device,
            dtype=torch.long,
        )
        packed_kl_row_timesteps = torch.empty(
            0,
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
        packed_kl_row_next_timesteps = torch.empty(
            0,
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
        packed_kl_row_latents = torch.empty(
            (0, *latent_shape_tail),
            device=all_latents[0].device,
            dtype=all_latents[0].dtype,
        )

    # 8. Deterministic continuation and decode (pass 2b)
    final_latents, useful_row_steps_2b, processed_row_steps_2b = _run_global_ragged_2b(
        packed_row_step_idx,
        packed_row_sample_idx,
        packed_row_next_latents,
    )

    if final_latents.shape[0] != packed_row_next_latents.shape[0]:
        raise ValueError(
            "Pass-2b output row count mismatch, got "
            f"{final_latents.shape[0]} and {packed_row_next_latents.shape[0]}."
        )
    if processed_row_steps_2b < useful_row_steps_2b:
        raise ValueError(
            "Pass-2b accounting mismatch: processed row-steps cannot be smaller than useful row-steps, got "
            f"{processed_row_steps_2b} and {useful_row_steps_2b}."
        )

    packed_images = _decode_rows_in_chunks(final_latents)

    packed_rollout = {
        "row_step_idx": packed_row_step_idx,
        "row_sample_idx": packed_row_sample_idx,
        "row_exploration_k": packed_row_exploration_k,
        "row_timesteps": packed_row_timesteps,
        "row_next_timesteps": packed_row_next_timesteps,
        "row_latents": packed_row_latents,
        "row_next_latents": packed_row_next_latents,
        "row_log_probs": packed_row_log_probs,
        "row_noises": packed_row_noises,
        "row_noise_preds": packed_row_preds,
        "kl_row_step_idx": packed_kl_row_step_idx,
        "kl_row_sample_idx": packed_kl_row_sample_idx,
        "kl_row_timesteps": packed_kl_row_timesteps,
        "kl_row_next_timesteps": packed_kl_row_next_timesteps,
        "kl_row_latents": packed_kl_row_latents,
        "row_images": packed_images,
        "useful_row_steps_2b": int(useful_row_steps_2b),
        "processed_row_steps_2b": int(processed_row_steps_2b),
        "padded_row_steps_2b": int(processed_row_steps_2b - useful_row_steps_2b),
    }

    if not return_dict:
        return packed_rollout

    return packed_images, packed_rollout
