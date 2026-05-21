# Copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_3/pipeline_stable_diffusion_3.py
# with the following modifications:
# - It uses the patched version of `sde_step_with_logprob` from `sd3_sde_with_logprob.py`.
# - It returns packed transition-row tensors used by branching training.
from numbers import Integral
from typing import Any, Callable, Dict, List, Optional, Union
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import XLA_AVAILABLE
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput
from .sd3_pipeline_with_logprob_shared import (
    decode_sd3_latents,
    prepare_ip_adapter_embeddings,
    prepare_latents_and_timesteps,
    setup_sd3_pipeline_context,
)
from .sd3_sde_with_logprob import sde_step_with_logprob

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


def _normalize_exploration_schedule(exploration_k, num_inference_steps: int) -> List[int]:
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
        if len(exploration_k) == num_transitions:
            values = exploration_k
        elif len(exploration_k) == num_inference_steps:
            # Backward-compatible path: accept step-aligned lists and ignore the last entry.
            values = exploration_k[:-1]
        else:
            raise ValueError(
                "When exploration_k is a list, its length must be either num_inference_steps "
                f"({num_inference_steps}) or num_inference_steps - 1 ({num_transitions}), got {len(exploration_k)}."
            )

        schedule = []
        for i, value in enumerate(values):
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


@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    prompt_3: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 7.0,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt_3: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    skip_guidance_layers: List[int] = None,
    skip_layer_guidance_scale: float = 2.8,
    skip_layer_guidance_stop: float = 0.2,
    skip_layer_guidance_start: float = 0.01,
    mu: Optional[float] = None,
    determistic: bool = False,
    noise_level: float = 0.7,
    return_prev_sample_mean: bool = False,
    collect_matching_aux: bool = False,
    exploration_k: Union[int, List[int]] = 1,
    collect_kl_anchor_rows: bool = False,
    *,
    latent_chunk_size: int,
):
    r"""
    Function invoked when calling the pipeline for generation.
    Args:
        prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
            instead.
        prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
            will be used instead
        prompt_3 (`str` or `List[str]`, *optional*):
            The prompt or prompts to be sent to `tokenizer_3` and `text_encoder_3`. If not defined, `prompt` is
            will be used instead
        height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The height in pixels of the generated image. This is set to 1024 by default for the best results.
        width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
            The width in pixels of the generated image. This is set to 1024 by default for the best results.
        num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        sigmas (`List[float]`, *optional*):
            Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
            their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
            will be used.
        guidance_scale (`float`, *optional*, defaults to 7.0):
            Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
            `guidance_scale` is defined as `w` of equation 2. of [Imagen
            Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
            1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
            usually at the expense of lower image quality.
        negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used instead
        negative_prompt_3 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_3` and
            `text_encoder_3`. If not defined, `negative_prompt` is used instead
        num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
            to make generation deterministic.
        latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will ge generated by sampling using the supplied random `generator`.
        prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        ip_adapter_image (`PipelineImageInput`, *optional*):
            Optional image input to work with IP Adapters.
        ip_adapter_image_embeds (`torch.Tensor`, *optional*):
            Pre-generated image embeddings for IP-Adapter. Should be a tensor of shape `(batch_size, num_images,
            emb_dim)`. It should contain the negative image embedding if `do_classifier_free_guidance` is set to
            `True`. If not provided, embeddings are computed from the `ip_adapter_image` input argument.
        output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generate image. Choose between
            [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] instead of
            a plain tuple.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising steps during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        max_sequence_length (`int` defaults to 256): Maximum sequence length to use with the `prompt`.
        skip_guidance_layers (`List[int]`, *optional*):
            A list of integers that specify layers to skip during guidance. If not provided, all layers will be
            used for guidance. If provided, the guidance will only be applied to the layers specified in the list.
            Recommended value by StabiltyAI for Stable Diffusion 3.5 Medium is [7, 8, 9].
        skip_layer_guidance_scale (`int`, *optional*): The scale of the guidance for the layers specified in
            `skip_guidance_layers`. The guidance will be applied to the layers specified in `skip_guidance_layers`
            with a scale of `skip_layer_guidance_scale`. The guidance will be applied to the rest of the layers
            with a scale of `1`.
        skip_layer_guidance_stop (`int`, *optional*): The step at which the guidance for the layers specified in
            `skip_guidance_layers` will stop. The guidance will be applied to the layers specified in
            `skip_guidance_layers` until the fraction specified in `skip_layer_guidance_stop`. Recommended value by
            StabiltyAI for Stable Diffusion 3.5 Medium is 0.2.
        skip_layer_guidance_start (`int`, *optional*): The step at which the guidance for the layers specified in
            `skip_guidance_layers` will start. The guidance will be applied to the layers specified in
            `skip_guidance_layers` from the fraction specified in `skip_layer_guidance_start`. Recommended value by
            StabiltyAI for Stable Diffusion 3.5 Medium is 0.01.
        mu (`float`, *optional*): `mu` value used for `dynamic_shifting`.

    Examples:

    Returns:
        [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] or `tuple`:
        [`~pipelines.stable_diffusion_3.StableDiffusion3PipelineOutput`] if `return_dict` is True, otherwise a
        `tuple`. When returning a tuple, the first element is a list with the generated images.
    """

    ctx = setup_sd3_pipeline_context(
        self,
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
        guidance_scale=guidance_scale,
        skip_layer_guidance_scale=skip_layer_guidance_scale,
        clip_skip=clip_skip,
        joint_attention_kwargs=joint_attention_kwargs,
        num_images_per_prompt=num_images_per_prompt,
        skip_guidance_layers=skip_guidance_layers,
    )
    height = ctx["height"]
    width = ctx["width"]
    batch_size = ctx["batch_size"]
    device = ctx["device"]
    prompt_embeds = ctx["prompt_embeds"]
    negative_prompt_embeds = ctx["negative_prompt_embeds"]
    pooled_prompt_embeds = ctx["pooled_prompt_embeds"]
    negative_pooled_prompt_embeds = ctx["negative_pooled_prompt_embeds"]
    original_prompt_embeds = ctx["original_prompt_embeds"]
    original_pooled_prompt_embeds = ctx["original_pooled_prompt_embeds"]

    latents, timesteps, num_inference_steps, num_warmup_steps = prepare_latents_and_timesteps(
        self,
        batch_size=batch_size,
        num_images_per_prompt=num_images_per_prompt,
        height=height,
        width=width,
        prompt_embeds_dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=latents,
        num_inference_steps=num_inference_steps,
        sigmas=sigmas,
        mu=mu,
    )
    exploration_schedule = _normalize_exploration_schedule(exploration_k, num_inference_steps)

    prepare_ip_adapter_embeddings(
        self,
        ip_adapter_image=ip_adapter_image,
        ip_adapter_image_embeds=ip_adapter_image_embeds,
        device=device,
        batch_size=batch_size,
        num_images_per_prompt=num_images_per_prompt,
    )

    all_latents = [latents]
    ode_step_states = []

    # 7. Denoising loop (pass 1): collect ODE trajectory and per-step states.
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps[:-1]):
            if self.interrupt:
                continue

            step_timestep = t.expand(latents.shape[0]).clone()
            ode_step_states.append((i, step_timestep))

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = (
                torch.cat([step_timestep, step_timestep], dim=0)
                if self.do_classifier_free_guidance
                else step_timestep
            )
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.to(prompt_embeds.dtype)
            # perform guidance
            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                should_skip_layers = (
                    True
                    if i > num_inference_steps * skip_layer_guidance_start
                    and i < num_inference_steps * skip_layer_guidance_stop
                    else False
                )
                if skip_guidance_layers is not None and should_skip_layers:
                    latent_model_input = latents
                    noise_pred_skip_layers = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=step_timestep,
                        encoder_hidden_states=original_prompt_embeds,
                        pooled_projections=original_pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        skip_layers=skip_guidance_layers,
                    )[0]
                    noise_pred = (
                        noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                    )
            latents_dtype = latents.dtype
            
            # ODE process
            ode_latents, _, _, _ = sde_step_with_logprob(
                self.scheduler, 
                noise_pred.float(), 
                step_timestep,
                latents.float(),
                noise_level=noise_level,
                sde_type="deterministic",
            )
            latents = ode_latents
            prev_latents = latents.clone()
            
            all_latents.append(latents)
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)
            
            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                negative_pooled_prompt_embeds = callback_outputs.pop(
                    "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                )

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    # 8. Branching loop (pass 2a): sample SDE branches in rollout-sized chunks.
    # Prompt embeddings are shared across timesteps, and step latents come from all_latents.
    # To keep transformer usage aligned with training, we batch pass 2a using `latent_chunk_size`.

    def _select_model_embeddings(embeds: torch.Tensor, sample_indices: torch.LongTensor) -> torch.Tensor:
        if not self.do_classifier_free_guidance:
            return embeds.index_select(0, sample_indices)
        half = embeds.shape[0] // 2
        return torch.cat(
            [embeds.index_select(0, sample_indices), embeds.index_select(0, sample_indices + half)],
            dim=0,
        )

    def _materialize_row_timestep(
        all_timesteps: torch.Tensor, step_positions: torch.LongTensor
    ) -> torch.Tensor:
        if step_positions.numel() == 0:
            return torch.empty(0, device=all_timesteps.device, dtype=all_timesteps.dtype)
        return all_timesteps.index_select(0, step_positions)

    def _run_global_ragged_2b(
        row_step_idx: torch.LongTensor,
        row_sample_idx: torch.LongTensor,
        row_next_latents: torch.Tensor,
    ) -> tuple:
        row_count = int(row_next_latents.shape[0])
        if row_count == 0:
            return row_next_latents, 0, 0

        remaining_steps = torch.clamp(
            timesteps.shape[0] - (row_step_idx + 1),
            min=0,
        )

        useful_row_steps = int(remaining_steps.sum().item())
        processed_padded_row_steps = 0

        # Hold exactly one full-row working copy and update rows in place.
        final_latents = row_next_latents.clone()
        expected_row_steps = remaining_steps.clone()
        finalized_rows = remaining_steps <= 0
        executed_row_steps = torch.zeros_like(remaining_steps)

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
                    inner_latents = torch.cat([inner_latents, inner_latents[:1].repeat(pad_count, 1, 1, 1)], dim=0)
                    branch_indices = torch.cat([branch_indices, branch_indices[:1].repeat(pad_count)], dim=0)
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
                branch_prompt_embeds = _select_model_embeddings(prompt_embeds, branch_indices)
                branch_pooled_prompt_embeds = _select_model_embeddings(pooled_prompt_embeds, branch_indices)

                latent_model_input = (
                    torch.cat([inner_latents] * 2) if self.do_classifier_free_guidance else inner_latents
                )
                timestep = (
                    torch.cat([row_timestep, row_timestep], dim=0)
                    if self.do_classifier_free_guidance
                    else row_timestep
                )
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=branch_prompt_embeds,
                    pooled_projections=branch_pooled_prompt_embeds,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                inner_latents_dtype = inner_latents.dtype
                inner_latents, _, _, _ = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred.float(),
                    row_timestep,
                    inner_latents.float(),
                    noise_level=noise_level,
                    sde_type="deterministic",
                )
                if inner_latents.dtype != inner_latents_dtype:
                    inner_latents = inner_latents.to(inner_latents_dtype)
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

            if XLA_AVAILABLE:
                xm.mark_step()

        if not torch.equal(executed_row_steps, expected_row_steps):
            raise ValueError(
                "Pass-2b step-count invariant failed: executed row-steps do not match expected remaining steps."
            )
        if not bool(torch.all(finalized_rows).item()):
            raise ValueError(
                "Pass-2b finalization invariant failed: some rows were never finalized."
            )

        return final_latents, useful_row_steps, processed_padded_row_steps

    def _decode_rows_in_chunks(latent_rows: torch.Tensor) -> Union[torch.Tensor, list]:
        if latent_rows.shape[0] == 0:
            return []

        tensor_chunks = []
        list_chunks = []
        decode_rows = max(int(latent_chunk_size), 1)
        for start in range(0, latent_rows.shape[0], decode_rows):
            end = min(start + decode_rows, latent_rows.shape[0])
            decoded_chunk = decode_sd3_latents(
                self,
                latent_rows[start:end],
                output_type=output_type,
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

    packed_step_idx_chunks = []
    packed_sample_idx_chunks = []
    packed_exploration_k_chunks = []
    packed_timestep_chunks = []
    packed_next_timestep_chunks = []
    packed_latent_chunks = []
    packed_next_latent_chunks = []
    packed_log_prob_chunks = []
    packed_kl_step_idx_chunks = []
    packed_kl_sample_idx_chunks = []
    packed_kl_timestep_chunks = []
    packed_kl_next_timestep_chunks = []
    packed_kl_latent_chunks = []
    packed_prev_mean_chunks = [] if return_prev_sample_mean else None
    packed_noise_chunks = [] if collect_matching_aux else None
    packed_pred_chunks = [] if collect_matching_aux else None
    for (i, step_timestep), step_latents, step_k in zip(
        ode_step_states,
        all_latents[:-1],
        exploration_schedule,
    ):
        if int(step_k) == 0:
            if not collect_kl_anchor_rows:
                continue
            row_count = int(step_latents.shape[0])
            step_indices = torch.arange(
                row_count,
                device=step_latents.device,
                dtype=torch.long,
            )
            packed_kl_step_idx_chunks.append(
                torch.full(
                    (row_count,),
                    fill_value=int(i),
                    device=step_latents.device,
                    dtype=torch.long,
                )
            )
            packed_kl_sample_idx_chunks.append(step_indices)
            packed_kl_timestep_chunks.append(step_timestep.clone())
            packed_kl_next_timestep_chunks.append(
                timesteps[i + 1].expand(row_count).clone()
            )
            packed_kl_latent_chunks.append(step_latents.clone())
            continue
        should_skip_layers = (
            True
            if i > num_inference_steps * skip_layer_guidance_start
            and i < num_inference_steps * skip_layer_guidance_stop
            else False
        )
        expanded_step_indices = torch.arange(
            step_latents.shape[0],
            device=step_latents.device,
            dtype=torch.long,
        ).repeat_interleave(step_k)
        expanded_step_timestep = step_timestep.repeat_interleave(step_k)

        step_sde_latent_chunks = []
        step_log_prob_chunks = []
        step_prev_latent_mean_chunks = [] if return_prev_sample_mean else None
        step_noise_chunks = [] if collect_matching_aux else None
        step_pred_chunks = [] if collect_matching_aux else None
        step_branch_index_chunks = []
        for branch_start in range(0, expanded_step_indices.shape[0], latent_chunk_size):
            branch_end = min(branch_start + latent_chunk_size, expanded_step_indices.shape[0])
            step_indices = expanded_step_indices[branch_start:branch_end]
            step_timestep_chunk = expanded_step_timestep[branch_start:branch_end]
            actual_chunk_rows = step_indices.numel()
            if actual_chunk_rows < latent_chunk_size:
                pad_count = latent_chunk_size - actual_chunk_rows
                step_indices_model = torch.cat([step_indices, step_indices[:1].repeat(pad_count)], dim=0)
                step_timestep_model = torch.cat(
                    [step_timestep_chunk, step_timestep_chunk[:1].repeat(pad_count)],
                    dim=0,
                )
            else:
                step_indices_model = step_indices
                step_timestep_model = step_timestep_chunk
            step_latents_chunk = step_latents.index_select(0, step_indices_model)

            latent_model_input = (
                torch.cat([step_latents_chunk] * 2) if self.do_classifier_free_guidance else step_latents_chunk
            )
            timestep = (
                torch.cat([step_timestep_model, step_timestep_model], dim=0)
                if self.do_classifier_free_guidance
                else step_timestep_model
            )
            chunk_prompt_embeds = _select_model_embeddings(prompt_embeds, step_indices_model)
            chunk_pooled_prompt_embeds = _select_model_embeddings(pooled_prompt_embeds, step_indices_model)
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=chunk_prompt_embeds,
                pooled_projections=chunk_pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.to(prompt_embeds.dtype)

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                if skip_guidance_layers is not None and should_skip_layers:
                    noise_pred_skip_layers = self.transformer(
                        hidden_states=step_latents_chunk,
                        timestep=step_timestep_model,
                        encoder_hidden_states=original_prompt_embeds.index_select(0, step_indices_model),
                        pooled_projections=original_pooled_prompt_embeds.index_select(0, step_indices_model),
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        skip_layers=skip_guidance_layers,
                    )[0]
                    noise_pred = (
                        noise_pred + (noise_pred_text - noise_pred_skip_layers) * self._skip_layer_guidance_scale
                    )

            if collect_matching_aux:
                (
                    sde_latents_chunk,
                    log_prob_chunk,
                    prev_sample_mean_chunk,
                    _,
                    noise_chunk,
                ) = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred.float(),
                    step_timestep_model,
                    step_latents_chunk.float(),
                    noise_level=noise_level,
                    sde_type="sde",
                    return_noise=True,
                )
            elif return_prev_sample_mean:
                sde_latents_chunk, log_prob_chunk, prev_sample_mean_chunk, _ = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred.float(),
                    step_timestep_model,
                    step_latents_chunk.float(),
                    noise_level=noise_level,
                    sde_type="sde",
                )
            else:
                sde_latents_chunk, log_prob_chunk, _, _ = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred.float(),
                    step_timestep_model,
                    step_latents_chunk.float(),
                    noise_level=noise_level,
                    sde_type="sde",
                )
            if collect_matching_aux:
                noise_pred_chunk = noise_pred
            if actual_chunk_rows < latent_chunk_size:
                sde_latents_chunk = sde_latents_chunk[:actual_chunk_rows]
                log_prob_chunk = log_prob_chunk[:actual_chunk_rows]
                if return_prev_sample_mean:
                    prev_sample_mean_chunk = prev_sample_mean_chunk[:actual_chunk_rows]
                if collect_matching_aux:
                    noise_pred_chunk = noise_pred_chunk[:actual_chunk_rows]
                    noise_chunk = noise_chunk[:actual_chunk_rows]
            step_sde_latent_chunks.append(sde_latents_chunk)
            step_log_prob_chunks.append(log_prob_chunk)
            if return_prev_sample_mean:
                step_prev_latent_mean_chunks.append(prev_sample_mean_chunk)
            if collect_matching_aux:
                step_pred_chunks.append(noise_pred_chunk)
                step_noise_chunks.append(noise_chunk)
            step_branch_index_chunks.append(step_indices)

        sde_latents = torch.cat(step_sde_latent_chunks, dim=0)
        log_prob = torch.cat(step_log_prob_chunks, dim=0)
        if return_prev_sample_mean:
            prev_sample_mean = torch.cat(step_prev_latent_mean_chunks, dim=0)
        if collect_matching_aux:
            noise_pred = torch.cat(step_pred_chunks, dim=0)
            noise = torch.cat(step_noise_chunks, dim=0)
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
                fill_value=int(i),
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
        packed_next_timestep_chunks.append(
            timesteps[i + 1].expand(expected_step_rows).clone()
        )
        packed_latent_chunks.append(step_latents.index_select(0, step_branch_indices))
        packed_next_latent_chunks.append(sde_latents)
        packed_log_prob_chunks.append(log_prob)
        if return_prev_sample_mean:
            packed_prev_mean_chunks.append(prev_sample_mean)
        if collect_matching_aux:
            packed_noise_chunks.append(noise)
            packed_pred_chunks.append(noise_pred)

        if XLA_AVAILABLE:
            xm.mark_step()

    if len(packed_step_idx_chunks) == 0:
        raise ValueError(
            "No SDE transition rows were generated because exploration_k is zero for all transitions. "
            "Provide at least one transition with exploration_k > 0."
        )

    packed_row_step_idx = torch.cat(packed_step_idx_chunks, dim=0)
    packed_row_sample_idx = torch.cat(packed_sample_idx_chunks, dim=0)
    packed_row_exploration_k = torch.cat(packed_exploration_k_chunks, dim=0)
    packed_row_timesteps = torch.cat(packed_timestep_chunks, dim=0)
    packed_row_next_timesteps = torch.cat(packed_next_timestep_chunks, dim=0)
    packed_row_latents = torch.cat(packed_latent_chunks, dim=0)
    packed_row_next_latents = torch.cat(packed_next_latent_chunks, dim=0)
    packed_row_log_probs = torch.cat(packed_log_prob_chunks, dim=0)
    packed_row_prev_mean = (
        torch.cat(packed_prev_mean_chunks, dim=0) if return_prev_sample_mean else None
    )
    packed_row_noises = (
        torch.cat(packed_noise_chunks, dim=0) if collect_matching_aux else None
    )
    packed_row_preds = (
        torch.cat(packed_pred_chunks, dim=0) if collect_matching_aux else None
    )
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

    # Drop per-step buffers before pass 2b to lower peak memory.
    del packed_step_idx_chunks
    del packed_sample_idx_chunks
    del packed_exploration_k_chunks
    del packed_timestep_chunks
    del packed_next_timestep_chunks
    del packed_latent_chunks
    del packed_next_latent_chunks
    del packed_log_prob_chunks
    del packed_kl_step_idx_chunks
    del packed_kl_sample_idx_chunks
    del packed_kl_timestep_chunks
    del packed_kl_next_timestep_chunks
    del packed_kl_latent_chunks
    del packed_prev_mean_chunks
    del packed_noise_chunks
    del packed_pred_chunks
    del ode_step_states
    del all_latents

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
    if return_prev_sample_mean:
        packed_rollout["row_prev_sample_mean"] = packed_row_prev_mean
    if collect_matching_aux:
        packed_rollout["row_noises"] = packed_row_noises
        packed_rollout["row_noise_preds"] = packed_row_preds

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return packed_rollout

    return StableDiffusion3PipelineOutput(images=packed_images), packed_rollout
