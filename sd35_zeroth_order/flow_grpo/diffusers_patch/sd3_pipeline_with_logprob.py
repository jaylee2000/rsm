# Copied from diffusers SD3 pipeline and patched to expose rollout/log-prob internals.
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import XLA_AVAILABLE

from .sd3_pipeline_with_logprob_shared import (
    build_sd3_pipeline_output,
    decode_sd3_latents,
    prepare_ip_adapter_embeddings,
    prepare_latents_and_timesteps,
    setup_sd3_pipeline_context,
)
from .sd3_sde_with_logprob import sde_step_with_logprob

if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


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
):
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

    prepare_ip_adapter_embeddings(
        self,
        ip_adapter_image=ip_adapter_image,
        ip_adapter_image_embeds=ip_adapter_image_embeds,
        device=device,
        batch_size=batch_size,
        num_images_per_prompt=num_images_per_prompt,
    )

    all_latents = [latents]
    all_log_probs = []
    all_prev_latents_mean = []
    all_noises = []
    all_preds = []

    sde_type = "deterministic" if determistic else "sde"

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            latent_model_input = (
                torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
            )
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred.to(prompt_embeds.dtype)

            if self.do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                should_skip_layers = (
                    i > num_inference_steps * skip_layer_guidance_start
                    and i < num_inference_steps * skip_layer_guidance_stop
                )
                if skip_guidance_layers is not None and should_skip_layers:
                    skip_timestep = t.expand(latents.shape[0])
                    noise_pred_skip_layers = self.transformer(
                        hidden_states=latents,
                        timestep=skip_timestep,
                        encoder_hidden_states=original_prompt_embeds,
                        pooled_projections=original_pooled_prompt_embeds,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                        skip_layers=skip_guidance_layers,
                    )[0]
                    noise_pred = noise_pred + (
                        noise_pred_text - noise_pred_skip_layers
                    ) * self._skip_layer_guidance_scale

            latents_dtype = latents.dtype

            step_timestep = t.expand(latents.shape[0])
            if collect_matching_aux:
                latents, log_prob, prev_latents_mean, _, noise = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred.float(),
                    step_timestep,
                    latents.float(),
                    noise_level=noise_level,
                    sde_type=sde_type,
                    return_noise=True,
                )
                all_noises.append(noise)
                all_preds.append(noise_pred)
            else:
                latents, log_prob, prev_latents_mean, _ = sde_step_with_logprob(
                    self.scheduler,
                    noise_pred.float(),
                    step_timestep,
                    latents.float(),
                    noise_level=noise_level,
                    sde_type=sde_type,
                )

            all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_prev_latents_mean.append(prev_latents_mean)
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for key in callback_on_step_end_tensor_inputs:
                    callback_kwargs[key] = locals()[key]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop(
                    "negative_prompt_embeds", negative_prompt_embeds
                )
                negative_pooled_prompt_embeds = callback_outputs.pop(
                    "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                )

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    if output_type == "latent":
        image = latents
    else:
        image = decode_sd3_latents(self, latents, output_type=output_type)

    self.maybe_free_model_hooks()

    image_output = build_sd3_pipeline_output(image, return_dict=return_dict)

    if return_prev_sample_mean:
        if collect_matching_aux:
            return (
                image_output,
                all_latents,
                all_log_probs,
                all_noises,
                all_preds,
                all_prev_latents_mean,
            )
        return image_output, all_latents, all_log_probs, all_prev_latents_mean

    if collect_matching_aux:
        return image_output, all_latents, all_log_probs, all_noises, all_preds

    return image_output, all_latents, all_log_probs
