from typing import Any, Dict, List, Optional, Union

import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_3 import StableDiffusion3PipelineOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    calculate_shift,
    retrieve_timesteps,
)


def setup_sd3_pipeline_context(
    self,
    *,
    prompt: Union[str, List[str], None],
    prompt_2: Optional[Union[str, List[str]]],
    prompt_3: Optional[Union[str, List[str]]],
    height: Optional[int],
    width: Optional[int],
    negative_prompt: Optional[Union[str, List[str]]],
    negative_prompt_2: Optional[Union[str, List[str]]],
    negative_prompt_3: Optional[Union[str, List[str]]],
    prompt_embeds: Optional[torch.FloatTensor],
    negative_prompt_embeds: Optional[torch.FloatTensor],
    pooled_prompt_embeds: Optional[torch.FloatTensor],
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor],
    callback_on_step_end_tensor_inputs: List[str],
    max_sequence_length: int,
    guidance_scale: float,
    skip_layer_guidance_scale: float,
    clip_skip: Optional[int],
    joint_attention_kwargs: Optional[Dict[str, Any]],
    num_images_per_prompt: int,
    skip_guidance_layers: Optional[List[int]],
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    self.check_inputs(
        prompt,
        prompt_2,
        prompt_3,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._skip_layer_guidance_scale = skip_layer_guidance_scale
    self._clip_skip = clip_skip
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None)
        if self.joint_attention_kwargs is not None
        else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_3=prompt_3,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        negative_prompt_3=negative_prompt_3,
        do_classifier_free_guidance=self.do_classifier_free_guidance,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        device=device,
        clip_skip=self.clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    original_prompt_embeds = None
    original_pooled_prompt_embeds = None
    if self.do_classifier_free_guidance:
        if skip_guidance_layers is not None:
            original_prompt_embeds = prompt_embeds
            original_pooled_prompt_embeds = pooled_prompt_embeds
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

    return {
        "height": height,
        "width": width,
        "batch_size": batch_size,
        "device": device,
        "prompt_embeds": prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        "original_prompt_embeds": original_prompt_embeds,
        "original_pooled_prompt_embeds": original_pooled_prompt_embeds,
    }


def prepare_latents_and_timesteps(
    self,
    *,
    batch_size: int,
    num_images_per_prompt: int,
    height: int,
    width: int,
    prompt_embeds_dtype: torch.dtype,
    device: torch.device,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]],
    latents: Optional[torch.FloatTensor],
    num_inference_steps: int,
    sigmas: Optional[List[float]],
    mu: Optional[float],
):
    num_channels_latents = self.transformer.config.in_channels
    latents = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds_dtype,
        device,
        generator,
        latents,
    )

    scheduler_kwargs = {}
    if self.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        _, _, latent_height, latent_width = latents.shape
        image_seq_len = (latent_height // self.transformer.config.patch_size) * (
            latent_width // self.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.16),
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu

    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        **scheduler_kwargs,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    return latents, timesteps, num_inference_steps, num_warmup_steps


def prepare_ip_adapter_embeddings(
    self,
    *,
    ip_adapter_image: Optional[PipelineImageInput],
    ip_adapter_image_embeds: Optional[torch.Tensor],
    device: torch.device,
    batch_size: int,
    num_images_per_prompt: int,
):
    if (ip_adapter_image is not None and self.is_ip_adapter_active) or ip_adapter_image_embeds is not None:
        ip_adapter_image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            self.do_classifier_free_guidance,
        )

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {"ip_adapter_image_embeds": ip_adapter_image_embeds}
        else:
            self._joint_attention_kwargs.update(ip_adapter_image_embeds=ip_adapter_image_embeds)


def decode_sd3_latents(self, latents: torch.FloatTensor, output_type: str):
    latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
    latents = latents.to(dtype=self.vae.dtype)
    image = self.vae.decode(latents, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type=output_type)
    return image


def build_sd3_pipeline_output(image, return_dict: bool):
    if not return_dict:
        return image
    return StableDiffusion3PipelineOutput(images=image)
