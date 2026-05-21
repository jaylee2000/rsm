import torch

from typing import Any, Callable, Dict, Optional, Union
from accelerate.logging import get_logger
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.image_processor import VaeImageProcessor
from diffusion.pipeline_helpers import encode_prompt, prepare_latents, prepare_extra_step_kwargs
from tqdm.auto import tqdm

from diffusion.ddim_step import ddim_step_with_logprob

logger = get_logger(__name__, log_level="INFO")

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
    guidance_rescale: float = 0.0,
    unet_ref: Optional[torch.nn.Module] = None,
    disable_progress_bar: bool = False,
    const_weight: Optional[float] = None,
    algorithm: str = "uw-sigma",
    generator: Union[torch.Generator, list[torch.Generator], None] = None,
    eval: bool = False,
):
    # unwrap DDP if present, so we can read .config
    raw_unet = unet.module if hasattr(unet, "module") else unet
    raw_vae = vae.module if hasattr(vae, "module") else vae

    # 1. Default height and width to unet
    vae_scale_factor = 2 ** (len(raw_vae.config.block_out_channels) - 1) # 8
    height = raw_unet.config.sample_size * vae_scale_factor
    width = raw_unet.config.sample_size * vae_scale_factor

    # 2. Define call parameters
    batch_size = prompt_embeds.shape[0]

    device = unet.device
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
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

    # 6. Prepare extra step kwargs.
    # TODO: Logic should be moved out of the pipeline
    extra_step_kwargs = prepare_extra_step_kwargs(scheduler, generator, eta=eta)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order
    all_latents = [latents]
    all_log_probs = []
    all_noises = []
    all_noise_preds = []
    all_noise_pred_texts = []
    all_noise_pred_refs = []
    all_noise_pred_ref_texts = []
    progress_bar = tqdm(
        desc="DDIM Sampler",
        total=num_inference_steps,
        dynamic_ncols=True,
        position=1,
        leave=False,
        disable=disable_progress_bar,
    )
    with progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

            noise_pred_ref = None
            noise_pred_ref_text = None
            if unet_ref is not None and ('lambda' in algorithm or eval):
                with torch.no_grad():
                    noise_pred_ref = unet_ref(
                        latent_model_input, # tensor of [batch_size*2, 4, 64, 64]
                        t, # tensor of 1 value
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        return_dict=False,
                    )[0]
                if do_classifier_free_guidance:
                    noise_pred_ref_uncond, noise_pred_ref_text = noise_pred_ref.chunk(2)
                    noise_pred_ref = noise_pred_ref_uncond + guidance_scale * (
                        noise_pred_ref_text - noise_pred_ref_uncond
                    )
                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred_ref = rescale_noise_cfg(
                        noise_pred_ref, noise_pred_ref_text, guidance_rescale=guidance_rescale
                    )

            # compute the previous noisy sample x_t -> x_t-1
            latents, log_prob, noise = ddim_step_with_logprob(
                scheduler,
                noise_pred,
                noise_pred_ref,
                noise_pred_text,
                noise_pred_ref_text,
                None,
                t,
                latents,
                rg_scale,
                num_train_timesteps,
                num_inference_steps,
                const_weight=const_weight,
                algorithm=algorithm,
                **extra_step_kwargs # contains eta, generator
            )

            all_latents.append(latents)
            all_log_probs.append(log_prob)
            all_noises.append(noise)
            all_noise_preds.append(noise_pred)
            all_noise_pred_texts.append(noise_pred_text)
            all_noise_pred_refs.append(noise_pred_ref)
            all_noise_pred_ref_texts.append(noise_pred_ref_text)

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    if not output_type == "latent":
        image = vae.decode(
            latents / raw_vae.config.scaling_factor, return_dict=False
        )[0]
    else:
        image = latents
    has_nsfw_concept = None
    do_denormalize = [True] * image.shape[0]

    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
    image = image_processor.postprocess(
        image, output_type=output_type, do_denormalize=do_denormalize
    )

    return image, has_nsfw_concept, all_latents, all_log_probs, all_noises, \
        all_noise_pred_texts, all_noise_pred_refs, all_noise_pred_ref_texts, \
        all_noise_preds
