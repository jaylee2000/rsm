#!/usr/bin/env python3
"""Generate qualitative SD3 images for multiple TempFlow checkpoints.

This script mirrors the checkpoint loading and SD3 generation path used by
``eval_dreamsim_diversity.py`` but focuses on qualitative comparison:

1. Resolve a set of checkpoint epochs into ``checkpoint-<epoch>`` directories.
2. Load each LoRA checkpoint into the same SD3 base pipeline.
3. Generate images for one fixed prompt across multiple seeds.
4. Save images into ``--output-dir`` grouped by epoch.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel


def parse_early_seed(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args(argv)
    return args.seed


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def parse_args() -> argparse.Namespace:
    default_output_dir = Path(__file__).resolve().parent / "qualitative_results"
    default_checkpoint_root = Path(__file__).resolve().parent / "checkpoints" / "tempflow"

    parser = argparse.ArgumentParser(
        description=(
            "Generate qualitative images for a single prompt using multiple "
            "TempFlow-GRPO SD3 checkpoints."
        )
    )
    parser.add_argument(
        "--checkpoint-epochs",
        "--checkpoint_epochs",
        dest="checkpoint_epochs",
        type=int,
        nargs="+",
        required=True,
        help="Epoch numbers to load as checkpoint-<epoch> under --checkpoint-root.",
    )
    parser.add_argument(
        "--checkpoint-root",
        "--checkpoint_root",
        dest="checkpoint_root",
        type=str,
        default=str(default_checkpoint_root),
        help="Directory that contains checkpoint-<epoch> subdirectories.",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="stabilityai/stable-diffusion-3.5-medium",
        help="Base SD3 model for loading the LoRA checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Single prompt used for qualitative comparison across checkpoints.",
    )
    parser.add_argument(
        "--images-per-prompt",
        "--images_per_prompt",
        dest="images_per_prompt",
        type=int,
        default=8,
        help="Number of images to generate for the prompt at each checkpoint.",
    )
    parser.add_argument(
        "--image-batch-size",
        "--image_batch_size",
        dest="image_batch_size",
        type=int,
        default=2,
        help="Generation batch size per forward pass.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument(
        "--attention-slicing",
        type=str,
        default="auto",
        choices=("auto", "off"),
        help="SD3 attention slicing mode. Use 'off' to maximize throughput on high-VRAM GPUs.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=("fp16", "bf16", "fp32"),
        help="Compute dtype for SD3 generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="",
        help="Negative prompt text used for generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Generated seeds are seed + image_index and reused across epochs.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=str,
        default=str(default_output_dir),
        help="Directory where generated images and run metadata are saved.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Compatibility flag accepted by eval_qualitative.sh. Images are always saved.",
    )
    return parser.parse_args()


def resolve_lora_dir(checkpoint_path: Path) -> Path:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if (checkpoint_path / "adapter_config.json").is_file() and (
        checkpoint_path / "adapter_model.safetensors"
    ).is_file():
        return checkpoint_path

    lora_dir = checkpoint_path / "lora"
    if (lora_dir / "adapter_config.json").is_file() and (
        lora_dir / "adapter_model.safetensors"
    ).is_file():
        return lora_dir

    raise FileNotFoundError(
        "Could not find LoRA weights. Expected either:\n"
        f"  - {checkpoint_path}/adapter_config.json and adapter_model.safetensors, or\n"
        f"  - {checkpoint_path}/lora/adapter_config.json and adapter_model.safetensors"
    )


def dtype_from_string(dtype_name: str) -> torch.dtype:
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def load_generation_pipeline(
    pretrained_model: str,
    lora_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    attention_slicing: str,
) -> StableDiffusion3Pipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(pretrained_model, torch_dtype=dtype)
    if lora_dir is not None:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, str(lora_dir))
    pipe.transformer.eval()
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if attention_slicing == "auto":
        pipe.enable_attention_slicing("auto")
    return pipe


def _make_generator(seed: int, device: torch.device) -> torch.Generator:
    if device.type in {"cpu", "mps"}:
        return torch.Generator(device="cpu").manual_seed(seed)
    return torch.Generator(device=str(device)).manual_seed(seed)


def build_initial_latents(
    pipe: StableDiffusion3Pipeline,
    seeds: Sequence[int],
    height: int,
    width: int,
) -> Dict[int, torch.Tensor]:
    latent_height = height // pipe.vae_scale_factor
    latent_width = width // pipe.vae_scale_factor
    latent_channels = pipe.transformer.config.in_channels
    latents_by_seed: Dict[int, torch.Tensor] = {}

    for seed in seeds:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents = torch.randn(
            (1, latent_channels, latent_height, latent_width),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        latents_by_seed[seed] = latents

    return latents_by_seed


def generate_images_for_prompt(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    negative_prompt: str,
    epoch: int,
    seeds: Sequence[int],
    latents_by_seed: Dict[int, torch.Tensor],
    image_batch_size: int,
    device: torch.device,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: List[Path] = []
    generated = 0
    batch_id = 0

    while generated < len(seeds):
        batch_seeds = list(seeds[generated : generated + image_batch_size])
        prompts = [prompt] * len(batch_seeds)
        batch_latents = torch.cat(
            [latents_by_seed[seed].clone() for seed in batch_seeds], dim=0
        ).to(device=device, dtype=pipe.dtype)
        kwargs: Dict[str, Any] = {
            "prompt": prompts,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "height": height,
            "width": width,
            "latents": batch_latents,
        }
        if negative_prompt != "":
            kwargs["negative_prompt"] = [negative_prompt] * len(batch_seeds)

        with torch.inference_mode():
            outputs = pipe(**kwargs)
        if not hasattr(outputs, "images"):
            raise RuntimeError(
                f"Unexpected SD3 output in batch {batch_id}: missing `images` field."
            )

        for seed, image in zip(batch_seeds, outputs.images):
            image_path = output_dir / f"{prompt}_seed_{seed:06d}_epoch_{epoch}.png"
            image.save(image_path)
            image_paths.append(image_path)

        del outputs
        generated += len(batch_seeds)
        batch_id += 1

    return image_paths


def build_checkpoint_paths(checkpoint_root: Path, epochs: Sequence[int]) -> List[Path]:
    checkpoint_paths: List[Path] = []
    for epoch in epochs:
        checkpoint_path = checkpoint_root / f"checkpoint-{epoch}"
        if not checkpoint_path.exists():
            checkpoint_paths.append(None)
            #raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found: {checkpoint_path}")
        checkpoint_paths.append(checkpoint_path)
    return checkpoint_paths


def main() -> None:
    set_global_seed(parse_early_seed(sys.argv[1:]))
    args = parse_args()

    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = dtype_from_string(args.dtype)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        print("[warn] CPU device does not support fast half precision well; switching to fp32.")
        dtype = torch.float32

    checkpoint_paths = build_checkpoint_paths(checkpoint_root, args.checkpoint_epochs)
    seeds = [args.seed + i for i in range(args.images_per_prompt)]

    metadata = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_epochs": list(args.checkpoint_epochs),
        "pretrained_model": args.pretrained_model,
        "images_per_prompt": args.images_per_prompt,
        "image_batch_size": args.image_batch_size,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "dtype": args.dtype,
        "device": str(device),
        "seeds": seeds,
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with (output_dir / "prompt.txt").open("w", encoding="utf-8") as f:
        f.write(args.prompt.strip() + "\n")

    for epoch, checkpoint_path in zip(args.checkpoint_epochs, checkpoint_paths):
        print(f"Loading epoch {epoch} from {checkpoint_path}...")
        if epoch == 0: lora_dir = None
        else:   lora_dir = resolve_lora_dir(checkpoint_path)
        pipe = load_generation_pipeline(
            pretrained_model=args.pretrained_model,
            lora_dir=lora_dir,
            device=device,
            dtype=dtype,
            attention_slicing=args.attention_slicing,
        )

        latents_by_seed = build_initial_latents(
            pipe=pipe,
            seeds=seeds,
            height=args.height,
            width=args.width,
        )

        print(f"Generating {len(seeds)} images for epoch {epoch}...")
        generate_images_for_prompt(
            pipe=pipe,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            epoch=epoch,
            seeds=seeds,
            latents_by_seed=latents_by_seed,
            image_batch_size=args.image_batch_size,
            device=device,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            output_dir=output_dir,
        )

        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Saved qualitative results to {output_dir}")


if __name__ == "__main__":
    main()
