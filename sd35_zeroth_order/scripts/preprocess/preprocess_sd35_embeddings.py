import argparse
import json
import os

import h5py
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt


logger = get_logger(__name__)


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


class PromptDataset(Dataset):
    def __init__(self, file_path):
        self.prompts = []
        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.endswith(".jsonl"):
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.prompts.append(json.loads(line)["prompt"])
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.prompts.append(line)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "index": idx}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        indices = [example["index"] for example in examples]
        return {"prompts": prompts, "indices": indices}


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-compute SD3.5 prompt embeddings.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_prompt_file", type=str, required=True)
    parser.add_argument("--test_prompt_file", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    torch_dtype = torch.float32
    if args.mixed_precision == "fp16":
        torch_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16

    logger.info("Loading Stable Diffusion 3 pipeline...", main_process_only=True)
    pipeline = StableDiffusion3Pipeline.from_pretrained(args.pretrained_model, torch_dtype=torch_dtype)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    for encoder in text_encoders:
        encoder.to(accelerator.device)
        encoder.eval()

    split_to_file = {
        "train": args.train_prompt_file,
        "test": args.test_prompt_file,
    }

    for split, prompt_file in split_to_file.items():
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        dataset = PromptDataset(prompt_file)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=PromptDataset.collate_fn,
            shuffle=False,
        )
        dataloader = accelerator.prepare(dataloader)

        temp_h5_path = os.path.join(args.output_dir, f"{split}_rank_{accelerator.process_index}.h5")
        with h5py.File(temp_h5_path, "w") as hf:
            for batch in tqdm(
                dataloader,
                desc=f"Encoding {split} (rank {accelerator.process_index})",
                disable=not accelerator.is_local_main_process,
            ):
                with torch.no_grad():
                    prompts = batch["prompts"]
                    prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                        prompts,
                        text_encoders,
                        tokenizers,
                        max_sequence_length=128,
                        device=accelerator.device,
                    )
                    prompt_ids = tokenizers[0](
                        prompts,
                        padding="max_length",
                        max_length=256,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids

                for i in range(len(prompts)):
                    index = int(batch["indices"][i])
                    group = hf.create_group(str(index))
                    group.create_dataset("prompt_embeds", data=prompt_embeds[i].cpu().numpy())
                    group.create_dataset("pooled_prompt_embeds", data=pooled_prompt_embeds[i].cpu().numpy())
                    group.create_dataset("prompt_ids", data=prompt_ids[i].cpu().numpy())

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Consolidating temporary files...", main_process_only=True)
        for split in split_to_file.keys():
            final_h5_path = os.path.join(args.output_dir, f"{split}_embeddings.h5")
            with h5py.File(final_h5_path, "w") as final_hf:
                with torch.no_grad():
                    neg_prompt_embeds, neg_pooled_embeds = compute_text_embeddings(
                        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
                    )
                    neg_prompt_ids = tokenizers[0](
                        [""],
                        padding="max_length",
                        max_length=256,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids
                neg_group = final_hf.create_group("negative")
                neg_group.create_dataset("prompt_embeds", data=neg_prompt_embeds.cpu().numpy())
                neg_group.create_dataset("pooled_prompt_embeds", data=neg_pooled_embeds.cpu().numpy())
                neg_group.create_dataset("prompt_ids", data=neg_prompt_ids.cpu().numpy())

                for i in range(accelerator.num_processes):
                    temp_h5_path = os.path.join(args.output_dir, f"{split}_rank_{i}.h5")
                    if os.path.exists(temp_h5_path):
                        with h5py.File(temp_h5_path, "r") as temp_hf:
                            for key in temp_hf.keys():
                                h5py.h5o.copy(temp_hf.id, key.encode("utf-8"), final_hf.id, key.encode("utf-8"))
                        os.remove(temp_h5_path)

        logger.info(f"Done. Saved embeddings to: {args.output_dir}", main_process_only=True)


if __name__ == "__main__":
    main()
