import json
import os
from typing import Dict, List

import h5py
import torch
from torch.utils.data import Dataset


class PrecomputedEmbeddingDataset(Dataset):
    """Loads prompt embeddings and ids from an HDF5 file."""

    def __init__(self, hdf5_path: str, prompt_file_path: str):
        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found at: {hdf5_path}")
        if not os.path.exists(prompt_file_path):
            raise FileNotFoundError(f"Prompt file not found at: {prompt_file_path}")

        self.hdf5_path = hdf5_path
        self.prompts, self.metadatas = self._load_prompts_and_metadata(prompt_file_path)
        self.num_samples = len(self.prompts)

    @staticmethod
    def _load_prompts_and_metadata(prompt_file_path: str):
        prompts = []
        metadatas = []

        if prompt_file_path.endswith(".jsonl"):
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    prompts.append(item["prompt"])
                    metadatas.append(item)
        else:
            with open(prompt_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    prompts.append(line)
                    metadatas.append({})

        return prompts, metadatas

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with h5py.File(self.hdf5_path, "r") as hf:
            sample_group = hf[str(idx)]
            prompt_embeds = torch.from_numpy(sample_group["prompt_embeds"][:])
            pooled_prompt_embeds = torch.from_numpy(sample_group["pooled_prompt_embeds"][:])
            prompt_ids = torch.from_numpy(sample_group["prompt_ids"][:])

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompts": self.prompts[idx],
            "metadatas": self.metadatas[idx],
            "prompt_ids": prompt_ids,
            "indices": idx,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        return {
            "prompt_embeds": torch.stack([item["prompt_embeds"] for item in batch]),
            "pooled_prompt_embeds": torch.stack([item["pooled_prompt_embeds"] for item in batch]),
            "prompts": [item["prompts"] for item in batch],
            "metadatas": [item["metadatas"] for item in batch],
            "prompt_ids": torch.stack([item["prompt_ids"] for item in batch]),
            "indices": torch.tensor([item["indices"] for item in batch]),
        }
