# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

from importlib import resources
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

ASSETS_PATH = resources.files("utils.assets")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLP()
        state_dict = torch.load(
            ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth")
        )
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @staticmethod
    def _extract_image_embed(embed_output):
        # transformers>=4.57 may return a modeling output instead of a plain tensor.
        if torch.is_tensor(embed_output):
            return embed_output

        pooler_output = getattr(embed_output, "pooler_output", None)
        if torch.is_tensor(pooler_output):
            return pooler_output

        image_embeds = getattr(embed_output, "image_embeds", None)
        if torch.is_tensor(image_embeds):
            return image_embeds

        if isinstance(embed_output, (tuple, list)):
            tensor_values = [value for value in embed_output if torch.is_tensor(value)]
            if len(tensor_values) > 0:
                # Prefer pooled embeddings [B, D] over token grids [B, T, D].
                for value in tensor_values:
                    if value.ndim == 2:
                        return value
                return tensor_values[0]

        raise TypeError(
            "CLIP image feature output is not a tensor-compatible type: "
            f"{type(embed_output).__name__}"
        )

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            k: (
                v.to(dtype=self.dtype, device=device)
                if torch.is_floating_point(v)
                else v.to(device=device)
            )
            for k, v in inputs.items()
        }
        embed = self._extract_image_embed(self.clip.get_image_features(**inputs))
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True).clamp_min(1e-12)
        return self.mlp(embed).squeeze(1)
