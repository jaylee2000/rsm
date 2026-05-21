# Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py

from importlib import resources
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

ASSETS_PATH = resources.files("utils.assets")


def _preprocess_like_clip(
    images: torch.Tensor,
    *,
    dtype: torch.dtype,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    # Expected input: float tensor in [0,1], shape [B,3,H,W]
    images = images.to(dtype=dtype)

    # Match CLIPProcessor: Resize shortest edge to 224 (keep aspect ratio), then center crop 224
    _, _, h, w = images.shape
    short = min(h, w)
    if short != 224:
        scale = 224.0 / float(short)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        images = F.interpolate(
            images,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
        )

    _, _, h, w = images.shape
    top = max((h - 224) // 2, 0)
    left = max((w - 224) // 2, 0)
    images = images[:, :, top : top + 224, left : left + 224]

    images = (images - mean.to(dtype=images.dtype)) / std.to(dtype=images.dtype)
    return images


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
        self.mlp = MLP()
        state_dict = torch.load(
            ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth")
        )
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

        self.register_buffer(
            "openai_clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "openai_clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)

        # Accept uint8 [0,255] or float [0,1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.to(torch.float32)
            if images.max() > 1.0:
                images = images / 255.0

        images = images.to(device)
        pixel_values = _preprocess_like_clip(
            images,
            dtype=self.dtype,
            mean=self.openai_clip_mean,
            std=self.openai_clip_std,
        )

        embed = self.clip.get_image_features(pixel_values=pixel_values)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


class MLPDiff(nn.Module):
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

    def forward(self, embed):
        return self.layers(embed)


class AestheticScorerDiff(torch.nn.Module):
    def __init__(self, dtype, distributed=True):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.mlp = MLPDiff()
        state_dict = torch.load(
            ASSETS_PATH.joinpath("sac+logos+ava1-l14-linearMSE.pth")
        )
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

        self.register_buffer(
            "openai_clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "openai_clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    # @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device

        if not isinstance(images, torch.Tensor):
            images = torch.as_tensor(images)

        # Accept uint8 [0,255] or float [0,1]
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.to(torch.float32)
            if images.max() > 1.0:
                images = images / 255.0

        images = images.to(device)
        pixel_values = _preprocess_like_clip(
            images,
            dtype=self.dtype,
            mean=self.openai_clip_mean,
            std=self.openai_clip_std,
        )

        embed = self.clip.get_image_features(pixel_values=pixel_values)
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)
