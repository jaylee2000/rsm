from PIL import Image
import io
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms
import logging
import re

class RetryLogFilter(logging.Filter):
    _pattern = re.compile(r"Retrying \(Retry\(total=(\d+),")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        m = self._pattern.search(msg)
        if m:
            total_left = int(m.group(1))
            return (total_left % 10) == 0
        return True

urllib3_logger = logging.getLogger("urllib3.connectionpool")
urllib3_logger.addFilter(RetryLogFilter())

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew, meta

    return _fn


from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
def hps_v2():
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        '../cache/huggingface/hub/open_clip_pytorch_model.bin',
        precision='amp',
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )
    cp = "../cache/huggingface/hub/HPS_v2.1_compressed.pt"
    checkpoint = torch.load(cp, map_location=f'cuda')
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer('ViT-H-14')

    hps_scorer = model.cuda().eval()

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        # Tensor [B,C,H,W] in [0,1] or [0,255]
        if isinstance(images, torch.Tensor):
            imgs = images.float()
            if imgs.max() > 1.0:
                imgs = imgs / 255.0
            # PIL for open_clip preprocess
            pils = [Image.fromarray(
                (img.cpu().numpy().transpose(1,2,0) * 255).round().astype('uint8')
            ) for img in imgs]
        else: # assume numpy NHWC uint8
            pils = [Image.fromarray(img) for img in images]

        device = next(hps_scorer.parameters()).device
        image_batch = torch.stack(
            [preprocess_val(p.convert("RGB")) for p in pils]
        ).to(device, non_blocking=True)
        text_batch = tokenizer(prompts).to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = hps_scorer(image_batch, text_batch)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            scores = torch.diagonal(logits_per_image) # [B]
        return scores.detach().cpu().numpy(), {}
    return _fn


def aesthetic_score():
    from utils.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clipScore():
    model_name = "openai/clip-vit-base-patch16"
    model = CLIPModel.from_pretrained(model_name).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    to_pil = transforms.ToPILImage()
    model_device = None

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        nonlocal model_device
        if not isinstance(images, torch.Tensor):
            arr = torch.from_numpy(images)  # assume numpy [B,H,W,3] uint8
            imgs = arr.permute(0,3,1,2).float() / 255.0
        else:
            imgs = images.float()
            if imgs.max() > 1.0:
                imgs = imgs / 255.0

        device = imgs.device
        if model_device != device:
            model.to(device)
            model_device = device
        pil_images = [to_pil(img.cpu()) for img in imgs]
        image_inputs = processor(
            images=pil_images,
            return_tensors="pt",
        ).to(device)
        text_inputs = processor(
            text=prompts,
            return_tensors="pt",
        ).to(device)

        image_embeds = model.get_image_features(**image_inputs)
        text_embeds = model.get_text_features(**text_inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        sims = (image_embeds * text_embeds).sum(dim=-1) # [B]

        return sims.cpu().numpy(), {}

    return _fn


def pickScore():
    from transformers import AutoProcessor, AutoModel
    from PIL import Image
    import torch
    import numpy as np

    # names from the PickScore README
    processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_name     = "yuvalkirstain/PickScore_v1"

    processor = AutoProcessor.from_pretrained(processor_name)
    model     = AutoModel.from_pretrained(model_name).eval()

    model_device = None

    @torch.no_grad()
    def _fn(images, prompts, metadata):
        nonlocal model_device
        # -- convert to list of PIL images
        if isinstance(images, torch.Tensor):
            imgs = images.detach().cpu().float().clamp(0, 1).mul(255).round().to(torch.uint8)
            # NCHW -> NHWC
            imgs = imgs.permute(0, 2, 3, 1).numpy()
            pil_images = [Image.fromarray(img) for img in imgs]
        else:
            # assume list/array of NHWC floats in [0,1]
            pil_images = [Image.fromarray((img * 255).round().astype("uint8")) for img in images]

        device = imgs.device

        if model_device != device:
            model.to(device)
            model_device = device

        # preprocess
        image_inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        text_inputs  = processor(text=prompts, return_tensors="pt", padding=True)

        # embed & normalize
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

        text_embs  = model.get_text_features(**text_inputs)
        text_embs  = text_embs / text_embs.norm(dim=-1, keepdim=True)

        # scaled dot‐product
        logit_scale = model.logit_scale.exp()
        # elementwise dot then scale → (batch,)
        scores = (text_embs * image_embs).sum(dim=-1) * logit_scale.squeeze()

        return scores.cpu().numpy(), {}

    return _fn


def qwen_bertscore():
    """
    Submits images to Qwen and computes a reward by comparing the responses to the prompts using BERTScore.
    See qwen-server repo for server-side code.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 16
    url = "http://127.0.0.1:8085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del metadata
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        all_info = {
            "precision": [],
            "f1": [],
            "outputs": [],
        }
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "queries": [["Answer concisely: what is going on in this image?"]]
                * len(image_batch),
                "answers": [
                    [f"The image contains {prompt}"] for prompt in prompt_batch
                ],
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)

            response_data = pickle.loads(response.content)

            # use the recall score as the reward
            scores = np.array(response_data["recall"])
            scores = scores.reshape(-1).tolist()
            all_scores += scores

            # save the precision and f1 scores for analysis
            precisions = np.array(response_data["precision"])
            precisions = precisions.reshape(-1).tolist()
            all_info["precision"] += precisions

            f1s = np.array(response_data["f1"])
            f1s = f1s.reshape(-1).tolist()
            all_info["f1"] += f1s

            outputs = np.array(response_data["outputs"])
            outputs = outputs.reshape(-1).tolist()
            all_info["outputs"] += outputs

        return np.array(all_scores), {k: np.array(v) for k, v in all_info.items()}

    return _fn
