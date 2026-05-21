# SD1.5 Zeroth-Order Experiments

This component contains the Stable Diffusion 1.5 zeroth-order RSM experiments used for Figure 4(b, c). It is adapted from the PCPO codebase.

## Setup

From this directory:

```bash
conda create -n rsm-sd15-zo python=3.12
conda activate rsm-sd15-zo
conda install nvidia::cuda-toolkit
pip install -r requirements.txt
pip install trl
```

For HPS-v2.1 rewards, install the HPSv2 package locally and place the required checkpoints under `./hps_ckpt/`:

```bash
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
pip install -e .
cd ..
```

Download:

- `HPS_v2.1_compressed.pt` from `https://huggingface.co/xswu/HPSv2/tree/main`
- `open_clip_pytorch_model.bin` from `https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main`

Weights & Biases logging is expected by the current training code. Set `WANDB_API_KEY` before running.

## Paper Commands

Figure 4(b), Ours:

```bash
accelerate launch train_grpo_pr.py --config configs/train/base_grpo_pr_uwsigma_lora10.yaml
```

Figure 4(b, c), Baseline:

```bash
accelerate launch train_grpo.py --config configs/train/hpsv2_1_lora_grpo.yaml
```

## Implementation Notes

`train_grpo_pr.py` expects branching keys in the config, including `sampling_mode=ddim_branching`, `group_strategy`, `exploration_k`, `collection_batch_size`, `latent_chunk_size`, and `updates_per_epoch`.

## Troubleshooting

Package errors involving `wandb` or `protobuf` can often be fixed with:

```bash
pip uninstall -y wandb protobuf
pip install "protobuf>=4.25.3,<7" "wandb==0.25.1"
pip check
```

If HPS/OpenCLIP checkpoints are not found, review local cache paths and checkpoint placement before running:

```bash
mkdir -p ../cache/huggingface/hub
cp -n ./hps_ckpt/open_clip_pytorch_model.bin ../cache/huggingface/hub/
cp -n ./hps_ckpt/HPS_v2.1_compressed.pt ../cache/huggingface/hub/
```
