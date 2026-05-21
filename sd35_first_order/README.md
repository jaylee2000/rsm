# SD3.5-M First-Order Experiments

This component contains the Stable Diffusion 3.5 Medium first-order RSM experiments used for Figure 5(a, b). It is adapted from the VGG-Flow code path, but this directory is part of the Reward Score Matching release.

## Setup

From this directory:

```bash
conda create -n rsm-sd35-fo python=3.10
conda activate rsm-sd35-fo
pip install -r requirements.txt
```

Training loads Stable Diffusion 3.5 Medium weights through Diffusers, so make sure your Hugging Face environment has access to the required model weights.

Before training, review `config/default_config.py` and update local values as needed:

- `config.logging.wandb_key`: replace the placeholder with your W&B key if using Weights & Biases.
- `config.logging.wandb_dir`: local W&B output directory.
- `config.saving.output_dir`: checkpoint/output directory.

## Paper Commands

Figure 5(a, b), Ours:

```bash
torchrun --standalone --nproc_per_node=4 train_vggflow.py \
    --config=config/hpsv2_geneval_ours.py \
    --exp_name=OURS
```

Figure 5(a, b), Pruned Baseline:

```bash
torchrun --standalone --nproc_per_node=4 train_vggflow.py \
    --config=config/hpsv2_geneval.py \
    --exp_name=PRUNED_BASELINE
```

## Configuration

Config presets live under `config/`. The paper commands above use:

- `config/hpsv2_geneval_ours.py`
- `config/hpsv2_geneval.py`

You can override config values from the command line in the same style as the paper commands.

## Notes

- SD3.5-M runs were tested on CUDA 12.8 with 4 x H200 GPUs.
- The largest SD3.5-M setup can nearly saturate 140GB H200 VRAM.
- Lower-VRAM GPUs can be used by reducing per-device batch sizes and increasing gradient accumulation.
- Review output, cache, and checkpoint paths before launching long runs.
