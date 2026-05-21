# SD1.5 First-Order Experiments

This component contains the Stable Diffusion 1.5 first-order RSM experiments used for Figure 5(c, d). It is adapted from the Nabla-GFlowNet codebase.

## Setup

From this directory:

```bash
conda create -n rsm-sd15-fo python=3.12
conda activate rsm-sd15-fo
conda install nvidia::cuda-toolkit
pip install -r requirements.txt
pip install trl
pip install xformers
```

For HPS-v2.1 rewards, install the HPSv2 package locally and place the required checkpoints under `./hps_ckpt/`.

```bash
git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
pip install -e .
cd ..
```

Weights & Biases logging is expected by the current training code. Set `WANDB_API_KEY` before running.

## Paper Commands

Figure 5(c, d), Ours:

```bash
torchrun --nproc_per_node=4 --master_port=29501 simple_res-nabladb.py --config config/simple_res-nabladb_sd_hps_usez0_basesnr.yaml
```

Figure 5(c, d), Pruned Baseline:

```bash
torchrun --nproc_per_node=4 --master_port=29501 simple_res-nabladb.py --config config/simple_res-nabladb_sd_hps.yaml
```

## Notes

- SD1.5 experiments were tested on CUDA 12.x with RTX 4090 GPUs.
- Review cache, checkpoint, and output paths before launching long runs.
- This component intentionally keeps only the Figure 5(c, d) paper entrypoint and configs.
