<h2 align="center">
    Reward Score Matching
</h2>

<div align="center">
  <a href='https://arxiv.org/abs/2604.17415'><img src='https://img.shields.io/badge/ArXiv-red?logo=arxiv'></a>  &nbsp;
<a href="https://jeongsol-kim.github.io/rsm_projectpage/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/💻_Project-RSM-blue.svg" height="20" /></a>
</div>


**TL;DR:** We unify reward-based fine-tuning algorithms for diffusion and flow generative models. This allows us to distinguish the fundamental design choices from others.

> [Jeongjae Lee*](https://jaylee2000.github.io/), [Jinho Chang*](https://chang-jinho.github.io/), [Jeongsol Kim†](https://www.jeongsol.dev/), [Jong Chul Ye†](https://bispl.weebly.com/professor.html).
>
> KAIST



## 🔥 News

- [2026.06.02] RSM has been accepted as an **Oral presentation** at the SPIGM Workshop, ICML 2026!
- [2026.05.21] Code released on Github!
- [2026.05.07] Preprint updated on arXiv!
- [2026.04.19] Preprint released on arXiv!


## Repository Layout

| Directory | Purpose | Model family |
| --- | --- | --- |
| `sd35_zeroth_order/` | Zeroth-order experiments against [TempFlow-GRPO](https://github.com/Shredded-Pork/TempFlow-GRPO) baseline for Figure 4(a). Most users should start here. | Stable Diffusion 3.5 Medium |
| `sd15_zeroth_order/` | Zeroth-order experiments against [PCPO](https://github.com/jaylee2000/pcpo/tree/main/dancegrpo) baseline for Figure 4(b, c). | Stable Diffusion 1.5 |
| `sd35_first_order/` | First-order experiments against [VGG-Flow](https://github.com/lzzcd001/vggflow) baseline for Figure 5(a, b). | Stable Diffusion 3.5 Medium |
| `sd15_first_order/` | First-order experiments against [Nabla-GFlowNet](https://github.com/lzzcd001/nabla-gfn) baseline for Figure 5(c, d). | Stable Diffusion 1.5 |

Each is an independent component, with its own setup notes, configs, and launch scripts.

## Setup Notes

Install dependencies inside the component you want to run. Model access, reward-model setup, and component-specific packages are described in each subdirectory README and config files.

Weights & Biases logging is expected by the current code. Set `WANDB_API_KEY` or replace the placeholder fields in component configs before running.

Review cache, checkpoint, and output paths before launching experiments. Some configs/scripts have hardcoded paths and may need local path edits.

## Hardware Notes

SD3.5-M experiments were tested on CUDA 12.8 with 4 x H200 GPUs. These runs can nearly saturate 140GB of H200 VRAM when hosting the GenEval server simultaneously. They can be adapted to lower-VRAM GPUs (as low as 1 x 24GB GPU), by increasing gradient accumulation. BatchSampler is flexible; you can train with the same *effective* batch size.

SD1.5 experiments were tested on CUDA 12.x with RTX 4090 GPUs (24GB VRAM).

Package versions may need adjustment for different CUDA versions, CUDA 13.x, PyTorch wheels, xformers builds, or GPU microarchitectures.

## Reproducing Paper Runs

### Figure 2

See subdirectory ```toy_experiments``` to reproduce the analyses on different value gradient estimators.

### Figure 4(a), Ours

```bash
cd sd35_zeroth_order
bash scripts/single_node/run_sd3.sh --profile lowsnr2 --sampler branch --reward geneval --loss matching --reweight fairclip2 --num-processes 4
```

### Figure 4(b), Ours

```bash
cd sd15_zeroth_order
accelerate launch train_grpo_pr.py --config configs/train/base_grpo_pr_uwsigma_lora10.yaml
```

### Figure 4(b, c), Baseline

```bash
cd sd15_zeroth_order
accelerate launch train_grpo.py --config configs/train/hpsv2_1_lora_grpo.yaml
```

### Figure 5(a, b), Ours

```bash
cd sd35_first_order
torchrun --standalone --nproc_per_node=4 train_vggflow.py \
    --config=config/hpsv2_geneval_ours.py \
    --exp_name=OURS
```

### Figure 5(a, b), Pruned Baseline

```bash
cd sd35_first_order
torchrun --standalone --nproc_per_node=4 train_vggflow.py \
    --config=config/hpsv2_geneval.py \
    --exp_name=PRUNED_BASELINE
```

### Figure 5(c, d), Ours

```bash
cd sd15_first_order
torchrun --nproc_per_node=4 --master_port=29501 simple_res-nabladb.py --config config/simple_res-nabladb_sd_hps_usez0_basesnr.yaml
```

### Figure 5(c, d), Pruned Baseline

```bash
cd sd15_first_order
torchrun --nproc_per_node=4 --master_port=29501 simple_res-nabladb.py --config config/simple_res-nabladb_sd_hps.yaml
```

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{lee2026rewardscorematchingunifying,
      title={Reward Score Matching: Unifying Reward-based Fine-tuning for Flow and Diffusion Models}, 
      author={Jeongjae Lee and Jinho Chang and Jeongsol Kim and Jong Chul Ye},
      year={2026},
      eprint={2604.17415},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.17415}, 
}
```

## Acknowledgements

This repo is based on [ddpo-pytorch](https://github.com/kvablack/ddpo-pytorch/tree/main), [flow_grpo](https://github.com/yifan123/flow_grpo), [pcpo](https://github.com/jaylee2000/pcpo/), [nabla-gfn](https://github.com/lzzcd001/nabla-gfn), [vggflow](https://github.com/lzzcd001/vggflow), [TempFlow-GRPO](https://github.com/Shredded-Pork/TempFlow-GRPO).

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
