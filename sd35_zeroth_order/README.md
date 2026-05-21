# SD3.5-M Zeroth-Order Experiments

This component contains the Stable Diffusion 3.5 Medium zeroth-order RSM experiments used for Figure 4(a) of the paper. It is adapted from the TempFlow-GRPO codebase.

Most users should start here.

## Setup

From this directory, create an environment and install the local package:

```bash
conda create -n rsm-sd35-zo python=3.10.16
conda activate rsm-sd35-zo
pip install -e .
pip install h5py
```

Training loads `stabilityai/stable-diffusion-3.5-medium`, so make sure your Hugging Face environment has access to the required model weights.

Weights & Biases logging is expected by the current training code. Set `WANDB_API_KEY` or update the relevant config before running.

## Reward Setup

Each reward model may require its own dependencies. For GenEval and DeQA-style reward servers, follow the [reward-server](https://github.com/yifan123/reward-server) setup used by the [Flow-GRPO](https://github.com/yifan123/flow_grpo) ecosystem. UnifiedReward can be served with `sglang`:

```bash
conda create -n sglang python=3.10.16
conda activate sglang
pip install "sglang[all]"
python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5 --api-key flowgrpo --port 17140 --chat-template chatml-llava --enable-p2p-check --mem-fraction-static 0.85
```

## Prompt Preprocessing

The training presets use preprocessed prompts/embeddings.

```bash
# pickscore/default prompts
bash scripts/preprocess/preprocess_sd35_embeddings.sh

# OCR prompts
PROMPT_PRESET=ocr bash scripts/preprocess/preprocess_sd35_embeddings.sh

# GenEval metadata prompts
PROMPT_PRESET=geneval bash scripts/preprocess/preprocess_sd35_embeddings.sh
```

## Paper Command

Figure 4(a), Ours:

```bash
bash scripts/single_node/run_sd3.sh --profile lowsnr2 --sampler branch --reward geneval --loss matching --reweight fairclip2 --num-processes 4
```

You can inspect available launcher choices with:

```bash
bash scripts/single_node/run_sd3.sh --list
```

The launcher writes outputs according to the selected config. Review cache, checkpoint, and output paths before launching long runs.

## Notes

- SD3.5-M runs were tested on CUDA 12.8 with 4 x H200 GPUs.
- The largest SD3.5-M setup can nearly saturate 140GB H200 VRAM when the GenEval server is hosted simultaneously.
- Lower-VRAM GPUs can be used by reducing per-device batch sizes and increasing gradient accumulation to keep the same effective batch size.
- Package versions may need adjustment for your CUDA version, PyTorch wheel, xformers build, and GPU architecture.
