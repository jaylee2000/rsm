#!/bin/bash

set -euo pipefail

# Keep long runs alive if the parent terminal disconnects.
trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PROFILE="base"
SAMPLER="default"
REWARD="pickscore"
LOSS="ppo"
REWEIGHT="base"
NUM_PROCESSES=""
MAIN_PROCESS_PORT=""
ACCELERATE_CONFIG=""
DRY_RUN=0
LIST_VALUES=0

SUPPORTED_PROFILES=(base highsnr highsnr2 base2 lowsnr lowsnr2)
SUPPORTED_SAMPLERS=(default branch)
SUPPORTED_REWARDS=(geneval ocr pickscore deqa imagereward qwenvl aesthetic jpeg_compressibility unifiedreward)
SUPPORTED_LOSSES=(ppo matching)
SUPPORTED_REWEIGHTS=(base tempflow pcpo guard fairclip fairclip2)

usage() {
  cat <<'EOF'
Usage:
  bash scripts/single_node/run_sd3.sh [options]

Options:
  --profile <base|highsnr|highsnr2|base2|lowsnr|lowsnr2>
  --sampler <default|branch>
  --reward <geneval|ocr|pickscore|deqa|imagereward|qwenvl|aesthetic|jpeg_compressibility|unifiedreward>
  --loss <ppo|matching>
  --reweight <base|tempflow|pcpo|guard|fairclip|fairclip2>
  --num-processes <int>
  --main-process-port <int>
  --accelerate-config <path>
  --dry-run
  --list
  -h, --help

Config id format:
  sd3.<profile>.<sampler>.<reward>.<loss>.<reweight>
EOF
}

list_values() {
  echo "profiles:   ${SUPPORTED_PROFILES[*]}"
  echo "samplers:   ${SUPPORTED_SAMPLERS[*]}"
  echo "rewards:    ${SUPPORTED_REWARDS[*]}"
  echo "losses:     ${SUPPORTED_LOSSES[*]}"
  echo "reweights:  ${SUPPORTED_REWEIGHTS[*]}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --sampler)
      SAMPLER="$2"
      shift 2
      ;;
    --reward)
      REWARD="$2"
      shift 2
      ;;
    --loss)
      LOSS="$2"
      shift 2
      ;;
    --reweight)
      REWEIGHT="$2"
      shift 2
      ;;
    --num-processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --main-process-port)
      MAIN_PROCESS_PORT="$2"
      shift 2
      ;;
    --accelerate-config)
      ACCELERATE_CONFIG="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --list)
      LIST_VALUES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ "${LIST_VALUES}" == "1" ]]; then
  list_values
  exit 0
fi

if [[ "${SAMPLER}" == "default" ]]; then
  TRAIN_SCRIPT="scripts/train_sd3.py"
  NUM_PROCESSES="${NUM_PROCESSES:-2}"
  MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29502}"
elif [[ "${SAMPLER}" == "branch" ]]; then
  TRAIN_SCRIPT="scripts/train_sd3_pr.py"
  NUM_PROCESSES="${NUM_PROCESSES:-2}"
  MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29700}"
else
  echo "Unsupported sampler: ${SAMPLER}"
  exit 1
fi

# Common NCCL defaults.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"

CONFIG_ID="sd3.${PROFILE}.${SAMPLER}.${REWARD}.${LOSS}.${REWEIGHT}"
CONFIG_TARGET="config/sd3_matrix.py:${CONFIG_ID}"

CMD=(
  accelerate launch
  --num_machines 1
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
)

if [[ -n "${ACCELERATE_CONFIG}" ]]; then
  CMD+=(--config_file "${ACCELERATE_CONFIG}")
fi

CMD+=("${TRAIN_SCRIPT}" --config "${CONFIG_TARGET}")

echo "[run_sd3.sh] profile=${PROFILE} sampler=${SAMPLER} reward=${REWARD} loss=${LOSS} reweight=${REWEIGHT}"
echo "[run_sd3.sh] config_id=${CONFIG_ID}"
echo "[run_sd3.sh] train_script=${TRAIN_SCRIPT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[run_sd3.sh] dry-run command:"
  printf '  %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
