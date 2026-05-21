#!/bin/bash

set -euo pipefail

trap '' HUP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE="${PROFILE:-base}"
REWARD="${REWARD:-pickscore}"
LOSS_TYPES=(${LOSS_TYPES:-ppo matching})
REWEIGHTS=(${REWEIGHTS:-base guard pcpo})
SAMPLERS=(${SAMPLERS:-default branch})
DRY_RUN="${DRY_RUN:-0}"

total_runs=0
for sampler in "${SAMPLERS[@]}"; do
  for loss in "${LOSS_TYPES[@]}"; do
    for reweight in "${REWEIGHTS[@]}"; do
      total_runs=$((total_runs + 1))
    done
  done
done

run_idx=0
for sampler in "${SAMPLERS[@]}"; do
  for loss in "${LOSS_TYPES[@]}"; do
    for reweight in "${REWEIGHTS[@]}"; do
      run_idx=$((run_idx + 1))
      echo "[$run_idx/$total_runs] profile=${PROFILE} sampler=${sampler} reward=${REWARD} loss=${loss} reweight=${reweight}"
      CMD=(
        bash "${SCRIPT_DIR}/run_sd3.sh"
        --profile "${PROFILE}" \
        --sampler "${sampler}" \
        --reward "${REWARD}" \
        --loss "${loss}" \
        --reweight "${reweight}"
      )
      if [[ "${DRY_RUN}" == "1" ]]; then
        CMD+=(--dry-run)
      fi
      "${CMD[@]}"
    done
  done
done

echo "Completed all ${total_runs} runs."
