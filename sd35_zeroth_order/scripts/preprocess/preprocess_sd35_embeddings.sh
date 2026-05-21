#!/bin/bash
set -e

GPU_NUM=${GPU_NUM:-1}
PROMPT_PRESET=${PROMPT_PRESET:-pickscore}
MIXED_PRECISION=${MIXED_PRECISION:-fp16}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-stabilityai/stable-diffusion-3.5-medium}

case "${PROMPT_PRESET}" in
  pickscore)
    DEFAULT_OUTPUT_DIR="../cache/sd35_embeddings/pickscore"
    DEFAULT_TRAIN_PROMPT_FILE="dataset/pickscore/train.txt"
    DEFAULT_TEST_PROMPT_FILE="dataset/pickscore/test.txt"
    ;;
  ocr)
    DEFAULT_OUTPUT_DIR="../cache/sd35_embeddings/ocr"
    DEFAULT_TRAIN_PROMPT_FILE="dataset/ocr/train.txt"
    DEFAULT_TEST_PROMPT_FILE="dataset/ocr/test.txt"
    ;;
  geneval)
    DEFAULT_OUTPUT_DIR="../cache/sd35_embeddings/geneval"
    DEFAULT_TRAIN_PROMPT_FILE="dataset/geneval/train_metadata.jsonl"
    DEFAULT_TEST_PROMPT_FILE="dataset/geneval/test_metadata.jsonl"
    ;;
  *)
    echo "Unsupported PROMPT_PRESET: ${PROMPT_PRESET}"
    echo "Supported presets: pickscore, ocr, geneval"
    exit 1
    ;;
esac

OUTPUT_DIR=${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}
TRAIN_PROMPT_FILE=${TRAIN_PROMPT_FILE:-${DEFAULT_TRAIN_PROMPT_FILE}}
TEST_PROMPT_FILE=${TEST_PROMPT_FILE:-${DEFAULT_TEST_PROMPT_FILE}}

echo "[preprocess_sd35_embeddings.sh] prompt_preset=${PROMPT_PRESET}"
echo "[preprocess_sd35_embeddings.sh] output_dir=${OUTPUT_DIR}"
echo "[preprocess_sd35_embeddings.sh] train_prompt_file=${TRAIN_PROMPT_FILE}"
echo "[preprocess_sd35_embeddings.sh] test_prompt_file=${TEST_PROMPT_FILE}"

accelerate launch --main_process_port 29501 --num_processes "${GPU_NUM}" \
  scripts/preprocess/preprocess_sd35_embeddings.py \
  --output_dir "${OUTPUT_DIR}" \
  --train_prompt_file "${TRAIN_PROMPT_FILE}" \
  --test_prompt_file "${TEST_PROMPT_FILE}" \
  --mixed_precision "${MIXED_PRECISION}" \
  --pretrained_model "${PRETRAINED_MODEL}"
