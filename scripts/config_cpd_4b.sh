#!/usr/bin/env bash
set -euo pipefail

# Qwen3-4B paper configuration on DAPO-Math-14k.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
export TEACHER_MODEL_NAME_OR_PATH="${TEACHER_MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"
export DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-13034}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
export RUN_CONFIG="${RUN_CONFIG:-cpd_qwen3_4b_dapo_math_14k}"

if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "Missing ${DATA_DIR}/train.jsonl" >&2
    echo "Run: python data/prepare_dapo_math_14k.py --output-dir data" >&2
    exit 1
fi

exec bash "${SCRIPT_DIR}/config_cpd_1p7b.sh"
