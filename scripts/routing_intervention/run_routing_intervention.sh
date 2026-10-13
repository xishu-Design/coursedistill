#!/usr/bin/env bash
set -euo pipefail

# Short Qwen3-1.7B intervention: CLS routing versus symmetric uncertainty.
# Run from any directory after setting the required paths below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RELEASE_ROOT="${RELEASE_ROOT:-${PROJECT_ROOT}/coursedistill/release/cpd_anonymous_release}"
export PYTHONPATH="${RELEASE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

: "${BASE_MODEL:?Set BASE_MODEL to the Qwen3-1.7B base model path}"
: "${TEACHER_MODEL:?Set TEACHER_MODEL to the teacher model path}"
: "${START_CHECKPOINT:?Set START_CHECKPOINT to the common PEFT checkpoint}"
: "${MATH500_FILE:?Set MATH500_FILE to the full MATH-500 JSONL file}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/routing_intervention}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${RELEASE_ROOT}/data}"
HOLDOUT_FILE="${HOLDOUT_FILE:-${OUTPUT_ROOT}/data/math500_holdout_256.jsonl}"
HOLDOUT_SIZE="${HOLDOUT_SIZE:-256}"
BRANCH_STEPS="${BRANCH_STEPS:-120}"
NUM_SAMPLES="${NUM_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
SEED="${SEED:-2027}"
STAGES="${STAGES:-prepare,train,eval,summarize}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
EVAL_GPUS="${EVAL_GPUS:-0}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
GRADER="${GRADER:-math_verify}"

has_stage() { [[ ",${STAGES}," == *",$1,"* ]]; }

mkdir -p "${OUTPUT_ROOT}/data" "${OUTPUT_ROOT}/eval" "${OUTPUT_ROOT}/results"

if has_stage prepare; then
  python "${SCRIPT_DIR}/sample_math_holdout.py" \
    --input "${MATH500_FILE}" \
    --output "${HOLDOUT_FILE}" \
    --size "${HOLDOUT_SIZE}" \
    --seed "${SEED}"
fi

START_STEP="$(python - "${START_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
state = checkpoint / "trainer_state.json"
if state.exists():
    print(int(json.loads(state.read_text(encoding="utf-8"))["global_step"]))
else:
    name = checkpoint.name
    if not name.startswith("checkpoint-"):
        raise SystemExit("START_CHECKPOINT needs trainer_state.json or a checkpoint-N name")
    print(int(name.rsplit("-", 1)[1]))
PY
)"
TARGET_STEP="$((START_STEP + BRANCH_STEPS))"
echo "Common start step: ${START_STEP}; branch target step: ${TARGET_STEP}"

train_branch() {
  local rule="$1"
  local port="$2"
  echo "Training ${rule} branch"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  PROJECT_ROOT="${RELEASE_ROOT}" \
  MODEL_NAME_OR_PATH="${BASE_MODEL}" \
  TEACHER_MODEL_NAME_OR_PATH="${TEACHER_MODEL}" \
  DATA_DIR="${TRAIN_DATA_DIR}" \
  OUTPUT_DIR="${OUTPUT_ROOT}/train" \
  RUN_CONFIG="${rule}" \
  RESUME_FROM_CHECKPOINT="${START_CHECKPOINT}" \
  MAX_STEPS="${TARGET_STEP}" \
  SEED="${SEED}" \
  MAIN_PROCESS_PORT="${port}" \
  CURRICULUM_ROUTING_RULE="${rule}" \
  CURRICULUM_REVIEW_MIX_MAX="0.0" \
  bash "${RELEASE_ROOT}/scripts/config_cpd_1p7b.sh"
}

if has_stage train; then
  train_branch cls 13127
  train_branch uncertainty 13128
fi

resolve_adapter() {
  local branch="$1"
  local root="${OUTPUT_ROOT}/train/${branch}"
  if [[ -f "${root}/adapter_config.json" ]]; then
    printf '%s\n' "${root}"
    return
  fi
  local checkpoint="${root}/checkpoint-${TARGET_STEP}"
  if [[ -f "${checkpoint}/adapter_config.json" ]]; then
    printf '%s\n' "${checkpoint}"
    return
  fi
  echo "No adapter found for ${branch} under ${root}" >&2
  return 1
}

eval_adapter() {
  local name="$1"
  local adapter="$2"
  echo "Evaluating ${name}: ${adapter}"
  CUDA_VISIBLE_DEVICES="${EVAL_GPUS}" python "${RELEASE_ROOT}/evaluation/eval_math500.py" \
    --model_name_or_path "${BASE_MODEL}" \
    --adapter_path "${adapter}" \
    --test_file "${HOLDOUT_FILE}" \
    --output_dir "${OUTPUT_ROOT}/eval" \
    --run_name "${name}" \
    --grader "${GRADER}" \
    --backend "${EVAL_BACKEND}" \
    --num_samples "${NUM_SAMPLES}" \
    --do_sample \
    --temperature 1.0 \
    --top_p 0.8 \
    --top_k 50 \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --vllm_max_lora_rank 64 \
    --seed "${SEED}" \
    --save_every 16
}

if has_stage eval; then
  eval_adapter start "${START_CHECKPOINT}"
  eval_adapter cls "$(resolve_adapter cls)"
  eval_adapter uncertainty "$(resolve_adapter uncertainty)"
fi

if has_stage summarize; then
  python "${SCRIPT_DIR}/paired_bootstrap.py" \
    --start "${OUTPUT_ROOT}/eval/start/generations.jsonl" \
    --cls "${OUTPUT_ROOT}/eval/cls/generations.jsonl" \
    --uncertainty "${OUTPUT_ROOT}/eval/uncertainty/generations.jsonl" \
    --num-samples "${NUM_SAMPLES}" \
    --replicates 10000 \
    --seed "${SEED}" \
    --output-json "${OUTPUT_ROOT}/results/paired_results.json" \
    --output-md "${OUTPUT_ROOT}/results/paired_results.md"
fi
