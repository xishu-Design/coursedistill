#!/usr/bin/env bash
set -euo pipefail

# Common CPD training entry point. The verifier determines update direction,
# while the teacher provides bounded positive credit on selected spans.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-1.7B}"
TEACHER_MODEL_NAME_OR_PATH="${TEACHER_MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-cpd}"
export SWANLAB_MODE="${SWANLAB_MODE:-disabled}"

NUM_PROCESSES="${NUM_PROCESSES:-4}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12983}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}"
NUM_STUDENT_SAMPLES="${NUM_STUDENT_SAMPLES:-16}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-768}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.14}"

LEARNING_RATE="${LEARNING_RATE:-3e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.08}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.25}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"

# Sampling parameters used by the released configurations.
TEMPERATURE="${TEMPERATURE:-0.95}"
TOP_P="${TOP_P:-0.85}"
TOP_K="${TOP_K:-30}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.05}"

GRPO_COEF="${GRPO_COEF:-1.5}"
GRPO_LOSS_CLIP="${GRPO_LOSS_CLIP:-0.5}"
GRPO_LOGP_CLIP="${GRPO_LOGP_CLIP:-10.0}"

# Decay local span supervision late in training.
SPAN_DISTILL_COEF="${SPAN_DISTILL_COEF:-0.012}"
SPAN_COEF_FLOOR="${SPAN_COEF_FLOOR:-0.0015}"
SPAN_DECAY_START_RATIO="${SPAN_DECAY_START_RATIO:-0.50}"
SPAN_DECAY_END_RATIO="${SPAN_DECAY_END_RATIO:-0.90}"

# Keep error suppression as a small guardrail rather than the main objective.
ERROR_SUPPRESS_COEF="${ERROR_SUPPRESS_COEF:-0.003}"
ERROR_SUPPRESS_MAX_RATIO="${ERROR_SUPPRESS_MAX_RATIO:-0.08}"
ERROR_SUPPRESS_MAX_SPAN_RATIO="${ERROR_SUPPRESS_MAX_SPAN_RATIO:-0.0}"
ERROR_SUPPRESS_START_RATIO="${ERROR_SUPPRESS_START_RATIO:-0.35}"

ALL_WRONG_PENALTY="${ALL_WRONG_PENALTY:-0.0}"
ALL_WRONG_DATA_WEIGHT="${ALL_WRONG_DATA_WEIGHT:-0.0}"
ALL_CORRECT_BONUS="${ALL_CORRECT_BONUS:-0.0}"
ALL_CORRECT_DATA_WEIGHT="${ALL_CORRECT_DATA_WEIGHT:-0.0}"
ADVANTAGE_CLIP="${ADVANTAGE_CLIP:-3.0}"

USE_PRIVILEGED_GAIN="${USE_PRIVILEGED_GAIN:-True}"
FINAL_SPAN_WEIGHT="${FINAL_SPAN_WEIGHT:--1.0}"
CONTRAST_SPAN_WEIGHT="${CONTRAST_SPAN_WEIGHT:--1.0}"
CONTEXT_SPAN_WEIGHT="${CONTEXT_SPAN_WEIGHT:--1.0}"
ANSWER_SPAN_WEIGHT="${ANSWER_SPAN_WEIGHT:-0.50}"
DIFF_SPAN_WEIGHT="${DIFF_SPAN_WEIGHT:-1.25}"
CALCULATION_SPAN_WEIGHT="${CALCULATION_SPAN_WEIGHT:-0.0}"
SUPPORT_SPAN_WEIGHT="${SUPPORT_SPAN_WEIGHT:-0.03}"
REASON_SPAN_WEIGHT="${REASON_SPAN_WEIGHT:-0.06}"
ANSWER_SPAN_TOKENS="${ANSWER_SPAN_TOKENS:-64}"
DIFF_SPAN_TOKENS="${DIFF_SPAN_TOKENS:-96}"
DIFF_SPAN_TOP_K="${DIFF_SPAN_TOP_K:-3}"
CALCULATION_SPAN_TOKENS="${CALCULATION_SPAN_TOKENS:-64}"
SUPPORT_SPAN_TOKENS="${SUPPORT_SPAN_TOKENS:-64}"
REASON_SPAN_TOKENS="${REASON_SPAN_TOKENS:-128}"
SUPPORT_START_RATIO="${SUPPORT_START_RATIO:-0.68}"
REASON_START_RATIO="${REASON_START_RATIO:-0.45}"
SPAN_EVIDENCE_FLOOR="${SPAN_EVIDENCE_FLOOR:--1.0}"
SPAN_RANK_FLOOR="${SPAN_RANK_FLOOR:-0.25}"
SPAN_GAIN_WEIGHT_FLOOR="${SPAN_GAIN_WEIGHT_FLOOR:-1.0}"
SPAN_GAIN_WEIGHT_SCALE="${SPAN_GAIN_WEIGHT_SCALE:-1.0}"
SPAN_TRAJ_NORMALIZE="${SPAN_TRAJ_NORMALIZE:-False}"
SPAN_MAX_TRAJ_WEIGHT="${SPAN_MAX_TRAJ_WEIGHT:-1.0}"

# Preserve coverage while prioritizing mixed response groups.
CURRICULUM_ALPHA_START="${CURRICULUM_ALPHA_START:-1.20}"
CURRICULUM_ALPHA_END="${CURRICULUM_ALPHA_END:-0.40}"
CURRICULUM_STAGE1_RATIO="${CURRICULUM_STAGE1_RATIO:-0.20}"
CURRICULUM_STAGE2_RATIO="${CURRICULUM_STAGE2_RATIO:-0.62}"
CURRICULUM_STAGE3_RATIO="${CURRICULUM_STAGE3_RATIO:-0.84}"
CURRICULUM_MEMORY_MOMENTUM="${CURRICULUM_MEMORY_MOMENTUM:-0.50}"
CURRICULUM_MEMORY_BLEND="${CURRICULUM_MEMORY_BLEND:-0.30}"
CURRICULUM_WEIGHT_FLOOR="${CURRICULUM_WEIGHT_FLOOR:-0.15}"
CURRICULUM_REVIEW_MIX_MAX="${CURRICULUM_REVIEW_MIX_MAX:-0.35}"
CURRICULUM_SAMPLER_ENABLED="${CURRICULUM_SAMPLER_ENABLED:-False}"
CURRICULUM_SAMPLER_WARMUP_RATIO="${CURRICULUM_SAMPLER_WARMUP_RATIO:-0.15}"
CURRICULUM_SAMPLER_FLOOR="${CURRICULUM_SAMPLER_FLOOR:-0.10}"
CURRICULUM_SAMPLER_POWER="${CURRICULUM_SAMPLER_POWER:-1.0}"
CURRICULUM_SAMPLER_REPLACEMENT="${CURRICULUM_SAMPLER_REPLACEMENT:-True}"
CURRICULUM_SAMPLER_UNIQUE_FRACTION="${CURRICULUM_SAMPLER_UNIQUE_FRACTION:-0.0}"
ADAPTIVE_CURRICULUM_ENABLED="${ADAPTIVE_CURRICULUM_ENABLED:-False}"
ADAPTIVE_CURRICULUM_MIN_COVERAGE="${ADAPTIVE_CURRICULUM_MIN_COVERAGE:-0.20}"
ADAPTIVE_SUPPORT_MIN_CORRECT_RATE="${ADAPTIVE_SUPPORT_MIN_CORRECT_RATE:-0.42}"
ADAPTIVE_REASON_MIN_CORRECT_RATE="${ADAPTIVE_REASON_MIN_CORRECT_RATE:-0.50}"
MIXED_BALANCE_BONUS="${MIXED_BALANCE_BONUS:-0.0}"

# Modestly reduce the influence of truncated and overlong responses.
WRONG_LENGTH_BOOST="${WRONG_LENGTH_BOOST:-0.25}"
TRUNCATED_NEGATIVE_WEIGHT="${TRUNCATED_NEGATIVE_WEIGHT:-0.0}"
NEAR_FULL_NEGATIVE_RATIO="${NEAR_FULL_NEGATIVE_RATIO:-1.0}"
CORRECT_NEAR_FULL_POSITIVE_WEIGHT="${CORRECT_NEAR_FULL_POSITIVE_WEIGHT:-1.0}"
POST_ANSWER_GRPO_WEIGHT="${POST_ANSWER_GRPO_WEIGHT:-1.0}"
ANSWER_STOP_LOSS_COEF="${ANSWER_STOP_LOSS_COEF:-0.0}"
ANSWER_STOP_GRACE_TOKENS="${ANSWER_STOP_GRACE_TOKENS:-2}"

TEACHER_STEP_WINDOW="${TEACHER_STEP_WINDOW:-21}"
TEACHER_STUDENT_GAP_COEF="${TEACHER_STUDENT_GAP_COEF:-0.0}"
TEACHER_PRIVILEGED_CONFIDENCE_COEF="${TEACHER_PRIVILEGED_CONFIDENCE_COEF:-0.0}"
TEACHER_GROUP_SUPPORT_FLOOR="${TEACHER_GROUP_SUPPORT_FLOOR:-0.20}"
TEACHER_SUPPORT_MAX="${TEACHER_SUPPORT_MAX:-1.45}"
TEACHER_GRPO_MAGNITUDE_COEF="${TEACHER_GRPO_MAGNITUDE_COEF:-1.0}"
TEACHER_CORRECT_LENGTH_DAMPEN="${TEACHER_CORRECT_LENGTH_DAMPEN:-0.0}"
TEACHER_STEP_RANK_FLOOR="${TEACHER_STEP_RANK_FLOOR:-0.70}"
TEACHER_TOKEN_WEIGHT_CLIP="${TEACHER_TOKEN_WEIGHT_CLIP:-1.25}"
TEACHER_SCORE_MICRO_BATCH="${TEACHER_SCORE_MICRO_BATCH:-1}"
POSITIVE_TEACHER_SPAN_ONLY="${POSITIVE_TEACHER_SPAN_ONLY:-True}"
POSITIVE_REASON_SPAN_ONLY="${POSITIVE_REASON_SPAN_ONLY:-True}"
NO_ANSWER_FULL_LENGTH_WEIGHT="${NO_ANSWER_FULL_LENGTH_WEIGHT:-0.0}"
VERIFIER_CONFIDENCE_ENABLED="${VERIFIER_CONFIDENCE_ENABLED:-False}"
VERIFIER_ANSWER_MARKER_NEGATIVE_CONFIDENCE="${VERIFIER_ANSWER_MARKER_NEGATIVE_CONFIDENCE:-0.85}"
VERIFIER_TRUNCATED_ANSWER_NEGATIVE_CONFIDENCE="${VERIFIER_TRUNCATED_ANSWER_NEGATIVE_CONFIDENCE:-0.65}"
VERIFIER_CORRECT_NO_ANSWER_CONFIDENCE="${VERIFIER_CORRECT_NO_ANSWER_CONFIDENCE:-0.75}"
TRACE_CONTEXT_TOPK="${TRACE_CONTEXT_TOPK:-0}"
TRACE_CONTEXT_MAX_TOKENS="${TRACE_CONTEXT_MAX_TOKENS:-64}"

SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-16}"
RUN_CONFIG="${RUN_CONFIG:-cpd_qwen3_1p7b}"
SEED="${SEED:-42}"
CURRICULUM_ROUTING_RULE="${CURRICULUM_ROUTING_RULE:-cls}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
MAX_STEPS="${MAX_STEPS:--1}"

EXTRA_TRAIN_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    EXTRA_TRAIN_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${MAX_STEPS}" -gt 0 ]]; then
    EXTRA_TRAIN_ARGS+=(--max_steps "${MAX_STEPS}")
fi

cd "${PROJECT_ROOT}"
"${PYTHON:-python}" -m accelerate.commands.launch \
    --config_file "${PROJECT_ROOT}/accelerate.yaml" \
    --num_processes "${NUM_PROCESSES}" \
    --gradient_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    coursedistill/train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --teacher_model_name_or_path "${TEACHER_MODEL_NAME_OR_PATH}" \
    --data_dir "${DATA_DIR}" \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --weight_decay 0.01 \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --per_device_train_batch_size 1 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --gradient_accumulation_steps "${GRAD_ACCUMULATION_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --run_config "${RUN_CONFIG}" \
    --seed "${SEED}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps 5 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --load_in_4bit False \
    --load_in_8bit False \
    --max_length 8192 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --top_k "${TOP_K}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --num_student_samples "${NUM_STUDENT_SAMPLES}" \
    --student_thinking False \
    --teacher_thinking True \
    --grpo_coef "${GRPO_COEF}" \
    --grpo_loss_clip "${GRPO_LOSS_CLIP}" \
    --grpo_logp_clip "${GRPO_LOGP_CLIP}" \
    --span_distill_coef "${SPAN_DISTILL_COEF}" \
    --span_coef_floor "${SPAN_COEF_FLOOR}" \
    --span_decay_start_ratio "${SPAN_DECAY_START_RATIO}" \
    --span_decay_end_ratio "${SPAN_DECAY_END_RATIO}" \
    --error_suppress_coef "${ERROR_SUPPRESS_COEF}" \
    --error_suppress_max_ratio "${ERROR_SUPPRESS_MAX_RATIO}" \
    --error_suppress_max_span_ratio "${ERROR_SUPPRESS_MAX_SPAN_RATIO}" \
    --error_suppress_start_ratio "${ERROR_SUPPRESS_START_RATIO}" \
    --all_wrong_penalty "${ALL_WRONG_PENALTY}" \
    --all_wrong_data_weight "${ALL_WRONG_DATA_WEIGHT}" \
    --all_correct_bonus "${ALL_CORRECT_BONUS}" \
    --all_correct_data_weight "${ALL_CORRECT_DATA_WEIGHT}" \
    --advantage_clip "${ADVANTAGE_CLIP}" \
    --use_privileged_gain "${USE_PRIVILEGED_GAIN}" \
    --curriculum_alpha_start "${CURRICULUM_ALPHA_START}" \
    --curriculum_alpha_end "${CURRICULUM_ALPHA_END}" \
    --curriculum_routing_rule "${CURRICULUM_ROUTING_RULE}" \
    --curriculum_stage1_ratio "${CURRICULUM_STAGE1_RATIO}" \
    --curriculum_stage2_ratio "${CURRICULUM_STAGE2_RATIO}" \
    --curriculum_stage3_ratio "${CURRICULUM_STAGE3_RATIO}" \
    --curriculum_memory_momentum "${CURRICULUM_MEMORY_MOMENTUM}" \
    --curriculum_memory_blend "${CURRICULUM_MEMORY_BLEND}" \
    --curriculum_weight_floor "${CURRICULUM_WEIGHT_FLOOR}" \
    --curriculum_review_mix_max "${CURRICULUM_REVIEW_MIX_MAX}" \
    --curriculum_sampler_enabled "${CURRICULUM_SAMPLER_ENABLED}" \
    --curriculum_sampler_warmup_ratio "${CURRICULUM_SAMPLER_WARMUP_RATIO}" \
    --curriculum_sampler_floor "${CURRICULUM_SAMPLER_FLOOR}" \
    --curriculum_sampler_power "${CURRICULUM_SAMPLER_POWER}" \
    --curriculum_sampler_replacement "${CURRICULUM_SAMPLER_REPLACEMENT}" \
    --curriculum_sampler_unique_fraction "${CURRICULUM_SAMPLER_UNIQUE_FRACTION}" \
    --adaptive_curriculum_enabled "${ADAPTIVE_CURRICULUM_ENABLED}" \
    --adaptive_curriculum_min_coverage "${ADAPTIVE_CURRICULUM_MIN_COVERAGE}" \
    --adaptive_support_min_correct_rate "${ADAPTIVE_SUPPORT_MIN_CORRECT_RATE}" \
    --adaptive_reason_min_correct_rate "${ADAPTIVE_REASON_MIN_CORRECT_RATE}" \
    --mixed_balance_bonus "${MIXED_BALANCE_BONUS}" \
    --wrong_length_boost "${WRONG_LENGTH_BOOST}" \
    --truncated_negative_weight "${TRUNCATED_NEGATIVE_WEIGHT}" \
    --near_full_negative_ratio "${NEAR_FULL_NEGATIVE_RATIO}" \
    --correct_near_full_positive_weight "${CORRECT_NEAR_FULL_POSITIVE_WEIGHT}" \
    --post_answer_grpo_weight "${POST_ANSWER_GRPO_WEIGHT}" \
    --answer_stop_loss_coef "${ANSWER_STOP_LOSS_COEF}" \
    --answer_stop_grace_tokens "${ANSWER_STOP_GRACE_TOKENS}" \
    --answer_span_tokens "${ANSWER_SPAN_TOKENS}" \
    --diff_span_tokens "${DIFF_SPAN_TOKENS}" \
    --diff_span_top_k "${DIFF_SPAN_TOP_K}" \
    --calculation_span_tokens "${CALCULATION_SPAN_TOKENS}" \
    --support_span_tokens "${SUPPORT_SPAN_TOKENS}" \
    --reason_span_tokens "${REASON_SPAN_TOKENS}" \
    --support_start_ratio "${SUPPORT_START_RATIO}" \
    --reason_start_ratio "${REASON_START_RATIO}" \
    --span_evidence_floor "${SPAN_EVIDENCE_FLOOR}" \
    --span_rank_floor "${SPAN_RANK_FLOOR}" \
    --span_gain_weight_floor "${SPAN_GAIN_WEIGHT_FLOOR}" \
    --span_gain_weight_scale "${SPAN_GAIN_WEIGHT_SCALE}" \
    --span_traj_normalize "${SPAN_TRAJ_NORMALIZE}" \
    --span_max_traj_weight "${SPAN_MAX_TRAJ_WEIGHT}" \
    --teacher_step_window "${TEACHER_STEP_WINDOW}" \
    --teacher_student_gap_coef "${TEACHER_STUDENT_GAP_COEF}" \
    --teacher_privileged_confidence_coef "${TEACHER_PRIVILEGED_CONFIDENCE_COEF}" \
    --teacher_group_support_floor "${TEACHER_GROUP_SUPPORT_FLOOR}" \
    --teacher_support_max "${TEACHER_SUPPORT_MAX}" \
    --teacher_grpo_magnitude_coef "${TEACHER_GRPO_MAGNITUDE_COEF}" \
    --teacher_correct_length_dampen "${TEACHER_CORRECT_LENGTH_DAMPEN}" \
    --teacher_step_rank_floor "${TEACHER_STEP_RANK_FLOOR}" \
    --teacher_token_weight_clip "${TEACHER_TOKEN_WEIGHT_CLIP}" \
    --teacher_score_micro_batch "${TEACHER_SCORE_MICRO_BATCH}" \
    --positive_teacher_span_only "${POSITIVE_TEACHER_SPAN_ONLY}" \
    --positive_reason_span_only "${POSITIVE_REASON_SPAN_ONLY}" \
    --no_answer_full_length_weight "${NO_ANSWER_FULL_LENGTH_WEIGHT}" \
    --verifier_confidence_enabled "${VERIFIER_CONFIDENCE_ENABLED}" \
    --verifier_answer_marker_negative_confidence "${VERIFIER_ANSWER_MARKER_NEGATIVE_CONFIDENCE}" \
    --verifier_truncated_answer_negative_confidence "${VERIFIER_TRUNCATED_ANSWER_NEGATIVE_CONFIDENCE}" \
    --verifier_correct_no_answer_confidence "${VERIFIER_CORRECT_NO_ANSWER_CONFIDENCE}" \
    --trace_context_topk "${TRACE_CONTEXT_TOPK}" \
    --trace_context_max_tokens "${TRACE_CONTEXT_MAX_TOKENS}" \
    --final_span_weight "${FINAL_SPAN_WEIGHT}" \
    --contrast_span_weight "${CONTRAST_SPAN_WEIGHT}" \
    --context_span_weight "${CONTEXT_SPAN_WEIGHT}" \
    --answer_span_weight "${ANSWER_SPAN_WEIGHT}" \
    --diff_span_weight "${DIFF_SPAN_WEIGHT}" \
    --calculation_span_weight "${CALCULATION_SPAN_WEIGHT}" \
    --support_span_weight "${SUPPORT_SPAN_WEIGHT}" \
    --reason_span_weight "${REASON_SPAN_WEIGHT}" \
    --swanlab_project "${SWANLAB_PROJECT}" \
    --disable_swanlab True \
    "${EXTRA_TRAIN_ARGS[@]}"
