#!/usr/bin/env bash
set -euo pipefail

# Qwen3-0.6B paper configuration. Method hyperparameters are inherited from
# the common 1.7B configuration to keep the comparison controlled.
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-0.6B}"
export TEACHER_MODEL_NAME_OR_PATH="${TEACHER_MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-13032}"
export RUN_CONFIG="${RUN_CONFIG:-cpd_qwen3_0p6b}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/config_cpd_1p7b.sh"
