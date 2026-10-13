# Course-Aware Privileged Distillation

This anonymous artifact contains the training, evaluation, and analysis code
for Course-Aware Privileged Distillation (CPD). It excludes model weights,
datasets, checkpoints, generated responses, logs, and paper sources.

## Contents

- `coursedistill/`: CPD curriculum routing and privileged local credit.
- `scripts/train_cpd.sh`: common training entry point.
- `scripts/config_cpd_{0p6b,1p7b,4b}.sh`: paper-scale configurations.
- `evaluation/`: benchmark generation and deterministic answer verification.
- `scripts/analyze_*.py`: selector, future-progress, and span-control analyses.
- `scripts/routing_intervention/`: matched short-horizon CLS-versus-uncertainty intervention.
- `analysis/`: self-contained derived-data analyses used by the appendix.
- `data/prepare_dapo_math_14k.py`: preparation for public DAPO-Math data.

## Environment

The code targets Linux, Python 3.10+, CUDA, and four GPUs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model paths default to public Hugging Face identifiers and can be overridden
with `MODEL_NAME_OR_PATH` and `TEACHER_MODEL_NAME_OR_PATH`. Online experiment
tracking is disabled by default.

## Data

Training expects `data/train.jsonl`. Each record must contain:

```json
{
  "id": "example-0",
  "student_input": "Problem statement",
  "ground_truth": "Reference answer",
  "verifier_type": "math",
  "reference_solution": "Privileged training-only solution"
}
```

Prepare the public DAPO-Math-14k split with:

```bash
python data/prepare_dapo_math_14k.py --output-dir data
```

For MATH training, adapt the public source to the same schema.

## Training

Run a paper configuration from the repository root:

```bash
bash scripts/config_cpd_0p6b.sh
bash scripts/config_cpd_1p7b.sh
bash scripts/config_cpd_4b.sh
```

Common overrides can be passed as environment variables:

```bash
NUM_PROCESSES=4 DATA_DIR="$PWD/data" OUTPUT_DIR="$PWD/outputs" \
  bash scripts/config_cpd_1p7b.sh
```

The 0.6B and 1.7B configurations use three epochs on MATH. The 4B
configuration uses two epochs on DAPO-Math-14k.

## Evaluation

The evaluators consume public benchmark files and a local or Hugging Face model
identifier:

```bash
python evaluation/eval_math500.py --help
python evaluation/eval_minerva.py --help
python evaluation/eval_olympiadbench.py --help
```

The two benchmark-specific entry points prevent accidental mixing of grading
protocols. Minerva Math uses last-boxed-answer extraction and
lm-evaluation-harness-style normalization. OlympiadBench requires the public
official `AutoScoringJudge`; set `OLYMPIADBENCH_EVAL_DIR` to its `eval`
directory. The shared `evaluation/eval_benchmark.py` remains available for
advanced use and regrading.

## Analysis

Analysis scripts expose their inputs through `--help`:

```bash
python scripts/analyze_cpd_selection.py --help
python scripts/analyze_cls_learning_progress.py --help
python scripts/analyze_span_offline_controls.py --help
```

Saved rollout files are not included in the anonymous artifact.

The CLS-shape sensitivity analysis is reproducible without model weights or
saved rollouts:

```bash
python analysis/analyze_cls_shape_sensitivity.py \
  --pairs analysis/data/future_pairs.csv \
  --output analysis/outputs/cls_shape_sensitivity.csv
```

The included `future_pairs.csv` is a compact derived table containing only
benchmark item identifiers, rollout statistics, and next-occurrence outcomes.
It contains no prompts, responses, model paths, or author metadata.

The matched routing intervention and its aggregate result are documented in
`scripts/routing_intervention/README.md`. No checkpoints or generated
trajectories are included.

## Reproducibility Notes

This release preserves the implementation and paper configurations while
removing machine-specific paths and author-identifying metadata. Random seeds,
hardware kernels, and public model/data availability can affect exact results.
