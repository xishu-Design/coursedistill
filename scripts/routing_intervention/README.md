# Fast CLS versus uncertainty intervention

This experiment starts two Qwen3-1.7B branches from one CPD checkpoint. Both
branches retain the same rollout budget, GRPO objective, local distillation,
teacher, optimizer, seed, and training data. Only problem routing changes:

- `cls`: `4 * min(p, 1 - p) * (1 - p)`
- `uncertainty`: `4 * p * (1 - p)`

Each branch runs for 120 additional optimizer steps. Evaluation uses the same
deterministic 256-problem MATH holdout and Avg@4. The analysis clusters the
paired bootstrap by problem and reports endpoint accuracy, gain from the shared
checkpoint, and CLS minus uncertainty.

## Run

From the repository root on the training machine:

```bash
export BASE_MODEL=/path/to/Qwen3-1.7B
export TEACHER_MODEL=/path/to/Qwen3-8B
export START_CHECKPOINT=/path/to/cpd/checkpoint-350
export MATH500_FILE=/path/to/math500.jsonl
export TRAIN_DATA_DIR=/path/to/cpd_training_data
export RELEASE_ROOT="$PWD"
export TRAIN_GPUS=0,1,2,3
export EVAL_GPUS=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

bash scripts/routing_intervention/run_routing_intervention.sh
```

The default output is `outputs/routing_intervention`. To resume individual
phases without repeating completed work:

```bash
STAGES=train bash scripts/routing_intervention/run_routing_intervention.sh
STAGES=eval bash scripts/routing_intervention/run_routing_intervention.sh
STAGES=summarize bash scripts/routing_intervention/run_routing_intervention.sh
```

Useful cost controls are `BRANCH_STEPS=80`, `HOLDOUT_SIZE=128`, and
`NUM_SAMPLES=2`. The recommended evidence setting remains 120, 256, and 4.

If the original run used path or hardware overrides, export the same values
before launching. The script calls the released `config_cpd_1p7b.sh`, so all
published CPD objective settings are inherited. It deliberately sets
`CURRICULUM_REVIEW_MIX_MAX=0` in both branches to isolate asymmetric CLS from
symmetric uncertainty.

## Interpretation

The strongest result is a positive `CLS minus uncertainty` estimate whose
paired 95% interval excludes zero, accompanied by a larger CLS gain from the
shared checkpoint. If the interval crosses zero, report the experiment as
short-horizon directional evidence rather than a significant superiority
claim. This intervention is much stronger than an offline selector diagnostic,
but it is not a substitute for a full-training baseline.

Curriculum memory is not serialized by the current trainer. Both branches
therefore rebuild it from the same checkpoint, seed, and online observations;
this keeps the intervention symmetric but should be disclosed when reporting
the result.

The default evaluator is `math_verify`. To use the paper's PRM800K grading
path when that package is already importable, set `GRADER=prm800k`. Evaluation
protocol and grader must be identical for the start, CLS, and uncertainty runs.

## Aggregate result

Using the recommended 120-step, 256-problem, Avg@4 setting, the shared start
checkpoint scores 60.84%, the CLS endpoint scores 62.30%, and the uncertainty
endpoint scores 59.67%. The paired CLS-minus-uncertainty difference is +2.64
percentage points with a 95% problem-cluster bootstrap interval of [0.20,
5.08]. The corresponding gains from the shared start are +1.46 points
([-1.27, 4.20]) for CLS and -1.17 points ([-3.71, 1.46]) for uncertainty.
These results are short-horizon intervention evidence rather than a
replacement for the full-training comparisons.
