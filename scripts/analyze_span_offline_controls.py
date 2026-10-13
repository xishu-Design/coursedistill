#!/usr/bin/env python3
"""Offline controls for CPD privileged local credit.

This script does not update model parameters. It reconstructs the released
AnswerSpan, DiffSpan, and SupportSpan masks on saved online rollouts, scores
the same verified-correct response with an 8B teacher both with and without
privileged reference information, and compares CPD localization against:

* position-matched random spans with the same run lengths,
* AnswerSpan only,
* the CPD mask shifted left/right by a fixed number of tokens, and
* the full verified-correct trajectory.

The independent outcome is the correctness change when the same problem next
appears in a saved online rollout. Teacher scores are cached after every case,
so an interrupted run can be resumed safely.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


# Running ``python scripts/<name>.py`` places ``scripts/`` rather than the
# repository root on sys.path. Add the root explicitly so local project modules
# resolve identically under direct, nohup, and scheduler launches.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_RUN = "outputs/cpd_qwen3_1p7b"

STRATEGIES = (
    "cpd_local",
    "random_position_matched",
    "answer_only",
    "shifted",
    "full_trace",
)

DISPLAY_NAMES = {
    "cpd_local": "CPD local",
    "random_position_matched": "Random matched",
    "answer_only": "Answer only",
    "shifted": "Shifted local",
    "full_trace": "Full trace",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-training offline span-control diagnostic for CPD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_RUN))
    parser.add_argument("--train-file", type=Path, default=Path("coursedistill/data/train.jsonl"))
    parser.add_argument(
        "--future-pairs-csv",
        type=Path,
        default=Path("training_curves_recovered/cls_learning_progress_1p7b/future_pairs.csv"),
    )
    parser.add_argument("--teacher-model", type=str, default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_curves_recovered/span_offline_controls_1p7b"),
    )
    parser.add_argument("--samples", type=int, default=200, help="0 uses every eligible observation.")
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--random-repeats", type=int, default=20)
    parser.add_argument("--shift-tokens", type=int, default=32)
    parser.add_argument("--top-fraction", type=float, default=0.30)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--teacher-temperature", type=float, default=0.95)
    parser.add_argument("--max-completion-tokens", type=int, default=768)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--answer-span-tokens", type=int, default=64)
    parser.add_argument("--diff-span-tokens", type=int, default=96)
    parser.add_argument("--diff-span-top-k", type=int, default=2)
    parser.add_argument("--support-span-tokens", type=int, default=48)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager", "auto"),
        default="flash_attention_2",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate joins and freeze the sample list without loading a tokenizer or teacher.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from exc


def load_training_data(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("id", ""))
        if sample_id:
            result[sample_id] = row
    if not result:
        raise RuntimeError(f"No training rows found in {path}")
    return result


def load_future_pairs(path: Path) -> dict[tuple[int, str], dict[str, float]]:
    result: dict[tuple[int, str], dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["step"]), str(row["id"]))
            result[key] = {
                "current_p": float(row["p"]),
                "future_step": float(row["future_step"]),
                "future_p": float(row["future_p"]),
                "future_gain": float(row["gain"]),
            }
    if not result:
        raise RuntimeError(f"No future pairs found in {path}")
    return result


def load_eligible_groups(
    run_dir: Path,
    training_rows: dict[str, dict[str, Any]],
    future_pairs: dict[tuple[int, str], dict[str, float]],
) -> list[dict[str, Any]]:
    generation_dir = run_dir / "generations"
    if not generation_dir.is_dir():
        raise FileNotFoundError(f"Missing generation directory: {generation_dir}")

    groups: list[dict[str, Any]] = []
    for source in sorted(generation_dir.glob("generations_step_*.json")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        step = int(payload.get("step", -1))
        by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in payload.get("generations", []):
            by_id[str(row.get("sample_id", ""))].append(row)

        for sample_id, rows in by_id.items():
            if not sample_id or sample_id not in training_rows:
                continue
            if step < 0 and rows:
                step = int(rows[0].get("step", -1))
            future = future_pairs.get((step, sample_id))
            if future is None or not (0.0 < future["current_p"] < 1.0):
                continue
            correct = [row for row in rows if bool(row.get("outcome"))]
            wrong = [row for row in rows if not bool(row.get("outcome"))]
            if not correct or not wrong:
                continue
            groups.append(
                {
                    "key": f"{step}:{sample_id}",
                    "step": step,
                    "sample_id": sample_id,
                    "source_file": source.name,
                    "correct": correct,
                    "wrong": wrong,
                    "train": training_rows[sample_id],
                    **future,
                }
            )
    groups.sort(key=lambda item: (item["step"], item["sample_id"]))
    return groups


def choose_cases(groups: list[dict[str, Any]], samples: int, seed: int) -> list[dict[str, Any]]:
    if samples <= 0 or samples >= len(groups):
        return list(groups)
    return sorted(
        random.Random(seed).sample(groups, samples),
        key=lambda item: (item["step"], item["sample_id"]),
    )


def write_case_manifest(cases: list[dict[str, Any]], path: Path) -> None:
    fields = (
        "key",
        "source_file",
        "step",
        "sample_id",
        "current_p",
        "future_step",
        "future_p",
        "future_gain",
        "num_correct",
        "num_wrong",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "key": case["key"],
                    "source_file": case["source_file"],
                    "step": case["step"],
                    "sample_id": case["sample_id"],
                    "current_p": case["current_p"],
                    "future_step": int(case["future_step"]),
                    "future_p": case["future_p"],
                    "future_gain": case["future_gain"],
                    "num_correct": len(case["correct"]),
                    "num_wrong": len(case["wrong"]),
                }
            )


def encode_completion(tokenizer: Any, text: str, limit: int) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=limit,
        return_attention_mask=False,
    )
    return list(encoded["input_ids"])


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def choose_representative_correct(
    tokenizer: Any,
    case: dict[str, Any],
    max_completion_tokens: int,
) -> tuple[int, list[list[int]], list[bool]]:
    rows = case["correct"] + case["wrong"]
    encoded = [
        encode_completion(tokenizer, str(row.get("completion", "")), max_completion_tokens)
        for row in rows
    ]
    correct_count = len(case["correct"])
    candidates: list[tuple[int, int, int]] = []
    for correct_index in range(correct_count):
        best_lcp = max(
            longest_common_prefix(encoded[correct_index], encoded[wrong_index])
            for wrong_index in range(correct_count, len(rows))
        )
        candidates.append((best_lcp, -len(encoded[correct_index]), correct_index))
    _, _, selected_index = max(candidates)
    outcomes = [True] * correct_count + [False] * len(case["wrong"])
    return selected_index, encoded, outcomes


def tensor_indices(mask: Any, row: int, valid_length: int) -> set[int]:
    values = mask[row, :valid_length].detach().cpu().tolist()
    return {index for index, value in enumerate(values) if float(value) > 0.0}


def build_cpd_masks(
    tokenizer: Any,
    encoded: list[list[int]],
    outcomes: list[bool],
    selected_index: int,
    args: argparse.Namespace,
    torch: Any,
) -> dict[str, set[int]]:
    from coursedistill.span_utils import SpanConfig, build_candidate_span_masks

    valid_length = len(encoded[selected_index])
    width = max(max((len(ids) for ids in encoded), default=1), 1)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    ids_tensor = torch.full((len(encoded), width), int(pad_id), dtype=torch.long)
    response_mask = torch.zeros((len(encoded), width), dtype=torch.float32)
    for row, ids in enumerate(encoded):
        if ids:
            ids_tensor[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            response_mask[row, : len(ids)] = 1.0
    outcome_tensor = torch.tensor(outcomes, dtype=torch.bool)
    parent_indices = torch.zeros(len(encoded), dtype=torch.long)
    config = SpanConfig(
        answer_span_tokens=args.answer_span_tokens,
        diff_span_tokens=args.diff_span_tokens,
        diff_span_top_k=args.diff_span_top_k,
        calculation_span_tokens=64,
        support_span_tokens=args.support_span_tokens,
        reason_span_tokens=48,
    )
    masks = build_candidate_span_masks(
        ids_tensor,
        response_mask,
        outcome_tensor,
        parent_indices,
        tokenizer,
        config,
        include_support=True,
        include_reason=False,
    )
    answer = tensor_indices(masks["answer"], selected_index, valid_length)
    diff = tensor_indices(masks["diff"], selected_index, valid_length)
    support = tensor_indices(masks["support"], selected_index, valid_length)
    local = answer | diff | support
    return {
        "answer": answer,
        "diff": diff,
        "support": support,
        "local": local,
        "full": set(range(valid_length)),
    }


def contiguous_runs(indices: set[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    ordered = sorted(indices)
    runs: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value == previous + 1:
            previous = value
            continue
        runs.append((start, previous + 1))
        start = previous = value
    runs.append((start, previous + 1))
    return runs


def shifted_indices(indices: set[int], length: int, offset: int) -> set[int]:
    if not indices or length <= 0:
        return set()
    shifted = {index + offset for index in indices if 0 <= index + offset < length}
    target = len(indices)
    if len(shifted) < target:
        anchor = sum(indices) / max(1, len(indices)) + offset
        candidates = sorted(
            (index for index in range(length) if index not in shifted),
            key=lambda index: (abs(index - anchor), index),
        )
        shifted.update(candidates[: target - len(shifted)])
    return shifted


def random_matched_indices(
    indices: set[int],
    length: int,
    rng: random.Random,
) -> set[int]:
    """Preserve run lengths and approximate normalized positions."""
    if not indices or length <= 0:
        return set()
    chosen: set[int] = set()
    original = set(indices)
    for start, end in contiguous_runs(indices):
        run_length = end - start
        midpoint = (start + end - 1) / 2.0
        quartile = min(3, int(4.0 * midpoint / max(1, length)))
        lower = int(quartile * length / 4)
        upper = int((quartile + 1) * length / 4)
        starts = list(range(lower, max(lower, upper - run_length) + 1))
        rng.shuffle(starts)
        selected: set[int] | None = None
        for candidate_start in starts:
            candidate = set(range(candidate_start, min(length, candidate_start + run_length)))
            if len(candidate) == run_length and not (candidate & chosen) and not (candidate & original):
                selected = candidate
                break
        if selected is None:
            all_starts = list(range(0, max(1, length - run_length + 1)))
            rng.shuffle(all_starts)
            for candidate_start in all_starts:
                candidate = set(range(candidate_start, min(length, candidate_start + run_length)))
                if len(candidate) == run_length and not (candidate & chosen):
                    selected = candidate
                    break
        if selected is not None:
            chosen.update(selected)

    if len(chosen) < len(indices):
        candidates = [index for index in range(length) if index not in chosen]
        rng.shuffle(candidates)
        chosen.update(candidates[: len(indices) - len(chosen)])
    if len(chosen) > len(indices):
        chosen = set(sorted(chosen)[: len(indices)])
    return chosen


def score_mask(gains: list[float], indices: set[int]) -> dict[str, float]:
    valid = [index for index in sorted(indices) if 0 <= index < len(gains)]
    if not valid:
        return {
            "token_count": 0.0,
            "token_fraction": 0.0,
            "mean_gain": 0.0,
            "positive_gain_density": 0.0,
            "positive_token_rate": 0.0,
            "positive_mass": 0.0,
            "positive_mass_capture": 0.0,
        }
    selected = [float(gains[index]) for index in valid]
    positive = [max(value, 0.0) for value in selected]
    full_positive_mass = sum(max(float(value), 0.0) for value in gains)
    positive_mass = sum(positive)
    return {
        "token_count": float(len(valid)),
        "token_fraction": len(valid) / max(1, len(gains)),
        "mean_gain": statistics.fmean(selected),
        "positive_gain_density": statistics.fmean(positive),
        "positive_token_rate": sum(value > 0.0 for value in selected) / len(selected),
        "positive_mass": positive_mass,
        "positive_mass_capture": positive_mass / max(full_positive_mass, 1e-12),
    }


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: statistics.fmean(item[key] for item in metrics)
        for key in metrics[0]
    }


def cache_key(
    case: dict[str, Any],
    completion: str,
    reference: str,
    args: argparse.Namespace,
) -> str:
    payload = {
        "case": case["key"],
        "completion": completion,
        "reference": reference,
        "teacher_model": args.teacher_model,
        "temperature": args.teacher_temperature,
        "max_completion_tokens": args.max_completion_tokens,
        "max_seq_length": args.max_seq_length,
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "cache_key" in row and "gains" in row:
                result[str(row["cache_key"])] = row
    return result


def teacher_prompt(collator: Any, problem: str, reference: str) -> tuple[str, str]:
    return collator._teacher_prompt(problem, ""), collator._teacher_prompt(problem, reference)


def sampled_logps(
    model: Any,
    tokenizer: Any,
    context: str,
    completion_ids: list[int],
    args: argparse.Namespace,
    torch: Any,
) -> list[float]:
    prompt_ids = tokenizer(
        context,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    response_ids = completion_ids[: args.max_completion_tokens]
    prompt_budget = args.max_seq_length - len(response_ids)
    if prompt_budget < 1:
        raise RuntimeError(
            f"max_seq_length={args.max_seq_length} leaves no room for "
            f"{len(response_ids)} response tokens"
        )
    # The training tokenizer uses right-side truncation for teacher contexts.
    prompt_ids = list(prompt_ids[:prompt_budget])
    all_ids = prompt_ids + response_ids
    input_ids = torch.tensor([all_ids], dtype=torch.long, device=args.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=len(response_ids) + 1,
            )
        except TypeError:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
        logits = outputs.logits
        if logits.shape[1] == len(response_ids) + 1:
            logits = logits[:, :-1, :]
        elif logits.shape[1] == len(all_ids):
            logits = logits[:, len(prompt_ids) - 1 : -1, :]
        elif logits.shape[1] > len(response_ids):
            logits = logits[:, -len(response_ids) - 1 : -1, :]
        else:
            logits = logits[:, -len(response_ids) :, :]
        log_probs = torch.log_softmax(logits.float() / args.teacher_temperature, dim=-1)
        targets = torch.tensor([response_ids], dtype=torch.long, device=args.device)
        selected = torch.gather(log_probs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        result = selected[0].detach().cpu().tolist()
    del input_ids, attention_mask, outputs, logits, log_probs, targets, selected
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [float(value) for value in result]


def load_teacher(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Full scoring requires torch and transformers. Run --dry-run locally "
            "or execute the full command in the training environment."
        ) from exc

    if not args.teacher_model:
        raise ValueError("--teacher-model is required unless --dry-run is used")
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.teacher_model, **model_kwargs)
    model.to(args.device)
    model.eval()
    return model, tokenizer, torch


def build_rows(
    cases: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    torch: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    from coursedistill.collator import CourseDistillDataCollator

    collator = CourseDistillDataCollator(
        tokenizer,
        max_length=args.max_seq_length,
        student_thinking=False,
        teacher_thinking=True,
    )
    cache_path = args.output_dir / "token_gain_cache.jsonl"
    cache = load_cache(cache_path)
    result: list[dict[str, Any]] = []

    with cache_path.open("a", encoding="utf-8") as cache_handle:
        for case_index, case in enumerate(cases, start=1):
            selected_index, encoded, outcomes = choose_representative_correct(
                tokenizer,
                case,
                args.max_completion_tokens,
            )
            masks = build_cpd_masks(tokenizer, encoded, outcomes, selected_index, args, torch)
            selected_row = case["correct"][selected_index]
            completion = str(selected_row.get("completion", ""))
            completion_ids = encoded[selected_index]
            train_row = case["train"]
            problem = str(
                train_row.get("student_input")
                or train_row.get("problem")
                or train_row.get("question")
                or ""
            )
            reference = str(train_row.get("reference_solution") or train_row.get("solution") or "")
            key = cache_key(case, completion, reference, args)

            if key in cache:
                gains = [float(value) for value in cache[key]["gains"]]
                cache_state = "cached"
            else:
                base_context, privileged_context = teacher_prompt(collator, problem, reference)
                privileged_logps = sampled_logps(
                    model,
                    tokenizer,
                    privileged_context,
                    completion_ids,
                    args,
                    torch,
                )
                base_logps = sampled_logps(
                    model,
                    tokenizer,
                    base_context,
                    completion_ids,
                    args,
                    torch,
                )
                gains = [
                    privileged - base
                    for privileged, base in zip(privileged_logps, base_logps)
                ]
                cache_row = {
                    "cache_key": key,
                    "case_key": case["key"],
                    "sample_id": case["sample_id"],
                    "step": case["step"],
                    "response_tokens": len(gains),
                    "gains": gains,
                }
                cache_handle.write(json.dumps(cache_row, ensure_ascii=False) + "\n")
                cache_handle.flush()
                cache[key] = cache_row
                cache_state = "scored"

            length = len(gains)
            local = {index for index in masks["local"] if index < length}
            answer = {index for index in masks["answer"] if index < length}
            full = set(range(length))
            shift_metrics = [
                score_mask(gains, shifted_indices(local, length, -abs(args.shift_tokens))),
                score_mask(gains, shifted_indices(local, length, abs(args.shift_tokens))),
            ]
            random_metrics = []
            for repeat in range(args.random_repeats):
                repeat_seed = int(
                    hashlib.sha256(
                        f"{args.seed}:{case['key']}:{repeat}".encode("utf-8")
                    ).hexdigest()[:16],
                    16,
                )
                random_mask = random_matched_indices(local, length, random.Random(repeat_seed))
                random_metrics.append(score_mask(gains, random_mask))

            strategy_metrics = {
                "cpd_local": score_mask(gains, local),
                "random_position_matched": average_metrics(random_metrics),
                "answer_only": score_mask(gains, answer),
                "shifted": average_metrics(shift_metrics),
                "full_trace": score_mask(gains, full),
            }
            for strategy, metrics in strategy_metrics.items():
                result.append(
                    {
                        "case_key": case["key"],
                        "sample_id": case["sample_id"],
                        "step": case["step"],
                        "current_p": case["current_p"],
                        "future_step": int(case["future_step"]),
                        "future_p": case["future_p"],
                        "future_gain": case["future_gain"],
                        "strategy": strategy,
                        "display_name": DISPLAY_NAMES[strategy],
                        "selector_score": metrics["positive_gain_density"],
                        **metrics,
                    }
                )
            print(
                f"[{case_index:4d}/{len(cases)}] {case['key']} "
                f"tokens={length} local={len(local)} {cache_state}",
                flush=True,
            )
    return result


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def standard_error(values: list[float]) -> float:
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values) / math.sqrt(len(values))


def tied_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and ordered[end][1] == ordered[cursor][1]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for index in range(cursor, end):
            ranks[ordered[index][0]] = rank
        cursor = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) < 2 or len(right) != len(left):
        return math.nan
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator > 0 else math.nan


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(tied_ranks(left), tied_ranks(right))


def select_top(rows: list[dict[str, Any]], fraction: float) -> list[dict[str, Any]]:
    count = max(1, math.ceil(len(rows) * fraction))
    return sorted(
        rows,
        key=lambda row: (-float(row["selector_score"]), str(row["case_key"])),
    )[:count]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_future(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, tuple[float, float]]:
    by_strategy: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    clusters: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_strategy[str(row["strategy"])][str(row["case_key"])] = row
        if str(row["case_key"]) not in clusters[str(row["sample_id"])]:
            clusters[str(row["sample_id"])].append(str(row["case_key"]))

    cluster_ids = sorted(clusters)
    draws: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(args.seed + 991)
    for _ in range(args.bootstrap_repeats):
        sampled_clusters = [rng.choice(cluster_ids) for _ in cluster_ids]
        sampled_keys: list[str] = []
        for cluster in sampled_clusters:
            sampled_keys.extend(clusters[cluster])
        local_rows = [by_strategy["cpd_local"][key] for key in sampled_keys]
        local_value = mean(
            [float(row["future_gain"]) for row in select_top(local_rows, args.top_fraction)]
        )
        draws["cpd_local"].append(local_value)
        for strategy in STRATEGIES:
            strategy_rows = [by_strategy[strategy][key] for key in sampled_keys]
            value = mean(
                [float(row["future_gain"]) for row in select_top(strategy_rows, args.top_fraction)]
            )
            draws[strategy].append(value)
            draws[f"diff:{strategy}"].append(local_value - value)

    result: dict[str, tuple[float, float]] = {}
    for key, values in draws.items():
        result[key] = (percentile(values, 0.025), percentile(values, 0.975))
    return result


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_strategy[str(row["strategy"])].append(row)

    descriptive: list[dict[str, Any]] = []
    predictive: list[dict[str, Any]] = []
    bootstrap = bootstrap_future(rows, args)
    local_top = mean(
        [
            float(row["future_gain"])
            for row in select_top(by_strategy["cpd_local"], args.top_fraction)
        ]
    )
    for strategy in STRATEGIES:
        strategy_rows = by_strategy[strategy]
        descriptive.append(
            {
                "strategy": strategy,
                "display_name": DISPLAY_NAMES[strategy],
                "observations": len(strategy_rows),
                "mean_token_fraction": mean(
                    [float(row["token_fraction"]) for row in strategy_rows]
                ),
                "mean_privileged_gain": mean(
                    [float(row["mean_gain"]) for row in strategy_rows]
                ),
                "positive_gain_density": mean(
                    [float(row["positive_gain_density"]) for row in strategy_rows]
                ),
                "positive_token_rate": mean(
                    [float(row["positive_token_rate"]) for row in strategy_rows]
                ),
                "positive_mass_capture": mean(
                    [float(row["positive_mass_capture"]) for row in strategy_rows]
                ),
            }
        )
        selected = select_top(strategy_rows, args.top_fraction)
        top_gain = mean([float(row["future_gain"]) for row in selected])
        gain_ci = bootstrap[strategy]
        diff_ci = bootstrap[f"diff:{strategy}"]
        predictive.append(
            {
                "strategy": strategy,
                "display_name": DISPLAY_NAMES[strategy],
                "observations": len(strategy_rows),
                "top_fraction": args.top_fraction,
                "top_count": len(selected),
                "spearman_future_gain": spearman(
                    [float(row["selector_score"]) for row in strategy_rows],
                    [float(row["future_gain"]) for row in strategy_rows],
                ),
                "top_future_gain": top_gain,
                "top_future_gain_se": standard_error(
                    [float(row["future_gain"]) for row in selected]
                ),
                "top_future_gain_ci_low": gain_ci[0],
                "top_future_gain_ci_high": gain_ci[1],
                "cpd_minus_strategy": local_top - top_gain,
                "difference_ci_low": diff_ci[0],
                "difference_ci_high": diff_ci[1],
            }
        )
    return descriptive, predictive


def format_float(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "--"
    return f"{number:.{digits}f}"


def write_latex_table(
    descriptive: list[dict[str, Any]],
    predictive: list[dict[str, Any]],
    path: Path,
) -> None:
    predictive_by_strategy = {row["strategy"]: row for row in predictive}
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Span rule & Tokens & $g^+$ dens. & Capture & Future $\Delta$ \\",
        r"\midrule",
    ]
    for row in descriptive:
        prediction = predictive_by_strategy[row["strategy"]]
        label = str(row["display_name"]).replace("_", r"\_")
        if row["strategy"] == "cpd_local":
            label = r"\textbf{CPD local}"
        lines.append(
            f"{label} & "
            f"{100.0 * float(row['mean_token_fraction']):.1f} & "
            f"{float(row['positive_gain_density']):.3f} & "
            f"{100.0 * float(row['positive_mass_capture']):.1f} & "
            f"{100.0 * float(prediction['top_future_gain']):+.2f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Offline span controls on saved Qwen3-1.7B CPD rollouts. "
            r"Tokens and capture are percentages. Future $\Delta$ is the "
            r"next-occurrence correctness gain for the top-30\% groups ranked "
            r"by positive privileged-gain density. No model is updated.}",
            r"\label{tab:offline_span_controls}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_report(
    descriptive: list[dict[str, Any]],
    predictive: list[dict[str, Any]],
    args: argparse.Namespace,
    path: Path,
) -> None:
    lines = [
        "# Offline span-control diagnostic",
        "",
        f"- Cases: {predictive[0]['observations']}",
        f"- Top-budget fraction: {args.top_fraction:.0%}",
        f"- Random controls per case: {args.random_repeats}",
        f"- Shift: +/-{args.shift_tokens} tokens, averaged",
        f"- Cluster bootstrap replicates: {args.bootstrap_repeats}",
        "- No student or teacher parameter is updated.",
        "",
        "## Span evidence",
        "",
        "| Rule | Token fraction | Positive-gain density | Positive-token rate | Positive-mass capture |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in descriptive:
        lines.append(
            f"| {row['display_name']} | "
            f"{100.0 * float(row['mean_token_fraction']):.1f}% | "
            f"{format_float(row['positive_gain_density'])} | "
            f"{100.0 * float(row['positive_token_rate']):.1f}% | "
            f"{100.0 * float(row['positive_mass_capture']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Independent future-progress check",
            "",
            "| Rule | Spearman | Top-budget future gain | 95% cluster CI | CPD minus rule | Difference CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in predictive:
        lines.append(
            f"| {row['display_name']} | "
            f"{format_float(row['spearman_future_gain'])} | "
            f"{100.0 * float(row['top_future_gain']):+.2f} pp | "
            f"[{100.0 * float(row['top_future_gain_ci_low']):+.2f}, "
            f"{100.0 * float(row['top_future_gain_ci_high']):+.2f}] | "
            f"{100.0 * float(row['cpd_minus_strategy']):+.2f} pp | "
            f"[{100.0 * float(row['difference_ci_low']):+.2f}, "
            f"{100.0 * float(row['difference_ci_high']):+.2f}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            "This is an offline localization diagnostic, not an end-to-end training "
            "ablation. Positive privileged gain measures how much reference "
            "information increases teacher support on selected tokens. The "
            "next-occurrence correctness gain is independent of that token-level "
            "score, but remains observational because no control rule is used to "
            "retrain the student.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_rows = load_training_data(args.train_file)
    future_pairs = load_future_pairs(args.future_pairs_csv)
    eligible = load_eligible_groups(args.run_dir, training_rows, future_pairs)
    if not eligible:
        raise RuntimeError("No mixed saved groups could be joined to future_pairs.csv")
    cases = choose_cases(eligible, args.samples, args.seed)
    write_case_manifest(cases, args.output_dir / "selected_cases.csv")

    metadata = {
        "run_dir": str(args.run_dir),
        "train_file": str(args.train_file),
        "future_pairs_csv": str(args.future_pairs_csv),
        "teacher_model": args.teacher_model,
        "eligible_observations": len(eligible),
        "selected_observations": len(cases),
        "unique_selected_problems": len({case["sample_id"] for case in cases}),
        "seed": args.seed,
        "random_repeats": args.random_repeats,
        "shift_tokens": args.shift_tokens,
        "top_fraction": args.top_fraction,
        "bootstrap_repeats": args.bootstrap_repeats,
        "teacher_temperature": args.teacher_temperature,
        "span_config": {
            "answer_span_tokens": args.answer_span_tokens,
            "diff_span_tokens": args.diff_span_tokens,
            "diff_span_top_k": args.diff_span_top_k,
            "support_span_tokens": args.support_span_tokens,
        },
        "dry_run": args.dry_run,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Eligible mixed observations: {len(eligible)}; "
        f"selected: {len(cases)}; "
        f"unique problems: {metadata['unique_selected_problems']}"
    )
    if args.dry_run:
        print(f"Dry run complete: {args.output_dir / 'selected_cases.csv'}")
        return

    model, tokenizer, torch = load_teacher(args)
    rows = build_rows(cases, model, tokenizer, torch, args)
    write_csv(rows, args.output_dir / "span_control_rows.csv")
    descriptive, predictive = summarize(rows, args)
    write_csv(descriptive, args.output_dir / "span_control_summary.csv")
    write_csv(predictive, args.output_dir / "future_predictive_summary.csv")
    write_latex_table(descriptive, predictive, args.output_dir / "table_span_controls.tex")
    write_report(descriptive, predictive, args, args.output_dir / "summary.md")
    print(f"Finished. Report: {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
