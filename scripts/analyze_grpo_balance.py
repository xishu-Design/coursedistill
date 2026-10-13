#!/usr/bin/env python
"""Show why balanced correct/wrong rollouts give stronger GRPO signal.

For binary rewards inside a group of M rollouts, the amount of usable
group-relative signal is controlled by the correct/wrong balance. This script
reports both observed rollout buckets and the theoretical curve:

  pair_count = C * W
  balanced_pair_capacity = min(C, W)
  reward_variance = p(1-p)
  mean_abs_normalized_advantage = 2 * sqrt(p(1-p))

All quantities peak when the group has a balanced number of correct and wrong
rollouts, and collapse when all rollouts are correct or all are wrong.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "correct"}
    return False


def _correct(row: dict[str, Any]) -> bool:
    for key in ("official_correct", "correct", "is_correct", "outcome"):
        if key in row:
            return _as_bool(row[key])
    for key in ("reward", "score"):
        if key in row:
            return _as_bool(row[key])
    raise KeyError("No correctness field found.")


def _group_key(row: dict[str, Any]) -> str:
    for key in ("unique_id", "id", "sample_id"):
        value = row.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(row.get("problem", "unknown"))


def load_group_counts(path: Path, num_samples: int | None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped[_group_key(row)].append(row)

    counts = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row.get("sample_index", len(rows))))
        if num_samples is not None and num_samples > 0:
            rows = rows[:num_samples]
        flags = [_correct(row) for row in rows]
        total = len(flags)
        if total == 0:
            continue
        correct = sum(flags)
        wrong = total - correct
        p = correct / total
        counts.append({"id": key, "n": total, "correct": correct, "wrong": wrong, "p": p})
    return counts


def signal_stats(correct: int, total: int) -> dict[str, float]:
    wrong = total - correct
    p = correct / total if total > 0 else 0.0
    variance = p * (1.0 - p)
    return {
        "n": float(total),
        "correct": float(correct),
        "wrong": float(wrong),
        "p": p,
        "pair_count": float(correct * wrong),
        "pair_density": 4.0 * p * (1.0 - p),
        "balanced_pair_capacity": float(min(correct, wrong)),
        "balanced_pair_capacity_norm": min(p, 1.0 - p),
        "reward_variance": variance,
        "reward_std": math.sqrt(variance),
        "mean_abs_normalized_advantage": 2.0 * math.sqrt(variance) if 0.0 < p < 1.0 else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze correct/wrong balance and GRPO signal.")
    parser.add_argument("--generations_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument(
        "--theory_m",
        type=int,
        default=0,
        help="M for theoretical curve. Defaults to observed median group size.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = load_group_counts(Path(args.generations_file), args.num_samples if args.num_samples > 0 else None)
    if not groups:
        raise ValueError("No groups loaded from generations_file.")

    observed_by_bucket: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        observed_by_bucket[(int(group["n"]), int(group["correct"]))].append(group)

    observed_rows = []
    for (total, correct), bucket in sorted(observed_by_bucket.items()):
        row = signal_stats(correct, total)
        row["num_problems"] = len(bucket)
        row["observed_fraction"] = len(bucket) / len(groups)
        observed_rows.append(row)

    sorted_ns = sorted(int(group["n"]) for group in groups)
    default_m = sorted_ns[len(sorted_ns) // 2]
    theory_m = args.theory_m if args.theory_m > 0 else default_m
    theory_rows = []
    for correct in range(theory_m + 1):
        row = signal_stats(correct, theory_m)
        row["num_problems"] = ""
        row["observed_fraction"] = ""
        theory_rows.append(row)

    write_csv(output_dir / "observed_balance_by_correct_count.csv", observed_rows)
    write_csv(output_dir / "theoretical_balance_curve.csv", theory_rows)

    aggregate = {
        "num_problems": len(groups),
        "mean_p": mean(group["p"] for group in groups),
        "mean_pair_density": mean(4.0 * group["p"] * (1.0 - group["p"]) for group in groups),
        "mean_balanced_pair_capacity": mean(min(group["correct"], group["wrong"]) for group in groups),
        "mean_abs_normalized_advantage": mean(
            2.0 * math.sqrt(group["p"] * (1.0 - group["p"])) if 0.0 < group["p"] < 1.0 else 0.0
            for group in groups
        ),
        "all_wrong_rate": mean(1.0 if group["correct"] == 0 else 0.0 for group in groups),
        "all_correct_rate": mean(1.0 if group["wrong"] == 0 else 0.0 for group in groups),
        "mixed_rate": mean(1.0 if group["correct"] > 0 and group["wrong"] > 0 else 0.0 for group in groups),
        "theory_m": theory_m,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    best_theory = max(theory_rows, key=lambda row: row["mean_abs_normalized_advantage"])
    md_lines = [
        "# Correct/Wrong Balance and GRPO Signal",
        "",
        f"- Problems: {len(groups)}",
        f"- Theory M: {theory_m}",
        f"- Observed mixed rate: {aggregate['mixed_rate']:.4f}",
        f"- Observed all-wrong rate: {aggregate['all_wrong_rate']:.4f}",
        f"- Observed all-correct rate: {aggregate['all_correct_rate']:.4f}",
        "",
        "For binary rewards, all useful signal terms peak when correct and wrong rollouts are balanced:",
        "",
        "| Correct | Wrong | Pair count C*W | Pair density 4p(1-p) | Mean abs norm advantage |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in theory_rows:
        if row["correct"] in {0, 1, math.floor(theory_m / 4), math.floor(theory_m / 2), math.ceil(3 * theory_m / 4), theory_m - 1, theory_m}:
            md_lines.append(
                "| "
                f"{int(row['correct'])} | {int(row['wrong'])} | "
                f"{row['pair_count']:.0f} | {row['pair_density']:.4f} | "
                f"{row['mean_abs_normalized_advantage']:.4f} |"
            )
    md_lines += [
        "",
        f"Peak mean absolute normalized advantage occurs near C={int(best_theory['correct'])}, "
        f"W={int(best_theory['wrong'])}. This is why balanced correct/wrong groups are more useful "
        "for GRPO-style relative updates than all-correct/all-wrong groups.",
    ]
    (output_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
