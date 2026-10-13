#!/usr/bin/env python
"""Analyze pass-based vs CLS data selection from rollout generations.

Input JSONL should contain one row per sampled response and include a problem id
(`unique_id`, `id`, or `problem`) plus a correctness field (`correct`,
`official_correct`, `reward`, or `score`). The script groups responses by
problem, computes p = #correct / M, and compares selection rules under the same
budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
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
    raise KeyError(
        "No correctness field found. Expected one of: official_correct, correct, "
        "is_correct, reward, score."
    )


def _group_key(row: dict[str, Any]) -> str:
    for key in ("unique_id", "id", "sample_id"):
        value = row.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(row.get("problem", "unknown"))


def load_groups(path: Path, num_samples: int | None) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped[_group_key(row)].append(row)

    groups = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: int(row.get("sample_index", len(rows))))
        if num_samples is not None and num_samples > 0:
            rows = rows[:num_samples]
        flags = [_correct(row) for row in rows]
        total = len(flags)
        if total == 0:
            continue
        correct = sum(flags)
        p = correct / total
        groups.append(
            {
                "id": key,
                "n": total,
                "correct": correct,
                "wrong": total - correct,
                "p": p,
                "pass_score": 1.0 if correct > 0 else 0.0,
                "pass_rate_score": p,
                "uncertainty_score": 4.0 * p * (1.0 - p),
                "cls_score": 4.0 * min(p, 1.0 - p) * (1.0 - p),
            }
        )
    return groups


def select_top(groups: list[dict[str, Any]], score_key: str, budget: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    scored = [(float(group[score_key]), rng.random(), group) for group in groups]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:budget]]


def summarize(selected: list[dict[str, Any]]) -> dict[str, float]:
    if not selected:
        return {}
    values = {
        "selected_count": float(len(selected)),
        "anchor_rate": mean(1.0 if group["correct"] > 0 else 0.0 for group in selected),
        "contrastive_rate": mean(1.0 if 0.0 < group["p"] < 1.0 else 0.0 for group in selected),
        "grpo_valid_rate": mean(1.0 if 0.0 < group["p"] < 1.0 else 0.0 for group in selected),
        "solved_pass_rate": mean(1.0 if group["p"] == 1.0 else 0.0 for group in selected),
        "no_anchor_rate": mean(1.0 if group["p"] == 0.0 else 0.0 for group in selected),
        "mean_error_room": mean(1.0 - group["p"] for group in selected),
        "mean_p": mean(group["p"] for group in selected),
        "mean_pair_capacity": mean(min(group["correct"], group["wrong"]) for group in selected),
        "mean_pair_capacity_norm": mean(min(group["p"], 1.0 - group["p"]) for group in selected),
        "mean_cls_score": mean(group["cls_score"] for group in selected),
    }
    return values


def aggregate(metric_runs: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(metric_runs[0].keys())
    out = {}
    for key in keys:
        vals = [run[key] for run in metric_runs]
        out[key] = mean(vals)
        out[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
    return out


def p_histogram(selected: list[dict[str, Any]]) -> dict[str, int]:
    hist: dict[str, int] = defaultdict(int)
    for group in selected:
        hist[f"{group['correct']}/{group['n']}"] += 1
    return dict(sorted(hist.items(), key=lambda item: (int(item[0].split("/")[1]), int(item[0].split("/")[0]))))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare pass-based and CLS data selection.")
    parser.add_argument("--generations_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_fraction", type=float, default=0.30)
    parser.add_argument("--budget", type=int, default=0, help="Overrides top_fraction when >0.")
    parser.add_argument("--num_samples", type=int, default=0, help="Use at most this many samples per problem.")
    parser.add_argument("--tie_seeds", type=int, default=5, help="Tie-breaking repeats for binary pass score.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = load_groups(Path(args.generations_file), args.num_samples if args.num_samples > 0 else None)
    if not groups:
        raise ValueError("No groups loaded from generations_file.")

    budget = args.budget if args.budget > 0 else max(1, math.ceil(len(groups) * args.top_fraction))
    budget = min(budget, len(groups))

    specs = [
        ("Pass-only", "pass_score", max(1, args.tie_seeds)),
        ("Pass-rate p", "pass_rate_score", 1),
        ("Uncertainty 4p(1-p)", "uncertainty_score", 1),
        ("CLS (ours)", "cls_score", 1),
    ]

    metrics: dict[str, dict[str, float]] = {}
    hists: dict[str, dict[str, int]] = {}
    selected_examples: dict[str, list[str]] = {}

    for name, score_key, repeats in specs:
        runs = []
        first_selection = None
        for seed in range(repeats):
            selected = select_top(groups, score_key, budget, seed)
            if first_selection is None:
                first_selection = selected
            runs.append(summarize(selected))
        metrics[name] = aggregate(runs)
        hists[name] = p_histogram(first_selection or [])
        selected_examples[name] = [group["id"] for group in (first_selection or [])[:50]]

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_problems": len(groups),
                "budget": budget,
                "top_fraction": args.top_fraction,
                "metrics": metrics,
                "p_histogram": hists,
                "selected_examples_head": selected_examples,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    metric_rows = []
    for name, vals in metrics.items():
        row = {"selection": name}
        row.update(vals)
        metric_rows.append(row)
    write_csv(output_dir / "selection_metrics.csv", metric_rows)

    hist_keys = sorted({key for hist in hists.values() for key in hist})
    hist_rows = []
    for key in hist_keys:
        row = {"correct_over_total": key}
        for name in hists:
            row[name] = hists[name].get(key, 0)
        hist_rows.append(row)
    write_csv(output_dir / "p_histogram.csv", hist_rows)

    md_lines = [
        "# Pass-Based Selection Limitation Analysis",
        "",
        f"- Problems: {len(groups)}",
        f"- Selection budget: {budget}",
        "",
        "| Selection | Anchor ↑ | Contrastive/GRPO-valid ↑ | Solved-pass ↓ | Error room ↑ | Pair capacity ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, vals in metrics.items():
        md_lines.append(
            "| "
            f"{name} | "
            f"{vals['anchor_rate']:.4f} | "
            f"{vals['contrastive_rate']:.4f} | "
            f"{vals['solved_pass_rate']:.4f} | "
            f"{vals['mean_error_room']:.4f} | "
            f"{vals['mean_pair_capacity']:.4f} |"
        )
    md_lines += [
        "",
        "Interpretation: pass-only only checks whether a correct rollout exists. "
        "CLS additionally prefers problems with correct-wrong contrast and remaining error room, "
        "which are the data conditions needed by GRPO direction learning and span-level distillation.",
    ]
    (output_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
