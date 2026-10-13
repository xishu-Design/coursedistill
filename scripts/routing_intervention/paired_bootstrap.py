#!/usr/bin/env python3
"""Paired, problem-clustered comparison of short routing interventions."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_problem_scores(path: Path, expected_samples: int) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("unique_id") or row.get("problem_id") or row.get("id")
            if key is None:
                raise ValueError(f"{path}:{line_no} has no problem identifier")
            grouped[str(key)].append(float(bool(row["correct"])))
    bad = {key: len(values) for key, values in grouped.items() if len(values) != expected_samples}
    if bad:
        preview = list(bad.items())[:5]
        raise ValueError(f"Expected {expected_samples} samples per problem in {path}; examples: {preview}")
    return {key: mean(values) for key, values in grouped.items()}


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def ci(values: list[float]) -> list[float]:
    return [percentile(values, 0.025), percentile(values, 0.975)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--cls", type=Path, required=True)
    parser.add_argument("--uncertainty", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    scores = {
        "start": load_problem_scores(args.start, args.num_samples),
        "cls": load_problem_scores(args.cls, args.num_samples),
        "uncertainty": load_problem_scores(args.uncertainty, args.num_samples),
    }
    ids = set(scores["start"])
    for name in ("cls", "uncertainty"):
        if set(scores[name]) != ids:
            raise ValueError(f"Problem IDs differ between start and {name}")
    ids = sorted(ids)

    def statistic(sampled_ids: list[str]) -> dict[str, float]:
        start = mean(scores["start"][key] for key in sampled_ids)
        cls = mean(scores["cls"][key] for key in sampled_ids)
        uncertainty = mean(scores["uncertainty"][key] for key in sampled_ids)
        return {
            "start": 100 * start,
            "cls": 100 * cls,
            "uncertainty": 100 * uncertainty,
            "cls_gain": 100 * (cls - start),
            "uncertainty_gain": 100 * (uncertainty - start),
            "cls_minus_uncertainty": 100 * (cls - uncertainty),
        }

    observed = statistic(ids)
    rng = random.Random(args.seed)
    boot = {key: [] for key in observed}
    for _ in range(args.replicates):
        sampled = [ids[rng.randrange(len(ids))] for _ in ids]
        values = statistic(sampled)
        for key, value in values.items():
            boot[key].append(value)
    result = {
        "problems": len(ids),
        "samples_per_problem": args.num_samples,
        "bootstrap_replicates": args.replicates,
        "seed": args.seed,
        "observed_pp": observed,
        "ci95_pp": {key: ci(values) for key, values in boot.items()},
        "bootstrap_probability_cls_gt_uncertainty": mean(
            value > 0 for value in boot["cls_minus_uncertainty"]
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    def row(label: str, key: str) -> str:
        low, high = result["ci95_pp"][key]
        return f"| {label} | {observed[key]:+.2f} | [{low:+.2f}, {high:+.2f}] |"

    lines = [
        "# Short-horizon routing intervention",
        "",
        f"Paired over {len(ids)} problems with Avg@{args.num_samples}; intervals use "
        f"{args.replicates:,} problem-cluster bootstrap replicates.",
        "",
        "| Quantity | Estimate (pp) | 95% paired CI |",
        "|---|---:|---:|",
        row("Start checkpoint", "start"),
        row("CLS endpoint", "cls"),
        row("Uncertainty endpoint", "uncertainty"),
        row("CLS gain from start", "cls_gain"),
        row("Uncertainty gain from start", "uncertainty_gain"),
        row("CLS minus uncertainty", "cls_minus_uncertainty"),
        "",
        "This is a short-horizon intervention, not a replacement for a full-training baseline.",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
