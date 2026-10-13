#!/usr/bin/env python
"""Analyze hardness/learnability buckets from saved online rollouts.

This is a lightweight, offline diagnostic for the CPD curriculum idea.  It
groups sampled rollouts by problem, computes the empirical correctness rate
``p`` and the CPD learnability score

    CLS = 4 * min(p, 1 - p) * (1 - p),

then compares which buckets would be selected by pass-only, pass-rate,
uncertainty, and CLS selectors under the same budget.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


BUCKETS = [
    "all-wrong",
    "hard-mixed",
    "balanced-mixed",
    "easy-mixed",
    "solved",
]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "correct"}
    return False


def correct(row: dict[str, Any]) -> bool:
    for key in ("outcome", "official_correct", "correct", "is_correct"):
        if key in row:
            return as_bool(row[key])
    for key in ("reward", "score"):
        if key in row:
            return as_bool(row[key])
    raise KeyError("No correctness field found in rollout row.")


def group_key(row: dict[str, Any]) -> str:
    for key in ("sample_id", "unique_id", "id"):
        value = row.get(key)
        if value is not None and str(value) != "":
            return str(value)
    return str(row.get("problem", "unknown"))


def step_from_file(path: Path, payload: Any) -> int:
    if isinstance(payload, dict) and payload.get("step") is not None:
        return int(payload["step"])
    stem = path.stem
    for part in reversed(stem.replace("-", "_").split("_")):
        if part.isdigit():
            return int(part)
    return -1


def load_rows(path: Path) -> tuple[int, list[dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        payload = json.loads(text)
        step = step_from_file(path, payload)
        if isinstance(payload, dict) and isinstance(payload.get("generations"), list):
            return step, payload["generations"]
        if isinstance(payload, list):
            return step, payload
        raise ValueError(f"Unsupported JSON payload in {path}")

    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return step_from_file(path, rows), rows


def bucket_name(p: float) -> str:
    if p <= 0.0:
        return "all-wrong"
    if p >= 1.0:
        return "solved"
    if p <= 0.25:
        return "hard-mixed"
    if p <= 0.75:
        return "balanced-mixed"
    return "easy-mixed"


def load_groups(paths: list[Path], num_samples: int) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for path in paths:
        step, rows = load_rows(path)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[group_key(row)].append(row)
        for key, samples in grouped.items():
            samples = samples[:num_samples] if num_samples > 0 else samples
            flags = [correct(row) for row in samples]
            if not flags:
                continue
            n = len(flags)
            c = sum(flags)
            p = c / n
            cls = 4.0 * min(p, 1.0 - p) * (1.0 - p)
            groups.append(
                {
                    "step": step,
                    "id": key,
                    "n": n,
                    "correct": c,
                    "wrong": n - c,
                    "p": p,
                    "bucket": bucket_name(p),
                    "pass_score": 1.0 if c > 0 else 0.0,
                    "pass_rate_score": p,
                    "uncertainty_score": 4.0 * p * (1.0 - p),
                    "cls_score": cls,
                    "pair_capacity": min(c, n - c),
                    "pair_capacity_norm": min(p, 1.0 - p),
                }
            )
    return groups


def select_top(groups: list[dict[str, Any]], score_key: str, budget: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    scored = [(float(group[score_key]), rng.random(), group) for group in groups]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:budget]]


def summarize(groups: list[dict[str, Any]]) -> dict[str, float]:
    if not groups:
        return {}
    return {
        "count": float(len(groups)),
        "anchor_rate": mean(1.0 if group["correct"] > 0 else 0.0 for group in groups),
        "mixed_rate": mean(1.0 if 0.0 < group["p"] < 1.0 else 0.0 for group in groups),
        "solved_rate": mean(1.0 if group["p"] >= 1.0 else 0.0 for group in groups),
        "all_wrong_rate": mean(1.0 if group["p"] <= 0.0 else 0.0 for group in groups),
        "mean_p": mean(group["p"] for group in groups),
        "mean_error_room": mean(1.0 - group["p"] for group in groups),
        "mean_pair_capacity": mean(group["pair_capacity"] for group in groups),
        "mean_pair_capacity_norm": mean(group["pair_capacity_norm"] for group in groups),
        "mean_cls": mean(group["cls_score"] for group in groups),
    }


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    out: dict[str, float] = {}
    for key in sorted(rows[0]):
        vals = [row[key] for row in rows]
        out[key] = mean(vals)
        out[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
    return out


def bucket_distribution(groups: list[dict[str, Any]]) -> dict[str, float]:
    total = max(1, len(groups))
    counts = Counter(group["bucket"] for group in groups)
    return {bucket: counts.get(bucket, 0) / total for bucket in BUCKETS}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    selectors = [row["selector"] for row in rows]
    bottoms = [0.0 for _ in rows]
    colors = {
        "all-wrong": "#B9B9B9",
        "hard-mixed": "#72B7B2",
        "balanced-mixed": "#4C78A8",
        "easy-mixed": "#F58518",
        "solved": "#E45756",
    }

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    xs = range(len(rows))
    for bucket in BUCKETS:
        vals = [float(row[bucket]) for row in rows]
        ax.bar(xs, vals, bottom=bottoms, color=colors[bucket], label=bucket)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_xticks(list(xs))
    ax.set_xticklabels(selectors, rotation=12, ha="right")
    ax.set_ylabel("Selected fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(ncol=3, frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CPD learnability buckets.")
    parser.add_argument("--generations_glob", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--top_fraction", type=float, default=0.30)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--tie_seeds", type=int, default=8)
    args = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(args.generations_glob))]
    if not paths:
        raise FileNotFoundError(f"No files matched {args.generations_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = load_groups(paths, args.num_samples)
    if not groups:
        raise ValueError("No grouped rollouts found.")
    budget = args.budget if args.budget > 0 else max(1, math.ceil(len(groups) * args.top_fraction))
    budget = min(budget, len(groups))

    with (output_dir / "groups.jsonl").open("w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group, ensure_ascii=False) + "\n")

    step_rows = []
    for step in sorted({group["step"] for group in groups}):
        step_groups = [group for group in groups if group["step"] == step]
        row = {"step": step, "count": len(step_groups)}
        row.update(bucket_distribution(step_groups))
        row.update({f"all_{k}": v for k, v in summarize(step_groups).items()})
        step_rows.append(row)
    write_csv(output_dir / "bucket_distribution_by_step.csv", step_rows)

    specs = [
        ("Pass-only", "pass_score", max(1, args.tie_seeds)),
        ("Pass-rate", "pass_rate_score", 1),
        ("Uncertainty", "uncertainty_score", 1),
        ("CLS (ours)", "cls_score", 1),
    ]

    metric_rows = []
    dist_rows = []
    for name, score_key, repeats in specs:
        summaries = []
        distributions = []
        for seed in range(repeats):
            selected = select_top(groups, score_key, budget, seed)
            summaries.append(summarize(selected))
            distributions.append(bucket_distribution(selected))
        metrics = aggregate(summaries)
        dist = aggregate(distributions)

        metric_row = {"selector": name, "budget": budget}
        metric_row.update(metrics)
        metric_rows.append(metric_row)

        dist_row = {"selector": name}
        for bucket in BUCKETS:
            dist_row[bucket] = dist[bucket]
        dist_rows.append(dist_row)

    write_csv(output_dir / "selector_metrics.csv", metric_rows)
    write_csv(output_dir / "selected_bucket_distribution.csv", dist_rows)
    make_plot(output_dir / "selected_bucket_distribution", dist_rows)

    md_lines = [
        "# Learnability Bucket Analysis",
        "",
        f"- Rollout files: {len(paths)}",
        f"- Problem groups: {len(groups)}",
        f"- Samples per problem: {args.num_samples}",
        f"- Selection budget: {budget} ({budget / len(groups):.1%})",
        "",
        "| Selector | Mixed ↑ | Solved ↓ | All-wrong ↓ | Error room ↑ | Pair cap. ↑ | CLS ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        md_lines.append(
            "| "
            f"{row['selector']} | "
            f"{row['mixed_rate']:.3f} | "
            f"{row['solved_rate']:.3f} | "
            f"{row['all_wrong_rate']:.3f} | "
            f"{row['mean_error_room']:.3f} | "
            f"{row['mean_pair_capacity_norm']:.3f} | "
            f"{row['mean_cls']:.3f} |"
        )
    md_lines += [
        "",
        "| Selector | All-wrong | Hard-mixed | Balanced-mixed | Easy-mixed | Solved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in dist_rows:
        md_lines.append(
            "| "
            f"{row['selector']} | "
            f"{row['all-wrong']:.3f} | "
            f"{row['hard-mixed']:.3f} | "
            f"{row['balanced-mixed']:.3f} | "
            f"{row['easy-mixed']:.3f} | "
            f"{row['solved']:.3f} |"
        )

    md_lines += [
        "",
        "Interpretation: a useful curriculum selector should avoid solved and "
        "all-wrong groups, while concentrating budget on mixed groups that provide "
        "both an anchor and remaining error room. CLS explicitly optimizes this "
        "learnability region.",
    ]
    (output_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
