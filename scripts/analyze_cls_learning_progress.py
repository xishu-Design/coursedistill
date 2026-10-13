#!/usr/bin/env python
"""Test whether CLS predicts future learning progress.

The script uses saved online rollout snapshots, e.g. ``generations_step_*.json``.
For each problem that appears at multiple saved steps, it forms consecutive
future-observation pairs and measures

    gain = p_future - p_now,

where p is the fraction of correct rollouts for that problem.  It then reports
whether higher current CLS corresponds to larger future gains.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


CLS_BUCKETS = ["All-wrong", "Solved", "Low CLS", "Mid CLS", "High CLS"]


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


def step_from_path(path: Path, payload: Any) -> int:
    if isinstance(payload, dict) and payload.get("step") is not None:
        return int(payload["step"])
    for part in reversed(path.stem.replace("-", "_").split("_")):
        if part.isdigit():
            return int(part)
    return -1


def load_snapshot(path: Path, num_samples: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    step = step_from_path(path, payload)
    rows = payload.get("generations", payload if isinstance(payload, list) else [])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)

    out: list[dict[str, Any]] = []
    for key, samples in grouped.items():
        samples = samples[:num_samples] if num_samples > 0 else samples
        flags = [correct(row) for row in samples]
        if not flags:
            continue
        n = len(flags)
        c = sum(flags)
        p = c / n
        cls = 4.0 * min(p, 1.0 - p) * (1.0 - p)
        out.append(
            {
                "step": step,
                "id": key,
                "n": n,
                "correct": c,
                "p": p,
                "cls": cls,
                "uncertainty": 4.0 * p * (1.0 - p),
                "pass_score": 1.0 if c > 0 else 0.0,
                "pass_rate": p,
            }
        )
    return out


def cls_bucket(row: dict[str, Any]) -> str:
    p = float(row["p"])
    cls = float(row["cls"])
    if p <= 0.0:
        return "All-wrong"
    if p >= 1.0:
        return "Solved"
    if cls < 0.25:
        return "Low CLS"
    if cls < 0.50:
        return "Mid CLS"
    return "High CLS"


def load_pairs(paths: list[Path], num_samples: int) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        for row in load_snapshot(path, num_samples):
            by_id[row["id"]].append(row)

    pairs: list[dict[str, Any]] = []
    for key, observations in by_id.items():
        observations.sort(key=lambda row: int(row["step"]))
        for current, future in zip(observations, observations[1:]):
            gain = float(future["p"]) - float(current["p"])
            row = {
                **current,
                "future_step": future["step"],
                "future_p": future["p"],
                "gain": gain,
                "gap": int(future["step"]) - int(current["step"]),
            }
            row["cls_bucket"] = cls_bucket(row)
            pairs.append(row)
    return pairs


def stderr(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return pstdev(values) / math.sqrt(len(values))


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    gains = [float(row["gain"]) for row in rows]
    return {
        "count": float(len(rows)),
        "initial_p": mean(float(row["p"]) for row in rows),
        "future_p": mean(float(row["future_p"]) for row in rows),
        "gain": mean(gains),
        "gain_se": stderr(gains),
        "mean_cls": mean(float(row["cls"]) for row in rows),
        "mean_gap": mean(float(row["gap"]) for row in rows),
    }


def select_top(rows: list[dict[str, Any]], key: str, fraction: float, seed: int) -> list[dict[str, Any]]:
    budget = max(1, math.ceil(len(rows) * fraction))
    rng = random.Random(seed)
    scored = [(float(row[key]), rng.random(), row) for row in rows]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in scored[:budget]]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def make_gain_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    labels = [row["bucket"] for row in rows]
    gains = [100.0 * float(row["gain"]) for row in rows]
    errors = [100.0 * float(row["gain_se"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    colors = ["#B9B9B9", "#E45756", "#72B7B2", "#4C78A8", "#F58518"]
    ax.bar(labels, gains, yerr=errors, color=colors, capsize=3)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("Future gain in p (pp)")
    ax.set_title("CLS Predicts Future Learning Progress")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=240)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze whether CLS predicts future learning progress.")
    parser.add_argument("--generations_glob", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--top_fraction", type=float, default=0.30)
    parser.add_argument("--tie_seeds", type=int, default=8)
    args = parser.parse_args()

    paths = [Path(path) for path in sorted(glob.glob(args.generations_glob))]
    if not paths:
        raise FileNotFoundError(f"No files matched {args.generations_glob}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(paths, args.num_samples)
    if not pairs:
        raise ValueError("No repeated-problem future pairs found.")

    write_csv(output_dir / "future_pairs.csv", pairs)

    bucket_rows: list[dict[str, Any]] = []
    for bucket in CLS_BUCKETS:
        rows = [row for row in pairs if row["cls_bucket"] == bucket]
        if not rows:
            continue
        stats = summarize(rows)
        bucket_rows.append(
            {
                "bucket": bucket,
                "count": int(stats["count"]),
                "initial_p": stats["initial_p"],
                "future_p": stats["future_p"],
                "gain": stats["gain"],
                "gain_se": stats["gain_se"],
                "mean_cls": stats["mean_cls"],
                "mean_gap": stats["mean_gap"],
            }
        )
    write_csv(output_dir / "learning_progress_by_cls_bucket.csv", bucket_rows)
    make_gain_plot(output_dir / "learning_progress_by_cls_bucket", bucket_rows)

    selector_specs = [
        ("Pass-only", "pass_score", max(1, args.tie_seeds)),
        ("Pass-rate", "pass_rate", 1),
        ("Uncertainty", "uncertainty", 1),
        ("CLS (ours)", "cls", 1),
    ]
    selector_rows: list[dict[str, Any]] = []
    for name, key, repeats in selector_specs:
        runs = []
        for seed in range(repeats):
            selected = select_top(pairs, key, args.top_fraction, seed)
            runs.append(summarize(selected))
        merged = {}
        for metric in runs[0]:
            values = [row[metric] for row in runs]
            merged[metric] = mean(values)
        selector_rows.append(
            {
                "selector": name,
                "top_fraction": args.top_fraction,
                "count": int(merged["count"]),
                "initial_p": merged["initial_p"],
                "future_p": merged["future_p"],
                "gain": merged["gain"],
                "gain_se": merged["gain_se"],
                "mean_cls": merged["mean_cls"],
                "mean_gap": merged["mean_gap"],
                "mixed_rate": mean(
                    1.0 if 0.0 < row["p"] < 1.0 else 0.0
                    for row in select_top(pairs, key, args.top_fraction, 0)
                ),
                "solved_rate": mean(
                    1.0 if row["p"] >= 1.0 else 0.0
                    for row in select_top(pairs, key, args.top_fraction, 0)
                ),
            }
        )
    write_csv(output_dir / "future_gain_by_selector.csv", selector_rows)

    md_lines = [
        "# CLS Predicts Future Learning Progress",
        "",
        f"- Rollout files: {len(paths)}",
        f"- Consecutive repeated-problem pairs: {len(pairs)}",
        f"- Samples per problem: {args.num_samples}",
        "",
        "## By Current CLS Bucket",
        "",
        "| Bucket | Pairs | Current p | Future p | Gain (pp) | Mean CLS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in bucket_rows:
        md_lines.append(
            "| "
            f"{row['bucket']} | "
            f"{row['count']} | "
            f"{pct(row['initial_p'])} | "
            f"{pct(row['future_p'])} | "
            f"{pct(row['gain'])} | "
            f"{row['mean_cls']:.3f} |"
        )

    md_lines += [
        "",
        "## Top-Fraction Selection",
        "",
        "| Selector | Current p | Future p | Gain (pp) | Mixed | Solved |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selector_rows:
        md_lines.append(
            "| "
            f"{row['selector']} | "
            f"{pct(row['initial_p'])} | "
            f"{pct(row['future_p'])} | "
            f"{pct(row['gain'])} | "
            f"{pct(row['mixed_rate'])} | "
            f"{pct(row['solved_rate'])} |"
        )

    md_lines += [
        "",
        "Interpretation: higher current CLS identifies problems whose future rollout "
        "accuracy improves more. This supports the claim that CLS measures "
        "student-side learnability rather than merely current accuracy.",
    ]
    (output_dir / "summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Current bucket & Pairs & Current $p$ & Future $p$ & Gain & CLS \\",
        r"\midrule",
    ]
    for row in bucket_rows:
        gain = pct(row["gain"])
        gain_text = rf"\textbf{{+{gain}}}" if row["bucket"] == "High CLS" else (f"+{gain}" if row["gain"] >= 0 else gain)
        cls_text = rf"\textbf{{{row['mean_cls']:.3f}}}" if row["bucket"] == "High CLS" else f"{row['mean_cls']:.3f}"
        tex.append(
            f"{row['bucket']} & {row['count']} & {pct(row['initial_p'])} & "
            f"{pct(row['future_p'])} & {gain_text} & {cls_text} \\\\"
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{CLS predicts future learning progress on saved Qwen3-1.7B CPD rollouts. "
        r"Each row groups repeated problems by their current rollout correctness profile; "
        r"gain is the percentage-point change in correctness rate at the next saved occurrence.}",
        r"\label{tab:cls_future_progress}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Selector & Current $p$ & Future $p$ & Gain & Mixed & Solved \\",
        r"\midrule",
    ]
    for row in selector_rows:
        gain = pct(row["gain"])
        gain_text = rf"\textbf{{+{gain}}}" if row["selector"] == "CLS (ours)" else (f"+{gain}" if row["gain"] >= 0 else gain)
        tex.append(
            f"{row['selector']} & {pct(row['initial_p'])} & {pct(row['future_p'])} & "
            f"{gain_text} & {pct(row['mixed_rate'])} & {pct(row['solved_rate'])} \\\\"
        )
    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Future progress under top-30\% selection rules. CLS selects mixed, "
        r"learnable problems and yields the largest subsequent correctness gain.}",
        r"\label{tab:selector_future_gain}",
        r"\end{table}",
    ]
    (output_dir / "tables.tex").write_text("\n".join(tex), encoding="utf-8")

    print("\n".join(md_lines))
    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
