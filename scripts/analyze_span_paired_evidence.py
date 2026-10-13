#!/usr/bin/env python3
"""Paired cluster-bootstrap analysis for offline CPD span controls."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "positive_gain_density",
    "positive_mass_capture",
    "positive_token_rate",
    "mean_gain",
)

CONTROLS = (
    "random_position_matched",
    "shifted",
    "answer_only",
    "full_trace",
)

DISPLAY = {
    "cpd_local": "CPD local",
    "random_position_matched": "Random matched",
    "shifted": "Shifted local",
    "answer_only": "Answer only",
    "full_trace": "Full trace",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("span_control_rows.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("training_curves_recovered/span_offline_controls_paired"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260727)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_rows(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    cases: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            case_key = str(row["case_key"])
            strategy = str(row["strategy"])
            cases[case_key][strategy] = row
    required = {"cpd_local", *CONTROLS}
    incomplete = [key for key, rows in cases.items() if not required.issubset(rows)]
    if incomplete:
        raise RuntimeError(f"{len(incomplete)} cases do not contain all strategies")
    return cases


def paired_values(
    cases: dict[str, dict[str, dict[str, Any]]],
    control: str,
    metric: str,
) -> list[dict[str, Any]]:
    result = []
    for case_key, strategies in sorted(cases.items()):
        local = float(strategies["cpd_local"][metric])
        baseline = float(strategies[control][metric])
        result.append(
            {
                "case_key": case_key,
                "sample_id": str(strategies["cpd_local"]["sample_id"]),
                "local": local,
                "control": baseline,
                "difference": local - baseline,
            }
        )
    return result


def cluster_bootstrap(
    pairs: list[dict[str, Any]],
    repeats: int,
    seed: int,
) -> tuple[float, float, float]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for row in pairs:
        clusters[str(row["sample_id"])].append(float(row["difference"]))
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(repeats):
        sampled = [rng.choice(cluster_ids) for _ in cluster_ids]
        differences = [
            value
            for cluster_id in sampled
            for value in clusters[cluster_id]
        ]
        draws.append(statistics.fmean(differences))
    probability_positive = sum(value > 0.0 for value in draws) / len(draws)
    return percentile(draws, 0.025), percentile(draws, 0.975), probability_positive


def analyze(
    cases: dict[str, dict[str, dict[str, Any]]],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    output = []
    for metric_index, metric in enumerate(METRICS):
        for control_index, control in enumerate(CONTROLS):
            pairs = paired_values(cases, control, metric)
            local_values = [float(row["local"]) for row in pairs]
            control_values = [float(row["control"]) for row in pairs]
            differences = [float(row["difference"]) for row in pairs]
            local_mean = statistics.fmean(local_values)
            control_mean = statistics.fmean(control_values)
            difference = statistics.fmean(differences)
            ci_low, ci_high, probability_positive = cluster_bootstrap(
                pairs,
                repeats,
                seed + metric_index * 101 + control_index,
            )
            output.append(
                {
                    "metric": metric,
                    "control": control,
                    "control_name": DISPLAY[control],
                    "cases": len(pairs),
                    "unique_problems": len({row["sample_id"] for row in pairs}),
                    "cpd_mean": local_mean,
                    "control_mean": control_mean,
                    "paired_difference": difference,
                    "relative_lift_percent": (
                        100.0 * difference / abs(control_mean)
                        if abs(control_mean) > 1e-12
                        else math.nan
                    ),
                    "case_win_rate": sum(value > 0.0 for value in differences) / len(differences),
                    "case_tie_rate": sum(value == 0.0 for value in differences) / len(differences),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "bootstrap_probability_positive": probability_positive,
                }
            )
    return output


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def write_report(rows: list[dict[str, Any]], path: Path, repeats: int) -> None:
    selected = [
        row
        for row in rows
        if row["metric"] in {"positive_gain_density", "positive_mass_capture"}
        and row["control"] in {"random_position_matched", "shifted"}
    ]
    lines = [
        "# Paired offline span evidence",
        "",
        f"Paired by saved observation and clustered by problem ID with {repeats:,} bootstrap replicates.",
        "",
        "| Metric | Control | CPD | Control | Difference | Relative lift | 95% cluster CI | Case win rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        scale = 100.0 if row["metric"] == "positive_mass_capture" else 1.0
        suffix = " pp" if scale == 100.0 else ""
        lines.append(
            f"| {row['metric']} | {row['control_name']} | "
            f"{fmt(scale * row['cpd_mean'])} | {fmt(scale * row['control_mean'])} | "
            f"{scale * row['paired_difference']:+.4f}{suffix} | "
            f"{row['relative_lift_percent']:+.1f}% | "
            f"[{scale * row['ci_low']:+.4f}, {scale * row['ci_high']:+.4f}]{suffix} | "
            f"{100.0 * row['case_win_rate']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "A paired difference whose 95% interval lies entirely above zero supports",
            "greater reference-supported evidence under the same saved observations.",
            "This remains an offline localization diagnostic and is not an end-to-end",
            "training ablation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(rows: list[dict[str, Any]], path: Path) -> None:
    lookup = {(row["metric"], row["control"]): row for row in rows}
    random_density = lookup[("positive_gain_density", "random_position_matched")]
    shifted_density = lookup[("positive_gain_density", "shifted")]
    random_capture = lookup[("positive_mass_capture", "random_position_matched")]
    shifted_capture = lookup[("positive_mass_capture", "shifted")]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Control & CPD & Control & Paired $\Delta$ [95\% CI] \\",
        r"\midrule",
        (
            "Random density & "
            f"{random_density['cpd_mean']:.3f} & "
            f"{random_density['control_mean']:.3f} & "
            f"{random_density['paired_difference']:+.3f} "
            f"[{random_density['ci_low']:+.3f}, {random_density['ci_high']:+.3f}] \\\\"
        ),
        (
            "Shifted density & "
            f"{shifted_density['cpd_mean']:.3f} & "
            f"{shifted_density['control_mean']:.3f} & "
            f"{shifted_density['paired_difference']:+.3f} "
            f"[{shifted_density['ci_low']:+.3f}, {shifted_density['ci_high']:+.3f}] \\\\"
        ),
        (
            "Random capture & "
            f"{100.0 * random_capture['cpd_mean']:.1f} & "
            f"{100.0 * random_capture['control_mean']:.1f} & "
            f"{100.0 * random_capture['paired_difference']:+.1f} "
            f"[{100.0 * random_capture['ci_low']:+.1f}, "
            f"{100.0 * random_capture['ci_high']:+.1f}] \\\\"
        ),
        (
            "Shifted capture & "
            f"{100.0 * shifted_capture['cpd_mean']:.1f} & "
            f"{100.0 * shifted_capture['control_mean']:.1f} & "
            f"{100.0 * shifted_capture['paired_difference']:+.1f} "
            f"[{100.0 * shifted_capture['ci_low']:+.1f}, "
            f"{100.0 * shifted_capture['ci_high']:+.1f}] \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Paired offline controls for CPD span localization. Density is",
        r"positive privileged gain per selected token; capture is reported in",
        r"percentage points. Intervals use problem-clustered bootstrap resampling.}",
        r"\label{tab:paired_span_evidence}",
        r"\end{table}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_rows(args.input)
    rows = analyze(cases, args.bootstrap_repeats, args.seed)
    write_csv(rows, args.output_dir / "paired_span_evidence.csv")
    write_report(rows, args.output_dir / "paired_span_evidence.md", args.bootstrap_repeats)
    write_latex(rows, args.output_dir / "table_paired_span_evidence.tex")
    metadata = {
        "input": str(args.input),
        "cases": len(cases),
        "unique_problems": len(
            {
                strategies["cpd_local"]["sample_id"]
                for strategies in cases.values()
            }
        ),
        "bootstrap_repeats": args.bootstrap_repeats,
        "seed": args.seed,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(args.output_dir / "paired_span_evidence.md")


if __name__ == "__main__":
    main()
