"""Offline sensitivity analysis for the asymmetric CLS exponent.

The analysis re-ranks saved repeated-problem observations without updating a
model. Ties are averaged over fixed random seeds because rollout success rates
lie on a discrete grid.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("analysis/data/future_pairs.csv"),
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 1.5, 2.0],
    )
    parser.add_argument("--top-fraction", type=float, default=0.30)
    parser.add_argument("--tie-seeds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/outputs/cls_shape_sensitivity.csv"),
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def main() -> None:
    args = parse_args()
    with args.pairs.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["p"] = float(row["p"])
        row["gain"] = float(row["gain"])

    budget = max(1, math.ceil(args.top_fraction * len(rows)))
    output_rows: list[dict[str, float | int]] = []
    for gamma in args.gammas:
        trials: list[dict[str, float]] = []
        for offset in range(args.tie_seeds):
            rng = random.Random(args.seed + offset)
            scored = []
            for row in rows:
                p = float(row["p"])
                score = 4.0 * min(p, 1.0 - p) * (1.0 - p) ** gamma
                scored.append((score, rng.random(), row))
            selected = [
                item[2]
                for item in sorted(scored, key=lambda item: (-item[0], item[1]))[
                    :budget
                ]
            ]
            trials.append(
                {
                    "pair_capacity": 100.0
                    * mean([min(row["p"], 1.0 - row["p"]) for row in selected]),
                    "residual_error": 100.0
                    * mean([1.0 - row["p"] for row in selected]),
                    "hard_mixed": 100.0
                    * mean(
                        [1.0 if 0.0 < row["p"] <= 0.25 else 0.0 for row in selected]
                    ),
                    "easy_mixed": 100.0
                    * mean(
                        [1.0 if 0.75 < row["p"] < 1.0 else 0.0 for row in selected]
                    ),
                    "future_gain": 100.0
                    * mean([float(row["gain"]) for row in selected]),
                }
            )

        output_rows.append(
            {
                "gamma": gamma,
                "observations": len(rows),
                "selected": budget,
                **{
                    key: mean([trial[key] for trial in trials])
                    for key in trials[0]
                },
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    for row in output_rows:
        print(
            f"gamma={row['gamma']:.1f} pair={row['pair_capacity']:.2f} "
            f"error={row['residual_error']:.2f} "
            f"hard={row['hard_mixed']:.2f} easy={row['easy_mixed']:.2f} "
            f"future={row['future_gain']:+.2f}"
        )


if __name__ == "__main__":
    main()
