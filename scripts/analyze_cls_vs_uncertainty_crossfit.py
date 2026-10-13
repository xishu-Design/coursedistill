"""Cross-fitted, training-free audit of CLS versus symmetric uncertainty.

The audit uses repeated problems in saved online rollout snapshots.  At each
current occurrence, the M outcomes are split in half.  One half determines
the selector score, while the held-out half estimates current correctness.
The next saved occurrence supplies future correctness.  This avoids reusing
the same Bernoulli samples for selection and for the gain baseline.

Two complementary tests are reported:

1. Equal-budget selector comparison within each saved step, including the
   disagreement sets selected by only CLS or only symmetric uncertainty.
2. A mirror test comparing p and 1-p groups, which have identical symmetric
   uncertainty, with optional exact matching by saved step.

Confidence intervals use a cluster bootstrap over problem IDs, so repeated
occurrences of the same problem are resampled together.
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
from statistics import mean
from typing import Any, Iterable


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) > 0.0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "correct"}
    return False


def correctness(row: dict[str, Any]) -> bool:
    for key in ("outcome", "official_correct", "correct", "is_correct"):
        if key in row:
            return as_bool(row[key])
    for key in ("reward", "score"):
        if key in row:
            return as_bool(row[key])
    raise KeyError("No correctness field found in rollout row")


def problem_id(row: dict[str, Any]) -> str:
    for key in ("sample_id", "unique_id", "id"):
        value = row.get(key)
        if value is not None and str(value):
            return str(value)
    return str(row.get("problem", "unknown"))


def snapshot_step(path: Path, payload: Any) -> int:
    if isinstance(payload, dict) and payload.get("step") is not None:
        return int(payload["step"])
    for part in reversed(path.stem.replace("-", "_").split("_")):
        if part.isdigit():
            return int(part)
    raise ValueError(f"Cannot infer step from {path}")


def load_observations(
    paths: list[Path], num_samples: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    observations: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            skipped.append({"path": str(path), "reason": str(error)})
            continue
        step = snapshot_step(path, payload)
        rows = payload.get("generations", payload if isinstance(payload, list) else [])
        grouped: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            grouped[problem_id(row)].append(correctness(row))
        for pid, flags in grouped.items():
            if len(flags) < num_samples:
                continue
            observations.append(
                {
                    "problem_id": pid,
                    "step": step,
                    "flags": tuple(flags[:num_samples]),
                }
            )
    return observations, skipped


def make_pairs(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        by_problem[row["problem_id"]].append(row)

    pairs: list[dict[str, Any]] = []
    for pid, rows in by_problem.items():
        rows.sort(key=lambda item: int(item["step"]))
        for pair_index, (current, future) in enumerate(zip(rows, rows[1:])):
            pairs.append(
                {
                    "problem_id": pid,
                    "cluster_id": pid,
                    "pair_id": f"{pid}:{current['step']}:{future['step']}:{pair_index}",
                    "current_step": int(current["step"]),
                    "future_step": int(future["step"]),
                    "gap": int(future["step"]) - int(current["step"]),
                    "current_flags": current["flags"],
                    "future_p": mean(float(flag) for flag in future["flags"]),
                }
            )
    return pairs


def load_pairs_csv(path: Path, num_samples: int) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_index, row in enumerate(csv.DictReader(handle)):
            n = int(row["n"])
            correct_count = int(row["correct"])
            if n != num_samples:
                continue
            pid = str(row["id"])
            current_step = int(row["step"])
            future_step = int(row["future_step"])
            pairs.append(
                {
                    "problem_id": pid,
                    "cluster_id": pid,
                    "pair_id": f"{pid}:{current_step}:{future_step}:{row_index}",
                    "current_step": current_step,
                    "future_step": future_step,
                    "gap": future_step - current_step,
                    # Randomly splitting this multiset is exactly a hypergeometric
                    # split of the original binary verifier outcomes.
                    "current_flags": tuple([True] * correct_count + [False] * (n - correct_count)),
                    "future_p": float(row["future_p"]),
                }
            )
    return pairs


def stable_rng(seed: int, *parts: Any) -> random.Random:
    token = ":".join([str(seed), *(str(part) for part in parts)])
    return random.Random(token)


def crossfit_rows(pairs: list[dict[str, Any]], seed: int, split_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        flags = list(pair["current_flags"])
        rng = stable_rng(seed, pair["pair_id"], pair.get("bootstrap_copy", 0))
        rng.shuffle(flags)
        scoring = flags[:split_size]
        holdout = flags[split_size : 2 * split_size]
        if len(scoring) != split_size or len(holdout) != split_size:
            continue
        p_score = mean(float(flag) for flag in scoring)
        p_holdout = mean(float(flag) for flag in holdout)
        uncertainty = 4.0 * p_score * (1.0 - p_score)
        cls = 4.0 * min(p_score, 1.0 - p_score) * (1.0 - p_score)
        rows.append(
            {
                **pair,
                "p_score": p_score,
                "p_holdout": p_holdout,
                "future_gain": float(pair["future_p"]) - p_holdout,
                "uncertainty": uncertainty,
                "cls": cls,
                "tie": stable_rng(seed, "tie", pair["pair_id"], pair.get("bootstrap_copy", 0)).random(),
            }
        )
    return rows


def selected_ids(rows: list[dict[str, Any]], score_key: str, fraction: float) -> set[str]:
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_step[int(row["current_step"])].append(row)

    selected: set[str] = set()
    for step_rows in by_step.values():
        eligible = [row for row in step_rows if 0.0 < float(row["p_score"]) < 1.0]
        if not eligible:
            continue
        budget = min(len(eligible), max(1, math.ceil(len(step_rows) * fraction)))
        eligible.sort(key=lambda row: (-float(row[score_key]), float(row["tie"])))
        selected.update(str(row["pair_id"]) for row in eligible[:budget])
    return selected


def subset_summary(rows: list[dict[str, Any]], ids: set[str]) -> dict[str, float]:
    chosen = [row for row in rows if str(row["pair_id"]) in ids]
    if not chosen:
        return {"count": 0.0, "p_holdout": math.nan, "future_p": math.nan, "gain": math.nan}
    return {
        "count": float(len(chosen)),
        "p_holdout": mean(float(row["p_holdout"]) for row in chosen),
        "future_p": mean(float(row["future_p"]) for row in chosen),
        "gain": mean(float(row["future_gain"]) for row in chosen),
    }


def selector_stat(rows: list[dict[str, Any]], fraction: float) -> dict[str, float]:
    cls_ids = selected_ids(rows, "cls", fraction)
    unc_ids = selected_ids(rows, "uncertainty", fraction)
    common = cls_ids & unc_ids
    cls_only = cls_ids - unc_ids
    unc_only = unc_ids - cls_ids
    cls_stats = subset_summary(rows, cls_ids)
    unc_stats = subset_summary(rows, unc_ids)
    cls_only_stats = subset_summary(rows, cls_only)
    unc_only_stats = subset_summary(rows, unc_only)

    return {
        "cls_count": cls_stats["count"],
        "uncertainty_count": unc_stats["count"],
        "common_count": float(len(common)),
        "cls_only_count": cls_only_stats["count"],
        "uncertainty_only_count": unc_only_stats["count"],
        "cls_gain": cls_stats["gain"],
        "uncertainty_gain": unc_stats["gain"],
        "selected_gain_difference": cls_stats["gain"] - unc_stats["gain"],
        "cls_only_gain": cls_only_stats["gain"],
        "uncertainty_only_gain": unc_only_stats["gain"],
        "disagreement_gain_difference": cls_only_stats["gain"] - unc_only_stats["gain"],
        "cls_holdout_p": cls_stats["p_holdout"],
        "uncertainty_holdout_p": unc_stats["p_holdout"],
        "cls_future_p": cls_stats["future_p"],
        "uncertainty_future_p": unc_stats["future_p"],
    }


def mirror_stat(rows: list[dict[str, Any]], match_step: bool) -> dict[str, float]:
    blocks: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(
        lambda: {"hard": [], "easy": []}
    )
    for row in rows:
        p = float(row["p_score"])
        if not (0.0 < p < 1.0) or math.isclose(p, 0.5):
            continue
        mirror_p = round(min(p, 1.0 - p), 8)
        key = (int(row["current_step"]), mirror_p) if match_step else (mirror_p,)
        side = "hard" if p < 0.5 else "easy"
        blocks[key][side].append(float(row["future_gain"]))

    weighted_difference = 0.0
    total_weight = 0.0
    hard_values: list[float] = []
    easy_values: list[float] = []
    matched_blocks = 0
    for sides in blocks.values():
        hard = sides["hard"]
        easy = sides["easy"]
        if not hard or not easy:
            continue
        weight = float(min(len(hard), len(easy)))
        weighted_difference += weight * (mean(hard) - mean(easy))
        total_weight += weight
        hard_values.extend(hard)
        easy_values.extend(easy)
        matched_blocks += 1

    return {
        "mirror_difference": weighted_difference / total_weight if total_weight else math.nan,
        "matched_weight": total_weight,
        "matched_blocks": float(matched_blocks),
        "hard_gain": mean(hard_values) if hard_values else math.nan,
        "easy_gain": mean(easy_values) if easy_values else math.nan,
        "hard_count": float(len(hard_values)),
        "easy_count": float(len(easy_values)),
    }


def finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def average_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    output: dict[str, float] = {}
    for key in keys:
        values = finite(row[key] for row in rows)
        output[key] = mean(values) if values else math.nan
    return output


def percentile(values: list[float], q: float) -> float:
    values = sorted(finite(values))
    if not values:
        return math.nan
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def two_sided_p(values: list[float]) -> float:
    values = finite(values)
    if not values:
        return math.nan
    non_positive = sum(value <= 0.0 for value in values) / len(values)
    non_negative = sum(value >= 0.0 for value in values) / len(values)
    return min(1.0, 2.0 * min(non_positive, non_negative))


def cluster_bootstrap_pairs(pairs: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_cluster[str(pair["cluster_id"])].append(pair)
    cluster_ids = sorted(by_cluster)
    sampled = rng.choices(cluster_ids, k=len(cluster_ids))
    output: list[dict[str, Any]] = []
    for copy_index, cluster_id in enumerate(sampled):
        for pair in by_cluster[cluster_id]:
            cloned = dict(pair)
            cloned["bootstrap_copy"] = copy_index
            cloned["pair_id"] = f"boot{copy_index}:{pair['pair_id']}"
            output.append(cloned)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_pp(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{100.0 * value:+.2f}"


def result_label(estimate: float, low: float, high: float) -> str:
    if low > 0.0:
        return "positive with a 95% interval above zero"
    if high < 0.0:
        return "negative with a 95% interval below zero"
    if estimate > 0.0:
        return "directionally positive but statistically inconclusive"
    if estimate < 0.0:
        return "directionally negative but statistically inconclusive"
    return "inconclusive"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations_glob")
    parser.add_argument("--pairs_csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--split_size", type=int, default=8)
    parser.add_argument("--split_repeats", type=int, default=100)
    parser.add_argument("--bootstrap_repeats", type=int, default=2000)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.20, 0.30, 0.40])
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()

    if 2 * args.split_size > args.num_samples:
        raise ValueError("2 * split_size must not exceed num_samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    observations: list[dict[str, Any]] = []
    skipped_files: list[dict[str, str]] = []
    if args.pairs_csv:
        source = f"validated pair table: {args.pairs_csv}"
        pairs = load_pairs_csv(Path(args.pairs_csv), args.num_samples)
    elif args.generations_glob:
        source = f"raw rollout glob: {args.generations_glob}"
        paths = [Path(path) for path in sorted(glob.glob(args.generations_glob))]
        if not paths:
            raise FileNotFoundError(f"No files matched {args.generations_glob}")
        observations, skipped_files = load_observations(paths, args.num_samples)
        pairs = make_pairs(observations)
    else:
        raise ValueError("Provide either --pairs_csv or --generations_glob")
    if not pairs:
        raise ValueError("No consecutive repeated-problem pairs were found")

    selector_split_runs: list[dict[str, Any]] = []
    mirror_split_runs: list[dict[str, Any]] = []
    point_by_fraction: dict[float, dict[str, float]] = {}
    point_mirror: dict[str, dict[str, float]] = {}
    for repeat in range(args.split_repeats):
        rows = crossfit_rows(pairs, args.seed + repeat, args.split_size)
        for fraction in args.fractions:
            stats = selector_stat(rows, fraction)
            selector_split_runs.append({"repeat": repeat, "fraction": fraction, **stats})
        for name, match_step in (("pooled", False), ("step_matched", True)):
            stats = mirror_stat(rows, match_step)
            mirror_split_runs.append({"repeat": repeat, "matching": name, **stats})

    for fraction in args.fractions:
        point_by_fraction[fraction] = average_dicts(
            [
                {key: float(value) for key, value in row.items() if key not in {"repeat", "fraction"}}
                for row in selector_split_runs
                if row["fraction"] == fraction
            ]
        )
    for name in ("pooled", "step_matched"):
        point_mirror[name] = average_dicts(
            [
                {key: float(value) for key, value in row.items() if key not in {"repeat", "matching"}}
                for row in mirror_split_runs
                if row["matching"] == name
            ]
        )

    bootstrap_rng = random.Random(args.seed + 999_983)
    bootstrap_rows: list[dict[str, Any]] = []
    for repeat in range(args.bootstrap_repeats):
        sampled_pairs = cluster_bootstrap_pairs(pairs, bootstrap_rng)
        rows = crossfit_rows(sampled_pairs, args.seed + 100_000 + repeat, args.split_size)
        for fraction in args.fractions:
            bootstrap_rows.append(
                {"repeat": repeat, "kind": "selector", "fraction": fraction, **selector_stat(rows, fraction)}
            )
        for name, match_step in (("pooled", False), ("step_matched", True)):
            bootstrap_rows.append(
                {"repeat": repeat, "kind": "mirror", "fraction": name, **mirror_stat(rows, match_step)}
            )

    selector_summary: list[dict[str, Any]] = []
    for fraction in args.fractions:
        point = point_by_fraction[fraction]
        boot = [row for row in bootstrap_rows if row["kind"] == "selector" and row["fraction"] == fraction]
        for metric in ("selected_gain_difference", "disagreement_gain_difference"):
            values = [float(row[metric]) for row in boot]
            low = percentile(values, 0.025)
            high = percentile(values, 0.975)
            selector_summary.append(
                {
                    "fraction": fraction,
                    "contrast": metric,
                    "estimate": point[metric],
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_p": two_sided_p(values),
                    "cls_gain": point["cls_gain"],
                    "uncertainty_gain": point["uncertainty_gain"],
                    "cls_only_gain": point["cls_only_gain"],
                    "uncertainty_only_gain": point["uncertainty_only_gain"],
                    "common_count": point["common_count"],
                    "cls_only_count": point["cls_only_count"],
                }
            )

    mirror_summary: list[dict[str, Any]] = []
    for name in ("pooled", "step_matched"):
        point = point_mirror[name]
        boot = [row for row in bootstrap_rows if row["kind"] == "mirror" and row["fraction"] == name]
        values = [float(row["mirror_difference"]) for row in boot]
        low = percentile(values, 0.025)
        high = percentile(values, 0.975)
        mirror_summary.append(
            {
                "matching": name,
                "estimate": point["mirror_difference"],
                "ci_low": low,
                "ci_high": high,
                "bootstrap_p": two_sided_p(values),
                "hard_gain": point["hard_gain"],
                "easy_gain": point["easy_gain"],
                "matched_weight": point["matched_weight"],
                "matched_blocks": point["matched_blocks"],
            }
        )

    write_csv(output_dir / "selector_split_runs.csv", selector_split_runs)
    write_csv(output_dir / "mirror_split_runs.csv", mirror_split_runs)
    write_csv(output_dir / "selector_summary.csv", selector_summary)
    write_csv(output_dir / "mirror_summary.csv", mirror_summary)

    primary = next(
        row for row in selector_summary if row["fraction"] == 0.30 and row["contrast"] == "disagreement_gain_difference"
    )
    matched = next(row for row in mirror_summary if row["matching"] == "step_matched")
    lines = [
        "# Cross-fitted CLS vs. Symmetric Uncertainty Audit",
        "",
        f"- Source: {source}",
        f"- Saved rollout files: {len(paths) if paths else 'recorded upstream in the pair table'}",
        f"- Parsed rollout files: {len(paths) - len(skipped_files) if paths else 'not reparsed'}",
        f"- Skipped malformed files: {len(skipped_files) if paths else 'not applicable'}",
        f"- Complete {args.num_samples}-sample group observations: {len(observations) if paths else 'not recomputed'}",
        f"- Consecutive repeated-problem pairs: {len(pairs)}",
        f"- Unique problem IDs: {len(set(pair['problem_id'] for pair in pairs))}",
        f"- Cross-fit: {args.split_size} scoring + {args.split_size} held-out current outcomes",
        f"- Split repeats: {args.split_repeats}",
        f"- Problem-cluster bootstrap repeats: {args.bootstrap_repeats}",
        "",
        "## Equal-budget selector comparison",
        "",
        "| Budget | Contrast | Difference (pp) | 95% CI (pp) | Bootstrap p | CLS gain | Uncertainty gain | CLS-only n |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selector_summary:
        contrast = "all selected" if row["contrast"] == "selected_gain_difference" else "disagreement only"
        lines.append(
            f"| {100 * row['fraction']:.0f}% | {contrast} | {fmt_pp(row['estimate'])} | "
            f"[{fmt_pp(row['ci_low'])}, {fmt_pp(row['ci_high'])}] | {row['bootstrap_p']:.4f} | "
            f"{fmt_pp(row['cls_gain'])} | {fmt_pp(row['uncertainty_gain'])} | {row['cls_only_count']:.1f} |"
        )

    lines += [
        "",
        "## Equal-uncertainty mirror test",
        "",
        "| Matching | Hard-minus-easy gain (pp) | 95% CI (pp) | Bootstrap p | Hard gain | Easy gain | Matched weight |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mirror_summary:
        lines.append(
            f"| {row['matching']} | {fmt_pp(row['estimate'])} | "
            f"[{fmt_pp(row['ci_low'])}, {fmt_pp(row['ci_high'])}] | {row['bootstrap_p']:.4f} | "
            f"{fmt_pp(row['hard_gain'])} | {fmt_pp(row['easy_gain'])} | {row['matched_weight']:.1f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "The primary top-30% disagreement-set contrast is "
        f"{result_label(primary['estimate'], primary['ci_low'], primary['ci_high'])} "
        f"({fmt_pp(primary['estimate'])} pp, 95% CI "
        f"[{fmt_pp(primary['ci_low'])}, {fmt_pp(primary['ci_high'])}]).",
        "",
        "The step-matched equal-uncertainty mirror contrast is "
        f"{result_label(matched['estimate'], matched['ci_low'], matched['ci_high'])} "
        f"({fmt_pp(matched['estimate'])} pp, 95% CI "
        f"[{fmt_pp(matched['ci_low'])}, {fmt_pp(matched['ci_high'])}]).",
        "",
        "This is a training-free predictive audit on trajectories collected under CPD. "
        "It tests whether CLS's asymmetric preference identifies greater delayed learning "
        "progress, but it is not an end-to-end causal training comparison.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "generations_glob": args.generations_glob,
        "pairs_csv": args.pairs_csv,
        "source": source,
        "files": len(paths),
        "parsed_files": len(paths) - len(skipped_files),
        "skipped_files": skipped_files,
        "observations": len(observations),
        "pairs": len(pairs),
        "unique_problems": len(set(pair["problem_id"] for pair in pairs)),
        "num_samples": args.num_samples,
        "split_size": args.split_size,
        "split_repeats": args.split_repeats,
        "bootstrap_repeats": args.bootstrap_repeats,
        "fractions": args.fractions,
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print((output_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
