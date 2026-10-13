#!/usr/bin/env python
"""Draw publication-quality mechanism views for CPD data selection.

The script exports three standalone plots and a full-width three-panel figure:
1. contrastive signal available at each group success rate;
2. an observed rollout distribution overlaid with CLS utility;
3. selector profiles for common alternatives and CLS.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


COLORS = {
    "ink": "#253247",
    "muted": "#667085",
    "grid": "#D8DEE8",
    "bar": "#A8C7E8",
    "bar_edge": "#4C78A8",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "amber": "#E69F00",
    "cpd": "#D55E00",
}


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


def load_correct_count_hist(path: Path, m: int) -> Counter[int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped[_group_key(row)].append(row)

    hist: Counter[int] = Counter()
    for rows in grouped.values():
        rows = sorted(rows, key=lambda row: int(row.get("sample_index", 0)))[:m]
        if rows:
            hist[sum(_correct(row) for row in rows)] += 1
    return hist


def load_histogram(path: Path, m: int) -> Counter[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    source_m = int(payload.get("m", m))
    if source_m != m:
        raise ValueError(f"Histogram uses M={source_m}, but --m={m}.")
    return Counter({int(key): int(value) for key, value in payload["histogram"].items()})


def configure_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.1,
            "axes.titlesize": 7.8,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.0,
            "axes.labelcolor": COLORS["ink"],
            "axes.edgecolor": COLORS["muted"],
            "axes.linewidth": 0.8,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "legend.fontsize": 5.9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: Any) -> None:
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=3.0, width=0.7)


def set_success_axis(ax: Any) -> None:
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_xlabel("Group success rate $p$")


def panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.13,
        1.30,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=COLORS["ink"],
        clip_on=False,
    )


def draw_signal(ax: Any, ps: list[float], pair_density: list[float], grpo_signal: list[float]) -> None:
    style_axis(ax)
    ax.plot(
        ps,
        pair_density,
        color=COLORS["green"],
        marker="o",
        markersize=4.0,
        linewidth=2.0,
        label="Pair density",
    )
    ax.plot(
        ps,
        grpo_signal,
        color=COLORS["blue"],
        marker="s",
        markersize=3.8,
        linewidth=2.0,
        linestyle="--",
        label="Group signal",
    )
    ax.axvline(0.5, color=COLORS["grid"], linewidth=0.9, linestyle=":")
    ax.set_ylim(0.0, 1.08)
    set_success_axis(ax)
    ax.set_ylabel("Signal strength")
    ax.set_title("Contrastive signal", pad=27)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        ncol=1,
        handlelength=2.0,
        labelspacing=0.25,
    )


def draw_observed(ax: Any, ps: list[float], empirical: list[float], cls: list[float]) -> None:
    style_axis(ax)
    width = 0.085
    bars = ax.bar(
        ps,
        empirical,
        width=width,
        color=COLORS["bar"],
        edgecolor=COLORS["bar_edge"],
        linewidth=0.8,
        label="Observed fraction",
        zorder=2,
    )
    ax.set_ylim(0.0, max(0.44, max(empirical) * 1.16))
    set_success_axis(ax)
    ax.set_ylabel("Group fraction")
    ax.set_title("Coverage vs. CLS", pad=27)

    twin = ax.twinx()
    twin.plot(
        ps,
        cls,
        color=COLORS["cpd"],
        marker="D",
        markersize=3.8,
        linewidth=2.2,
        label="CLS utility",
        zorder=3,
    )
    twin.set_ylim(0.0, 1.08)
    twin.set_ylabel("CLS utility", color=COLORS["cpd"])
    twin.tick_params(axis="y", colors=COLORS["cpd"], length=3.0, width=0.7)
    twin.spines["top"].set_visible(False)
    twin.spines["left"].set_visible(False)
    twin.spines["right"].set_color(COLORS["cpd"])

    handles: list[Any] = [bars, twin.lines[0]]
    labels = ["Observed fraction", "CLS utility"]
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        labelspacing=0.25,
    )


def draw_selectors(
    ax: Any,
    ps: list[float],
    pass_only: list[float],
    pass_rate: list[float],
    uncertainty: list[float],
    cls: list[float],
) -> None:
    style_axis(ax)
    ax.plot(
        ps,
        pass_only,
        color=COLORS["muted"],
        linewidth=1.6,
        linestyle=":",
        marker="o",
        markersize=3.2,
        label="Any success",
    )
    ax.plot(
        ps,
        pass_rate,
        color=COLORS["sky"],
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markersize=3.2,
        label="Success rate",
    )
    ax.plot(
        ps,
        uncertainty,
        color=COLORS["green"],
        linewidth=1.9,
        linestyle="-.",
        marker="^",
        markersize=3.5,
        label="Uncertainty",
    )
    ax.plot(
        ps,
        cls,
        color=COLORS["cpd"],
        linewidth=2.4,
        marker="D",
        markersize=3.8,
        label="CLS (ours)",
        zorder=4,
    )
    ax.axvline(0.5, color=COLORS["grid"], linewidth=0.9, linestyle=":")
    ax.set_ylim(0.0, 1.08)
    set_success_axis(ax)
    ax.set_ylabel("Selection utility")
    ax.set_title("Selector profiles", pad=27)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        columnspacing=0.65,
        handlelength=1.8,
        labelspacing=0.25,
    )


def save_figure(fig: Any, output_dir: Path, stem: str, dpi: int) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.03)


def normalized_histogram(hist: Counter[int], xs: Iterable[int]) -> list[float]:
    total = max(1, sum(hist.values()))
    return [hist.get(x, 0) / total for x in xs]


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CPD selection mechanisms.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--m", type=int, default=8, help="Online responses per problem.")
    parser.add_argument("--generations_file", default=None)
    parser.add_argument("--observed_histogram", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.output_dir) / ".matplotlib"))
    import matplotlib.pyplot as plt

    configure_style()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    m = max(1, args.m)

    xs = list(range(m + 1))
    ps = [value / m for value in xs]
    pass_only = [1.0 if value > 0 else 0.0 for value in xs]
    pass_rate = ps
    uncertainty = [4.0 * p * (1.0 - p) for p in ps]
    cls = [4.0 * min(p, 1.0 - p) * (1.0 - p) for p in ps]
    pair_density = uncertainty
    grpo_signal = [2.0 * (p * (1.0 - p)) ** 0.5 for p in ps]

    hist: Counter[int] = Counter()
    if args.generations_file:
        hist = load_correct_count_hist(Path(args.generations_file), m)
    elif args.observed_histogram:
        hist = load_histogram(Path(args.observed_histogram), m)
    empirical = normalized_histogram(hist, xs) if hist else [0.0] * len(xs)

    fig, ax = plt.subplots(figsize=(3.45, 2.70))
    draw_signal(ax, ps, pair_density, grpo_signal)
    fig.tight_layout(pad=0.5)
    save_figure(fig, output_dir, "grpo_balance_curves", args.dpi)
    plt.close(fig)

    if hist:
        fig, ax = plt.subplots(figsize=(3.45, 2.70))
        draw_observed(ax, ps, empirical, cls)
        fig.tight_layout(pad=0.5)
        save_figure(fig, output_dir, "observed_distribution_with_cls", args.dpi)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.45, 2.70))
    draw_selectors(ax, ps, pass_only, pass_rate, uncertainty, cls)
    fig.tight_layout(pad=0.5)
    save_figure(fig, output_dir, "selection_utility_curves", args.dpi)
    plt.close(fig)

    if hist:
        fig, axes = plt.subplots(1, 3, figsize=(7.10, 2.48))
        draw_signal(axes[0], ps, pair_density, grpo_signal)
        draw_observed(axes[1], ps, empirical, cls)
        draw_selectors(axes[2], ps, pass_only, pass_rate, uncertainty, cls)
        for label, ax in zip(("(a)", "(b)", "(c)"), axes):
            panel_label(ax, label)
        fig.subplots_adjust(left=0.060, right=0.945, bottom=0.20, top=0.74, wspace=0.52)
        save_figure(fig, output_dir, "fig_cpd_mechanisms", args.dpi)
        plt.close(fig)

    if args.generations_file and hist:
        payload = {
            "m": m,
            "histogram": {str(key): hist[key] for key in sorted(hist)},
            "total": sum(hist.values()),
            "provenance": f"Computed from {args.generations_file}",
        }
        (output_dir / "observed_histogram.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    print(f"Saved CPD mechanism figures to: {output_dir}")


if __name__ == "__main__":
    main()
