#!/usr/bin/env python3
"""Plot longitudinal evidence that CLS predicts future learning progress."""

from __future__ import annotations

import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training_curves_recovered" / "cls_learning_progress_1p7b"
OUT_DIR = ROOT / "figures" / "cls_future_progress"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "ink": "#253247",
    "muted": "#667085",
    "grid": "#D8DEE8",
    "sky": "#56B4E9",
    "green": "#009E73",
    "amber": "#E69F00",
    "cpd": "#D55E00",
}


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def add_value_labels(
    ax: plt.Axes, bars, values: np.ndarray, errors: np.ndarray
) -> None:
    for bar, value, error in zip(bars, values, errors):
        offset = 0.28 if value >= 0 else -0.28
        va = "bottom" if value >= 0 else "top"
        y = value + error + offset if value >= 0 else value - error + offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:+.2f}",
            ha="center",
            va=va,
            fontsize=7.4,
            fontweight="semibold",
            color=COLORS["ink"],
        )


def style_axis(ax: plt.Axes, ylim: tuple[float, float], title: str) -> None:
    ax.axhline(0, color=COLORS["muted"], linewidth=0.8, zorder=1)
    ax.set_ylim(*ylim)
    ax.set_title(title, pad=6, fontweight="semibold")
    ax.set_ylabel("Future correctness gain (pp)", labelpad=3)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.65, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["muted"])
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(axis="x", length=0, pad=4)
    ax.tick_params(axis="y", length=3.0, width=0.7, color=COLORS["muted"], pad=2)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.1,
            "axes.titlesize": 7.8,
            "axes.titleweight": "semibold",
            "axes.labelsize": 7.0,
            "axes.labelcolor": COLORS["ink"],
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    bucket_rows = read_rows("learning_progress_by_cls_bucket.csv")
    bucket_order = ["All-wrong", "Low CLS", "Mid CLS", "High CLS", "Solved"]
    bucket_by_name = {row["bucket"]: row for row in bucket_rows}
    bucket_rows = [bucket_by_name[name] for name in bucket_order]
    selector_rows = read_rows("future_gain_by_selector.csv")
    bucket_labels = [row["bucket"].replace(" ", "\n", 1) for row in bucket_rows]
    bucket_labels = [f"{label}\n(n={row['count']})" for label, row in zip(bucket_labels, bucket_rows)]
    bucket_gain = np.asarray([100 * float(row["gain"]) for row in bucket_rows])
    bucket_se = np.asarray([100 * float(row["gain_se"]) for row in bucket_rows])
    bucket_colors = ["#9CA3AF", "#F6C77B", COLORS["amber"], COLORS["cpd"], "#667085"]
    x = np.arange(len(bucket_rows))
    fig, ax = plt.subplots(figsize=(3.30, 2.35))
    ax.axvspan(0.5, 3.5, color="#FFF4E6", alpha=0.80, zorder=0)
    ax.errorbar(
        x,
        bucket_gain,
        yerr=bucket_se,
        color=COLORS["cpd"],
        linewidth=1.8,
        marker="o",
        markersize=5.2,
        markerfacecolor="white",
        markeredgecolor=COLORS["cpd"],
        markeredgewidth=1.1,
        capsize=2.5,
        elinewidth=0.9,
        ecolor="#374151",
        capthick=0.9,
        zorder=2,
    )
    ax.scatter(
        x,
        bucket_gain,
        s=25,
        c=bucket_colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    ax.set_xticks(x, bucket_labels)
    style_axis(ax, (-3.2, 12.6), "Future gain across CLS buckets")
    for xi, value, error in zip(x, bucket_gain, bucket_se):
        offset = 0.30 if value >= 0 else -0.30
        va = "bottom" if value >= 0 else "top"
        y = value + error + offset if value >= 0 else value - error + offset
        ax.text(
            xi,
            y,
            f"{value:+.2f}",
            ha="center",
            va=va,
            fontsize=7.4,
            fontweight="semibold",
            color=COLORS["ink"],
        )
    ax.text(
        2.0,
        12.15,
        "mixed learnable frontier",
        ha="center",
        va="top",
        fontsize=7.2,
        fontweight="semibold",
        color="#A94400",
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.25, top=0.90)
    stem = OUT_DIR / "fig_future_gain_cls_bucket"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    selector_labels = [row["selector"].replace(" (ours)", "\n(ours)") for row in selector_rows]
    selector_gain = np.asarray([100 * float(row["gain"]) for row in selector_rows])
    selector_se = np.asarray([100 * float(row["gain_se"]) for row in selector_rows])
    selector_colors = ["#9CA3AF", "#667085", COLORS["green"], COLORS["cpd"]]
    x = np.arange(len(selector_rows))
    fig, ax = plt.subplots(figsize=(3.30, 2.35))
    bars = ax.bar(
        x,
        selector_gain,
        yerr=selector_se,
        color=selector_colors,
        edgecolor="white",
        linewidth=0.6,
        width=0.68,
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "ecolor": "#374151", "capthick": 0.8},
        zorder=2,
    )
    ax.set_xticks(x, selector_labels)
    style_axis(ax, (-3.2, 9.8), "Equal-budget selector comparison")
    add_value_labels(ax, bars, selector_gain, selector_se)
    ax.text(
        0.02,
        0.97,
        "n = 368 per selector",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["muted"],
    )
    ax.text(
        3,
        8.35,
        "highest observed",
        ha="center",
        va="bottom",
        fontsize=7.0,
        fontweight="semibold",
        color="#A94400",
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.24, top=0.90)
    stem = OUT_DIR / "fig_future_gain_selector"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
