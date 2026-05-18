#!/usr/bin/env python3
"""Regenerate the main BERT/NOUN head-distance Linear ID plot."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = (
    ROOT
    / "tables_POS_HEAD_DIST"
    / "bert"
    / "first"
    / "all"
    / "noun"
    / "headdist_NOUN_lpca99_bert-base-uncased_first.csv"
)
OUT_PATH = ROOT / "plots_main" / "plots_features" / "bert_head_LID_NOUN.pdf"

LABEL_LAYER = {
    "-6": 6,
    "-5": 6,
    "-4": 4,
    "-3": 2,
    "-2": 10,
    "-1": 8,
    "1": 8,
    "2": 4,
    "3": 4,
    "4": 6,
    "5": 6,
    "6": 6,
}


def main() -> None:
    df = pd.read_csv(TABLE_PATH)
    classes = sorted(df["class"].astype(str).unique(), key=lambda value: int(value))

    sns.set_style("darkgrid")
    sns.set_context("paper", font_scale=1.6)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    cmap = plt.get_cmap("coolwarm")
    norm = TwoSlopeNorm(vmin=-6, vcenter=0, vmax=6)

    for class_name in classes:
        class_df = df[df["class"].astype(str) == class_name].sort_values("layer")
        color = cmap(norm(int(class_name)))
        ax.plot(class_df["layer"], class_df["mean"], lw=1.9, color=color)
        ax.fill_between(
            class_df["layer"].to_numpy(),
            class_df["ci_low"].to_numpy(),
            class_df["ci_high"].to_numpy(),
            alpha=0.14,
            color=color,
        )

    for class_name, label_layer in LABEL_LAYER.items():
        class_df = df[df["class"].astype(str) == class_name].sort_values("layer")
        row = class_df[class_df["layer"] == label_layer]
        if row.empty:
            continue
        color = cmap(norm(int(class_name)))
        ax.text(
            label_layer,
            float(row["mean"].iloc[0]),
            class_name,
            color=color,
            fontsize=20,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
        )

    ax.set_xlabel("")
    ax.set_ylabel("Linear ID")
    ax.set_xticks(np.arange(0, 13, 2))
    ax.margins(x=0.02)

    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)


if __name__ == "__main__":
    main()
