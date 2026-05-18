#!/usr/bin/env python3
"""Regenerate the lower-triangle metric correlation heatmap."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "correlation" / "tables" / "spearman_corr_with_pvals_lower_triangle.csv"
PLOT_DIR = ROOT / "correlation" / "plots"

METRIC_LABELS = {
    "iso": "IsoScore",
    "sf": "SF",
    "vmf_kappa": "vMF kappa",
    "erank": "eRank",
    "pr": "PR",
    "stable_rank": "Stable rank",
    "lpca": "LPCA",
    "lpca95": "LPCA95",
    "lpca99": "LPCA99",
    "twonn": "TwoNN",
    "gride": "GRIDE",
    "mle": "MLE",
    "mom": "MOM",
    "tle": "TLE",
    "ess": "ESS",
    "mada": "MADA",
    "corrint": "CorrInt",
    "fishers": "FisherS",
}

GROUPS = [
    ("isotropy", ["iso", "sf", "vmf_kappa"]),
    ("linear_id", ["erank", "pr", "stable_rank", "lpca", "lpca95", "lpca99"]),
    ("nonlinear_id", ["twonn", "gride", "mle", "mom", "tle", "ess", "mada", "corrint", "fishers"]),
]


def parse_numeric(value: object) -> Optional[float]:
    if pd.isna(value) or value == "":
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if not match:
        return None
    return float(match.group(0))


def main() -> None:
    annot = pd.read_csv(TABLE_PATH, index_col=0, dtype=str).fillna("")
    values = annot.map(parse_numeric)

    labels = [METRIC_LABELS.get(metric, metric) for metric in annot.index]
    annot.index = labels
    annot.columns = labels
    values.index = labels
    values.columns = labels

    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(11.06, 9.73))
    heatmap = sns.heatmap(
        values,
        vmin=-1,
        vmax=1,
        center=0,
        cmap="coolwarm",
        square=True,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 8},
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Spearman rho"},
        ax=ax,
    )

    ax.tick_params(axis="x", labelsize=12, length=0)
    ax.tick_params(axis="y", labelsize=12, length=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Draw only the family separators over populated lower-triangle cells.
    cut = 0
    cuts: List[int] = []
    metric_names = list(pd.read_csv(TABLE_PATH, index_col=0, nrows=0).columns)
    for _, group in GROUPS[:-1]:
        cut += sum(metric in metric_names for metric in group)
        cuts.append(cut)

    for cut in cuts:
        ax.plot([0, cut], [cut, cut], color="black", lw=2.0, solid_capstyle="butt")
        ax.plot([cut, cut], [cut, len(values)], color="black", lw=2.0, solid_capstyle="butt")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_frame_on(False)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Spearman rho", fontsize=14, rotation=270, labelpad=28)

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(PLOT_DIR / f"corr_heatmap_lower_triangle.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
