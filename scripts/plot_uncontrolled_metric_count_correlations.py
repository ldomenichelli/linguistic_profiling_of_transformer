#!/usr/bin/env python3
"""Plot uncontrolled pooled correlations between class metrics and class counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "correlation"
    / "class_frequency_type_correlations"
    / "feature_controlled_pooled_residualized_points_layer_mean_scores.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "correlation"
    / "class_frequency_type_correlations"
    / "uncontrolled_metric_count_correlations"
)

FAMILY_LABELS = {
    "isotropy": "IsoScore",
    "linear_id": "LID",
    "nonlinear_id": "NLID",
}
METRIC_LABELS = {
    "iso": "IsoScore",
    "lpca99": "LID (lPCA99)",
    "pca99": "LID (PCA99)",
    "gride": "NLID (GRIDE)",
}
METRIC_ORDER = ["IsoScore", "LID (lPCA99)", "LID (PCA99)", "NLID (GRIDE)"]
TARGET_LABELS = {
    "n_tokens": "Frequency",
    "n_types": "Type frequency",
}
TARGET_ORDER = ["Frequency", "Type frequency"]
MODEL_LABELS = {
    "bert-base-uncased": "BERT",
    "gpt2": "GPT-2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def significance_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return ""


def load_points(path: Path) -> pd.DataFrame:
    columns = [
        "model",
        "word_rep_mode",
        "feature",
        "metric_family",
        "metric",
        "class",
        "mean_score",
        "n_tokens",
        "n_types",
    ]
    points = pd.read_csv(path, usecols=columns)
    points["mean_score"] = pd.to_numeric(points["mean_score"], errors="coerce")
    points["n_tokens"] = pd.to_numeric(points["n_tokens"], errors="coerce")
    points["n_types"] = pd.to_numeric(points["n_types"], errors="coerce")
    points = points.dropna(subset=["mean_score", "n_tokens", "n_types"]).copy()
    points["family_label"] = points["metric_family"].map(FAMILY_LABELS).fillna(points["metric_family"])
    points["metric_label"] = points["metric"].map(METRIC_LABELS).fillna(points["metric"])
    points["model_label"] = points["model"].map(MODEL_LABELS).fillna(points["model"])
    points["row_label"] = points["model_label"] + " " + points["metric_label"]
    return points.reset_index(drop=True)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(data["x"], data["y"]) if method == "spearman" else pearsonr(data["x"], data["y"])
    statistic = getattr(result, "statistic", result[0])
    p_value = getattr(result, "pvalue", result[1])
    return float(statistic), float(p_value)


def compute_correlations(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "model_label", "word_rep_mode", "metric_family", "family_label", "metric", "metric_label"]
    for keys, group in points.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, keys))
        features = sorted(group["feature"].unique())
        for target, target_label in TARGET_LABELS.items():
            spearman_r, spearman_p = safe_corr(group["mean_score"], group[target], "spearman")
            pearson_r, pearson_p = safe_corr(group["mean_score"], group[target], "pearson")
            rows.append(
                {
                    **base,
                    "target": target,
                    "target_label": target_label,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "n_classes": int(len(group)),
                    "n_features": int(group["feature"].nunique()),
                    "features_used": ", ".join(features),
                }
            )
    out = pd.DataFrame(rows)
    out["metric_order"] = out["metric_label"].map({name: i for i, name in enumerate(METRIC_ORDER)})
    out["target_order"] = out["target_label"].map({name: i for i, name in enumerate(TARGET_ORDER)})
    out["model_order"] = out["model_label"].map({"BERT": 0, "GPT-2": 1}).fillna(99)
    return (
        out.sort_values(["model_order", "metric_order", "target_order", "metric"])
        .drop(columns=["metric_order", "target_order", "model_order"])
        .reset_index(drop=True)
    )


def heatmap_matrices(correlations: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_col = f"{method}_r"
    p_col = f"{method}_p"
    rows = []
    for model in ["BERT", "GPT-2"]:
        for metric in METRIC_ORDER:
            rows.append(f"{model} {metric}")
    matrix = pd.DataFrame(index=rows, columns=TARGET_ORDER, dtype=float)
    annot = matrix.astype(object).copy()

    for _, row in correlations.iterrows():
        row_label = f"{row['model_label']} {row['metric_label']}"
        target = row["target_label"]
        if row_label not in matrix.index or target not in matrix.columns:
            continue
        value = float(row[value_col])
        p_value = float(row[p_col])
        matrix.loc[row_label, target] = value
        annot.loc[row_label, target] = f"rho={value:.2f}{significance_stars(p_value)}" if method == "spearman" else f"r={value:.2f}{significance_stars(p_value)}"
    return matrix, annot


def save_heatmap(correlations: pd.DataFrame, out_base: Path, method: str = "spearman") -> None:
    matrix, annot = heatmap_matrices(correlations, method)
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=2.0,
        linecolor="white",
        cbar=True,
        annot_kws={"fontsize": 15, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=16, length=0, rotation=0)
    ax.tick_params(axis="y", labelsize=14, length=0, rotation=0)
    title_method = "Spearman rank" if method == "spearman" else "Pearson"
    ax.set_title(f"Uncontrolled pooled metric-count correlations ({title_method})", fontsize=15, pad=12)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_base.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(args.input)
    correlations = compute_correlations(points)
    points.to_csv(args.output_dir / "uncontrolled_metric_count_points.csv", index=False)
    correlations.to_csv(args.output_dir / "uncontrolled_metric_count_correlations.csv", index=False)
    save_heatmap(correlations, args.output_dir / "uncontrolled_metric_count_spearman_heatmap", "spearman")
    save_heatmap(correlations, args.output_dir / "uncontrolled_metric_count_pearson_heatmap", "pearson")

    print(correlations.to_string(index=False))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
