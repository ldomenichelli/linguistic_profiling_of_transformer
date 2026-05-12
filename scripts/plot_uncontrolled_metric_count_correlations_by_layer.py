#!/usr/bin/env python3
"""Plot uncontrolled per-layer metric/count correlations pooled across features."""

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

from correlate_class_scores_with_counts import (
    DEFAULT_GEOMETRY_ROOT,
    DEFAULT_SENTENCE_CSV,
    compute_class_counts,
    load_selected_metric_tables,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "correlation"
    / "class_frequency_type_correlations"
    / "uncontrolled_metric_count_correlations_by_layer"
)

METRIC_LABELS = {
    "iso": "IsoScore",
    "lpca99": "LID (lPCA99)",
    "pca99": "LID (PCA99)",
    "gride": "NLID (GRIDE)",
}
METRIC_ORDER = ["iso", "lpca99", "pca99", "gride"]
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
    parser.add_argument("--sentence-csv", type=Path, default=DEFAULT_SENTENCE_CSV)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics", nargs="+", default=METRIC_ORDER, help="Metric keys to include.")
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


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan, np.nan
    result = spearmanr(data["x"], data["y"]) if method == "spearman" else pearsonr(data["x"], data["y"])
    statistic = getattr(result, "statistic", result[0])
    p_value = getattr(result, "pvalue", result[1])
    return float(statistic), float(p_value)


def load_joined(sentence_csv: Path, geometry_root: Path, metrics_to_keep: list[str]) -> pd.DataFrame:
    counts = compute_class_counts(sentence_csv)
    metrics = load_selected_metric_tables(geometry_root)
    metrics = metrics[metrics["metric"].isin(metrics_to_keep)].copy()
    joined = metrics.merge(counts, on=["feature", "class"], how="left", validate="many_to_one")
    joined = joined.dropna(subset=["mean", "layer", "n_tokens", "n_types"]).copy()
    joined["mean"] = pd.to_numeric(joined["mean"], errors="coerce")
    joined["layer"] = pd.to_numeric(joined["layer"], errors="coerce").astype(int)
    joined["n_tokens"] = pd.to_numeric(joined["n_tokens"], errors="coerce").astype(int)
    joined["n_types"] = pd.to_numeric(joined["n_types"], errors="coerce").astype(int)
    joined["model_label"] = joined["model"].map(MODEL_LABELS).fillna(joined["model"])
    joined["metric_label"] = joined["metric"].map(METRIC_LABELS).fillna(joined["metric"])
    joined["row_label"] = joined["model_label"] + " " + joined["metric_label"]
    return joined.reset_index(drop=True)


def compute_layer_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["model", "model_label", "word_rep_mode", "metric_family", "metric", "metric_label", "layer"]
    for keys, group in joined.groupby(group_cols, dropna=False, sort=False):
        base = dict(zip(group_cols, keys))
        features = sorted(group["feature"].unique())
        for target, target_label in TARGET_LABELS.items():
            spearman_r, spearman_p = safe_corr(group["mean"], group[target], "spearman")
            pearson_r, pearson_p = safe_corr(group["mean"], group[target], "pearson")
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
    metric_order = {metric: i for i, metric in enumerate(METRIC_ORDER)}
    target_order = {label: i for i, label in enumerate(TARGET_ORDER)}
    model_order = {"BERT": 0, "GPT-2": 1}
    out["model_order"] = out["model_label"].map(model_order).fillna(99)
    out["metric_order"] = out["metric"].map(metric_order).fillna(99)
    out["target_order"] = out["target_label"].map(target_order).fillna(99)
    return (
        out.sort_values(["model_order", "metric_order", "target_order", "layer"])
        .drop(columns=["model_order", "metric_order", "target_order"])
        .reset_index(drop=True)
    )


def last_layer_correlations(layer_corr: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "word_rep_mode", "metric", "target"]
    max_layers = layer_corr.groupby(group_cols, dropna=False)["layer"].transform("max")
    return layer_corr[layer_corr["layer"].eq(max_layers)].copy().reset_index(drop=True)


def ordered_row_labels(data: pd.DataFrame) -> list[str]:
    rows = []
    for model in ["BERT", "GPT-2"]:
        for metric in METRIC_ORDER:
            label = METRIC_LABELS.get(metric, metric)
            row = f"{model} {label}"
            if row in set(data["model_label"] + " " + data["metric_label"]):
                rows.append(row)
    return rows


def save_last_layer_heatmap(last: pd.DataFrame, out_base: Path, method: str = "spearman") -> None:
    value_col = f"{method}_r"
    p_col = f"{method}_p"
    row_labels = ordered_row_labels(last)
    matrix = pd.DataFrame(index=row_labels, columns=TARGET_ORDER, dtype=float)
    annot = matrix.astype(object).copy()
    for _, row in last.iterrows():
        row_label = f"{row['model_label']} {row['metric_label']}"
        target = row["target_label"]
        value = float(row[value_col])
        p_value = float(row[p_col])
        matrix.loc[row_label, target] = value
        prefix = "rho" if method == "spearman" else "r"
        annot.loc[row_label, target] = f"{prefix}={value:.2f}{significance_stars(p_value)}"

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
    method_title = "Spearman rank" if method == "spearman" else "Pearson"
    ax.set_title(f"Last-layer uncontrolled metric-count correlations ({method_title})", fontsize=15, pad=12)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_base.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_layerwise_heatmap(layer_corr: pd.DataFrame, target_label: str, out_base: Path) -> None:
    data = layer_corr[layer_corr["target_label"].eq(target_label)].copy()
    data["row_label"] = data["model_label"] + " " + data["metric_label"]
    row_labels = ordered_row_labels(data)
    matrix = data.pivot_table(index="row_label", columns="layer", values="spearman_r", aggfunc="first")
    matrix = matrix.reindex(index=row_labels)
    matrix = matrix.reindex(columns=sorted(matrix.columns))
    annot = matrix.map(lambda value: "" if pd.isna(value) else f"{value:.2f}")

    sns.set_theme(style="white")
    fig_w = max(9.5, 0.55 * len(matrix.columns) + 4.5)
    fig_h = max(5.8, 0.5 * len(matrix.index) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        linewidths=1.2,
        linecolor="white",
        cbar=True,
        annot_kws={"fontsize": 9},
        ax=ax,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=11, length=0, rotation=0)
    ax.tick_params(axis="y", labelsize=12, length=0, rotation=0)
    ax.set_title(f"Layerwise uncontrolled Spearman correlations with {target_label.lower()}", fontsize=14, pad=12)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_base.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    joined = load_joined(args.sentence_csv, args.geometry_root, args.metrics)
    layer_corr = compute_layer_correlations(joined)
    last = last_layer_correlations(layer_corr)

    joined.to_csv(args.output_dir / "uncontrolled_metric_count_points_by_layer.csv", index=False)
    layer_corr.to_csv(args.output_dir / "uncontrolled_metric_count_correlations_by_layer.csv", index=False)
    last.to_csv(args.output_dir / "uncontrolled_metric_count_correlations_last_layer.csv", index=False)

    save_last_layer_heatmap(last, args.output_dir / "uncontrolled_metric_count_last_layer_spearman_heatmap", "spearman")
    save_last_layer_heatmap(last, args.output_dir / "uncontrolled_metric_count_last_layer_pearson_heatmap", "pearson")
    save_layerwise_heatmap(layer_corr, "Frequency", args.output_dir / "uncontrolled_metric_count_layerwise_frequency_spearman_heatmap")
    save_layerwise_heatmap(
        layer_corr,
        "Type frequency",
        args.output_dir / "uncontrolled_metric_count_layerwise_type_frequency_spearman_heatmap",
    )

    print(last.to_string(index=False))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
