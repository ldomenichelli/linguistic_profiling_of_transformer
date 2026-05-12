#!/usr/bin/env python3
"""Plot uncontrolled pooled correlations between class frequency and type count."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    / "uncontrolled_type_frequency"
)

FEATURE_LABELS = {
    "arity": "Arity",
    "head_dist": "Head distance",
    "index": "Token index",
    "length": "Token length",
    "pos": "POS",
    "relation_type": "Relation type",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_counts(path: Path) -> pd.DataFrame:
    required = ["feature", "class", "n_tokens", "n_types"]
    df = pd.read_csv(path, usecols=required)
    counts = df.drop_duplicates(["feature", "class"])[required].copy()
    counts["n_tokens"] = pd.to_numeric(counts["n_tokens"], errors="coerce")
    counts["n_types"] = pd.to_numeric(counts["n_types"], errors="coerce")
    counts = counts.dropna(subset=["n_tokens", "n_types"])
    counts["feature_label"] = counts["feature"].map(FEATURE_LABELS).fillna(counts["feature"])
    counts["class_label"] = counts["feature_label"] + ": " + counts["class"].astype(str)
    counts["log10_n_tokens"] = np.log10(counts["n_tokens"])
    counts["log10_n_types"] = np.log10(counts["n_types"])
    counts["token_rank"] = counts["n_tokens"].rank(method="average")
    counts["type_rank"] = counts["n_types"].rank(method="average")
    return counts.sort_values(["feature", "class"]).reset_index(drop=True)


def corr_rows(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [
        ("spearman_raw", "n_tokens", "n_types", spearmanr),
        ("pearson_raw", "n_tokens", "n_types", pearsonr),
        ("pearson_log10", "log10_n_tokens", "log10_n_types", pearsonr),
        ("spearman_log10", "log10_n_tokens", "log10_n_types", spearmanr),
    ]
    for name, x_col, y_col, fn in specs:
        result = fn(counts[x_col], counts[y_col])
        statistic = getattr(result, "statistic", result[0])
        p_value = getattr(result, "pvalue", result[1])
        rows.append(
            {
                "correlation": name,
                "x": x_col,
                "y": y_col,
                "r": float(statistic),
                "p_value": float(p_value),
                "n_classes": int(len(counts)),
                "n_features": int(counts["feature"].nunique()),
                "features_used": ", ".join(sorted(counts["feature"].unique())),
            }
        )
    return pd.DataFrame(rows)


def fit_line(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    coef = np.polyfit(x.to_numpy(dtype=float), y.to_numpy(dtype=float), deg=1)
    x_line = np.linspace(float(x.min()), float(x.max()), 100)
    y_line = coef[0] * x_line + coef[1]
    return x_line, y_line


def feature_colors(counts: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    features = list(counts["feature_label"].drop_duplicates())
    cmap = plt.get_cmap("tab10")
    return {feature: cmap(i % 10) for i, feature in enumerate(features)}


def save_scatter(counts: pd.DataFrame, correlations: pd.DataFrame, out_base: Path) -> None:
    colors = feature_colors(counts)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))

    for feature, group in counts.groupby("feature_label", sort=False):
        axes[0].scatter(
            group["n_tokens"],
            group["n_types"],
            s=48,
            alpha=0.84,
            color=colors[feature],
            edgecolor="white",
            linewidth=0.6,
            label=feature,
        )
        axes[1].scatter(
            group["log10_n_tokens"],
            group["log10_n_types"],
            s=48,
            alpha=0.84,
            color=colors[feature],
            edgecolor="white",
            linewidth=0.6,
        )

    pearson_raw = correlations.loc[correlations["correlation"].eq("pearson_raw"), "r"].iat[0]
    pearson_log = correlations.loc[correlations["correlation"].eq("pearson_log10"), "r"].iat[0]
    spearman_raw = correlations.loc[correlations["correlation"].eq("spearman_raw"), "r"].iat[0]

    x_line, y_line = fit_line(counts["log10_n_tokens"], counts["log10_n_types"])
    axes[1].plot(x_line, y_line, color="#222222", linewidth=2.0)

    axes[0].set_title(f"Raw counts: Pearson r = {pearson_raw:.2f}")
    axes[0].set_xlabel("Class frequency (tokens)")
    axes[0].set_ylabel("Unique word types")
    axes[0].grid(alpha=0.25)

    axes[1].set_title(f"Log counts: Pearson r = {pearson_log:.2f}; Spearman rho = {spearman_raw:.2f}")
    axes[1].set_xlabel(r"$\log_{10}$ class frequency")
    axes[1].set_ylabel(r"$\log_{10}$ unique word types")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Uncontrolled pooled type-count vs frequency correlation", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0.11, 1, 0.94])

    for suffix in [".png", ".pdf"]:
        fig.savefig(out_base.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_rank_scatter(counts: pd.DataFrame, correlations: pd.DataFrame, out_base: Path) -> None:
    colors = feature_colors(counts)
    fig, ax = plt.subplots(figsize=(6.9, 5.8))
    for feature, group in counts.groupby("feature_label", sort=False):
        ax.scatter(
            group["token_rank"],
            group["type_rank"],
            s=58,
            alpha=0.86,
            color=colors[feature],
            edgecolor="white",
            linewidth=0.6,
            label=feature,
        )

    x_line, y_line = fit_line(counts["token_rank"], counts["type_rank"])
    ax.plot(x_line, y_line, color="#222222", linewidth=2.0)

    spearman_raw = correlations.loc[correlations["correlation"].eq("spearman_raw"), "r"].iat[0]
    p_value = correlations.loc[correlations["correlation"].eq("spearman_raw"), "p_value"].iat[0]
    ax.set_title(f"Rank correlation: Spearman rho = {spearman_raw:.2f}, p = {p_value:.3g}")
    ax.set_xlabel("Frequency rank across all feature classes")
    ax.set_ylabel("Type-count rank across all feature classes")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()

    for suffix in [".png", ".pdf"]:
        fig.savefig(out_base.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


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


def save_heatmap(correlations: pd.DataFrame, out_base: Path) -> None:
    rows = [
        ("Raw counts", "spearman_raw", "pearson_raw"),
        (r"$\log_{10}$ counts", "spearman_log10", "pearson_log10"),
    ]
    cols = [("Spearman", "rho"), ("Pearson", "r")]
    matrix = pd.DataFrame(index=[row[0] for row in rows], columns=[col[0] for col in cols], dtype=float)
    annot = matrix.astype(object).copy()
    corr_lookup = correlations.set_index("correlation")

    for row_label, spearman_key, pearson_key in rows:
        for col_label, symbol, key in [
            ("Spearman", "rho", spearman_key),
            ("Pearson", "r", pearson_key),
        ]:
            value = float(corr_lookup.loc[key, "r"])
            p_value = float(corr_lookup.loc[key, "p_value"])
            matrix.loc[row_label, col_label] = value
            annot.loc[row_label, col_label] = f"{symbol}={value:.2f}{significance_stars(p_value)}"

    fig, ax = plt.subplots(figsize=(8.0, 4.1))
    im = ax.imshow(matrix.to_numpy(dtype=float), cmap="Reds", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, fontsize=18)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=18)
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iat[i, j]
            color = "white" if val >= 0.45 else "#5b0f16"
            ax.text(j, i, annot.iat[i, j], ha="center", va="center", fontsize=19, fontweight="bold", color=color)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title("Uncontrolled pooled type-count vs frequency correlation", fontsize=15, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    fig.tight_layout()

    for suffix in [".png", ".pdf"]:
        fig.savefig(out_base.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = load_counts(args.input)
    correlations = corr_rows(counts)

    counts.to_csv(args.output_dir / "uncontrolled_type_frequency_counts.csv", index=False)
    correlations.to_csv(args.output_dir / "uncontrolled_type_frequency_correlations.csv", index=False)
    save_scatter(counts, correlations, args.output_dir / "uncontrolled_type_frequency_scatter")
    save_rank_scatter(counts, correlations, args.output_dir / "uncontrolled_type_frequency_rank_scatter")
    save_heatmap(correlations, args.output_dir / "uncontrolled_type_frequency_heatmap")

    print(correlations.to_string(index=False))
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
