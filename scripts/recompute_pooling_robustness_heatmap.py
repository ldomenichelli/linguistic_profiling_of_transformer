#!/usr/bin/env python3
"""Recompute the mean-vs-first/last pooling robustness recap heatmap.

The older recap digitized the saved mean PNGs because table exports were not
available for every mean-pooling feature plot. The recomputation queue now
writes mean tables, so this script compares those tables directly against the
legacy first/last pooling tables and refreshes the existing recap filenames.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEAN_ROOT = PROJECT_ROOT / "plots_extra" / "metrics" / "features"
DEFAULT_OUT_DIR = PROJECT_ROOT / "plots_extra" / "plot_recap"
DEFAULT_LEGACY_ROOT = (
    PROJECT_ROOT.parent
    / "linguistic_profiling_of_the_geometry_copy"
    / "geometric_profiling_of_a_neural_language_model"
)


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    mean_dir: str
    suffix: str
    legacy_prefix: str
    legacy_table_dir: str


FEATURES = {
    "pos": FeatureSpec("pos", "pos", "pos", "pos", "tables_POS/pos_bootstrap"),
    "length": FeatureSpec("length", "length", "length", "length", "tables_LENGTH/length_bootstrap"),
    "index": FeatureSpec("index", "index", "index", "index", "tables_INDEX/index_bootstrap"),
    "relation_type": FeatureSpec(
        "relation_type",
        "relations",
        "relation",
        "relation",
        "tables_REL/relation_bootstrap",
    ),
    "arity": FeatureSpec("arity", "arity", "arity", "arity", "tables_ARITY/arity_bootstrap"),
}

METRICS = {
    "IsoScore": ("iso", "iso"),
    "LID": ("lpca99", "lpca99"),
    "NLID": ("gride", "gride"),
}
METRIC_ORDER = ["IsoScore", "LID", "NLID"]

COMPARISONS = {
    "bert": ("BERT: MEAN vs FIRST", "bert-base-uncased"),
    "gpt2": ("GPT-2: MEAN vs LAST", "gpt2"),
}
COMPARISON_ORDER = [COMPARISONS["bert"][0], COMPARISONS["gpt2"][0]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-root", type=Path, default=DEFAULT_MEAN_ROOT)
    parser.add_argument("--legacy-root", type=Path, default=DEFAULT_LEGACY_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--features",
        nargs="+",
        default=["pos", "length", "index", "relation_type", "arity"],
        choices=sorted(FEATURES),
        help="Feature tables to include in the pooled recap.",
    )
    parser.add_argument("--base-name", default="feature_png_digitized_pooling_robustness_heatmap")
    return parser.parse_args()


def normalize_class(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


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


def exact_metric_for(feature: str, metric_label: str, default_exact: str) -> str:
    if metric_label == "LID" and feature == "arity":
        return "pca99"
    return default_exact


def mean_table_path(mean_root: Path, model: str, feature: str, mean_metric: str) -> Path:
    spec = FEATURES[feature]
    return mean_root / spec.mean_dir / "mean" / model / "tables" / f"{mean_metric}_{model}_{spec.suffix}.csv"


def legacy_table_path(legacy_root: Path, model: str, feature: str, exact_metric: str) -> Path:
    spec = FEATURES[feature]
    if model == "bert":
        base = legacy_root / "bert" / spec.legacy_table_dir
        model_suffix = "bert-base-uncased"
    elif feature == "pos":
        base = legacy_root / "bert" / "tables_POS" / "pos_bootstrap"
        model_suffix = "gpt2"
    elif feature == "relation_type":
        base = legacy_root / "gpt2_no_index" / "tables_REL_GPT2_no_idx0" / "relation_bootstrap"
        model_suffix = "gpt2"
    else:
        base = legacy_root / "gpt2" / spec.legacy_table_dir
        model_suffix = "gpt2"
    return base / f"{spec.legacy_prefix}_raw_{exact_metric}_{model_suffix}.csv"


def load_table(path: Path, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"class", "layer", "mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    if "subset" in df.columns:
        df = df[df["subset"].astype(str) == "raw"].copy()
    out = pd.DataFrame(
        {
            "class_key": df["class"].map(normalize_class),
            "class": df["class"].map(normalize_class),
            "layer": pd.to_numeric(df["layer"], errors="coerce"),
            value_col: pd.to_numeric(df["mean"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["class_key", "layer", value_col])
    out["layer"] = out["layer"].astype(int)
    out = out.groupby(["class_key", "class", "layer"], as_index=False)[value_col].mean()
    return out


def rank_within_plot(values: pd.Series) -> pd.Series:
    return values.rank(method="average", ascending=True)


def finite_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    valid = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    if int(valid.sum()) < 3:
        return float("nan"), float("nan")
    if pd.Series(x.to_numpy(dtype=float)[valid]).nunique() < 2:
        return float("nan"), float("nan")
    if pd.Series(y.to_numpy(dtype=float)[valid]).nunique() < 2:
        return float("nan"), float("nan")
    rho, p_value = spearmanr(x.to_numpy(dtype=float)[valid], y.to_numpy(dtype=float)[valid])
    return float(rho), float(p_value)


def build_points(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_frames = []
    summaries = []

    for model, (comparison, _) in COMPARISONS.items():
        for feature in args.features:
            for metric_label, (mean_metric, default_exact_metric) in METRICS.items():
                exact_metric = exact_metric_for(feature, metric_label, default_exact_metric)
                mean_path = mean_table_path(args.mean_root, model, feature, mean_metric)
                exact_path = legacy_table_path(args.legacy_root, model, feature, exact_metric)
                if not mean_path.exists():
                    raise FileNotFoundError(f"Missing mean table: {mean_path}")
                if not exact_path.exists():
                    raise FileNotFoundError(f"Missing legacy table: {exact_path}")

                mean_df = load_table(mean_path, "mean_value")
                exact_df = load_table(exact_path, "reference_value")
                merged = mean_df.merge(
                    exact_df[["class_key", "layer", "reference_value"]],
                    on=["class_key", "layer"],
                    how="inner",
                )
                if merged.empty:
                    raise ValueError(f"No overlapping class/layer rows for {mean_path} vs {exact_path}")

                merged["mean_rank_within_plot"] = rank_within_plot(merged["mean_value"])
                merged["reference_rank_within_plot"] = rank_within_plot(merged["reference_value"])
                rho_plot, p_plot = finite_spearman(
                    merged["mean_rank_within_plot"], merged["reference_rank_within_plot"]
                )

                merged.insert(0, "model_key", model)
                merged.insert(1, "comparison", comparison)
                merged.insert(2, "feature", feature)
                merged.insert(3, "metric_label", metric_label)
                merged.insert(4, "metric_key", mean_metric)
                merged.insert(5, "reference_metric", exact_metric)
                merged["mean_path"] = str(mean_path)
                merged["reference_path"] = str(exact_path)
                point_frames.append(merged)

                summaries.append(
                    {
                        "comparison": comparison,
                        "model_key": model,
                        "feature": feature,
                        "metric_label": metric_label,
                        "metric_key": mean_metric,
                        "reference_metric": exact_metric,
                        "n_points": int(len(merged)),
                        "rho_plot": rho_plot,
                        "p_plot": p_plot,
                        "mean_path": str(mean_path),
                        "reference_path": str(exact_path),
                    }
                )

    points = pd.concat(point_frames, ignore_index=True)
    summaries_df = pd.DataFrame(summaries)
    return points, summaries_df


def build_values(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = points.groupby(["comparison", "metric_label"], sort=False)
    for (comparison, metric_label), group in grouped:
        rho, p_value = finite_spearman(group["mean_rank_within_plot"], group["reference_rank_within_plot"])
        features = sorted(group["feature"].unique())
        rows.append(
            {
                "comparison": comparison,
                "metric": metric_label,
                "rho": rho,
                "p_value": p_value,
                "stars": significance_stars(p_value),
                "n_points": int(len(group)),
                "n_feature_plots": int(group["feature"].nunique()),
                "features_used": ", ".join(features),
            }
        )
    values = pd.DataFrame(rows)
    values["comparison"] = pd.Categorical(values["comparison"], categories=COMPARISON_ORDER, ordered=True)
    values["metric"] = pd.Categorical(values["metric"], categories=METRIC_ORDER, ordered=True)
    return values.sort_values(["comparison", "metric"]).reset_index(drop=True)


def annotation_matrix(values: pd.DataFrame) -> pd.DataFrame:
    labels = values.copy()
    labels["annotation"] = labels.apply(
        lambda row: "" if not np.isfinite(row["rho"]) else f"rho={row['rho']:.2f}{row['stars']}",
        axis=1,
    )
    return labels.pivot(index="comparison", columns="metric", values="annotation").reindex(
        index=COMPARISON_ORDER, columns=METRIC_ORDER
    )


def value_matrix(values: pd.DataFrame) -> pd.DataFrame:
    return values.pivot(index="comparison", columns="metric", values="rho").reindex(
        index=COMPARISON_ORDER, columns=METRIC_ORDER
    )


def save_heatmap(values: pd.DataFrame, out_base: Path, *, uniform: bool = False) -> None:
    matrix = value_matrix(values)
    annot = annotation_matrix(values)
    sns.set_theme(style="white")
    if uniform:
        plot_matrix = pd.DataFrame(
            np.ones(matrix.shape),
            index=matrix.index,
            columns=matrix.columns,
        )
        cmap = sns.color_palette(["#cf202f"], as_cmap=True)
        cbar = False
    else:
        plot_matrix = matrix
        cmap = "Reds"
        cbar = True

    fig, ax = plt.subplots(figsize=(12.8, 5.2))
    sns.heatmap(
        plot_matrix,
        annot=annot,
        fmt="",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        linewidths=2.0,
        linecolor="white",
        cbar=cbar,
        annot_kws={"fontsize": 20, "fontweight": "bold", "color": "white"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=20, length=0, rotation=0)
    ax.tick_params(axis="y", labelsize=18, length=0, rotation=0)
    if not uniform:
        cbar_obj = ax.collections[0].colorbar
        if cbar_obj is not None:
            cbar_obj.ax.tick_params(labelsize=14)
    fig.tight_layout()
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_base.with_suffix(suffix), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    points, summaries = build_points(args)
    values = build_values(points)

    points.to_csv(args.out_dir / f"{args.base_name}_points.csv", index=False)
    summaries.to_csv(args.out_dir / f"{args.base_name}_plot_summaries.csv", index=False)
    values.to_csv(args.out_dir / f"{args.base_name}_values.csv", index=False)

    save_heatmap(values, args.out_dir / args.base_name)
    save_heatmap(values, args.out_dir / f"{args.base_name}_red")
    save_heatmap(values, args.out_dir / f"{args.base_name}_red_uniform", uniform=True)

    print(values.to_string(index=False))
    if values["rho"].isna().any():
        missing = values[values["rho"].map(lambda value: not math.isfinite(float(value)))]
        raise RuntimeError(f"Non-finite correlations:\n{missing}")


if __name__ == "__main__":
    main()
