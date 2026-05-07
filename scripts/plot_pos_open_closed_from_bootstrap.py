#!/usr/bin/env python3
"""Create POS open/closed-class plots from saved per-POS bootstrap tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIRS = (
    PROJECT_ROOT / "tables_POS" / "pos_bootstrap",
    PROJECT_ROOT.parent
    / "linguistic_profiling_of_the_geometry_copy"
    / "geometric_profiling_of_a_neural_language_model"
    / "bert"
    / "tables_POS"
    / "pos_bootstrap",
)

OPEN_CLASS_POS = {"ADJ", "ADV", "NOUN", "PROPN", "VERB"}
CLOSED_CLASS_POS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}

MODEL_SPECS = {
    "bert": "bert-base-uncased",
    "gpt": "gpt2",
}

METRIC_SPECS = {
    "isoscore": ("iso", "Isotropy"),
    "LID": ("lpca99", "Linear ID"),
    "NLID": ("gride", "Nonlinear ID"),
}

METRIC_DISPLAY_LABELS = {
    "isoscore": "IsoScore",
    "LID": "Linear ID",
    "NLID": "Nonlinear ID",
}

GROUP_COLORS = {
    "Open classes": "#287C8E",
    "Closed classes": "#D65F4A",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing pos_raw_<metric>_<model>.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "plots_main" / "plots_features" / "pos_open_closed",
        help="Directory for generated plots and aggregate tables.",
    )
    return parser.parse_args()


def resolve_source_dir(source_dir: Path | None) -> Path:
    if source_dir is not None:
        if not source_dir.exists():
            raise FileNotFoundError(f"source directory does not exist: {source_dir}")
        return source_dir

    for candidate in DEFAULT_SOURCE_DIRS:
        if candidate.exists() and any(candidate.glob("pos_raw_*_*.csv")):
            return candidate

    candidates = "\n".join(str(path) for path in DEFAULT_SOURCE_DIRS)
    raise FileNotFoundError(f"no POS bootstrap tables found in:\n{candidates}")


def weighted_group_curve(df: pd.DataFrame, classes: set[str], group_name: str) -> pd.DataFrame:
    sub = df[df["class"].isin(classes)].copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for layer, layer_df in sub.groupby("layer", sort=True):
        weights = layer_df["n_tokens"].astype(float)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue

        rows.append(
            {
                "class_group": group_name,
                "layer": int(layer),
                "mean": float((layer_df["mean"] * weights).sum() / weight_sum),
                "ci_low": float((layer_df["ci_low"] * weights).sum() / weight_sum),
                "ci_high": float((layer_df["ci_high"] * weights).sum() / weight_sum),
                "n_tokens": int(weight_sum),
                "pos_members": ",".join(sorted(layer_df["class"].unique())),
            }
        )

    return pd.DataFrame(rows)


def plot_grouped(groups_df: pd.DataFrame, ylabel: str, title: str, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for group_name, sub in groups_df.groupby("class_group", sort=False):
        sub = sub.sort_values("layer")
        color = GROUP_COLORS.get(group_name)
        x = sub["layer"].to_numpy()
        mean = sub["mean"].to_numpy()
        lo = sub["ci_low"].to_numpy()
        hi = sub["ci_high"].to_numpy()
        ax.plot(x, mean, label=group_name, lw=2.2, color=color)
        ax.fill_between(x, lo, hi, alpha=0.15, color=color)

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=240)
    plt.close(fig)


def plot_pos_subset(
    df: pd.DataFrame,
    classes: set[str],
    ylabel: str,
    title: str,
    out_base: Path,
) -> None:
    sub = df[df["class"].isin(classes)].copy()
    if sub.empty:
        return

    present_classes = sorted(sub["class"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {klass: cmap(i % 10) for i, klass in enumerate(present_classes)}

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for klass in present_classes:
        class_df = sub[sub["class"] == klass].sort_values("layer")
        x = class_df["layer"].to_numpy()
        mean = class_df["mean"].to_numpy()
        lo = class_df["ci_low"].to_numpy()
        hi = class_df["ci_high"].to_numpy()
        ax.plot(x, mean, label=klass, lw=1.8, color=colors[klass])
        ax.fill_between(x, lo, hi, alpha=0.12, color=colors[klass])

    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=3 if len(present_classes) > 5 else 2, fontsize="small", frameon=False)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=240)
    plt.close(fig)


def plot_histogram_summary(
    metric_tables: dict[str, tuple[pd.DataFrame, str]],
    model_short: str,
    output_dir: Path,
    layer_mode: str,
) -> list[Path]:
    if not metric_tables:
        return []

    sns.set_style("darkgrid")
    sns.set_context("paper", font_scale=1.8)

    fig, axes = plt.subplots(1, len(METRIC_SPECS), figsize=(13.0, 4.6))
    if len(METRIC_SPECS) == 1:
        axes = [axes]

    written: list[Path] = []
    for ax, (metric_label, (_metric_name, ylabel)) in zip(axes, METRIC_SPECS.items()):
        if metric_label not in metric_tables:
            ax.axis("off")
            continue

        groups, _table_path = metric_tables[metric_label]
        if layer_mode == "last":
            target_layer = int(groups["layer"].max())
            plot_df = groups[groups["layer"] == target_layer].copy()
            title_suffix = f"layer {target_layer}"
        elif layer_mode == "mean":
            plot_df = (
                groups.groupby("class_group", as_index=False)
                .agg(mean=("mean", "mean"), ci_low=("ci_low", "mean"), ci_high=("ci_high", "mean"))
            )
            title_suffix = "layer mean"
        else:
            raise ValueError("layer_mode must be 'last' or 'mean'")

        order = ["Open classes", "Closed classes"]
        plot_df["class_group"] = pd.Categorical(plot_df["class_group"], categories=order, ordered=True)
        plot_df = plot_df.sort_values("class_group")

        labels = ["Open", "Closed"]
        colors = [GROUP_COLORS.get(group, "#777777") for group in plot_df["class_group"].astype(str)]
        bars = ax.barh(
            labels,
            plot_df["mean"],
            color=colors,
            edgecolor="white",
            linewidth=1.2,
            height=0.58,
        )

        xerr_low = plot_df["mean"] - plot_df["ci_low"]
        xerr_high = plot_df["ci_high"] - plot_df["mean"]
        ax.errorbar(
            plot_df["mean"],
            range(len(plot_df)),
            xerr=[xerr_low, xerr_high],
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
        )

        xmax = float(plot_df["ci_high"].max())
        xmin = min(0.0, float(plot_df["ci_low"].min()))
        xrange = max(xmax - xmin, xmax, 1e-9)
        ax.set_xlim(xmin, xmax + 0.24 * xrange)

        for bar, value in zip(bars, plot_df["mean"]):
            ax.text(
                bar.get_width() + 0.045 * xrange,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3g}",
                ha="left",
                va="center",
                fontsize=13,
                fontweight="semibold",
                color="#2A2A2A",
            )

        ax.set_xlabel(ylabel)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()

    fig.tight_layout(rect=(0, 0.02, 1, 0.98), w_pad=2.0)
    out_base = output_dir / f"{model_short}_pos_open_closed_histogram_{layer_mode}"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    written.extend([out_base.with_suffix(".pdf"), out_base.with_suffix(".png")])
    return written


def main() -> None:
    args = parse_args()
    source_dir = resolve_source_dir(args.source_dir)
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for model_short, model_name in MODEL_SPECS.items():
        metric_tables: dict[str, tuple[pd.DataFrame, str]] = {}
        for metric_label, (metric_name, ylabel) in METRIC_SPECS.items():
            csv_path = source_dir / f"pos_raw_{metric_name}_{model_name}.csv"
            if not csv_path.exists():
                print(f"skip missing table: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            groups = pd.concat(
                [
                    weighted_group_curve(df, OPEN_CLASS_POS, "Open classes"),
                    weighted_group_curve(df, CLOSED_CLASS_POS, "Closed classes"),
                ],
                ignore_index=True,
            )
            if groups.empty:
                print(f"skip empty groups for: {csv_path}")
                continue

            groups.insert(0, "metric", metric_name)
            groups.insert(0, "model", model_name)
            groups["aggregation"] = "token_weighted_mean_of_pos_curves"

            table_out = tables_dir / f"{model_short}_pos_open_closed_{metric_label}.csv"
            groups.to_csv(table_out, index=False)
            metric_tables[metric_label] = (groups, str(table_out))
            written.append(table_out)

            plot_grouped(
                groups,
                ylabel=ylabel,
                title=f"{model_short.upper()} POS open vs closed classes ({ylabel})",
                out_base=output_dir / f"{model_short}_pos_open_closed_{metric_label}",
            )
            written.extend(
                [
                    output_dir / f"{model_short}_pos_open_closed_{metric_label}.pdf",
                    output_dir / f"{model_short}_pos_open_closed_{metric_label}.png",
                ]
            )

            plot_pos_subset(
                df,
                OPEN_CLASS_POS,
                ylabel=ylabel,
                title=f"{model_short.upper()} open-class POS ({ylabel})",
                out_base=output_dir / f"{model_short}_pos_open_classes_{metric_label}",
            )
            written.extend(
                [
                    output_dir / f"{model_short}_pos_open_classes_{metric_label}.pdf",
                    output_dir / f"{model_short}_pos_open_classes_{metric_label}.png",
                ]
            )

            plot_pos_subset(
                df,
                CLOSED_CLASS_POS,
                ylabel=ylabel,
                title=f"{model_short.upper()} closed-class POS ({ylabel})",
                out_base=output_dir / f"{model_short}_pos_closed_classes_{metric_label}",
            )
            written.extend(
                [
                    output_dir / f"{model_short}_pos_closed_classes_{metric_label}.pdf",
                    output_dir / f"{model_short}_pos_closed_classes_{metric_label}.png",
                ]
            )

        written.extend(plot_histogram_summary(metric_tables, model_short, output_dir, "last"))
        written.extend(plot_histogram_summary(metric_tables, model_short, output_dir, "mean"))

    print(f"source tables: {source_dir}")
    print(f"wrote {len(written)} files")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
