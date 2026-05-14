#!/usr/bin/env python3
"""Create POS open/closed-class plots from saved per-POS summary tables."""

from __future__ import annotations

import argparse
import math
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
DEFAULT_MEAN_SOURCE_ROOT = PROJECT_ROOT / "plots_extra" / "metrics" / "features" / "pos" / "mean"

OPEN_CLASS_POS = {"ADJ", "ADV", "INTJ", "NOUN", "PROPN", "VERB"}
CLOSED_CLASS_POS = {"ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ"}

MODEL_SPECS = {
    "bert": "bert-base-uncased",
    "gpt": "gpt2",
}

MODEL_TABLE_SLUGS = {
    "bert": "bert",
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

CONTRAST_COLOR = "#2F5D9E"
CI_Z = 1.96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Directory containing pos_raw_<metric>_<model>.csv files.",
    )
    parser.add_argument(
        "--source-format",
        choices=("auto", "bootstrap", "mean"),
        default="auto",
        help=(
            "Input table layout. 'auto' prefers the newer mean tables when present, "
            "then falls back to legacy bootstrap tables."
        ),
    )
    parser.add_argument(
        "--mean-source-root",
        type=Path,
        default=DEFAULT_MEAN_SOURCE_ROOT,
        help="Root containing <model>/tables/<metric>_<model>_pos.csv mean tables.",
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


def mean_source_path(mean_source_root: Path, model_short: str, metric_name: str) -> Path:
    model_slug = MODEL_TABLE_SLUGS[model_short]
    return mean_source_root / model_slug / "tables" / f"{metric_name}_{model_slug}_pos.csv"


def mean_tables_available(mean_source_root: Path) -> bool:
    return all(
        mean_source_path(mean_source_root, model_short, metric_name).exists()
        for model_short in MODEL_SPECS
        for metric_name, _ylabel in METRIC_SPECS.values()
    )


def select_source(args: argparse.Namespace) -> tuple[str, Path]:
    if args.source_dir is not None and args.source_format == "mean":
        raise ValueError("--source-dir is only used with bootstrap tables; use --mean-source-root for mean tables")

    if args.source_format == "mean":
        if not mean_tables_available(args.mean_source_root):
            raise FileNotFoundError(f"missing one or more POS mean tables under: {args.mean_source_root}")
        return "mean", args.mean_source_root

    if args.source_format == "bootstrap":
        return "bootstrap", resolve_source_dir(args.source_dir)

    if args.source_dir is not None:
        return "bootstrap", resolve_source_dir(args.source_dir)

    if mean_tables_available(args.mean_source_root):
        return "mean", args.mean_source_root

    return "bootstrap", resolve_source_dir(None)


def source_table_path(
    source_kind: str,
    source_root: Path,
    model_short: str,
    model_name: str,
    metric_name: str,
) -> Path:
    if source_kind == "mean":
        return mean_source_path(source_root, model_short, metric_name)
    if source_kind == "bootstrap":
        return source_root / f"pos_raw_{metric_name}_{model_name}.csv"
    raise ValueError(f"unknown source kind: {source_kind}")


def warn_missing_requested_classes(df: pd.DataFrame, csv_path: Path) -> None:
    available = set(df["class"].astype(str))
    for group_name, classes in (
        ("open", OPEN_CLASS_POS),
        ("closed", CLOSED_CLASS_POS),
    ):
        missing = sorted(classes - available)
        if missing:
            print(f"warning: {csv_path} missing requested {group_name} POS: {', '.join(missing)}")


def ci_to_se(row: pd.Series) -> float:
    """Approximate a standard error from a two-sided 95% confidence interval."""
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    if not (math.isfinite(low) and math.isfinite(high)):
        return 0.0
    width = max(high - low, 0.0)
    return width / (2.0 * CI_Z)


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

        normalized_weights = weights / weight_sum
        class_ses = layer_df.apply(ci_to_se, axis=1).astype(float)
        group_mean = float((layer_df["mean"] * normalized_weights).sum())
        # Approximate uncertainty from the available per-POS bootstrap summaries.
        # The original bootstrap replicates are not stored in the CSVs, so the
        # contrast CI below propagates the reported POS-level intervals.
        group_se = float(math.sqrt(((normalized_weights * class_ses) ** 2).sum()))
        if group_se > 0:
            group_ci_low = group_mean - CI_Z * group_se
            group_ci_high = group_mean + CI_Z * group_se
            ci_method = "normal_approx_from_pos_level_ci"
        else:
            group_ci_low = group_mean
            group_ci_high = group_mean
            ci_method = "no_source_uncertainty"

        rows.append(
            {
                "class_group": group_name,
                "layer": int(layer),
                "mean": group_mean,
                "ci_low": group_ci_low,
                "ci_high": group_ci_high,
                "se": group_se,
                "n_tokens": int(weight_sum),
                "pos_members": ",".join(sorted(layer_df["class"].unique())),
                "n_pos_members": int(layer_df["class"].nunique()),
                "ci_method": ci_method,
            }
        )

    return pd.DataFrame(rows)


def support_label(diff: float, ci_low: float, ci_high: float, has_uncertainty: bool) -> str:
    if has_uncertainty:
        if ci_low > 0:
            return "supports_open_greater_than_closed"
        if ci_high < 0:
            return "supports_closed_greater_than_open"
        return "inconclusive"
    if diff > 0:
        return "observed_open_greater_than_closed_no_ci"
    if diff < 0:
        return "observed_closed_greater_than_open_no_ci"
    return "observed_no_difference_no_ci"


def contrast_curve(groups_df: pd.DataFrame) -> pd.DataFrame:
    required_groups = {"Open classes", "Closed classes"}
    if not required_groups.issubset(set(groups_df["class_group"])):
        return pd.DataFrame()

    open_df = groups_df[groups_df["class_group"] == "Open classes"].set_index("layer")
    closed_df = groups_df[groups_df["class_group"] == "Closed classes"].set_index("layer")
    common_layers = sorted(set(open_df.index).intersection(set(closed_df.index)))
    rows = []
    for layer in common_layers:
        open_row = open_df.loc[layer]
        closed_row = closed_df.loc[layer]
        diff = float(open_row["mean"] - closed_row["mean"])
        se = float(math.sqrt(float(open_row["se"]) ** 2 + float(closed_row["se"]) ** 2))
        has_uncertainty = se > 0
        if has_uncertainty:
            ci_low = diff - CI_Z * se
            ci_high = diff + CI_Z * se
            z_value = diff / se
            p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
            ci_method = "normal_approx_from_independent_pos_group_ci"
        else:
            ci_low = diff
            ci_high = diff
            z_value = math.nan
            p_value = math.nan
            ci_method = "no_source_uncertainty"

        rows.append(
            {
                "layer": int(layer),
                "open_mean": float(open_row["mean"]),
                "closed_mean": float(closed_row["mean"]),
                "open_minus_closed": diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "se": se,
                "z_value": z_value,
                "p_value_two_sided": p_value,
                "support": support_label(diff, ci_low, ci_high, has_uncertainty),
                "ci_method": ci_method,
            }
        )

    return pd.DataFrame(rows)


def summarize_contrast(
    contrast_df: pd.DataFrame,
    model_name: str,
    metric_name: str,
    metric_label: str,
    ylabel: str,
) -> pd.DataFrame:
    if contrast_df.empty:
        return pd.DataFrame()

    rows = []
    last_layer = int(contrast_df["layer"].max())
    for scope, scoped in (
        ("last_layer", contrast_df[contrast_df["layer"] == last_layer]),
        ("layer_mean", contrast_df),
    ):
        diff = float(scoped["open_minus_closed"].mean())
        se_values = scoped["se"].astype(float)
        has_uncertainty = bool((se_values > 0).any())
        if scope == "last_layer":
            ci_low = float(scoped["ci_low"].iloc[0])
            ci_high = float(scoped["ci_high"].iloc[0])
            se = float(scoped["se"].iloc[0])
            p_value = float(scoped["p_value_two_sided"].iloc[0])
            layer = last_layer
        elif has_uncertainty:
            # Descriptive layer-mean CI. Layers are correlated, so use it as a
            # compact uncertainty summary rather than an independent-layer test.
            se = float(math.sqrt((se_values**2).sum()) / len(scoped))
            ci_low = diff - CI_Z * se
            ci_high = diff + CI_Z * se
            p_value = math.erfc(abs(diff / se) / math.sqrt(2.0)) if se > 0 else math.nan
            layer = math.nan
        else:
            se = 0.0
            ci_low = diff
            ci_high = diff
            p_value = math.nan
            layer = math.nan

        rows.append(
            {
                "model": model_name,
                "metric": metric_name,
                "metric_label": metric_label,
                "metric_family": ylabel,
                "scope": scope,
                "layer": layer,
                "open_minus_closed": diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "se": se,
                "p_value_two_sided": p_value,
                "n_layers": int(len(scoped)),
                "n_positive_layers": int((scoped["open_minus_closed"] > 0).sum()),
                "n_supported_layers": int((scoped["support"] == "supports_open_greater_than_closed").sum()),
                "support": support_label(diff, ci_low, ci_high, has_uncertainty),
                "ci_method": "normal_approx_from_contrast_ci" if has_uncertainty else "no_source_uncertainty",
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


def plot_contrast(contrast_df: pd.DataFrame, ylabel: str, title: str, out_base: Path) -> None:
    if contrast_df.empty:
        return

    sub = contrast_df.sort_values("layer")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = sub["layer"].to_numpy()
    diff = sub["open_minus_closed"].to_numpy()
    lo = sub["ci_low"].to_numpy()
    hi = sub["ci_high"].to_numpy()
    ax.axhline(0.0, color="#333333", lw=1.0, ls="--", alpha=0.8)
    ax.plot(x, diff, label="Open - closed", lw=2.2, color=CONTRAST_COLOR)
    if (hi - lo).max() > 0:
        ax.fill_between(x, lo, hi, alpha=0.18, color=CONTRAST_COLOR)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Open - closed {ylabel}")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=240)
    plt.close(fig)


def plot_contrast_grid(contrasts_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    if contrasts_df.empty:
        return []

    model_order = list(MODEL_SPECS.keys())
    metric_order = list(METRIC_SPECS.keys())
    fig, axes = plt.subplots(
        len(model_order),
        len(metric_order),
        figsize=(13.5, 6.8),
        sharex=False,
        squeeze=False,
    )

    for row_idx, model_short in enumerate(model_order):
        for col_idx, metric_label in enumerate(metric_order):
            ax = axes[row_idx][col_idx]
            sub = contrasts_df[
                (contrasts_df["model_short"] == model_short)
                & (contrasts_df["metric_label"] == metric_label)
            ].sort_values("layer")
            if sub.empty:
                ax.axis("off")
                continue

            x = sub["layer"].to_numpy()
            diff = sub["open_minus_closed"].to_numpy()
            lo = sub["ci_low"].to_numpy()
            hi = sub["ci_high"].to_numpy()
            ax.axhline(0.0, color="#333333", lw=0.9, ls="--", alpha=0.8)
            ax.plot(x, diff, lw=2.0, color=CONTRAST_COLOR)
            if (hi - lo).max() > 0:
                ax.fill_between(x, lo, hi, alpha=0.18, color=CONTRAST_COLOR)
            ax.margins(x=0.03)
            ax.set_xlabel("Layer")
            if col_idx == 0:
                ax.set_ylabel(f"{model_short.upper()}\nOpen - closed")
            if row_idx == 0:
                ax.set_title(METRIC_DISPLAY_LABELS[metric_label])

    fig.suptitle("POS open-class minus closed-class contrast", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_base = output_dir / "pos_open_minus_closed_contrast_grid"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [out_base.with_suffix(".pdf"), out_base.with_suffix(".png")]


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


def format_number(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    abs_value = abs(value)
    if abs_value != 0 and abs_value < 0.001:
        return f"{value:.2e}"
    if abs_value >= 100:
        return f"{value:.1f}"
    return f"{value:.4g}"


def write_markdown_summary(summary_df: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# POS Open-vs-Closed Contrast",
        "",
        "**Hypothesis.** Open-class POS (`ADJ`, `ADV`, `INTJ`, `NOUN`, `PROPN`, `VERB`) have higher geometry scores than closed-class POS (`ADP`, `AUX`, `CCONJ`, `DET`, `NUM`, `PART`, `PRON`, `SCONJ`). The contrast is operationalized as a token-weighted open-class mean minus a token-weighted closed-class mean at each layer.",
        "",
        "**Decision rule.** A layer-level contrast is marked as supporting the hypothesis when the approximate 95% confidence interval for open-minus-closed is strictly above zero. If the source POS table has zero-width intervals, the result is reported as an observed contrast without an uncertainty decision.",
        "",
        "## Main Summary",
        "",
        "| Model | Metric | Scope | Contrast | 95% CI | Support |",
        "|---|---|---|---:|---:|---|",
    ]

    for _, row in summary_df.sort_values(["model", "metric_label", "scope"]).iterrows():
        ci = f"[{format_number(row['ci_low'])}, {format_number(row['ci_high'])}]"
        lines.append(
            "| {model} | {metric} | {scope} | {diff} | {ci} | {support} |".format(
                model=row["model"],
                metric=row["metric_label"],
                scope=row["scope"].replace("_", " "),
                diff=format_number(row["open_minus_closed"]),
                ci=ci,
                support=str(row["support"]).replace("_", " "),
            )
        )

    lines.extend(
        [
            "",
            "## Manuscript-Ready Wording",
            "",
            "To make the POS observation directly quantitative, we collapsed POS tags into two predefined groups: open-class tags (`ADJ`, `ADV`, `INTJ`, `NOUN`, `PROPN`, `VERB`) and closed-class tags (`ADP`, `AUX`, `CCONJ`, `DET`, `NUM`, `PART`, `PRON`, `SCONJ`). For each model, metric, and layer, we computed the token-weighted mean score for each group and tested the directional contrast open minus closed. This turns the qualitative reading of the POS curves into a layer-wise hypothesis test against zero.",
            "",
        ]
    )

    no_ci_models = sorted(summary_df.loc[summary_df["ci_method"].eq("no_source_uncertainty"), "model"].unique())
    all_layers_supported = bool((summary_df["n_supported_layers"] == summary_df["n_layers"]).all())
    if no_ci_models:
        lines.append(
            "The available POS-level intervals support a positive open-minus-closed contrast where the approximate confidence interval is strictly above zero. Rows for "
            + ", ".join(no_ci_models)
            + " contain zero-width intervals and should be described as observed contrasts unless those metric tables are regenerated with bootstrap replicates."
        )
    elif all_layers_supported:
        lines.append(
            "For both models, the available POS-level intervals support a positive open-minus-closed contrast in all reported layers and metric families."
        )
    else:
        lines.append(
            "The available POS-level intervals support a positive open-minus-closed contrast for rows whose approximate confidence interval is strictly above zero; rows crossing zero should be described as inconclusive."
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_kind, source_root = select_source(args)
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    summary_frames: list[pd.DataFrame] = []
    contrast_frames: list[pd.DataFrame] = []
    for model_short, model_name in MODEL_SPECS.items():
        metric_tables: dict[str, tuple[pd.DataFrame, str]] = {}
        for metric_label, (metric_name, ylabel) in METRIC_SPECS.items():
            csv_path = source_table_path(source_kind, source_root, model_short, model_name, metric_name)
            if not csv_path.exists():
                print(f"skip missing table: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            warn_missing_requested_classes(df, csv_path)
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

            contrast = contrast_curve(groups)
            if not contrast.empty:
                contrast.insert(0, "metric_label", metric_label)
                contrast.insert(0, "metric", metric_name)
                contrast.insert(0, "model", model_name)
                contrast["hypothesis"] = "open_class_greater_than_closed_class"
                contrast["aggregation"] = "token_weighted_open_minus_closed"
                contrast_out = tables_dir / f"{model_short}_pos_open_minus_closed_{metric_label}.csv"
                contrast.to_csv(contrast_out, index=False)
                written.append(contrast_out)
                summary_frames.append(summarize_contrast(contrast, model_name, metric_name, metric_label, ylabel))
                grid_contrast = contrast.copy()
                grid_contrast["model_short"] = model_short
                grid_contrast["metric_family"] = ylabel
                contrast_frames.append(grid_contrast)

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

            plot_contrast(
                contrast,
                ylabel=ylabel,
                title=f"{model_short.upper()} POS open - closed classes ({ylabel})",
                out_base=output_dir / f"{model_short}_pos_open_minus_closed_{metric_label}",
            )
            written.extend(
                [
                    output_dir / f"{model_short}_pos_open_minus_closed_{metric_label}.pdf",
                    output_dir / f"{model_short}_pos_open_minus_closed_{metric_label}.png",
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

    if contrast_frames:
        written.extend(plot_contrast_grid(pd.concat(contrast_frames, ignore_index=True), output_dir))

    if summary_frames:
        summary = pd.concat(summary_frames, ignore_index=True)
        summary_out = tables_dir / "pos_open_minus_closed_summary.csv"
        summary.to_csv(summary_out, index=False)
        written.append(summary_out)
        markdown_out = tables_dir / "paper_section_pos_open_closed_contrasts.md"
        write_markdown_summary(summary, markdown_out)
        written.append(markdown_out)

    print(f"source format: {source_kind}")
    print(f"source tables: {source_root}")
    print(f"wrote {len(written)} files")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
