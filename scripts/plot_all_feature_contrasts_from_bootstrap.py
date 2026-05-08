#!/usr/bin/env python3
"""Create hypothesis-driven contrast plots for all linguistic features.

The script reads saved per-class bootstrap tables and collapses each feature
into one interpretable binary contrast:

- POS: open-class minus closed-class POS
- length: long tokens minus short tokens
- index: late sentence positions minus early positions
- arity: mid dependency arity minus low arity
- head distance: before-head tokens minus after-head tokens
- relation type: content relations minus function relations
"""

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
DEFAULT_GEOMETRY_ROOT = (
    PROJECT_ROOT.parent
    / "linguistic_profiling_of_the_geometry_copy"
    / "geometric_profiling_of_a_neural_language_model"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots_main" / "plots_features" / "feature_contrasts"

CI_Z = 1.96
CONTRAST_COLOR = "#2F5D9E"
GROUP_COLORS = {
    "positive": "#287C8E",
    "negative": "#D65F4A",
}

MODEL_NAMES = {
    "bert": "bert-base-uncased",
    "gpt": "gpt2",
}

METRIC_SPECS = {
    "isoscore": {
        "candidates": ("iso",),
        "display": "IsoScore",
        "family": "Isotropy",
    },
    "LID": {
        "candidates": ("lpca99", "pca99"),
        "display": "Linear ID",
        "family": "Linear ID",
    },
    "NLID": {
        "candidates": ("gride",),
        "display": "Nonlinear ID",
        "family": "Nonlinear ID",
    },
}

OPEN_CLASS_POS = {"ADJ", "ADV", "NOUN", "PROPN", "VERB"}
CLOSED_CLASS_POS = {"ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"}

CONTENT_RELATIONS = {
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "ccomp",
    "compound",
    "conj",
    "csubj",
    "iobj",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "root",
    "xcomp",
}
FUNCTION_RELATIONS = {"aux", "case", "cc", "cop", "det", "fixed", "mark"}

FEATURE_SPECS = {
    "pos": {
        "display": "POS",
        "prefix": "pos",
        "contrast": "open_closed",
        "positive_label": "Open classes",
        "negative_label": "Closed classes",
        "positive_short": "Open",
        "negative_short": "Closed",
        "hypothesis": "Open-class POS have higher geometry scores than closed-class POS.",
        "positive_name": "open_class",
        "negative_name": "closed_class",
    },
    "length": {
        "display": "Token Length",
        "prefix": "length",
        "contrast": "long_short",
        "positive_label": "Long tokens (length >= 7)",
        "negative_label": "Short tokens (length <= 3)",
        "positive_short": "Long",
        "negative_short": "Short",
        "hypothesis": "Long tokens have higher geometry scores than short tokens.",
        "positive_name": "long",
        "negative_name": "short",
    },
    "index": {
        "display": "Token Index",
        "prefix": "index",
        "contrast": "late_early",
        "positive_label": "Late positions (index >= 8)",
        "negative_label": "Early positions (index <= 3)",
        "positive_short": "Late",
        "negative_short": "Early",
        "hypothesis": "Late sentence positions differ from early positions.",
        "positive_name": "late",
        "negative_name": "early",
    },
    "arity": {
        "display": "Dependency Arity",
        "prefix": "arity",
        "contrast": "mid_low",
        "positive_label": "Mid arity (2-4 dependents)",
        "negative_label": "Low arity (<= 1)",
        "positive_short": "Mid",
        "negative_short": "Low",
        "hypothesis": "Mid-arity tokens (2-4 dependents) occupy higher-dimensional and more isotropic subspaces than low-arity tokens.",
        "positive_name": "mid_arity",
        "negative_name": "low_arity",
    },
    "head_dist": {
        "display": "Head Distance",
        "prefix": "headdist",
        "contrast": "head_after_before",
        "positive_label": "Before Head (dist > 0)",
        "negative_label": "After Head (dist < 0)",
        "positive_short": "Before Head",
        "negative_short": "After Head",
        "histogram_order": ["negative", "positive"],
        "hypothesis": "Tokens before their syntactic head differ from tokens after their syntactic head.",
        "positive_name": "head_follows",
        "negative_name": "head_precedes",
    },
    "relation_type": {
        "display": "Relation Type",
        "prefix": "relation",
        "contrast": "content_function",
        "positive_label": "Content relations",
        "negative_label": "Function relations",
        "positive_short": "Content",
        "negative_short": "Function",
        "hypothesis": "Content-bearing dependency relations have higher geometry scores than function relations.",
        "positive_name": "content_relation",
        "negative_name": "function_relation",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def source_dirs(geometry_root: Path) -> dict[tuple[str, str], Path]:
    return {
        ("bert", "pos"): geometry_root / "bert" / "tables_POS" / "pos_bootstrap",
        ("bert", "length"): geometry_root / "bert" / "tables_LENGTH" / "length_bootstrap",
        ("bert", "index"): geometry_root / "bert" / "tables_INDEX" / "index_bootstrap",
        ("bert", "arity"): geometry_root / "bert" / "tables_ARITY" / "arity_bootstrap",
        ("bert", "head_dist"): geometry_root / "bert" / "tables_HEADDIST" / "headdist_bootstrap",
        ("bert", "relation_type"): geometry_root / "bert" / "tables_REL" / "relation_bootstrap",
        ("gpt", "pos"): geometry_root / "bert" / "tables_POS" / "pos_bootstrap",
        ("gpt", "length"): geometry_root / "gpt2_no_index" / "tables_LENGTH_no_index" / "length_bootstrap",
        ("gpt", "index"): geometry_root / "gpt2" / "tables_INDEX" / "index_bootstrap",
        ("gpt", "arity"): geometry_root / "gpt2_no_index" / "tables_ARITY_no_index" / "arity_bootstrap",
        ("gpt", "head_dist"): geometry_root / "gpt2_no_index" / "tables_HEADDIST_no_index" / "headdist_bootstrap",
        ("gpt", "relation_type"): geometry_root
        / "gpt2_no_index"
        / "tables_REL_GPT2_no_idx0"
        / "relation_bootstrap",
    }


def normalize_class(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def numeric_class(value: object) -> int | None:
    text = normalize_class(value)
    try:
        return int(text)
    except ValueError:
        return None


def class_group(feature: str, value: object) -> str | None:
    text = normalize_class(value)
    lower = text.lower()
    number = numeric_class(value)

    if feature == "pos":
        if text in OPEN_CLASS_POS:
            return "positive"
        if text in CLOSED_CLASS_POS:
            return "negative"
        return None
    if feature == "length":
        if number is None:
            return None
        if number >= 7:
            return "positive"
        if number <= 3:
            return "negative"
        return None
    if feature == "index":
        if number is None:
            return None
        if number >= 8:
            return "positive"
        if number <= 3:
            return "negative"
        return None
    if feature == "arity":
        if number is None:
            return None
        if 2 <= number <= 4:
            return "positive"
        if number <= 1:
            return "negative"
        return None
    if feature == "head_dist":
        if number is None:
            return None
        if number > 0:
            return "positive"
        if number < 0:
            return "negative"
        return None
    if feature == "relation_type":
        if lower in CONTENT_RELATIONS:
            return "positive"
        if lower in FUNCTION_RELATIONS:
            return "negative"
        return None
    raise KeyError(feature)


def ci_to_se(row: pd.Series) -> float:
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    if not (math.isfinite(low) and math.isfinite(high)):
        return 0.0
    return max(high - low, 0.0) / (2.0 * CI_Z)


def metric_file(source_dir: Path, feature: str, metric_label: str, model_name: str) -> tuple[Path | None, str | None]:
    prefix = FEATURE_SPECS[feature]["prefix"]
    for metric_name in METRIC_SPECS[metric_label]["candidates"]:
        path = source_dir / f"{prefix}_raw_{metric_name}_{model_name}.csv"
        if path.exists():
            return path, metric_name
    return None, None


def support_label(diff: float, ci_low: float, ci_high: float, has_uncertainty: bool) -> str:
    if has_uncertainty:
        if ci_low > 0:
            return "supports_positive_group_greater"
        if ci_high < 0:
            return "supports_negative_group_greater"
        return "inconclusive"
    if diff > 0:
        return "observed_positive_group_greater_no_ci"
    if diff < 0:
        return "observed_negative_group_greater_no_ci"
    return "observed_no_difference_no_ci"


def weighted_group_curves(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    spec = FEATURE_SPECS[feature]
    work = df.copy()
    work["normalized_class"] = work["class"].map(normalize_class)
    work["group_key"] = work["class"].map(lambda value: class_group(feature, value))
    work = work[work["group_key"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    label_map = {
        "positive": spec["positive_label"],
        "negative": spec["negative_label"],
    }
    rows = []
    for (group_key, layer), layer_df in work.groupby(["group_key", "layer"], sort=True):
        weights = layer_df["n_tokens"].astype(float)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue

        normalized_weights = weights / weight_sum
        class_ses = layer_df.apply(ci_to_se, axis=1).astype(float)
        mean = float((layer_df["mean"] * normalized_weights).sum())
        se = float(math.sqrt(((normalized_weights * class_ses) ** 2).sum()))
        if se > 0:
            ci_low = mean - CI_Z * se
            ci_high = mean + CI_Z * se
            ci_method = "normal_approx_from_class_level_ci"
        else:
            ci_low = mean
            ci_high = mean
            ci_method = "no_source_uncertainty"

        rows.append(
            {
                "group_key": group_key,
                "class_group": label_map[group_key],
                "layer": int(layer),
                "mean": mean,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "se": se,
                "n_tokens": int(weight_sum),
                "class_members": ",".join(sorted(layer_df["normalized_class"].unique())),
                "n_class_members": int(layer_df["normalized_class"].nunique()),
                "ci_method": ci_method,
            }
        )

    return pd.DataFrame(rows)


def contrast_curve(groups_df: pd.DataFrame) -> pd.DataFrame:
    if groups_df.empty:
        return pd.DataFrame()
    if not {"positive", "negative"}.issubset(set(groups_df["group_key"])):
        return pd.DataFrame()

    positive_df = groups_df[groups_df["group_key"] == "positive"].set_index("layer")
    negative_df = groups_df[groups_df["group_key"] == "negative"].set_index("layer")
    rows = []
    for layer in sorted(set(positive_df.index).intersection(set(negative_df.index))):
        positive = positive_df.loc[layer]
        negative = negative_df.loc[layer]
        diff = float(positive["mean"] - negative["mean"])
        se = float(math.sqrt(float(positive["se"]) ** 2 + float(negative["se"]) ** 2))
        has_uncertainty = se > 0
        if has_uncertainty:
            ci_low = diff - CI_Z * se
            ci_high = diff + CI_Z * se
            z_value = diff / se
            p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
            ci_method = "normal_approx_from_independent_group_ci"
        else:
            ci_low = diff
            ci_high = diff
            z_value = math.nan
            p_value = math.nan
            ci_method = "no_source_uncertainty"

        rows.append(
            {
                "layer": int(layer),
                "positive_mean": float(positive["mean"]),
                "negative_mean": float(negative["mean"]),
                "positive_minus_negative": diff,
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


def summarize_contrast(contrast_df: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    if contrast_df.empty:
        return pd.DataFrame()

    rows = []
    last_layer = int(contrast_df["layer"].max())
    for scope, scoped in (
        ("last_layer", contrast_df[contrast_df["layer"] == last_layer]),
        ("layer_mean", contrast_df),
    ):
        diff = float(scoped["positive_minus_negative"].mean())
        se_values = scoped["se"].astype(float)
        has_uncertainty = bool((se_values > 0).any())
        if scope == "last_layer":
            ci_low = float(scoped["ci_low"].iloc[0])
            ci_high = float(scoped["ci_high"].iloc[0])
            se = float(scoped["se"].iloc[0])
            p_value = float(scoped["p_value_two_sided"].iloc[0])
            layer = last_layer
        elif has_uncertainty:
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

        row = dict(metadata)
        row.update(
            {
                "scope": scope,
                "layer": layer,
                "positive_minus_negative": diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "se": se,
                "p_value_two_sided": p_value,
                "n_layers": int(len(scoped)),
                "n_positive_layers": int((scoped["positive_minus_negative"] > 0).sum()),
                "n_supported_positive_layers": int(
                    (scoped["support"] == "supports_positive_group_greater").sum()
                ),
                "n_supported_negative_layers": int(
                    (scoped["support"] == "supports_negative_group_greater").sum()
                ),
                "support": support_label(diff, ci_low, ci_high, has_uncertainty),
                "ci_method": "normal_approx_from_contrast_ci" if has_uncertainty else "no_source_uncertainty",
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def plot_feature_grid(feature: str, contrast_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    if contrast_df.empty:
        return []

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.8), squeeze=False)
    for row_idx, model_short in enumerate(("bert", "gpt")):
        for col_idx, metric_label in enumerate(METRIC_SPECS):
            ax = axes[row_idx][col_idx]
            sub = contrast_df[
                (contrast_df["model_short"] == model_short)
                & (contrast_df["metric_label"] == metric_label)
            ].sort_values("layer")
            if sub.empty:
                ax.axis("off")
                continue

            x = sub["layer"].to_numpy()
            diff = sub["positive_minus_negative"].to_numpy()
            lo = sub["ci_low"].to_numpy()
            hi = sub["ci_high"].to_numpy()
            ax.axhline(0.0, color="#333333", lw=0.9, ls="--", alpha=0.8)
            ax.plot(x, diff, lw=2.0, color=CONTRAST_COLOR)
            if (hi - lo).max() > 0:
                ax.fill_between(x, lo, hi, alpha=0.18, color=CONTRAST_COLOR)
            ax.set_xlabel("Layer")
            if col_idx == 0:
                ax.set_ylabel(f"{model_short.upper()}\npositive - negative")
            if row_idx == 0:
                ax.set_title(METRIC_SPECS[metric_label]["display"])
            ax.margins(x=0.03)

    title = f"{FEATURE_SPECS[feature]['display']} contrast"
    fig.suptitle(title, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_base = output_dir / f"{feature}_{FEATURE_SPECS[feature]['contrast']}_contrast_grid"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [out_base.with_suffix(".pdf"), out_base.with_suffix(".png")]


def plot_histogram_summary(
    feature: str,
    model_short: str,
    metric_groups: dict[str, pd.DataFrame],
    output_dir: Path,
    layer_mode: str,
) -> list[Path]:
    if not metric_groups:
        return []

    sns.set_style("darkgrid")
    sns.set_context("paper", font_scale=1.8)

    spec = FEATURE_SPECS[feature]
    metric_order = list(METRIC_SPECS)
    fig, axes = plt.subplots(1, len(metric_order), figsize=(13.0, 4.6), squeeze=False)
    axes_row = axes[0]

    for ax, metric_label in zip(axes_row, metric_order):
        if metric_label not in metric_groups:
            ax.axis("off")
            continue

        groups = metric_groups[metric_label].copy()
        if layer_mode == "last":
            target_layer = int(groups["layer"].max())
            plot_df = groups[groups["layer"] == target_layer].copy()
        elif layer_mode == "mean":
            plot_df = (
                groups.groupby(["group_key", "class_group"], as_index=False)
                .agg(mean=("mean", "mean"), ci_low=("ci_low", "mean"), ci_high=("ci_high", "mean"))
            )
        else:
            raise ValueError("layer_mode must be 'last' or 'mean'")

        order = spec.get("histogram_order", ["positive", "negative"])
        plot_df["group_key"] = pd.Categorical(plot_df["group_key"], categories=order, ordered=True)
        plot_df = plot_df.sort_values("group_key")

        short_labels = {
            "positive": spec["positive_short"],
            "negative": spec["negative_short"],
        }
        labels = [short_labels[str(group_key)] for group_key in plot_df["group_key"].astype(str)]
        colors = [GROUP_COLORS.get(str(group_key), "#777777") for group_key in plot_df["group_key"].astype(str)]
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
        xrange = max(xmax - xmin, abs(xmax), 1e-9)
        ax.set_xlim(xmin, xmax + 0.24 * xrange)

        for bar, value in zip(bars, plot_df["mean"]):
            ax.text(
                bar.get_width() + 0.045 * xrange,
                bar.get_y() + bar.get_height() / 2,
                format_number(value),
                ha="left",
                va="center",
                fontsize=13,
                fontweight="semibold",
                color="#2A2A2A",
        )

        ax.set_xlabel(METRIC_SPECS[metric_label]["family"])
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()

    fig.tight_layout(rect=(0, 0.02, 1, 0.98), w_pad=2.0)
    out_base = output_dir / f"{feature}_{spec['contrast']}_{model_short}_histogram_{layer_mode}"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [out_base.with_suffix(".pdf"), out_base.with_suffix(".png")]


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
        "# All Feature Contrasts",
        "",
        "Each qualitative feature observation is represented as a binary contrast. The positive direction is listed explicitly so the hypothesis can be accepted, rejected, or marked as inconclusive by checking whether the approximate 95% CI for the contrast excludes zero.",
        "",
        "| Feature | Contrast | Model | Metric | Scope | Effect | 95% CI | Support |",
        "|---|---|---|---|---|---:|---:|---|",
    ]

    order = ["pos", "length", "index", "arity", "head_dist", "relation_type"]
    for _, row in summary_df.sort_values(["feature_order", "model_short", "metric_label", "scope"]).iterrows():
        ci = f"[{format_number(row['ci_low'])}, {format_number(row['ci_high'])}]"
        contrast = f"{row['positive_label']} - {row['negative_label']}"
        lines.append(
            "| {feature} | {contrast} | {model} | {metric} | {scope} | {effect} | {ci} | {support} |".format(
                feature=row["feature_display"],
                contrast=contrast,
                model=row["model"],
                metric=row["metric_label"],
                scope=row["scope"].replace("_", " "),
                effect=format_number(row["positive_minus_negative"]),
                ci=ci,
                support=str(row["support"]).replace("_", " "),
            )
        )

    lines.extend(["", "## Hypotheses", ""])
    for feature in order:
        spec = FEATURE_SPECS[feature]
        lines.append(f"- **{spec['display']}**: {spec['hypothesis']}")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Layer-mean intervals are descriptive summaries across correlated layers, not independent-layer tests.",
            "- Intervals are approximated from the saved class-level CIs because the original bootstrap replicates are not stored in these CSVs.",
            "- Middle or neutral classes are excluded from each binary contrast, for example medium token lengths, root-distance 0, punctuation, and relation labels outside the content/function sets.",
            "- The arity contrast uses the available all-token arity tables. The manuscript wording should say `predicate` only if predicate-conditioned arity tables are regenerated; otherwise describe it as mid-arity tokens.",
            "- Rows marked `no ci` come from source tables with zero-width intervals and should be described as observed contrasts rather than CI-supported effects.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    dirs = source_dirs(args.geometry_root)
    written: list[Path] = []
    all_summaries: list[pd.DataFrame] = []
    all_contrasts: dict[str, list[pd.DataFrame]] = {feature: [] for feature in FEATURE_SPECS}
    histogram_groups: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}

    for feature_order, feature in enumerate(FEATURE_SPECS):
        spec = FEATURE_SPECS[feature]
        for model_short, model_name in MODEL_NAMES.items():
            source_dir = dirs[(model_short, feature)]
            for metric_label in METRIC_SPECS:
                path, metric_name = metric_file(source_dir, feature, metric_label, model_name)
                if path is None or metric_name is None:
                    print(f"skip missing {feature} {model_short} {metric_label}: {source_dir}")
                    continue

                df = pd.read_csv(path)
                groups = weighted_group_curves(df, feature)
                if groups.empty:
                    print(f"skip empty grouping for {path}")
                    continue

                metadata = {
                    "feature": feature,
                    "feature_display": spec["display"],
                    "feature_order": feature_order,
                    "contrast": spec["contrast"],
                    "positive_label": spec["positive_label"],
                    "negative_label": spec["negative_label"],
                    "positive_name": spec["positive_name"],
                    "negative_name": spec["negative_name"],
                    "model_short": model_short,
                    "model": model_name,
                    "metric": metric_name,
                    "metric_label": metric_label,
                    "metric_family": METRIC_SPECS[metric_label]["family"],
                    "source_path": str(path),
                }

                for key, value in reversed(metadata.items()):
                    groups.insert(0, key, value)
                groups["aggregation"] = "token_weighted_mean_of_feature_groups"
                group_out = tables_dir / f"{feature}_{spec['contrast']}_{model_short}_{metric_label}_groups.csv"
                groups.to_csv(group_out, index=False)
                written.append(group_out)
                histogram_groups.setdefault((feature, model_short), {})[metric_label] = groups

                contrast = contrast_curve(groups)
                if contrast.empty:
                    print(f"skip empty contrast for {path}")
                    continue

                for key, value in reversed(metadata.items()):
                    contrast.insert(0, key, value)
                contrast["hypothesis"] = spec["hypothesis"]
                contrast["aggregation"] = "token_weighted_positive_minus_negative"
                contrast_out = tables_dir / f"{feature}_{spec['contrast']}_{model_short}_{metric_label}.csv"
                contrast.to_csv(contrast_out, index=False)
                written.append(contrast_out)
                all_contrasts[feature].append(contrast)
                all_summaries.append(summarize_contrast(contrast, metadata))

    for feature, frames in all_contrasts.items():
        if not frames:
            continue
        feature_contrasts = pd.concat(frames, ignore_index=True)
        written.extend(plot_feature_grid(feature, feature_contrasts, output_dir))

    for (feature, model_short), metric_groups in sorted(histogram_groups.items()):
        written.extend(plot_histogram_summary(feature, model_short, metric_groups, output_dir, "last"))
        written.extend(plot_histogram_summary(feature, model_short, metric_groups, output_dir, "mean"))

    if all_summaries:
        summary = pd.concat(all_summaries, ignore_index=True)
        summary_out = tables_dir / "all_feature_contrasts_summary.csv"
        summary.to_csv(summary_out, index=False)
        written.append(summary_out)
        markdown_out = tables_dir / "paper_section_all_feature_contrasts.md"
        write_markdown_summary(summary, markdown_out)
        written.append(markdown_out)

    print(f"geometry root: {args.geometry_root}")
    print(f"wrote {len(written)} files")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
