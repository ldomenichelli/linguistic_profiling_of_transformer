#!/usr/bin/env python3
"""Create verbal-valency plots from verb-only arity bootstrap tables.

This script is intentionally separate from the all-token arity contrast script:
verbal valency is defined only over tokens whose POS tag is VERB. The main
contrast is:

    mid-valency verbs (2-4 dependents) minus low-valency verbs (0-1 dependents)

Saved arity tables are used only when their class counts match the verb-only
profile in the sentence-level CSV. This avoids accidentally treating all-token
arity tables as verbal-valency evidence.
"""

from __future__ import annotations

import argparse
import ast
import math
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
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
DEFAULT_SENTENCE_CSV = DEFAULT_GEOMETRY_ROOT / "en_ewt-ud-train_sentences.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots_main" / "plots_features" / "verbal_valency"

CI_Z = 1.96
CONTRAST_COLOR = "#2F5D9E"
GROUP_COLORS = {
    "mid": "#287C8E",
    "low": "#D65F4A",
}
CURVE_PALETTE = {
    "0": "#6B7280",
    "1": "#D65F4A",
    "2": "#287C8E",
    "3": "#3B82F6",
    "4": "#8B5CF6",
    "5": "#C084FC",
    "6": "#7C3AED",
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

LOW_VALENCY = {"0", "1"}
MID_VALENCY = {"2", "3", "4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY_ROOT)
    parser.add_argument("--sentence-csv", type=Path, default=DEFAULT_SENTENCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-legacy-pdf-fallback",
        action="store_true",
        help="Do not digitize legacy VERB arity PDFs when verb-only CSV tables are missing.",
    )
    return parser.parse_args()


def to_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.startswith("["):
        return ast.literal_eval(value)
    return []


def pick_arity_col(columns: pd.Index) -> str:
    for candidate in ("arity", "ariety", "ARITY", "Arity"):
        if candidate in columns:
            return candidate
    raise ValueError("No arity column found. Expected one of: arity, ariety, ARITY, Arity.")


def verb_valency_counts(sentence_csv: Path) -> dict[str, int]:
    df = pd.read_csv(sentence_csv)
    arity_col = pick_arity_col(df.columns)
    counts: dict[str, int] = {}
    for tokens, arities, pos_tags in df[["tokens", arity_col, "pos"]].itertuples(index=False):
        tokens_list = to_list(tokens)
        arity_list = to_list(arities)
        pos_list = to_list(pos_tags)
        length = min(len(tokens_list), len(arity_list), len(pos_list))
        for word_id in range(length):
            if word_id == 0:
                continue
            if pos_list[word_id] != "VERB":
                continue
            try:
                valency = int(arity_list[word_id])
            except Exception:
                valency = 0
            key = str(min(max(valency, 0), 6))
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def source_dirs(geometry_root: Path, model_short: str) -> list[Path]:
    if model_short == "bert":
        return [
            geometry_root / "tables_VERBAL_VALENCY" / "valency_bootstrap",
            geometry_root / "bert" / "tables_VERBAL_VALENCY" / "valency_bootstrap",
            geometry_root / "tables_ARITY_no_index" / "arity_bootstrap",
        ]
    if model_short == "gpt":
        return [
            geometry_root / "gpt2_no_index" / "tables_VERBAL_VALENCY" / "valency_bootstrap",
            geometry_root / "tables_VERBAL_VALENCY_GPT2_no_index" / "valency_bootstrap",
            geometry_root / "tables_ARITY_no_index" / "arity_bootstrap",
        ]
    raise KeyError(model_short)


def normalize_class(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def valid_verb_only_table(df: pd.DataFrame, expected_counts: dict[str, int]) -> bool:
    if df.empty or "class" not in df or "n_tokens" not in df:
        return False
    observed = (
        df.assign(normalized_class=df["class"].map(normalize_class))
        .groupby("normalized_class")["n_tokens"]
        .first()
        .astype(int)
        .to_dict()
    )
    if not observed:
        return False
    for cls, n_tokens in observed.items():
        if expected_counts.get(cls) != int(n_tokens):
            return False
    return set(MID_VALENCY).issubset(observed) and set(LOW_VALENCY).issubset(observed)


def metric_file(
    geometry_root: Path,
    model_short: str,
    model_name: str,
    metric_label: str,
    expected_counts: dict[str, int],
) -> tuple[Path | None, str | None]:
    for source_dir in source_dirs(geometry_root, model_short):
        for prefix in ("valency", "verbal_valency", "arity"):
            for metric_name in METRIC_SPECS[metric_label]["candidates"]:
                path = source_dir / f"{prefix}_raw_{metric_name}_{model_name}.csv"
                if not path.exists():
                    continue
                df = pd.read_csv(path)
                if valid_verb_only_table(df, expected_counts):
                    return path, metric_name
    return None, None


def legacy_pdf_file(geometry_root: Path, model_short: str, metric_label: str) -> Path:
    file_metric = {
        "isoscore": "iso",
        "LID": "LID",
        "NLID": "NLID",
    }[metric_label]
    prefix = {"bert": "bert", "gpt": "gpt"}[model_short]
    return geometry_root / "plot_POS" / f"{prefix}_arity_{file_metric}_VERB.pdf"


def parse_svg_paths(svg_text: str) -> tuple[list[tuple[str, str]], float]:
    viewbox_match = re.search(r'viewBox="[^"]*?([0-9.]+)\s+([0-9.]+)"', svg_text)
    page_height = float(viewbox_match.group(2)) if viewbox_match else 360.0
    path_pattern = re.compile(r'<path[^>]+style="([^"]+)"[^>]+d="([^"]+)"')
    return [(match.group(1), match.group(2)) for match in path_pattern.finditer(svg_text)], page_height


def path_coords(path_d: str) -> list[tuple[float, float]]:
    values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path_d)]
    return list(zip(values[0::2], values[1::2]))


def svg_horizontal_grid(paths: list[tuple[str, str]]) -> tuple[float, list[float]]:
    horizontal: list[tuple[float, float, float]] = []
    for style, path_d in paths:
        if "stroke:rgb(100%,100%,100%)" not in style or "fill:none" not in style:
            continue
        coords = path_coords(path_d)
        if len(coords) != 2:
            continue
        (x0, y0), (x1, y1) = coords
        if abs(y0 - y1) < 1e-6 and abs(x1 - x0) > 100:
            horizontal.append((min(x0, x1), max(x0, x1), y0))
    if not horizontal:
        raise ValueError("Could not find horizontal grid lines in legacy PDF SVG.")
    longest = max(horizontal, key=lambda item: item[1] - item[0])
    plot_left = longest[0]
    grid_y = sorted({round(item[2], 6) for item in horizontal if abs(item[0] - plot_left) < 1.0})
    return plot_left, grid_y


def y_axis_tick_words(xhtml_path: Path, plot_left: float, page_height: float) -> list[tuple[float, float]]:
    root = ET.parse(xhtml_path).getroot()
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    ticks: list[tuple[float, float]] = []
    for word in root.findall(".//x:word", ns):
        text = "".join(word.itertext()).strip()
        if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
            continue
        x_max = float(word.attrib["xMax"])
        if x_max > plot_left - 1.0:
            continue
        y_center = (float(word.attrib["yMin"]) + float(word.attrib["yMax"])) / 2.0
        ticks.append((page_height - y_center, float(text)))
    return ticks


def fit_y_mapping(grid_y: list[float], ticks: list[tuple[float, float]]) -> tuple[float, float]:
    pairs: list[tuple[float, float]] = []
    for tick_y, value in ticks:
        nearest = min(grid_y, key=lambda candidate: abs(candidate - tick_y))
        if abs(nearest - tick_y) <= 6.0:
            pairs.append((nearest, value))
    dedup = {}
    for y_coord, value in pairs:
        dedup[y_coord] = value
    pairs = sorted(dedup.items())
    if len(pairs) < 2:
        raise ValueError("Could not align y-axis tick labels with SVG grid lines.")
    n = len(pairs)
    mean_x = sum(item[0] for item in pairs) / n
    mean_y = sum(item[1] for item in pairs) / n
    denom = sum((item[0] - mean_x) ** 2 for item in pairs)
    if denom <= 0:
        raise ValueError("Degenerate y-axis mapping in legacy PDF.")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept


def digitize_legacy_pdf(
    pdf_path: Path,
    expected_counts: dict[str, int],
    model_name: str,
    metric_label: str,
) -> pd.DataFrame:
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        svg_path = tmpdir_path / f"{pdf_path.stem}.svg"
        xhtml_path = tmpdir_path / f"{pdf_path.stem}.xhtml"
        subprocess.run(
            ["pdftocairo", "-svg", str(pdf_path), str(svg_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["pdftotext", "-bbox", str(pdf_path), str(xhtml_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        paths, page_height = parse_svg_paths(svg_path.read_text(encoding="utf-8"))
        plot_left, grid_y = svg_horizontal_grid(paths)
        ticks = y_axis_tick_words(xhtml_path, plot_left, page_height)
        slope, intercept = fit_y_mapping(grid_y, ticks)

    mean_paths = [
        path_coords(path_d)
        for style, path_d in paths
        if "stroke-width:1.8" in style and "fill:none" in style
    ]
    ci_paths = [
        path_coords(path_d)
        for style, path_d in paths
        if "fill-opacity:0.15" in style and "fill-rule:nonzero" in style
    ]
    if len(mean_paths) < 7:
        raise ValueError(f"Expected 7 valency curves in {pdf_path}, found {len(mean_paths)}.")
    mean_paths = mean_paths[:7]
    ci_paths = ci_paths[:7]

    def to_metric(svg_y: float) -> float:
        return slope * svg_y + intercept

    rows = []
    for class_id, mean_coords in enumerate(mean_paths):
        cls = str(class_id)
        ci_coords = ci_paths[class_id] if class_id < len(ci_paths) else []
        for layer, (x_coord, mean_y) in enumerate(mean_coords):
            matching_ci = [ci_y for ci_x, ci_y in ci_coords if abs(ci_x - x_coord) <= 0.5]
            mean = to_metric(mean_y)
            if len(matching_ci) >= 2:
                ci_low = to_metric(min(matching_ci))
                ci_high = to_metric(max(matching_ci))
            else:
                ci_low = mean
                ci_high = mean
            rows.append(
                {
                    "subset": "legacy_pdf_digitized",
                    "model": model_name,
                    "feature": "verbal_valency",
                    "class": cls,
                    "metric": metric_label,
                    "layer": layer,
                    "mean": mean,
                    "ci_low": min(ci_low, ci_high),
                    "ci_high": max(ci_low, ci_high),
                    "n_tokens": expected_counts[cls],
                }
            )
    return pd.DataFrame(rows)


def ci_to_se(row: pd.Series) -> float:
    low = float(row["ci_low"])
    high = float(row["ci_high"])
    if not (math.isfinite(low) and math.isfinite(high)):
        return 0.0
    return max(high - low, 0.0) / (2.0 * CI_Z)


def support_label(diff: float, ci_low: float, ci_high: float, has_uncertainty: bool) -> str:
    if has_uncertainty:
        if ci_low > 0:
            return "supports_mid_valency_greater"
        if ci_high < 0:
            return "supports_low_valency_greater"
        return "inconclusive"
    if diff > 0:
        return "observed_mid_valency_greater_no_ci"
    if diff < 0:
        return "observed_low_valency_greater_no_ci"
    return "observed_no_difference_no_ci"


def valency_group(value: object) -> str | None:
    cls = normalize_class(value)
    if cls in MID_VALENCY:
        return "mid"
    if cls in LOW_VALENCY:
        return "low"
    return None


def weighted_group_curves(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["normalized_class"] = work["class"].map(normalize_class)
    work["group_key"] = work["class"].map(valency_group)
    work = work[work["group_key"].notna()].copy()
    if work.empty:
        return pd.DataFrame()

    label_map = {
        "mid": "Mid Valency (2-4 dependents)",
        "low": "Low Valency (0-1 dependents)",
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
                "class_members": ",".join(sorted(layer_df["normalized_class"].unique(), key=int)),
                "n_class_members": int(layer_df["normalized_class"].nunique()),
                "ci_method": ci_method,
            }
        )
    return pd.DataFrame(rows)


def contrast_curve(groups_df: pd.DataFrame) -> pd.DataFrame:
    if groups_df.empty or not {"mid", "low"}.issubset(set(groups_df["group_key"])):
        return pd.DataFrame()

    mid_df = groups_df[groups_df["group_key"] == "mid"].set_index("layer")
    low_df = groups_df[groups_df["group_key"] == "low"].set_index("layer")
    rows = []
    for layer in sorted(set(mid_df.index).intersection(set(low_df.index))):
        mid = mid_df.loc[layer]
        low = low_df.loc[layer]
        diff = float(mid["mean"] - low["mean"])
        se = float(math.sqrt(float(mid["se"]) ** 2 + float(low["se"]) ** 2))
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
                "mid_mean": float(mid["mean"]),
                "low_mean": float(low["mean"]),
                "mid_minus_low": diff,
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
        diff = float(scoped["mid_minus_low"].mean())
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
                "mid_minus_low": diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "se": se,
                "p_value_two_sided": p_value,
                "n_layers": int(len(scoped)),
                "n_positive_layers": int((scoped["mid_minus_low"] > 0).sum()),
                "n_supported_mid_layers": int(
                    (scoped["support"] == "supports_mid_valency_greater").sum()
                ),
                "n_supported_low_layers": int(
                    (scoped["support"] == "supports_low_valency_greater").sum()
                ),
                "support": support_label(diff, ci_low, ci_high, has_uncertainty),
                "ci_method": "normal_approx_from_contrast_ci" if has_uncertainty else "no_source_uncertainty",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


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


def plot_raw_valency_curves(
    model_short: str,
    metric_label: str,
    df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    sns.set_style("darkgrid")
    sns.set_context("paper", font_scale=1.55)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    work = df.copy()
    work["normalized_class"] = work["class"].map(normalize_class)
    for cls in sorted(work["normalized_class"].unique(), key=int):
        sub = work[work["normalized_class"] == cls].sort_values("layer")
        color = CURVE_PALETTE.get(cls)
        x = sub["layer"].to_numpy()
        y = sub["mean"].to_numpy()
        lo = sub["ci_low"].to_numpy()
        hi = sub["ci_high"].to_numpy()
        ax.plot(x, y, lw=2.0, label=cls, color=color)
        if (hi - lo).max() > 0:
            ax.fill_between(x, lo, hi, alpha=0.14, color=color)

    ax.set_xlabel("Layer")
    ax.set_ylabel(METRIC_SPECS[metric_label]["family"])
    ax.legend(title="Valency", ncol=4, fontsize="small", frameon=False)
    ax.margins(x=0.03)
    fig.tight_layout()
    out_base = output_dir / f"verbal_valency_{model_short}_{metric_label}_curves"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [out_base.with_suffix(".pdf"), out_base.with_suffix(".png")]


def plot_contrast_grid(contrast_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    if contrast_df.empty:
        return []

    available = [
        (model_short, metric_label)
        for model_short in ("bert", "gpt")
        for metric_label in METRIC_SPECS
        if not contrast_df[
            (contrast_df["model_short"] == model_short)
            & (contrast_df["metric_label"] == metric_label)
        ].empty
    ]
    if not available:
        return []

    sns.set_style("darkgrid")
    sns.set_context("paper", font_scale=1.45)
    fig, axes = plt.subplots(1, len(available), figsize=(5.0 * len(available), 4.4), squeeze=False)
    for ax, (model_short, metric_label) in zip(axes[0], available):
        sub = contrast_df[
            (contrast_df["model_short"] == model_short)
            & (contrast_df["metric_label"] == metric_label)
        ].sort_values("layer")
        x = sub["layer"].to_numpy()
        diff = sub["mid_minus_low"].to_numpy()
        lo = sub["ci_low"].to_numpy()
        hi = sub["ci_high"].to_numpy()
        ax.axhline(0.0, color="#333333", lw=0.9, ls="--", alpha=0.8)
        ax.plot(x, diff, lw=2.0, color=CONTRAST_COLOR)
        if (hi - lo).max() > 0:
            ax.fill_between(x, lo, hi, alpha=0.18, color=CONTRAST_COLOR)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mid - Low")
        ax.set_title(f"{model_short.upper()} {METRIC_SPECS[metric_label]['display']}")
        ax.margins(x=0.03)

    fig.tight_layout()
    out_base = output_dir / "verbal_valency_mid_low_contrast_grid"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [out_base.with_suffix(".pdf"), out_base.with_suffix(".png")]


def plot_histogram_summary(
    model_short: str,
    metric_groups: dict[str, pd.DataFrame],
    output_dir: Path,
    layer_mode: str,
) -> list[Path]:
    if not metric_groups:
        return []

    sns.set_style("darkgrid")
    sns.set_context("paper", font_scale=1.8)

    metric_order = list(METRIC_SPECS)
    fig, axes = plt.subplots(1, len(metric_order), figsize=(13.0, 4.6), squeeze=False)
    axes_row = axes[0]

    for ax, metric_label in zip(axes_row, metric_order):
        if metric_label not in metric_groups:
            ax.text(
                0.5,
                0.52,
                "No verb-only\ntable",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=14,
                fontweight="semibold",
                color="#4A4A4A",
            )
            ax.set_xlabel(METRIC_SPECS[metric_label]["family"])
            ax.set_xticks([])
            ax.set_yticks([])
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

        order = ["mid", "low"]
        plot_df["group_key"] = pd.Categorical(plot_df["group_key"], categories=order, ordered=True)
        plot_df = plot_df.sort_values("group_key")
        label_map = {
            "mid": "Mid Valency",
            "low": "Low Valency",
        }
        labels = [label_map[str(group_key)] for group_key in plot_df["group_key"].astype(str)]
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
        ax.set_xlim(xmin, xmax + 0.36 * xrange)
        for bar, (_, row) in zip(bars, plot_df.iterrows()):
            label_x = max(float(row["mean"]), float(row["ci_high"])) + 0.085 * xrange
            ax.text(
                label_x,
                bar.get_y() + bar.get_height() / 2,
                format_number(row["mean"]),
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
    out_base = output_dir / f"verbal_valency_mid_low_{model_short}_histogram_{layer_mode}"
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return [out_base.with_suffix(".pdf"), out_base.with_suffix(".png")]


def write_markdown_summary(summary_df: pd.DataFrame, out_path: Path, missing: list[str]) -> None:
    lines = [
        "# Verbal Valency Contrast",
        "",
        "Verbal valency is defined over `VERB` tokens only, using the number of syntactic dependents attached to the verb. The contrast is mid-valency verbs (`2-4` dependents) minus low-valency verbs (`0-1` dependents).",
        "",
        "| Model | Metric | Scope | Effect | 95% CI | Support |",
        "|---|---|---|---:|---:|---|",
    ]
    for _, row in summary_df.sort_values(["model_short", "metric_label", "scope"]).iterrows():
        ci = f"[{format_number(row['ci_low'])}, {format_number(row['ci_high'])}]"
        lines.append(
            "| {model} | {metric} | {scope} | {effect} | {ci} | {support} |".format(
                model=row["model"],
                metric=row["metric_label"],
                scope=row["scope"].replace("_", " "),
                effect=format_number(row["mid_minus_low"]),
                ci=ci,
                support=str(row["support"]).replace("_", " "),
            )
        )
    if missing:
        lines.extend(["", "## Missing Tables", ""])
        lines.extend(f"- {item}" for item in missing)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    expected_counts = verb_valency_counts(args.sentence_csv)
    written: list[Path] = []
    missing: list[str] = []
    fallbacks: list[str] = []
    summaries: list[pd.DataFrame] = []
    contrast_frames: list[pd.DataFrame] = []
    histogram_groups: dict[str, dict[str, pd.DataFrame]] = {}

    counts_out = tables_dir / "verbal_valency_expected_counts.csv"
    pd.DataFrame(
        [{"class": key, "n_tokens": value} for key, value in expected_counts.items()]
    ).to_csv(counts_out, index=False)
    written.append(counts_out)

    for model_short, model_name in MODEL_NAMES.items():
        for metric_label in METRIC_SPECS:
            path, metric_name = metric_file(
                args.geometry_root,
                model_short,
                model_name,
                metric_label,
                expected_counts,
            )
            if path is None or metric_name is None:
                if args.no_legacy_pdf_fallback:
                    missing.append(f"{model_name} {metric_label}: no verb-only bootstrap table found")
                    continue
                pdf_path = legacy_pdf_file(args.geometry_root, model_short, metric_label)
                try:
                    df = digitize_legacy_pdf(pdf_path, expected_counts, model_name, metric_label)
                except Exception as exc:
                    missing.append(
                        f"{model_name} {metric_label}: no verb-only bootstrap table found; "
                        f"legacy PDF fallback failed ({exc})"
                    )
                    continue
                path = pdf_path
                metric_name = f"legacy_pdf_{metric_label}"
                source_kind = "legacy_pdf_digitized"
                fallbacks.append(f"{model_name} {metric_label}: digitized {pdf_path}")
            else:
                df = pd.read_csv(path)
                source_kind = "bootstrap_csv"

            df["normalized_class"] = df["class"].map(normalize_class)
            metadata = {
                "feature": "verbal_valency",
                "feature_display": "Verbal Valency",
                "contrast": "mid_low",
                "positive_label": "Mid Valency (2-4 dependents)",
                "negative_label": "Low Valency (0-1 dependents)",
                "positive_name": "mid_valency",
                "negative_name": "low_valency",
                "model_short": model_short,
                "model": model_name,
                "metric": metric_name,
                "metric_label": metric_label,
                "metric_family": METRIC_SPECS[metric_label]["family"],
                "source_kind": source_kind,
                "source_path": str(path),
            }

            written.extend(plot_raw_valency_curves(model_short, metric_label, df, output_dir))

            groups = weighted_group_curves(df)
            for key, value in reversed(metadata.items()):
                groups.insert(0, key, value)
            groups["aggregation"] = "token_weighted_mean_of_valency_groups"
            group_out = tables_dir / f"verbal_valency_mid_low_{model_short}_{metric_label}_groups.csv"
            groups.to_csv(group_out, index=False)
            written.append(group_out)
            histogram_groups.setdefault(model_short, {})[metric_label] = groups

            contrast = contrast_curve(groups)
            if contrast.empty:
                missing.append(f"{model_name} {metric_label}: contrast could not be computed")
                continue
            for key, value in reversed(metadata.items()):
                contrast.insert(0, key, value)
            contrast["hypothesis"] = (
                "Mid-valency verbs (2-4 dependents) occupy higher-dimensional "
                "and more isotropic subspaces than low-valency verbs."
            )
            contrast["aggregation"] = "token_weighted_mid_minus_low"
            contrast_out = tables_dir / f"verbal_valency_mid_low_{model_short}_{metric_label}.csv"
            contrast.to_csv(contrast_out, index=False)
            written.append(contrast_out)
            contrast_frames.append(contrast)
            summaries.append(summarize_contrast(contrast, metadata))

    if contrast_frames:
        all_contrasts = pd.concat(contrast_frames, ignore_index=True)
        written.extend(plot_contrast_grid(all_contrasts, output_dir))

    for model_short, metric_groups in sorted(histogram_groups.items()):
        written.extend(plot_histogram_summary(model_short, metric_groups, output_dir, "last"))
        written.extend(plot_histogram_summary(model_short, metric_groups, output_dir, "mean"))

    if summaries:
        summary = pd.concat(summaries, ignore_index=True)
        summary_out = tables_dir / "verbal_valency_summary.csv"
        summary.to_csv(summary_out, index=False)
        written.append(summary_out)
        markdown_out = tables_dir / "paper_section_verbal_valency_contrasts.md"
        write_markdown_summary(summary, markdown_out, missing)
        written.append(markdown_out)

    missing_out = tables_dir / "missing_verbal_valency_tables.txt"
    missing_out.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
    written.append(missing_out)
    fallbacks_out = tables_dir / "legacy_pdf_fallbacks.txt"
    fallbacks_out.write_text("\n".join(fallbacks) + ("\n" if fallbacks else ""), encoding="utf-8")
    written.append(fallbacks_out)

    print(f"sentence csv: {args.sentence_csv}")
    print(f"geometry root: {args.geometry_root}")
    print(f"verb-only valency counts: {expected_counts}")
    print(f"wrote {len(written)} files")
    for path in written:
        print(path)
    if missing:
        print("missing or rejected tables:")
        for item in missing:
            print(f"- {item}")
    if fallbacks:
        print("legacy PDF fallbacks:")
        for item in fallbacks:
            print(f"- {item}")


if __name__ == "__main__":
    main()
