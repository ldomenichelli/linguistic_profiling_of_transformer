#!/usr/bin/env python3
"""Matched control analysis for POS-conditioned head-distance contrasts.

This script addresses the corpus-vs-model concern for contrasts such as:

    NOUN tokens whose syntactic head is after them  (Before Head; head_dist > 0)
    vs.
    NOUN tokens whose syntactic head is before them (After Head;  head_dist < 0)

It first constructs matched Before/After groups from the corpus metadata using
coarsened exact matching over token frequency, token length, sentence position,
and dependency relation. It can then recompute representation geometry on the
matched token sets using the same metric functions as the POS x head-distance
experiments.
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import random
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from run_pos_head_dist_experiments import (
    LABELS,
    METRIC_FUNCTIONS,
    bootstrap_layer_loop,
    embed_subset,
    model_short_name,
    set_reproducibility,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "code" / "data" / "en_ewt-ud-train_sentences.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots_main" / "plots_features" / "matched_head_distance_controls"

DEFAULT_MATCH_COLS = (
    "relation_type",
    "token_length_bin",
    "sentence_position_bin",
    "frequency_bin",
    "abs_head_dist_bin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pos", default="NOUN", help="UPOS tag to control, e.g. NOUN.")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--word-rep-mode", default="first", choices=("first", "last", "mean"))
    parser.add_argument("--metrics", nargs="+", default=["iso", "lpca99", "gride"], choices=tuple(LABELS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face network checks/downloads. By default the script uses the local cache only.",
    )
    parser.add_argument("--n-bootstrap-fast", type=int, default=50)
    parser.add_argument("--n-bootstrap-heavy", type=int, default=20)
    parser.add_argument("--fast-bs-max-samp-per-class", type=int, default=int(1e12))
    parser.add_argument("--heavy-bs-max-samp-per-class", type=int, default=5000)
    parser.add_argument("--include-first-token", action="store_true")
    parser.add_argument("--max-head-dist-abs", type=int, default=6)
    parser.add_argument("--frequency-bins", type=int, default=6)
    parser.add_argument("--position-bin-width", type=int, default=5)
    parser.add_argument("--length-bin-width", type=int, default=2)
    parser.add_argument("--min-pairs-per-stratum", type=int, default=1)
    parser.add_argument(
        "--max-pairs-per-stratum",
        type=int,
        default=0,
        help="Optional cap per matched stratum. 0 means no cap beyond min(group counts).",
    )
    parser.add_argument(
        "--match-cols",
        nargs="+",
        default=list(DEFAULT_MATCH_COLS),
        help=(
            "Columns used for coarsened exact matching. Useful options include "
            "relation_type token_length_bin sentence_position_bin frequency_bin abs_head_dist_bin."
        ),
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write corpus matching/balance tables but do not load the transformer.",
    )
    return parser.parse_args()


def to_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return ast.literal_eval(value)
    return []


def bin_range(value: int, width: int, minimum: int = 1) -> str:
    value = max(int(value), minimum)
    width = max(int(width), 1)
    start = ((value - minimum) // width) * width + minimum
    end = start + width - 1
    return f"{start}-{end}"


def parse_sentence_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype={"sentence_id": str})
    required = {"sentence_id", "tokens", "pos", "head_dist", "relation_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    arity_col = "ariety" if "ariety" in df.columns else "arity" if "arity" in df.columns else None
    for col in ["tokens", "pos", "head_dist", "relation_type", "index", "length", arity_col]:
        if col and col in df.columns:
            df[col] = df[col].apply(to_list)
    return df


def token_frequency_map(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tokens in df["tokens"]:
        for token in tokens:
            key = str(token).lower()
            counts[key] = counts.get(key, 0) + 1
    return counts


def build_token_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = parse_sentence_rows(args.csv_path)
    freq = token_frequency_map(df)
    rows = []
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        sid = str(row_dict["sentence_id"])
        tokens = row_dict["tokens"]
        pos_tags = row_dict["pos"]
        dists = row_dict["head_dist"]
        rels = row_dict["relation_type"]
        indexes = row_dict.get("index", [])
        lengths = row_dict.get("length", [])
        limit = min(len(tokens), len(pos_tags), len(dists), len(rels))
        for word_id in range(limit):
            if not args.include_first_token and word_id == 0:
                continue
            pos_tag = str(pos_tags[word_id])
            if pos_tag != args.pos:
                continue
            try:
                raw_dist = int(dists[word_id])
            except Exception:
                continue
            if raw_dist == 0:
                continue
            dist = max(-args.max_head_dist_abs, min(args.max_head_dist_abs, raw_dist))
            direction = "Before Head" if dist > 0 else "After Head"
            token = str(tokens[word_id])
            lower = token.lower()
            token_length = int(lengths[word_id]) if word_id < len(lengths) else len(token)
            sent_position = int(indexes[word_id]) if word_id < len(indexes) else word_id + 1
            rows.append(
                {
                    "sentence_id": sid,
                    "word_id": int(word_id),
                    "word": token,
                    "word_lower": lower,
                    "pos": pos_tag,
                    "head_dist": int(dist),
                    "abs_head_dist": int(abs(dist)),
                    "direction": direction,
                    "token_frequency": int(freq.get(lower, 1)),
                    "log_token_frequency": math.log1p(int(freq.get(lower, 1))),
                    "token_length": token_length,
                    "sentence_position": sent_position,
                    "relation_type": str(rels[word_id]),
                    "token_length_bin": bin_range(token_length, args.length_bin_width),
                    "sentence_position_bin": bin_range(sent_position, args.position_bin_width),
                    "abs_head_dist_bin": bin_range(abs(dist), 2),
                }
            )

    token_df = pd.DataFrame(rows)
    if token_df.empty:
        raise ValueError(f"No {args.pos} tokens with non-zero head_dist were found.")

    token_df["frequency_bin"] = frequency_bins(token_df, args.frequency_bins)
    sent_df = df[["sentence_id", "tokens"]].drop_duplicates("sentence_id").copy()
    return sent_df, token_df


def frequency_bins(token_df: pd.DataFrame, n_bins: int) -> pd.Series:
    n_bins = max(int(n_bins), 1)
    values = token_df["log_token_frequency"].astype(float)
    try:
        bins = pd.qcut(values, q=min(n_bins, values.nunique()), duplicates="drop")
    except ValueError:
        return pd.Series(["freq_all"] * len(token_df), index=token_df.index)
    return bins.astype(str)


def validate_match_cols(token_df: pd.DataFrame, match_cols: Iterable[str]) -> list[str]:
    cols = list(match_cols)
    missing = [col for col in cols if col not in token_df.columns]
    if missing:
        raise ValueError(f"Unknown match columns: {missing}. Available columns: {sorted(token_df.columns)}")
    return cols


def matched_sample(args: argparse.Namespace, token_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_cols = validate_match_cols(token_df, args.match_cols)
    rng = np.random.default_rng(args.seed)
    matched_parts = []
    stratum_rows = []
    for stratum_values, stratum in token_df.groupby(match_cols, dropna=False, sort=True):
        before = stratum[stratum["direction"] == "Before Head"]
        after = stratum[stratum["direction"] == "After Head"]
        n_pairs = min(len(before), len(after))
        if args.max_pairs_per_stratum > 0:
            n_pairs = min(n_pairs, args.max_pairs_per_stratum)
        if n_pairs < args.min_pairs_per_stratum:
            continue
        before_idx = rng.choice(before.index.to_numpy(), size=n_pairs, replace=False)
        after_idx = rng.choice(after.index.to_numpy(), size=n_pairs, replace=False)
        matched_parts.append(token_df.loc[np.concatenate([before_idx, after_idx])])
        if not isinstance(stratum_values, tuple):
            stratum_values = (stratum_values,)
        row = {col: value for col, value in zip(match_cols, stratum_values)}
        row.update(
            {
                "n_before_raw": int(len(before)),
                "n_after_raw": int(len(after)),
                "n_pairs_kept": int(n_pairs),
            }
        )
        stratum_rows.append(row)

    if not matched_parts:
        raise ValueError("No matched strata survived. Try fewer --match-cols or lower --min-pairs-per-stratum.")
    matched = pd.concat(matched_parts, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    strata = pd.DataFrame(stratum_rows)
    matched["matched_stratum_id"] = pd.factorize(
        matched[match_cols].astype(str).agg("||".join, axis=1), sort=True
    )[0]
    return matched, strata


def numeric_balance(raw_df: pd.DataFrame, matched_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for data_name, data in (("raw", raw_df), ("matched", matched_df)):
        for variable in ["token_frequency", "log_token_frequency", "token_length", "sentence_position", "abs_head_dist"]:
            before = data[data["direction"] == "Before Head"][variable].astype(float)
            after = data[data["direction"] == "After Head"][variable].astype(float)
            pooled_sd = math.sqrt((before.var(ddof=1) + after.var(ddof=1)) / 2.0) if len(before) > 1 and len(after) > 1 else math.nan
            smd = (before.mean() - after.mean()) / pooled_sd if pooled_sd and np.isfinite(pooled_sd) else math.nan
            rows.append(
                {
                    "sample": data_name,
                    "variable": variable,
                    "before_mean": float(before.mean()),
                    "after_mean": float(after.mean()),
                    "before_sd": float(before.std(ddof=1)),
                    "after_sd": float(after.std(ddof=1)),
                    "standardized_mean_difference": float(smd) if np.isfinite(smd) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def categorical_balance(raw_df: pd.DataFrame, matched_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for data_name, data in (("raw", raw_df), ("matched", matched_df)):
        for variable in ["relation_type", "token_length_bin", "sentence_position_bin", "frequency_bin", "abs_head_dist_bin"]:
            table = pd.crosstab(data[variable], data["direction"], normalize="columns")
            before = table.get("Before Head", pd.Series(0.0, index=table.index))
            after = table.get("After Head", pd.Series(0.0, index=table.index))
            total_variation = 0.5 * float((before - after).abs().sum())
            rows.append(
                {
                    "sample": data_name,
                    "variable": variable,
                    "n_levels": int(table.shape[0]),
                    "total_variation_distance": total_variation,
                }
            )
    return pd.DataFrame(rows)


def corpus_summary(raw_df: pd.DataFrame, matched_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for data_name, data in (("raw", raw_df), ("matched", matched_df)):
        for direction, sub in data.groupby("direction"):
            rows.append(
                {
                    "sample": data_name,
                    "direction": direction,
                    "n_tokens": int(len(sub)),
                    "n_types": int(sub["word_lower"].nunique()),
                    "mean_token_frequency": float(sub["token_frequency"].mean()),
                    "mean_token_length": float(sub["token_length"].mean()),
                    "mean_sentence_position": float(sub["sentence_position"].mean()),
                    "mean_abs_head_dist": float(sub["abs_head_dist"].mean()),
                    "top_relations": ", ".join(sub["relation_type"].value_counts().head(5).index.tolist()),
                }
            )
    return pd.DataFrame(rows)


def write_corpus_outputs(
    args: argparse.Namespace,
    token_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    strata_df: pd.DataFrame,
) -> dict[str, Path]:
    out = args.output_dir / args.pos.lower() / model_short_name(args.model) / args.word_rep_mode
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw_tokens": tables / "raw_before_after_tokens.csv",
        "matched_tokens": tables / "matched_before_after_tokens.csv",
        "matched_strata": tables / "matched_strata.csv",
        "corpus_summary": tables / "corpus_summary.csv",
        "numeric_balance": tables / "numeric_balance.csv",
        "categorical_balance": tables / "categorical_balance.csv",
    }
    token_df.to_csv(paths["raw_tokens"], index=False)
    matched_df.to_csv(paths["matched_tokens"], index=False)
    strata_df.to_csv(paths["matched_strata"], index=False)
    corpus_summary(token_df, matched_df).to_csv(paths["corpus_summary"], index=False)
    numeric_balance(token_df, matched_df).to_csv(paths["numeric_balance"], index=False)
    categorical_balance(token_df, matched_df).to_csv(paths["categorical_balance"], index=False)
    return paths


def metric_bootstrap(
    reps: np.ndarray,
    group_labels: np.ndarray,
    metric: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if metric == "gride":
        n_bootstrap = args.n_bootstrap_heavy
        max_sample = args.heavy_bs_max_samp_per_class
    else:
        n_bootstrap = args.n_bootstrap_fast
        max_sample = args.fast_bs_max_samp_per_class

    rows = []
    for direction in ["Before Head", "After Head"]:
        idx = np.where(group_labels == direction)[0]
        if idx.size < 4:
            continue
        sample_size = min(max_sample, idx.size)
        mean, lo, hi = bootstrap_layer_loop(
            reps[:, idx],
            sample_size,
            n_bootstrap,
            METRIC_FUNCTIONS[metric],
            args.seed,
            desc=f"{metric} {direction}",
        )
        for layer, value in enumerate(mean):
            rows.append(
                {
                    "direction": direction,
                    "metric": metric,
                    "layer": int(layer),
                    "mean": float(value),
                    "ci_low": float(lo[layer]),
                    "ci_high": float(hi[layer]),
                    "n_tokens": int(idx.size),
                    "sample_size_per_bootstrap": int(sample_size),
                    "n_bootstrap": int(n_bootstrap),
                }
            )
    return pd.DataFrame(rows)


def contrast_from_metric_table(metric_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, sub in metric_df.groupby("metric"):
        before = sub[sub["direction"] == "Before Head"].set_index("layer")
        after = sub[sub["direction"] == "After Head"].set_index("layer")
        for layer in sorted(set(before.index).intersection(after.index)):
            before_row = before.loc[layer]
            after_row = after.loc[layer]
            before_se = max(float(before_row["ci_high"]) - float(before_row["ci_low"]), 0.0) / (2.0 * 1.96)
            after_se = max(float(after_row["ci_high"]) - float(after_row["ci_low"]), 0.0) / (2.0 * 1.96)
            se = math.sqrt(before_se**2 + after_se**2)
            diff = float(before_row["mean"] - after_row["mean"])
            rows.append(
                {
                    "metric": metric,
                    "layer": int(layer),
                    "before_head_mean": float(before_row["mean"]),
                    "after_head_mean": float(after_row["mean"]),
                    "before_minus_after": diff,
                    "ci_low": diff - 1.96 * se,
                    "ci_high": diff + 1.96 * se,
                    "se": se,
                    "support": "before_greater" if diff - 1.96 * se > 0 else "after_greater" if diff + 1.96 * se < 0 else "inconclusive",
                }
            )
    return pd.DataFrame(rows)


def summarize_contrast(contrast_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, sub in contrast_df.groupby("metric"):
        last_layer = int(sub["layer"].max())
        for scope, scoped in (
            ("last_layer", sub[sub["layer"] == last_layer]),
            ("layer_mean", sub),
        ):
            diff = float(scoped["before_minus_after"].mean())
            se = float(math.sqrt((scoped["se"] ** 2).sum()) / len(scoped))
            rows.append(
                {
                    "metric": metric,
                    "scope": scope,
                    "layer": last_layer if scope == "last_layer" else np.nan,
                    "before_minus_after": diff,
                    "ci_low": diff - 1.96 * se,
                    "ci_high": diff + 1.96 * se,
                    "n_layers": int(len(scoped)),
                    "n_positive_layers": int((scoped["before_minus_after"] > 0).sum()),
                    "support": "before_greater" if diff - 1.96 * se > 0 else "after_greater" if diff + 1.96 * se < 0 else "inconclusive",
                }
            )
    return pd.DataFrame(rows)


def plot_metric_histograms(metric_df: pd.DataFrame, out_dir: Path, args: argparse.Namespace, layer_mode: str) -> Path:
    sns.set_style("darkgrid")
    sns.set_context("paper", font_scale=1.75)
    metric_order = [metric for metric in args.metrics if metric in set(metric_df["metric"])]
    fig, axes = plt.subplots(1, len(metric_order), figsize=(4.6 * len(metric_order), 4.6), squeeze=False)
    for ax, metric in zip(axes[0], metric_order):
        sub = metric_df[metric_df["metric"] == metric].copy()
        if layer_mode == "last":
            layer = int(sub["layer"].max())
            plot_df = sub[sub["layer"] == layer].copy()
        elif layer_mode == "mean":
            plot_df = (
                sub.groupby("direction", as_index=False)
                .agg(mean=("mean", "mean"), ci_low=("ci_low", "mean"), ci_high=("ci_high", "mean"))
            )
        else:
            raise ValueError(layer_mode)
        order = ["Before Head", "After Head"]
        plot_df["direction"] = pd.Categorical(plot_df["direction"], categories=order, ordered=True)
        plot_df = plot_df.sort_values("direction")
        colors = ["#287C8E" if value == "Before Head" else "#D65F4A" for value in plot_df["direction"].astype(str)]
        bars = ax.barh(plot_df["direction"].astype(str), plot_df["mean"], color=colors, edgecolor="white", linewidth=1.2)
        xerr = [plot_df["mean"] - plot_df["ci_low"], plot_df["ci_high"] - plot_df["mean"]]
        ax.errorbar(plot_df["mean"], range(len(plot_df)), xerr=xerr, fmt="none", ecolor="black", capsize=3, lw=0.9)
        xmax = float(plot_df["ci_high"].max())
        xmin = min(0.0, float(plot_df["ci_low"].min()))
        span = max(xmax - xmin, abs(xmax), 1e-9)
        ax.set_xlim(xmin, xmax + 0.32 * span)
        for bar, (_, row) in zip(bars, plot_df.iterrows()):
            ax.text(
                max(float(row["mean"]), float(row["ci_high"])) + 0.06 * span,
                bar.get_y() + bar.get_height() / 2,
                f"{row['mean']:.4g}",
                va="center",
                ha="left",
                fontsize=12,
                fontweight="semibold",
            )
        ax.set_xlabel(LABELS.get(metric, metric))
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
    fig.tight_layout(w_pad=2.0)
    out = out_dir / f"matched_{args.pos.lower()}_head_direction_histogram_{layer_mode}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def write_markdown_summary(summary_df: pd.DataFrame, paths: dict[str, Path], out_path: Path) -> None:
    lines = [
        "# Matched Head-Direction Control",
        "",
        "The comparison fixes POS and compares `Before Head` (`head_dist > 0`) against `After Head` (`head_dist < 0`) after coarsened exact matching on corpus covariates.",
        "",
        "| Metric | Scope | Before - After | 95% CI | Support |",
        "|---|---|---:|---:|---|",
    ]
    for _, row in summary_df.sort_values(["metric", "scope"]).iterrows():
        lines.append(
            f"| {row['metric']} | {row['scope'].replace('_', ' ')} | "
            f"{row['before_minus_after']:.4g} | [{row['ci_low']:.4g}, {row['ci_high']:.4g}] | "
            f"{row['support'].replace('_', ' ')} |"
        )
    lines.extend(["", "## Files", ""])
    for name, path in sorted(paths.items()):
        lines.append(f"- {name}: `{path}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_geometry(args: argparse.Namespace, sent_df: pd.DataFrame, matched_df: pd.DataFrame, paths: dict[str, Path]) -> None:
    out = args.output_dir / args.pos.lower() / model_short_name(args.model) / args.word_rep_mode
    tables = out / "tables"
    set_reproducibility(args.seed)
    if not args.allow_download:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    reps, filled = embed_subset(sent_df, matched_df, args.model, args.word_rep_mode, args.batch_size)
    reps = reps[:, filled]
    matched_kept = matched_df.reset_index(drop=True).loc[filled].reset_index(drop=True)
    group_labels = matched_kept["direction"].to_numpy()
    matched_kept.to_csv(tables / "matched_tokens_with_embeddings_kept.csv", index=False)

    metric_frames = []
    for metric in args.metrics:
        metric_frames.append(metric_bootstrap(reps, group_labels, metric, args))
    metric_df = pd.concat(metric_frames, ignore_index=True)
    metric_path = tables / "matched_head_direction_metrics.csv"
    metric_df.to_csv(metric_path, index=False)
    paths["matched_metrics"] = metric_path

    contrast_df = contrast_from_metric_table(metric_df)
    contrast_path = tables / "matched_before_minus_after_contrast.csv"
    contrast_df.to_csv(contrast_path, index=False)
    paths["matched_contrast"] = contrast_path

    summary_df = summarize_contrast(contrast_df)
    summary_path = tables / "matched_before_minus_after_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    paths["matched_summary"] = summary_path

    paths["histogram_mean_pdf"] = plot_metric_histograms(metric_df, out, args, "mean")
    paths["histogram_last_pdf"] = plot_metric_histograms(metric_df, out, args, "last")
    markdown_path = tables / "paper_section_matched_head_direction_control.md"
    write_markdown_summary(summary_df, paths, markdown_path)
    paths["markdown_summary"] = markdown_path


def main() -> None:
    args = parse_args()
    args.pos = args.pos.upper()
    random.seed(args.seed)
    np.random.seed(args.seed)

    sent_df, token_df = build_token_frame(args)
    matched_df, strata_df = matched_sample(args, token_df)
    paths = write_corpus_outputs(args, token_df, matched_df, strata_df)

    print(f"POS: {args.pos}")
    print(f"Raw tokens: {len(token_df):,}")
    print(token_df["direction"].value_counts().to_dict())
    print(f"Matched tokens: {len(matched_df):,}")
    print(matched_df["direction"].value_counts().to_dict())
    print(f"Matched strata: {len(strata_df):,}")
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")

    if args.summary_only:
        return
    run_geometry(args, sent_df, matched_df, paths)
    print("Geometry outputs:")
    for name, path in sorted(paths.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
