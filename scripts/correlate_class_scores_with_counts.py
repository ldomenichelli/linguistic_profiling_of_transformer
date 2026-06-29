#!/usr/bin/env python3
"""Correlate class-level geometry scores with class frequency and type counts.

The script recomputes class counts from the sentence CSV using the same coarse
bins used by the metric tables, joins those counts to every selected metric row,
then computes correlations across all feature classes together.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SENTENCE_CSV = PROJECT_ROOT / "code" / "data" / "en_ewt-ud-train_sentences.csv"
DEFAULT_GEOMETRY_ROOT = PROJECT_ROOT
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "correlation" / "class_frequency_type_correlations"

ISOTROPY = {"iso", "sf", "vmf_kappa"}
LINEAR_ID = {
    "erank",
    "pr",
    "stable_rank",
    "lpca",
    "lpca95",
    "lpca99",
    "lpca99_skdim",
    "pca95",
    "pca99",
    "spect",
}
NONLINEAR_ID = {"twonn", "gride", "mle", "mom", "tle", "ess", "mada", "corrint", "fishers", "knn"}
METRIC_ORDER = [
    "iso",
    "sf",
    "vmf_kappa",
    "erank",
    "pr",
    "stable_rank",
    "lpca",
    "lpca95",
    "lpca99",
    "lpca99_skdim",
    "pca95",
    "pca99",
    "spect",
    "twonn",
    "gride",
    "mle",
    "mom",
    "tle",
    "ess",
    "mada",
    "corrint",
    "fishers",
    "knn",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentence-csv", type=Path, default=DEFAULT_SENTENCE_CSV)
    parser.add_argument("--geometry-root", type=Path, default=DEFAULT_GEOMETRY_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write CSVs; skip the compact last-layer heatmaps.",
    )
    parser.add_argument(
        "--skip-feature-permutation",
        action="store_true",
        help="Skip last-layer per-feature Spearman permutation tests.",
    )
    parser.add_argument(
        "--permutation-resamples",
        type=int,
        default=19_999,
        help="Monte Carlo permutations for per-feature Spearman p-values.",
    )
    return parser.parse_args()


def normalize_class(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    try:
        as_float = float(text)
    except ValueError:
        return text
    if as_float.is_integer():
        return str(int(as_float))
    return text


def metric_family(metric: str) -> str:
    metric = metric.strip()
    if metric in ISOTROPY:
        return "isotropy"
    if metric in LINEAR_ID:
        return "linear_id"
    if metric in NONLINEAR_ID:
        return "nonlinear_id"
    return "other"


def binned_value(feature: str, value: object) -> str:
    if feature in {"pos", "relation_type"}:
        return normalize_class(value)
    value_int = int(value)
    if feature in {"length", "index"}:
        return str(min(value_int, 10))
    if feature == "arity":
        return str(min(value_int, 4))
    if feature == "head_dist":
        if value_int <= -6:
            return "-6"
        if value_int >= 6:
            return "6"
        return str(value_int)
    return normalize_class(value)


def list_from_cell(value: str) -> list:
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list cell, got: {value[:80]}")
    return parsed


def compute_class_counts(sentence_csv: Path) -> pd.DataFrame:
    raw_to_feature = {
        "pos": "pos",
        "relation_type": "relation_type",
        "length": "length",
        "index": "index",
        "head_dist": "head_dist",
        "ariety": "arity",
    }
    counts: dict[tuple[str, str], int] = {}
    types: dict[tuple[str, str], set[str]] = {}

    sentences = pd.read_csv(sentence_csv)
    for _, row in sentences.iterrows():
        tokens = list_from_cell(row["tokens"])
        token_types = [str(token).casefold() for token in tokens]
        for raw_col, feature in raw_to_feature.items():
            values = list_from_cell(row[raw_col])
            if len(values) != len(tokens):
                raise ValueError(f"{raw_col} length mismatch in sentence {row.get('sentence_id', '<unknown>')}")
            for token_type, value in zip(token_types, values):
                key = (feature, binned_value(feature, value))
                counts[key] = counts.get(key, 0) + 1
                types.setdefault(key, set()).add(token_type)

    rows = [
        {
            "feature": feature,
            "class": klass,
            "n_tokens": n_tokens,
            "n_types": len(types[(feature, klass)]),
        }
        for (feature, klass), n_tokens in counts.items()
    ]
    out = pd.DataFrame(rows)
    out["class"] = out["class"].map(normalize_class)
    return out.sort_values(["feature", "class"]).reset_index(drop=True)


def is_selected_metric_file(path: Path, geometry_root: Path) -> bool:
    rel = path.relative_to(geometry_root).as_posix()
    if ".ipynb_checkpoints" in rel:
        return False
    if rel.startswith("no_index_1/tables/gpt2_no_index/"):
        return True
    if rel.startswith("no_index_1/tables/tables_"):
        return True
    if "/tables_POS/pos_bootstrap/" in rel and rel.startswith("bert/"):
        return True
    if rel.startswith("bert/tables_"):
        return True
    if rel.startswith("gpt2/tables_") and "/tables_REL/" not in rel:
        return True
    if rel.startswith("gpt2_no_index/tables_REL_GPT2_no_idx0/"):
        return True
    if rel.startswith("gpt2_no_index/tables_HEADDIST_no_index/"):
        return True
    return False


def load_selected_metric_tables(geometry_root: Path) -> pd.DataFrame:
    frames = []
    required = {"model", "feature", "class", "metric", "layer", "mean"}
    for path in sorted(geometry_root.rglob("*_raw_*.csv")):
        if not is_selected_metric_file(path, geometry_root):
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive for odd legacy files
            print(f"Skipping unreadable file {path}: {exc}")
            continue
        if not required.issubset(df.columns):
            continue
        df = df.copy()
        df["source_path"] = str(path)
        df["source_path_relative"] = path.relative_to(geometry_root).as_posix()
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No selected metric tables found under {geometry_root}")

    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics.rename(columns={"n_tokens": "metric_table_n_tokens"})
    metrics["feature"] = metrics["feature"].replace({"relation": "relation_type"})
    metrics["class"] = metrics["class"].map(normalize_class)
    metrics["metric"] = metrics["metric"].astype(str).str.strip()
    metrics["metric_family"] = metrics["metric"].map(metric_family)
    metrics["mean"] = pd.to_numeric(metrics["mean"], errors="coerce")
    metrics["layer"] = pd.to_numeric(metrics["layer"], errors="coerce").astype("Int64")
    if "word_rep_mode" not in metrics.columns:
        metrics["word_rep_mode"] = ""
    metrics["word_rep_mode"] = metrics["word_rep_mode"].fillna("")

    dedupe_cols = ["model", "feature", "class", "metric", "layer", "word_rep_mode", "source_csv"]
    dedupe_cols = [col for col in dedupe_cols if col in metrics.columns]
    metrics = metrics.drop_duplicates(dedupe_cols + ["mean"])
    return metrics.reset_index(drop=True)


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(data) < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan, np.nan
    if method == "spearman":
        result = spearmanr(data["x"], data["y"])
    elif method == "pearson":
        result = pearsonr(data["x"], data["y"])
    else:
        raise ValueError(method)
    stat = getattr(result, "statistic", result[0])
    pvalue = getattr(result, "pvalue", result[1])
    return float(stat), float(pvalue)


def pearson_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom == 0:
        return np.nan
    return float((x * y).sum() / denom)


def stable_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def spearman_permutation_test(
    x: pd.Series,
    y: pd.Series,
    *,
    seed: int,
    n_resamples: int,
    exact_max_n: int = 8,
) -> tuple[float, float, str, int, int]:
    data = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(data)
    if n < 3 or data["x"].nunique() < 2 or data["y"].nunique() < 2:
        return np.nan, np.nan, "insufficient", n, 0

    x_rank = rankdata(data["x"].to_numpy(), method="average")
    y_rank = rankdata(data["y"].to_numpy(), method="average")
    observed = pearson_np(x_rank, y_rank)
    if not np.isfinite(observed):
        return np.nan, np.nan, "insufficient", n, 0

    threshold = abs(observed) - 1e-12
    if n <= exact_max_n and math.factorial(n) <= 50_000:
        total = 0
        extreme = 0
        for permuted_y_rank in itertools.permutations(y_rank):
            total += 1
            statistic = pearson_np(x_rank, np.asarray(permuted_y_rank))
            if abs(statistic) >= threshold:
                extreme += 1
        return observed, extreme / total, "exact", n, total

    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_resamples):
        statistic = pearson_np(x_rank, rng.permutation(y_rank))
        if abs(statistic) >= threshold:
            extreme += 1
    return observed, (extreme + 1) / (n_resamples + 1), "monte_carlo", n, n_resamples


def compute_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    data = joined.copy()
    data["log10_n_tokens"] = np.log10(data["n_tokens"])
    data["log10_n_types"] = np.log10(data["n_types"])
    targets = ["n_tokens", "n_types", "log10_n_tokens", "log10_n_types"]
    group_cols = ["model", "word_rep_mode", "metric_family", "metric", "layer"]

    rows = []
    for keys, group in data.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        features = ",".join(sorted(group["feature"].dropna().unique()))
        for target in targets:
            spearman_r, spearman_p = safe_corr(group["mean"], group[target], "spearman")
            pearson_r, pearson_p = safe_corr(group["mean"], group[target], "pearson")
            rows.append(
                {
                    **base,
                    "target": target,
                    "n_feature_classes": int(group[["feature", "class"]].drop_duplicates().shape[0]),
                    "features_covered": features,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                }
            )
    return pd.DataFrame(rows).sort_values(group_cols + ["target"]).reset_index(drop=True)


def metric_sort_key(metric: str) -> int:
    try:
        return METRIC_ORDER.index(metric)
    except ValueError:
        return len(METRIC_ORDER)


def write_last_layer_pivots(correlations: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for (model, word_rep_mode, metric), group in correlations.groupby(["model", "word_rep_mode", "metric"]):
        max_layer = group["layer"].max()
        rows.append(group[group["layer"] == max_layer])
    last = pd.concat(rows, ignore_index=True) if rows else correlations.iloc[0:0].copy()
    last.to_csv(output_dir / "class_metric_frequency_type_correlations_last_layer.csv", index=False)

    pivot = last.pivot_table(
        index=["model", "word_rep_mode", "metric_family", "metric", "layer"],
        columns="target",
        values="spearman_r",
        aggfunc="first",
    ).reset_index()
    pivot["metric_order"] = pivot["metric"].map(metric_sort_key)
    pivot = pivot.sort_values(["model", "word_rep_mode", "metric_family", "metric_order", "metric"])
    pivot = pivot.drop(columns=["metric_order"])
    pivot.to_csv(output_dir / "class_metric_frequency_type_correlations_last_layer_pivot_spearman.csv", index=False)
    return pivot


def compute_by_feature_last_layer_correlations(
    joined: pd.DataFrame,
    *,
    n_resamples: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = joined.copy()
    data["log10_n_tokens"] = np.log10(data["n_tokens"])
    data["log10_n_types"] = np.log10(data["n_types"])
    group_cols = ["model", "word_rep_mode", "feature", "metric_family", "metric"]
    last_layers = data.groupby(group_cols, dropna=False)["layer"].transform("max")
    last = data[data["layer"].eq(last_layers)].copy()
    targets = ["n_tokens", "n_types", "log10_n_tokens", "log10_n_types"]

    rows = []
    for keys, group in last.groupby(group_cols + ["layer"], dropna=False):
        base = dict(zip(group_cols + ["layer"], keys))
        for target in targets:
            spearman_r, spearman_p = safe_corr(group["mean"], group[target], "spearman")
            _, permutation_p, permutation_method, n_classes, n_permutations = spearman_permutation_test(
                group["mean"],
                group[target],
                seed=stable_seed(*keys, target),
                n_resamples=n_resamples,
            )
            rows.append(
                {
                    **base,
                    "target": target,
                    "n_classes": n_classes,
                    "spearman_r": spearman_r,
                    "spearman_asymptotic_p": spearman_p,
                    "spearman_permutation_p": permutation_p,
                    "permutation_method": permutation_method,
                    "n_permutations": n_permutations,
                    "classes": ",".join(map(str, group["class"].tolist())),
                }
            )

    by_feature = pd.DataFrame(rows).sort_values(group_cols + ["layer", "target"]).reset_index(drop=True)
    pivot = by_feature.pivot_table(
        index=["model", "word_rep_mode", "feature", "metric_family", "metric", "layer", "n_classes"],
        columns="target",
        values="spearman_r",
        aggfunc="first",
    ).reset_index()
    key_metrics = by_feature[
        by_feature["target"].eq("n_tokens") & by_feature["metric"].isin(["iso", "lpca99", "pca99", "gride"])
    ].copy()
    return by_feature, pivot, key_metrics


def plot_last_layer_heatmaps(pivot: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = ["n_tokens", "n_types", "log10_n_tokens", "log10_n_types"]
    for (model, word_rep_mode), group in pivot.groupby(["model", "word_rep_mode"], dropna=False):
        plot_data = group.set_index("metric")[targets]
        if plot_data.empty:
            continue
        fig_h = max(5.0, 0.35 * len(plot_data))
        fig, ax = plt.subplots(figsize=(8.5, fig_h))
        masked = np.ma.masked_invalid(plot_data.to_numpy(dtype=float))
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("white")
        im = ax.imshow(masked, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, rotation=30, ha="right")
        ax.set_yticks(range(len(plot_data.index)))
        ax.set_yticklabels(plot_data.index)
        title_mode = f" ({word_rep_mode})" if word_rep_mode else ""
        ax.set_title(f"Last-layer class score vs frequency/type Spearman: {model}{title_mode}")
        for i in range(plot_data.shape[0]):
            for j in range(plot_data.shape[1]):
                val = plot_data.iat[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Spearman r")
        fig.tight_layout()
        safe_model = str(model).replace("/", "_").replace(" ", "_")
        safe_mode = str(word_rep_mode).replace("/", "_").replace(" ", "_") or "mode"
        base = output_dir / f"last_layer_spearman_heatmap_{safe_model}_{safe_mode}"
        fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts = compute_class_counts(args.sentence_csv)
    counts.to_csv(args.output_dir / "class_counts_binned.csv", index=False)

    metrics = load_selected_metric_tables(args.geometry_root)
    metrics.to_csv(args.output_dir / "selected_class_metric_scores_long.csv", index=False)

    joined = metrics.merge(counts, on=["feature", "class"], how="left", validate="many_to_one")
    unmatched = joined[joined["n_tokens"].isna()].copy()
    if not unmatched.empty:
        unmatched.to_csv(args.output_dir / "unmatched_class_metric_rows.csv", index=False)
    joined = joined.dropna(subset=["mean", "n_tokens", "n_types"]).copy()
    joined["n_tokens"] = joined["n_tokens"].astype(int)
    joined["n_types"] = joined["n_types"].astype(int)
    joined.to_csv(args.output_dir / "class_metric_scores_with_counts.csv", index=False)

    correlations = compute_correlations(joined)
    correlations.to_csv(args.output_dir / "class_metric_frequency_type_correlations.csv", index=False)
    pivot = write_last_layer_pivots(correlations, args.output_dir)
    if not args.no_plots:
        plot_last_layer_heatmaps(pivot, args.output_dir)

    by_feature_rows = 0
    if not args.skip_feature_permutation:
        by_feature, by_feature_pivot, key_metrics = compute_by_feature_last_layer_correlations(
            joined,
            n_resamples=args.permutation_resamples,
        )
        by_feature.to_csv(
            args.output_dir / "class_metric_frequency_type_correlations_by_feature_last_layer.csv",
            index=False,
        )
        by_feature_pivot.to_csv(
            args.output_dir / "class_metric_frequency_type_correlations_by_feature_last_layer_pivot_spearman.csv",
            index=False,
        )
        key_metrics.to_csv(
            args.output_dir / "feature_separate_frequency_key_metrics_last_layer.csv",
            index=False,
        )
        by_feature_rows = len(by_feature)

    print(f"Counts: {len(counts)} class bins")
    print(f"Metric rows selected: {len(metrics)}")
    print(f"Joined rows with counts: {len(joined)}")
    print(f"Correlation rows: {len(correlations)}")
    if not args.skip_feature_permutation:
        print(f"By-feature last-layer permutation correlation rows: {by_feature_rows}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
