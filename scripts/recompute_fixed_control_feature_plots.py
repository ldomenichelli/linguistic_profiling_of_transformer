#!/usr/bin/env python3
"""Recompute fixed-frequency and fixed-type feature metric plots.

The script mirrors the existing feature-metric notebooks, but makes the
balancing rule explicit:

- fixed_freq: keep classes whose token frequency is > min threshold, then
  sample each kept class to the smallest remaining class frequency.
- fixed_type: keep classes whose type frequency is > min threshold, then
  sample each kept class to the smallest remaining type frequency, using one
  token occurrence per sampled type.

Run one feature/model/control job at a time to keep memory bounded.
"""

from __future__ import annotations

import argparse
import gc
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import recompute_mean_feature_plots as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "plots_extra" / "metrics"
ORIGINAL_LOAD_TOKENIZER_AND_MODEL = base.load_tokenizer_and_model

CONTROL_DIR = {
    "fixed_freq": "fixed_freq",
    "fixed_type": "fixed_type",
}

FEATURE_DIR = {
    "pos": "pos",
    "length": "length",
    "index": "index",
    "arity": "arity",
    "head_dist": "head_dist",
    "relation_type": "relation",
}

FEATURE_SUFFIX = {
    "pos": "pos",
    "length": "length",
    "index": "index",
    "arity": "arity",
    "head_dist": "head_dist",
    "relation_type": "relation",
}

METRIC_ALIASES = {
    "rand": ("rand", "randcos"),
    "randcos": ("randcos", "rand"),
    "spect": ("spect", "spectral"),
    "spectral": ("spectral", "spect"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", required=True, choices=sorted(CONTROL_DIR))
    parser.add_argument("--feature", required=True, choices=sorted(FEATURE_DIR))
    parser.add_argument("--model", required=True, choices=sorted(base.MODEL_IDS))
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["iso", "lpca", "lpca99", "gride", "corrint", "fishers", "mom", "rand", "spect", "tle", "twonn"],
    )
    parser.add_argument("--csv", type=Path, default=base.DEFAULT_CSV)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--word-rep-mode", default="mean", choices=["first", "last", "mean"])
    parser.add_argument(
        "--output-rep-dir",
        default=None,
        help="Folder name for the representation mode. Use first_last for BERT-first/GPT2-last plots.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-class-count", type=int, default=1000)
    parser.add_argument("--min-type-count", type=int, default=1000)
    parser.add_argument("--fast-bootstrap", type=int, default=20)
    parser.add_argument("--heavy-bootstrap", type=int, default=8)
    parser.add_argument("--fast-sample-cap", type=int, default=30000)
    parser.add_argument("--heavy-sample-cap", type=int, default=3000)
    parser.add_argument("--relation-top-k", type=int, default=10)
    parser.add_argument("--length-max-class", type=int, default=10)
    parser.add_argument("--index-max-class", type=int, default=10)
    parser.add_argument("--arity-max-class", type=int, default=4)
    parser.add_argument("--head-dist-max-abs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-title", action="store_true")
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="Use the architecture config and tokenizer, but initialize model weights randomly.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads. By default, use local cache only.",
    )
    return parser.parse_args()


def output_dir(args: argparse.Namespace) -> Path:
    rep_dir = args.output_rep_dir or args.word_rep_mode
    return args.out_root / CONTROL_DIR[args.control] / FEATURE_DIR[args.feature] / rep_dir / args.model


def metric_filename(metric: str, args: argparse.Namespace) -> str:
    suffix = FEATURE_SUFFIX[args.feature]
    baseline_tag = "_untrained" if args.untrained else ""
    if args.control == "fixed_type":
        return f"{metric}_{suffix}_TYPES_balanced{baseline_tag}.png"
    return f"{metric}_{suffix}_balanced{baseline_tag}.png"


def csv_filename(metric: str, args: argparse.Namespace) -> str:
    return metric_filename(metric, args).replace(".png", ".csv")


def metric_tokens(metric: str) -> tuple[str, ...]:
    if metric == "lpca":
        return ("lpca",)
    return METRIC_ALIASES.get(metric, (metric,))


def looks_like_metric_plot(path: Path, metric: str, args: argparse.Namespace) -> bool:
    name = path.name.lower()
    if path.suffix.lower() != ".png":
        return False
    if "balanced" not in name:
        return False
    if args.untrained and "untrained" not in name:
        return False
    if not args.untrained and "untrained" in name:
        return False
    if args.control == "fixed_type" and "type" not in name:
        return False
    if args.control == "fixed_freq" and "type" in name:
        return False
    if metric == "lpca":
        return name.startswith("lpca_") or " lpca_" in name
    if metric == "lpca99":
        return "lpca99" in name
    return any(token in name for token in metric_tokens(metric))


def load_untrained_tokenizer_and_model(model_key: str, allow_download: bool, device: str):
    model_id = base.MODEL_IDS[model_key]
    local_files_only = not allow_download

    def load(mid: str):
        tok_kwargs = {"use_fast": True, "local_files_only": local_files_only}
        if model_key == "gpt2":
            tok_kwargs["add_prefix_space"] = True
        tokenizer = AutoTokenizer.from_pretrained(mid, **tok_kwargs)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        config = AutoConfig.from_pretrained(
            mid,
            output_hidden_states=True,
            local_files_only=local_files_only,
        )
        model = AutoModel.from_config(config)
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return tokenizer, model, f"{mid}-untrained"

    attempts = [model_id]
    if model_key == "gpt2":
        attempts.append("openai-community/gpt2")

    errors = []
    for mid in attempts:
        try:
            tokenizer, model, resolved = load(mid)
            model.eval().to(device)
            if device == "cuda":
                model.half()
            return tokenizer, model, resolved
        except Exception as exc:
            errors.append(f"{mid}: {exc}")
    raise RuntimeError("Could not load untrained model/tokenizer:\n" + "\n".join(errors))


def existing_plot(out_dir: Path, metric: str, args: argparse.Namespace) -> Path | None:
    exact = out_dir / metric_filename(metric, args)
    if exact.exists() and exact.stat().st_size > 0:
        return exact
    if not out_dir.exists():
        return None
    for path in out_dir.iterdir():
        if path.is_file() and looks_like_metric_plot(path, metric, args):
            return path
    return None


def normalize_word(value: object) -> str:
    return str(value).lower()


def sample_fixed_frequency(df_tok: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df_tok["class"].astype(str).value_counts()
    eligible = counts[counts > args.min_class_count].sort_index(key=lambda s: s.map(str))
    if eligible.empty:
        raise ValueError(
            f"No {args.feature} classes have token frequency > {args.min_class_count}."
        )
    balance_n = int(eligible.min())
    rng = np.random.default_rng(args.seed)
    parts = []
    rows = []
    for klass, n_raw in eligible.items():
        group = df_tok[df_tok["class"].astype(str) == str(klass)]
        take = rng.choice(group.index.to_numpy(), size=balance_n, replace=False)
        parts.append(group.loc[take])
        rows.append(
            {
                "class": str(klass),
                "raw_token_frequency": int(n_raw),
                "kept_token_frequency": balance_n,
                "raw_type_frequency": int(group["word"].map(normalize_word).nunique()),
                "control": args.control,
            }
        )
    sampled = pd.concat(parts, ignore_index=True)
    return sampled, pd.DataFrame(rows)


def sample_fixed_type(df_tok: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df_tok.copy()
    work["word_lower"] = work["word"].map(normalize_word)

    type_counts = work.groupby("class")["word_lower"].nunique()
    eligible = type_counts[type_counts > args.min_type_count].sort_index(key=lambda s: s.map(str))
    if eligible.empty:
        raise ValueError(
            f"No {args.feature} classes have type frequency > {args.min_type_count}."
        )
    balance_n = int(eligible.min())
    rng = np.random.default_rng(args.seed)
    parts = []
    rows = []
    for klass, n_types_raw in eligible.items():
        group = work[work["class"].astype(str) == str(klass)]
        unique_types = np.array(sorted(group["word_lower"].unique()))
        chosen_types = rng.choice(unique_types, size=balance_n, replace=False)
        chosen_parts = []
        for word_type in chosen_types:
            type_rows = group[group["word_lower"] == word_type]
            chosen_parts.append(type_rows.sample(n=1, random_state=int(rng.integers(0, 2**32 - 1))))
        selected = pd.concat(chosen_parts, ignore_index=True)
        parts.append(selected.drop(columns=["word_lower"]))
        rows.append(
            {
                "class": str(klass),
                "raw_token_frequency": int(len(group)),
                "raw_type_frequency": int(n_types_raw),
                "kept_type_frequency": balance_n,
                "kept_token_frequency": int(len(selected)),
                "control": args.control,
            }
        )
    sampled = pd.concat(parts, ignore_index=True)
    return sampled, pd.DataFrame(rows)


def plot_metric(
    stats: dict[str, dict[str, np.ndarray | int]],
    layers: np.ndarray,
    metric: str,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    classes = sorted(stats, key=base.numeric_sort_key)
    palette = sns.color_palette("tab20", n_colors=max(1, len(classes)))
    for idx, klass in enumerate(classes):
        mean = stats[klass]["mean"]
        low = stats[klass]["ci_low"]
        high = stats[klass]["ci_high"]
        color = palette[idx]
        ax.plot(layers, mean, label=klass, lw=1.8, color=color)
        ax.fill_between(layers, low, high, alpha=0.15, color=color)

    label = base.LABELS.get(metric, metric.upper())
    ax.set_xlabel("Layer")
    ax.set_ylabel(label)
    if not args.no_title:
        control_label = "fixed type" if args.control == "fixed_type" else "fixed frequency"
        ax.set_title(f"{label} - {base.MODEL_IDS[args.model]} - {args.feature} - {control_label}")
    ax.legend(ncol=5, fontsize="small", title=args.feature, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_metric_csv(
    stats: dict[str, dict[str, np.ndarray | int]],
    metric: str,
    class_col: str,
    resolved_model: str,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    rows = []
    for klass in sorted(stats, key=base.numeric_sort_key):
        item = stats[klass]
        mean = item["mean"]
        low = item["ci_low"]
        high = item["ci_high"]
        for layer, value in enumerate(mean):
            rows.append(
                {
                    "subset": args.control,
                    "model": resolved_model,
                    "feature": class_col,
                    "class": klass,
                    "metric": metric,
                    "layer": layer,
                    "mean": float(value),
                    "ci_low": float(low[layer]),
                    "ci_high": float(high[layer]),
                    "n_tokens": int(item["n_tokens"]),
                    "word_rep_mode": args.word_rep_mode,
                    "source_csv": args.csv.name,
                    "min_class_count": args.min_class_count,
                    "min_type_count": args.min_type_count,
                    "fast_bootstrap": args.fast_bootstrap,
                    "heavy_bootstrap": args.heavy_bootstrap,
                    "fast_sample_cap": args.fast_sample_cap,
                    "heavy_sample_cap": args.heavy_sample_cap,
                }
            )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not args.allow_download:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    fast_metrics, heavy_metrics = base.metric_registry()
    metrics = [name.lower() for name in args.metrics]
    unknown = [name for name in metrics if name not in fast_metrics and name not in heavy_metrics]
    if unknown:
        raise ValueError(f"Unknown or unavailable metrics: {unknown}")

    out_dir = output_dir(args)
    table_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    todo_metrics = []
    for metric in metrics:
        done_path = existing_plot(out_dir, metric, args) if args.skip_existing else None
        if done_path is not None:
            print(f"skip existing {metric}: {done_path}", flush=True)
        else:
            todo_metrics.append(metric)

    if not todo_metrics:
        print(
            f"all requested metrics already exist for control={args.control} "
            f"feature={args.feature} model={args.model}",
            flush=True,
        )
        return

    print(
        f"control={args.control} feature={args.feature} model={args.model} "
        f"word_rep_mode={args.word_rep_mode} untrained={args.untrained} metrics={todo_metrics}",
        flush=True,
    )
    print(f"device={device} out_dir={out_dir}", flush=True)
    if args.untrained:
        base.load_tokenizer_and_model = load_untrained_tokenizer_and_model
    else:
        base.load_tokenizer_and_model = ORIGINAL_LOAD_TOKENIZER_AND_MODEL

    df_sent, df_tok, class_col = base.build_token_frame(args)
    print(f"token rows before balancing: {len(df_tok):,}", flush=True)
    if args.control == "fixed_freq":
        df_tok, balance = sample_fixed_frequency(df_tok, args)
    else:
        df_tok, balance = sample_fixed_type(df_tok, args)

    df_tok = df_tok.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    balance.to_csv(table_dir / f"{args.control}_{args.feature}_balance_summary.csv", index=False)
    print(f"token rows after balancing: {len(df_tok):,}", flush=True)
    print(f"balance summary: {balance.to_dict(orient='records')}", flush=True)

    reps, filled, resolved_model = base.embed_subset(df_sent, df_tok, args, device)
    if not filled.all():
        missing = int((~filled).sum())
        print(f"warning: missing vectors for {missing:,} of {len(filled):,} rows", flush=True)
        reps = reps[:, filled, :]
        df_tok = df_tok.reset_index(drop=True).loc[filled].reset_index(drop=True)

    class_values = df_tok["class"].astype(str).to_numpy()
    layers = np.arange(reps.shape[0])

    for metric in todo_metrics:
        is_fast = metric in fast_metrics
        compute = fast_metrics[metric] if is_fast else heavy_metrics[metric]
        print(f"computing {metric} ({'fast' if is_fast else 'heavy'})", flush=True)
        stats: dict[str, dict[str, np.ndarray | int]] = {}
        for klass in sorted(pd.unique(class_values), key=base.numeric_sort_key):
            indices = np.where(class_values == klass)[0]
            if len(indices) < 3:
                print(f"skip class={klass}: only {len(indices)} rows", flush=True)
                continue
            mean, low, high = base.bootstrap_layers(reps, indices, metric, compute, is_fast, args)
            stats[klass] = {
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n_tokens": int(len(indices)),
            }
            gc.collect()

        png_path = out_dir / metric_filename(metric, args)
        csv_path = table_dir / csv_filename(metric, args)
        write_metric_csv(stats, metric, class_col, resolved_model, args, csv_path)
        plot_metric(stats, layers, metric, args, png_path)
        print(f"saved {png_path}", flush=True)
        print(f"saved {csv_path}", flush=True)
        del stats
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    del reps
    gc.collect()
    print("done", flush=True)


if __name__ == "__main__":
    main()
