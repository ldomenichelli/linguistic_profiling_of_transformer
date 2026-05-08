#!/usr/bin/env python3
"""Run POS-conditioned dependency head-distance geometry experiments.

This is the reusable version of the nominal head-distance experiment in
code/head_dist.ipynb. It filters tokens by POS first, then compares geometry
across signed head-distance classes inside that POS.
"""

from __future__ import annotations

import argparse
import ast
import gc
import inspect
import os
import random
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "code" / "data"

DEFAULT_POS = ("NOUN", "VERB", "ADJ", "PRON", "PROPN")
DEFAULT_METRICS = ("iso", "lpca99", "gride")

LABELS = {
    "iso": "IsoScore",
    "lpca99": "Linear ID",
    "pca99": "Linear ID",
    "gride": "Nonlinear ID",
}

MAIN_METRIC_NAMES = {
    "iso": "iso",
    "lpca99": "LID",
    "pca99": "LID",
    "gride": "NLID",
}

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DATA_DIR / "en_ewt-ud-train_sentences.csv",
        help="Sentence-level CSV with tokens, POS tags, and head_dist lists.",
    )
    parser.add_argument(
        "--model",
        default="bert-base-uncased",
        help="Hugging Face model id. Use gpt2 for the GPT experiment.",
    )
    parser.add_argument(
        "--word-rep-mode",
        default="first",
        choices=("first", "last", "mean"),
        help="Subword aggregation. BERT nominal runs usually use first; GPT uses last or mean.",
    )
    parser.add_argument(
        "--pos",
        nargs="+",
        default=list(DEFAULT_POS),
        help="UPOS tags to run, e.g. NOUN VERB ADJ PRON PROPN.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        choices=tuple(LABELS),
        help="Metrics to compute.",
    )
    parser.add_argument("--head-dist-col", default="head_dist")
    parser.add_argument("--pos-col", default="pos")
    parser.add_argument("--head-dist-clamp", type=int, default=6)
    parser.add_argument("--include-zero-class", action="store_true")
    parser.add_argument("--include-first-token", action="store_true")
    parser.add_argument("--raw-max-per-class", type=int, default=int(1e12))
    parser.add_argument("--n-bootstrap-fast", type=int, default=50)
    parser.add_argument("--n-bootstrap-heavy", type=int, default=20)
    parser.add_argument("--fast-bs-max-samp-per-class", type=int, default=int(1e12))
    parser.add_argument("--heavy-bs-max-samp-per-class", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=PROJECT_ROOT / "plots_extra" / "pos_x_head_dist",
        help="Root directory for per-POS plots.",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=PROJECT_ROOT / "tables_POS_HEAD_DIST",
        help="Root directory for per-POS bootstrap tables.",
    )
    parser.add_argument(
        "--main-plots-dir",
        type=Path,
        default=PROJECT_ROOT / "plots_main" / "plots_features",
        help="Directory for paper-style PDF aliases such as bert_head_LID_VERB.pdf.",
    )
    parser.add_argument(
        "--no-main-plots",
        action="store_true",
        help="Do not write PDF aliases into plots_main/plots_features.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write token counts by POS and head distance; do not load transformers.",
    )
    return parser.parse_args()


def _to_list(value):
    if isinstance(value, str) and value.startswith("["):
        return ast.literal_eval(value)
    return value


def _sanitize(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value)).strip("_")


def model_short_name(model: str) -> str:
    lower = model.lower()
    if "bert" in lower:
        return "bert"
    if "gpt" in lower:
        return "gpt2"
    return _sanitize(model)


def main_model_prefix(model: str) -> str:
    short = model_short_name(model)
    return "gpt" if short.startswith("gpt") else short


def set_reproducibility(seed: int) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def load_pos_head_dist_from_column(
    csv_path: Path,
    pos_keep: str,
    head_dist_col: str,
    pos_col: str,
    clamp: int,
    include_zero: bool,
    exclude_first_token: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    need = ["sentence_id", "tokens", head_dist_col, pos_col]
    df = pd.read_csv(csv_path, usecols=need, dtype={"sentence_id": str})
    df["tokens"] = df["tokens"].apply(_to_list)
    df[head_dist_col] = df[head_dist_col].apply(_to_list)
    df[pos_col] = df[pos_col].apply(_to_list)

    rows = []
    for sid, toks, dists, poss in df[["sentence_id", "tokens", head_dist_col, pos_col]].itertuples(index=False):
        length = min(len(toks), len(dists), len(poss))
        for word_id in range(length):
            if exclude_first_token and word_id == 0:
                continue
            pos = str(poss[word_id])
            if pos != pos_keep:
                continue
            try:
                dist = int(dists[word_id])
            except Exception:
                continue
            if dist == 0 and not include_zero:
                continue
            dist = max(-clamp, min(clamp, dist))
            rows.append((str(sid), int(word_id), str(dist), toks[word_id], pos))

    token_df = pd.DataFrame(rows, columns=["sentence_id", "word_id", "head_dist_class", "word", "pos"])
    if token_df.empty:
        raise ValueError(f"No token rows constructed for POS={pos_keep}.")

    sent_df = df[["sentence_id", "tokens"]].drop_duplicates("sentence_id")
    return sent_df, token_df


def sample_raw(token_df: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    if max_per_class >= len(token_df):
        return token_df.reset_index(drop=True)
    chunks = []
    for _class_name, class_df in token_df.groupby("head_dist_class", sort=False):
        n = min(max_per_class, len(class_df))
        chunks.append(class_df.sample(n=n, random_state=seed))
    return pd.concat(chunks, ignore_index=True)


def _load_tokenizer_and_model(model_name: str):
    from transformers import AutoModel, AutoTokenizer, GPT2TokenizerFast

    candidates: list[str] = []
    seen: set[str] = set()
    if "gpt2" in model_name.lower():
        for candidate in (model_name, "openai-community/gpt2", "gpt2"):
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)
    else:
        candidates = [model_name]

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            if "gpt2" in candidate.lower():
                tokenizer = GPT2TokenizerFast.from_pretrained(candidate, add_prefix_space=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=True)

            tokenizer.padding_side = "right"
            if tokenizer.pad_token is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModel.from_pretrained(candidate, output_hidden_states=True)
            if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
                model.config.pad_token_id = tokenizer.pad_token_id
            return tokenizer, model, candidate
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not load {model_name}; tried {candidates}. Last error: {last_error}")


def _num_hidden_layers(model) -> int:
    n_layers = getattr(model.config, "num_hidden_layers", None)
    if n_layers is None:
        n_layers = getattr(model.config, "n_layer", None)
    if n_layers is None:
        raise ValueError("Cannot determine number of hidden layers from model.config.")
    return int(n_layers)


def _hidden_size(model) -> int:
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(model.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Cannot determine hidden size from model.config.")
    return int(hidden_size)


def _autocast_context(torch_module, device: str, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch_module, "amp") and hasattr(torch_module.amp, "autocast"):
        return torch_module.amp.autocast(device_type=device, enabled=True)
    return torch_module.cuda.amp.autocast(enabled=True)


def embed_subset(
    sent_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    model_name: str,
    word_rep_mode: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if "gpt" in model_name.lower() and word_rep_mode == "first":
        raise ValueError("GPT-style models should use --word-rep-mode last or mean.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sent_df = sent_df.copy()
    subset_df = subset_df.copy()
    sent_df["sentence_id"] = sent_df["sentence_id"].astype(str)
    subset_df["sentence_id"] = subset_df["sentence_id"].astype(str)

    by_sid: dict[str, list[tuple[int, int]]] = {}
    for global_idx, (sid, word_id) in enumerate(subset_df[["sentence_id", "word_id"]].itertuples(index=False)):
        by_sid.setdefault(str(sid), []).append((global_idx, int(word_id)))

    sids = list(by_sid)
    selected_sent = (
        sent_df[sent_df["sentence_id"].isin(sids)]
        .drop_duplicates("sentence_id")
        .set_index("sentence_id")
        .loc[sids]
    )

    tokenizer, model, resolved_model = _load_tokenizer_and_model(model_name)
    model = model.eval().to(device)
    if device == "cuda":
        model.half()

    enc_kwargs = {"is_split_into_words": True, "return_tensors": "pt", "padding": True}
    if "add_prefix_space" in inspect.signature(tokenizer.__call__).parameters:
        enc_kwargs["add_prefix_space"] = True

    n_layers = _num_hidden_layers(model) + 1
    hidden_size = _hidden_size(model)
    n_tokens = len(subset_df)
    reps = np.zeros((n_layers, n_tokens, hidden_size), dtype=np.float16)
    filled = np.zeros(n_tokens, dtype=bool)

    use_amp = device == "cuda"
    with torch.no_grad():
        for start in tqdm(range(0, len(sids), batch_size), desc=f"{resolved_model} embed {subset_df['pos'].iloc[0]}"):
            batch_ids = sids[start : start + batch_size]
            batch_tokens = selected_sent.loc[batch_ids, "tokens"].tolist()
            encoded_batch = tokenizer(batch_tokens, **enc_kwargs)
            encoded_tensors = {key: value.to(device) for key, value in encoded_batch.items()}

            with _autocast_context(torch, device, use_amp):
                output = model(**encoded_tensors)
            hidden = torch.stack(output.hidden_states).detach().cpu().numpy().astype(np.float32)

            for batch_index, sid in enumerate(batch_ids):
                word_to_tokens: dict[int, list[int]] = {}
                word_ids = encoded_batch.word_ids(batch_index)
                if word_ids is None:
                    raise RuntimeError("A fast tokenizer with word_ids support is required.")
                for token_index, word_id in enumerate(word_ids):
                    if word_id is not None:
                        word_to_tokens.setdefault(int(word_id), []).append(int(token_index))

                for global_idx, word_id in by_sid.get(str(sid), []):
                    token_positions = word_to_tokens.get(word_id)
                    if not token_positions:
                        continue
                    if word_rep_mode == "first":
                        vec = hidden[:, batch_index, token_positions[0], :]
                    elif word_rep_mode == "last":
                        vec = hidden[:, batch_index, token_positions[-1], :]
                    elif word_rep_mode == "mean":
                        vec = hidden[:, batch_index, token_positions, :].mean(axis=1)
                    else:
                        raise ValueError("word_rep_mode must be first, last, or mean.")
                    reps[:, global_idx, :] = vec.astype(np.float16, copy=False)
                    filled[global_idx] = True

            del encoded_batch, encoded_tensors, output, hidden
            if device == "cuda":
                torch.cuda.empty_cache()

    missing = int((~filled).sum())
    if missing:
        print(f"Warning: missing vectors for {missing} of {n_tokens} tokens.")

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return reps, filled


def _center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(0, keepdims=True)


def _eigvals_from_x(x: np.ndarray) -> np.ndarray:
    x_centered = _center(x.astype(np.float32, copy=False))
    try:
        _, singular_values, _ = np.linalg.svd(x_centered, full_matrices=False)
    except Exception:
        return np.array([], dtype=np.float64)
    eigvals = (singular_values.astype(np.float64) ** 2)
    eigvals.sort()
    return eigvals[::-1]


def _fast_isoscore(x: np.ndarray) -> float:
    eigvals = _eigvals_from_x(x)
    eigvals = eigvals[np.isfinite(eigvals) & (eigvals >= 0)]
    n = int(eigvals.size)
    if n < 2:
        return float("nan")
    norm = float(np.linalg.norm(eigvals))
    if norm <= 0:
        return 0.0
    root_n = np.sqrt(n)
    denom = np.sqrt(2.0 * (n - root_n))
    if denom <= 0:
        return float("nan")
    delta = float(np.linalg.norm((root_n * eigvals) / norm - 1.0) / denom)
    delta = float(np.clip(delta, 0.0, 1.0))
    phi = (n - (delta**2) * (n - root_n)) ** 2 / (n**2)
    score = (n * phi - 1.0) / (n - 1.0)
    return float(np.clip(score, 0.0, 1.0))


def _pca99_once(x: np.ndarray) -> float:
    eigvals = _eigvals_from_x(x)
    if eigvals.size == 0:
        return float("nan")
    cumulative = np.cumsum(eigvals)
    threshold = cumulative[-1] * 0.99
    return float(np.searchsorted(cumulative, threshold) + 1)


def _gride_once(x: np.ndarray) -> float:
    if x.shape[0] < 4:
        return float("nan")
    try:
        from dadapy import Data
    except Exception:
        return float("nan")

    maxk = min(64, x.shape[0] - 1)
    try:
        data = Data(coordinates=_jitter_unique(x))
        data.compute_distances(maxk=maxk)
        ids, _, _ = data.return_id_scaling_gride(range_max=maxk)
        return float(ids[-1])
    except Exception:
        return float("nan")


def _jitter_unique(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    try:
        if np.unique(x, axis=0).shape[0] < x.shape[0]:
            x = x + np.random.normal(scale=eps, size=x.shape).astype(x.dtype)
    except Exception:
        pass
    return x


METRIC_FUNCTIONS: dict[str, Callable[[np.ndarray], float]] = {
    "iso": _fast_isoscore,
    "lpca99": _pca99_once,
    "pca99": _pca99_once,
    "gride": _gride_once,
}


def bootstrap_layer_loop(
    rep_sub: np.ndarray,
    sample_size: int,
    n_reps: int,
    compute_once: Callable[[np.ndarray], float],
    seed: int,
    desc: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_layers, n_tokens, _hidden_size = rep_sub.shape
    rng = np.random.default_rng(seed)
    values = np.full((n_reps, n_layers), np.nan, dtype=np.float32)
    rep_iter = tqdm(range(n_reps), desc=desc, leave=False, disable=desc is None)
    for rep_idx in rep_iter:
        idx = rng.integers(0, n_tokens, size=sample_size)
        for layer in range(n_layers):
            x = rep_sub[layer, idx].astype(np.float32, copy=False)
            try:
                values[rep_idx, layer] = float(compute_once(x))
            except Exception:
                values[rep_idx, layer] = np.nan
    mean = np.nanmean(values, axis=0).astype(np.float32)
    lo = np.nanpercentile(values, 2.5, axis=0).astype(np.float32)
    hi = np.nanpercentile(values, 97.5, axis=0).astype(np.float32)
    return mean, lo, hi


def make_dist_palette(classes: list[str]) -> dict[str, tuple[float, float, float]]:
    ordered = sorted(classes, key=lambda value: int(value))
    colors = sns.color_palette("vlag", n_colors=len(ordered))
    return dict(zip(ordered, colors))


def save_metric_csv(
    out_path: Path,
    pos_tag: str,
    metric: str,
    class_to_stats: dict[str, dict[str, np.ndarray | int]],
    layers: np.ndarray,
    model_name: str,
    word_rep_mode: str,
    source_csv: Path,
) -> None:
    rows = []
    for class_name, stats in class_to_stats.items():
        mean = stats["mean"]
        lo = stats["lo"]
        hi = stats["hi"]
        assert isinstance(mean, np.ndarray)
        assert isinstance(lo, np.ndarray)
        assert isinstance(hi, np.ndarray)
        for layer_idx, layer in enumerate(layers):
            rows.append(
                {
                    "subset": f"{pos_tag}_head_dist",
                    "model": model_name,
                    "word_rep_mode": word_rep_mode,
                    "feature": "head_dist",
                    "pos": pos_tag,
                    "class": class_name,
                    "metric": metric,
                    "layer": int(layer),
                    "mean": float(mean[layer_idx]) if np.isfinite(mean[layer_idx]) else np.nan,
                    "ci_low": float(lo[layer_idx]) if np.isfinite(lo[layer_idx]) else np.nan,
                    "ci_high": float(hi[layer_idx]) if np.isfinite(hi[layer_idx]) else np.nan,
                    "n_tokens": int(stats.get("n", 0)),
                    "source_csv": source_csv.name,
                }
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def plot_metric_with_ci(
    class_to_stats: dict[str, dict[str, np.ndarray | int]],
    layers: np.ndarray,
    metric: str,
    pos_tag: str,
    model_name: str,
    out_path: Path,
) -> None:
    classes = sorted(class_to_stats, key=lambda value: int(value))
    palette = make_dist_palette(classes)
    sns.set_style("darkgrid")
    sns.set_context("paper", font_scale=1.6)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for class_name in classes:
        stats = class_to_stats[class_name]
        mean = stats["mean"]
        lo = stats["lo"]
        hi = stats["hi"]
        assert isinstance(mean, np.ndarray)
        assert isinstance(lo, np.ndarray)
        assert isinstance(hi, np.ndarray)
        if np.all(np.isnan(mean)):
            continue
        color = palette[class_name]
        ax.plot(layers, mean, label=class_name, lw=1.9, color=color)
        if not np.all(np.isnan(lo)):
            ax.fill_between(layers, lo, hi, alpha=0.14, color=color)

    ax.set_xlabel("Layer")
    ax.set_ylabel(LABELS.get(metric, metric))
    ax.set_title(f"{pos_tag} head distance - {model_short_name(model_name).upper()}")
    ax.legend(ncol=4, fontsize="small", title="Head distance", frameon=False)
    ax.margins(x=0.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=240)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def write_main_plot_alias(source_base: Path, args: argparse.Namespace, pos_tag: str, metric: str) -> None:
    if args.no_main_plots or metric not in MAIN_METRIC_NAMES:
        return
    prefix = main_model_prefix(args.model)
    metric_name = MAIN_METRIC_NAMES[metric]
    out_base = args.main_plots_dir / f"{prefix}_head_{metric_name}_{pos_tag}"
    out_base.parent.mkdir(parents=True, exist_ok=True)

    # Re-save from the generated figure file through matplotlib's PDF output path by
    # copying bytes is less fragile for PDFs/PNGs and keeps aliases exact.
    for suffix in (".pdf", ".png"):
        src = source_base.with_suffix(suffix)
        dst = out_base.with_suffix(suffix)
        if src.exists():
            dst.write_bytes(src.read_bytes())


def write_summary_counts(args: argparse.Namespace) -> Path:
    rows = []
    for pos_tag in tqdm(args.pos, desc="Counting POS/head-distance tokens"):
        _sent_df, token_df = load_pos_head_dist_from_column(
            args.csv_path,
            pos_tag.upper(),
            args.head_dist_col,
            args.pos_col,
            args.head_dist_clamp,
            args.include_zero_class,
            not args.include_first_token,
        )
        counts = token_df["head_dist_class"].value_counts()
        for class_name, n_tokens in counts.items():
            rows.append({"pos": pos_tag.upper(), "head_dist_class": class_name, "n_tokens": int(n_tokens)})
    out = args.tables_dir / "pos_head_dist_token_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result["head_dist_class"] = pd.to_numeric(result["head_dist_class"], errors="coerce")
    result = result.sort_values(["pos", "head_dist_class"])
    result.to_csv(out, index=False)
    return out


def run_for_pos(args: argparse.Namespace, pos_tag: str) -> None:
    pos_tag = pos_tag.upper()
    model_short = model_short_name(args.model)
    output_group = "all"
    plot_dir = args.plots_dir / model_short / args.word_rep_mode / output_group / pos_tag.lower()
    table_dir = args.tables_dir / model_short / args.word_rep_mode / output_group / pos_tag.lower()

    sent_df, token_df = load_pos_head_dist_from_column(
        args.csv_path,
        pos_tag,
        args.head_dist_col,
        args.pos_col,
        args.head_dist_clamp,
        args.include_zero_class,
        not args.include_first_token,
    )
    classes = sorted(token_df["head_dist_class"].unique(), key=lambda value: int(value))
    raw_df = sample_raw(token_df, args.raw_max_per_class, args.seed)
    counts = raw_df["head_dist_class"].value_counts().reindex(classes)
    print(f"\nPOS={pos_tag}: {len(raw_df):,} tokens across head distances {classes}")
    print(counts.astype(int).to_dict())

    reps, filled = embed_subset(sent_df, raw_df, args.model, args.word_rep_mode, args.batch_size)
    reps = reps[:, filled]
    raw_df = raw_df.reset_index(drop=True).loc[filled].reset_index(drop=True)
    class_array = raw_df["head_dist_class"].values
    layers = np.arange(reps.shape[0])
    print(f"Embedded POS={pos_tag}: {len(raw_df):,} tokens, layers={len(layers)}")

    for metric in tqdm(args.metrics, desc=f"{pos_tag} metrics", leave=False):
        compute_once = METRIC_FUNCTIONS[metric]
        if metric == "gride":
            n_bootstrap = args.n_bootstrap_heavy
            max_sample = args.heavy_bs_max_samp_per_class
        else:
            n_bootstrap = args.n_bootstrap_fast
            max_sample = args.fast_bs_max_samp_per_class

        print(f"Computing {metric} for POS={pos_tag}")
        class_results: dict[str, dict[str, np.ndarray | int]] = {}
        class_iter = tqdm(classes, desc=f"{pos_tag} {metric} head distances", leave=False)
        for class_name in class_iter:
            idx = np.where(class_array == class_name)[0]
            if idx.size < 3:
                continue
            sample_size = min(max_sample, idx.size)
            class_iter.set_postfix({"hd": class_name, "n": idx.size})
            mean, lo, hi = bootstrap_layer_loop(
                reps[:, idx],
                sample_size,
                n_bootstrap,
                compute_once,
                args.seed,
                desc=f"{pos_tag} {metric} hd={class_name}",
            )
            class_results[class_name] = {"mean": mean, "lo": lo, "hi": hi, "n": int(idx.size)}

        csv_path = table_dir / f"headdist_{pos_tag}_{metric}_{_sanitize(args.model)}_{args.word_rep_mode}.csv"
        save_metric_csv(csv_path, pos_tag, metric, class_results, layers, args.model, args.word_rep_mode, args.csv_path)

        out_base = plot_dir / f"{metric}_{pos_tag}"
        plot_metric_with_ci(class_results, layers, metric, pos_tag, args.model, out_base)
        write_main_plot_alias(out_base, args, pos_tag, metric)
        print(f"Saved {csv_path}")
        print(f"Saved {out_base.with_suffix('.pdf')}")

        del class_results
        gc.collect()

    del reps
    gc.collect()


def main() -> None:
    args = parse_args()
    args.pos = [pos.upper() for pos in args.pos]
    set_reproducibility(args.seed)

    if args.summary_only:
        out = write_summary_counts(args)
        print(f"Wrote {out}")
        return

    for pos_tag in tqdm(args.pos, desc="POS experiments"):
        run_for_pos(args, pos_tag)


if __name__ == "__main__":
    main()
