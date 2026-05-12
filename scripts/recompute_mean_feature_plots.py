#!/usr/bin/env python3
"""Recompute feature metric plots with mean subtoken representations.

This is a lean, parameterized replacement for rerunning whole notebooks when
only the `WORD_REP_MODE="mean"` feature plots are needed. It intentionally
does one feature/model job at a time so a nohup queue can keep RAM bounded.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import gc
import inspect
import math
import os
import random
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    from dadapy import Data

    HAS_DADAPY = True
except Exception:  # pragma: no cover - optional dependency
    HAS_DADAPY = False

try:
    from skdim.id import CorrInt, FisherS, KNN, MADA, MLE, MOM, TLE, lPCA

    HAS_SKDIM = True
except Exception:  # pragma: no cover - optional dependency
    HAS_SKDIM = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "code" / "data" / "en_ewt-ud-train_sentences.csv"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "plots_extra" / "metrics" / "features"
EPS = 1e-12

MODEL_IDS = {
    "bert": "bert-base-uncased",
    "gpt2": "gpt2",
}

FEATURE_DIR = {
    "pos": "pos",
    "length": "length",
    "index": "index",
    "arity": "arity",
    "head_dist": "head_dist",
    "relation_type": "relations",
}

FEATURE_SUFFIX = {
    "pos": "pos",
    "length": "length",
    "index": "index",
    "arity": "arity",
    "head_dist": "head_dist",
    "relation_type": "relation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature",
        required=True,
        choices=sorted(FEATURE_DIR),
        help="Linguistic feature to recompute.",
    )
    parser.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["iso"],
        help="Metrics to compute, e.g. iso lpca99 gride corrint fishers mom rand spect tle twonn.",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--word-rep-mode", default="mean", choices=["first", "last", "mean"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=30000,
        help="Cap tokens per feature class before embedding. Use 0 for no cap.",
    )
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
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads. By default, use local cache only.",
    )
    return parser.parse_args()


def to_list(value: object) -> list:
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def normalize_class(value: object) -> str:
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def numeric_sort_key(value: object) -> tuple[int, float | str]:
    text = normalize_class(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def build_token_frame(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    df_all = pd.read_csv(args.csv)
    df_all["sentence_id"] = df_all["sentence_id"].astype(str)
    df_all["tokens"] = df_all["tokens"].apply(to_list)
    df_sent = df_all[["sentence_id", "tokens"]].drop_duplicates("sentence_id").copy()

    feature_col = args.feature
    class_col = args.feature
    if args.feature == "arity":
        feature_col = "ariety" if "ariety" in df_all.columns else "arity"
    if args.feature == "relation_type":
        feature_col = "relation_type"
    if feature_col not in df_all.columns:
        raise ValueError(f"Missing feature column {feature_col!r} in {args.csv}")
    df_all[feature_col] = df_all[feature_col].apply(to_list)

    top_relations: set[str] | None = None
    if args.feature == "relation_type":
        counts: dict[str, int] = {}
        for vals in df_all[feature_col]:
            for rel in vals:
                rel = str(rel)
                counts[rel] = counts.get(rel, 0) + 1
        top_relations = {
            rel
            for rel, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
                : args.relation_top_k
            ]
        }

    rows = []
    for sid, tokens, vals in df_all[["sentence_id", "tokens", feature_col]].itertuples(index=False):
        limit = min(len(tokens), len(vals))
        for word_id in range(limit):
            raw = vals[word_id]
            klass: str | None
            if args.feature == "length":
                try:
                    number = int(raw)
                except Exception:
                    continue
                if number <= 0:
                    continue
                klass = str(min(number, args.length_max_class))
                class_col = "length"
            elif args.feature == "index":
                try:
                    number = int(raw)
                except Exception:
                    continue
                if number <= 0:
                    continue
                klass = str(min(number, args.index_max_class))
                class_col = "index"
            elif args.feature == "arity":
                try:
                    number = int(raw)
                except Exception:
                    continue
                if number < 0:
                    continue
                klass = str(min(number, args.arity_max_class))
                class_col = "arity"
            elif args.feature == "head_dist":
                try:
                    number = int(raw)
                except Exception:
                    continue
                if number == 0:
                    continue
                number = max(-args.head_dist_max_abs, min(args.head_dist_max_abs, number))
                klass = str(number)
                class_col = "head_dist"
            elif args.feature == "relation_type":
                rel = str(raw)
                if top_relations is not None and rel not in top_relations:
                    continue
                klass = rel
                class_col = "relation_type"
            elif args.feature == "pos":
                klass = str(raw)
                class_col = "pos"
            else:  # pragma: no cover - argparse choices cover this
                raise ValueError(args.feature)

            rows.append((str(sid), int(word_id), klass, str(tokens[word_id])))

    df_tok = pd.DataFrame(rows, columns=["sentence_id", "word_id", "class", "word"])
    if df_tok.empty:
        raise ValueError(f"No token rows built for feature {args.feature}")
    return df_sent, df_tok, class_col


def sample_per_class(df_tok: pd.DataFrame, max_per_class: int, seed: int) -> pd.DataFrame:
    if max_per_class <= 0:
        return df_tok.reset_index(drop=True)
    parts = []
    for _, group in df_tok.groupby("class", sort=False):
        n = min(len(group), max_per_class)
        parts.append(group.sample(n=n, replace=False, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def load_tokenizer_and_model(model_key: str, allow_download: bool, device: str):
    model_id = MODEL_IDS[model_key]
    local_files_only = not allow_download

    def load(mid: str):
        tok_kwargs = {"use_fast": True, "local_files_only": local_files_only}
        if model_key == "gpt2":
            tok_kwargs["add_prefix_space"] = True
        tokenizer = AutoTokenizer.from_pretrained(mid, **tok_kwargs)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            mid,
            output_hidden_states=True,
            local_files_only=local_files_only,
        )
        if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return tokenizer, model, mid

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
    raise RuntimeError("Could not load model/tokenizer:\n" + "\n".join(errors))


def num_hidden_layers(model) -> int:
    value = getattr(model.config, "num_hidden_layers", None)
    if value is None:
        value = getattr(model.config, "n_layer", None)
    if value is None:
        raise ValueError("Cannot determine number of hidden layers")
    return int(value)


def hidden_size(model) -> int:
    value = getattr(model.config, "hidden_size", None)
    if value is None:
        value = getattr(model.config, "n_embd", None)
    if value is None:
        raise ValueError("Cannot determine hidden size")
    return int(value)


def embed_subset(
    df_sent: pd.DataFrame,
    df_tok: pd.DataFrame,
    args: argparse.Namespace,
    device: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    df_sent = df_sent.copy()
    df_tok = df_tok.copy()
    by_sid: dict[str, list[tuple[int, int]]] = {}
    for global_idx, (sid, word_id) in enumerate(df_tok[["sentence_id", "word_id"]].itertuples(index=False)):
        by_sid.setdefault(str(sid), []).append((global_idx, int(word_id)))

    sids = list(by_sid)
    df_selected = (
        df_sent[df_sent["sentence_id"].isin(sids)]
        .drop_duplicates("sentence_id")
        .set_index("sentence_id")
        .loc[sids]
    )

    tokenizer, model, resolved_id = load_tokenizer_and_model(args.model, args.allow_download, device)
    layers = num_hidden_layers(model) + 1
    dim = hidden_size(model)
    reps = np.zeros((layers, len(df_tok), dim), dtype=np.float16)
    filled = np.zeros(len(df_tok), dtype=bool)

    enc_kwargs = {
        "is_split_into_words": True,
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
    }
    max_len = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_len, int) and max_len < 1_000_000:
        enc_kwargs["max_length"] = max_len
    if "add_prefix_space" in inspect.signature(tokenizer.__call__).parameters and args.model == "gpt2":
        enc_kwargs["add_prefix_space"] = True

    autocast_ctx = torch.cuda.amp.autocast if device == "cuda" else contextlib.nullcontext
    with torch.no_grad(), autocast_ctx():
        for start in tqdm(range(0, len(sids), args.batch_size), desc=f"{resolved_id} mean embeddings"):
            batch_ids = sids[start : start + args.batch_size]
            batch_tokens = df_selected.loc[batch_ids, "tokens"].tolist()
            encoded = tokenizer(batch_tokens, **enc_kwargs)
            encoded_device = {key: value.to(device) for key, value in encoded.items()}
            out = model(**encoded_device)
            hidden = torch.stack(out.hidden_states).detach().cpu().numpy()

            for batch_idx, sid in enumerate(batch_ids):
                wordpiece_map: dict[int, list[int]] = {}
                for token_idx, word_id in enumerate(encoded.word_ids(batch_idx)):
                    if word_id is not None:
                        wordpiece_map.setdefault(int(word_id), []).append(int(token_idx))

                for global_idx, word_id in by_sid[sid]:
                    token_indices = wordpiece_map.get(word_id)
                    if not token_indices:
                        continue
                    if args.word_rep_mode == "first":
                        vector = hidden[:, batch_idx, token_indices[0], :]
                    elif args.word_rep_mode == "last":
                        vector = hidden[:, batch_idx, token_indices[-1], :]
                    else:
                        vector = hidden[:, batch_idx, token_indices, :].mean(axis=1)
                    reps[:, global_idx, :] = vector.astype(np.float16, copy=False)
                    filled[global_idx] = True

            del encoded, encoded_device, out, hidden
            if device == "cuda":
                torch.cuda.empty_cache()

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return reps, filled, resolved_id


def centered_eigenvalues(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 2:
        return np.array([], dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    try:
        _, singular_values, _ = np.linalg.svd(x, full_matrices=False)
    except Exception:
        return np.array([], dtype=np.float64)
    eigenvalues = np.sort((singular_values.astype(np.float64) ** 2))[::-1]
    return eigenvalues[np.isfinite(eigenvalues) & (eigenvalues >= 0)]


def jitter_unique(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    try:
        if np.unique(x, axis=0).shape[0] < x.shape[0]:
            rng = np.random.default_rng(123)
            x = x + rng.normal(scale=eps, size=x.shape).astype(x.dtype)
    except Exception:
        pass
    return x


def metric_iso(x: np.ndarray) -> float:
    eigenvalues = centered_eigenvalues(x)
    n = eigenvalues.size
    if n < 2:
        return float("nan")
    norm = float(np.linalg.norm(eigenvalues))
    if norm <= 0:
        return 0.0
    root_n = math.sqrt(n)
    denom = math.sqrt(2.0 * (n - root_n))
    delta = float(np.linalg.norm((root_n * eigenvalues) / norm - 1.0) / denom)
    delta = float(np.clip(delta, 0.0, 1.0))
    phi = (n - (delta**2) * (n - root_n)) ** 2 / (n**2)
    return float(np.clip((n * phi - 1.0) / (n - 1.0), 0.0, 1.0))


def metric_spect(x: np.ndarray) -> float:
    eigenvalues = centered_eigenvalues(x)
    if eigenvalues.size == 0:
        return float("nan")
    return float(eigenvalues.max() / (eigenvalues.mean() + EPS))


def metric_rand(x: np.ndarray, pairs: int = 2000) -> float:
    n = x.shape[0]
    if n < 2:
        return float("nan")
    rng = np.random.default_rng()
    pairs = min(pairs, max(1, n * (n - 1) // 2))
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    same = i == j
    if same.any():
        j[same] = rng.integers(0, n, size=int(same.sum()))
    a = x[i]
    b = x[j]
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + EPS
    return float(np.mean(np.abs(num / den)))


def metric_twonn(x: np.ndarray) -> float:
    if not HAS_DADAPY:
        return float("nan")
    data = Data(coordinates=jitter_unique(x))
    value, _, _ = data.compute_id_2NN()
    return float(value)


def metric_gride(x: np.ndarray) -> float:
    if not HAS_DADAPY:
        return float("nan")
    data = Data(coordinates=jitter_unique(x))
    range_max = min(64, max(2, x.shape[0] - 1))
    data.compute_distances(maxk=range_max)
    values, _, _ = data.return_id_scaling_gride(range_max=range_max)
    return float(values[-1])


def skdim_metric(name: str) -> Callable[[np.ndarray], float] | None:
    if not HAS_SKDIM:
        return None
    builders = {
        "mom": MOM,
        "tle": TLE,
        "corrint": CorrInt,
        "fishers": FisherS,
        "mle": MLE,
        "mada": MADA,
        "knn": KNN,
    }

    def build_lpca(metric_name: str):
        if metric_name == "lpca":
            return lPCA(ver="FO")
        if metric_name == "lpca99":
            return lPCA(ver="ratio", alphaRatio=0.99)
        return None

    def compute(x: np.ndarray) -> float:
        estimator = build_lpca(name) if name.startswith("lpca") else builders[name]()
        estimator.fit(jitter_unique(x))
        return float(getattr(estimator, "dimension_", np.nan))

    if name in {"lpca", "lpca99"} or name in builders:
        return compute
    return None


def metric_registry() -> tuple[dict[str, Callable[[np.ndarray], float]], dict[str, Callable[[np.ndarray], float]]]:
    fast = {
        "iso": metric_iso,
        "spect": metric_spect,
        "spectral": metric_spect,
        "rand": metric_rand,
        "randcos": metric_rand,
    }
    heavy: dict[str, Callable[[np.ndarray], float]] = {
        "twonn": metric_twonn,
        "gride": metric_gride,
    }
    for name in ["lpca", "lpca99", "mom", "tle", "corrint", "fishers", "mle", "mada", "knn"]:
        fn = skdim_metric(name)
        if fn is not None:
            heavy[name] = fn
    return fast, heavy


LABELS = {
    "iso": "IsoScore",
    "spect": "Spectral Ratio",
    "spectral": "Spectral Ratio",
    "rand": "RandCos",
    "randcos": "RandCos",
    "twonn": "TwoNN ID",
    "gride": "GRIDE",
    "lpca": "lPCA",
    "lpca99": "lPCA 0.99",
    "mom": "MOM",
    "tle": "TLE",
    "corrint": "CorrInt",
    "fishers": "FisherS",
    "mle": "MLE",
    "mada": "MADA",
    "knn": "KNN",
}


def bootstrap_layers(
    reps: np.ndarray,
    indices: np.ndarray,
    metric_name: str,
    compute: Callable[[np.ndarray], float],
    is_fast: bool,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    n_bootstrap = args.fast_bootstrap if is_fast else args.heavy_bootstrap
    sample_cap = args.fast_sample_cap if is_fast else args.heavy_sample_cap
    layers = reps.shape[0]
    n = len(indices)
    sample_size = min(n, sample_cap if sample_cap > 0 else n)
    out = np.full((n_bootstrap, layers), np.nan, dtype=np.float32)
    for boot in range(n_bootstrap):
        sample = rng.choice(indices, size=sample_size, replace=True)
        for layer in range(layers):
            x = np.asarray(reps[layer, sample], dtype=np.float32)
            try:
                out[boot, layer] = float(compute(x))
            except Exception as exc:
                print(f"warning: {metric_name} failed at bootstrap={boot} layer={layer}: {exc}", flush=True)
                out[boot, layer] = np.nan
            del x
    mean = np.nanmean(out, axis=0)
    low = np.nanpercentile(out, 2.5, axis=0)
    high = np.nanpercentile(out, 97.5, axis=0)
    return mean.astype(np.float32), low.astype(np.float32), high.astype(np.float32)


def plot_metric(
    stats: dict[str, dict[str, np.ndarray | int]],
    layers: np.ndarray,
    metric: str,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    classes = sorted(stats, key=numeric_sort_key)
    palette = sns.color_palette("tab20", n_colors=max(1, len(classes)))
    for idx, klass in enumerate(classes):
        mean = stats[klass]["mean"]
        low = stats[klass]["ci_low"]
        high = stats[klass]["ci_high"]
        color = palette[idx]
        ax.plot(layers, mean, label=klass, lw=1.8, color=color)
        ax.fill_between(layers, low, high, alpha=0.15, color=color)
    label = LABELS.get(metric, metric.upper())
    ax.set_xlabel("Layer")
    ax.set_ylabel(label)
    ax.set_title(f"{label} - {MODEL_IDS[args.model]} - {args.feature} - mean subtokens")
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
    for klass in sorted(stats, key=numeric_sort_key):
        item = stats[klass]
        mean = item["mean"]
        low = item["ci_low"]
        high = item["ci_high"]
        for layer, value in enumerate(mean):
            rows.append(
                {
                    "subset": "raw",
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
                    "max_per_class": args.max_per_class,
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    fast_metrics, heavy_metrics = metric_registry()
    metrics = [name.lower() for name in args.metrics]
    unknown = [name for name in metrics if name not in fast_metrics and name not in heavy_metrics]
    if unknown:
        raise ValueError(f"Unknown or unavailable metrics: {unknown}")

    print(f"feature={args.feature} model={args.model} metrics={metrics}", flush=True)
    print(f"device={device} word_rep_mode={args.word_rep_mode}", flush=True)
    print(
        "caps: "
        f"max_per_class={args.max_per_class} fast_bs={args.fast_bootstrap} "
        f"heavy_bs={args.heavy_bootstrap} fast_sample_cap={args.fast_sample_cap} "
        f"heavy_sample_cap={args.heavy_sample_cap}",
        flush=True,
    )

    df_sent, df_tok, class_col = build_token_frame(args)
    print(f"token rows before cap: {len(df_tok):,}", flush=True)
    df_tok = sample_per_class(df_tok, args.max_per_class, args.seed)
    counts = df_tok["class"].value_counts().sort_index(key=lambda s: s.map(str))
    print(f"token rows after cap: {len(df_tok):,}", flush=True)
    print(f"class counts: {counts.to_dict()}", flush=True)

    reps, filled, resolved_model = embed_subset(df_sent, df_tok, args, device)
    if not filled.all():
        missing = int((~filled).sum())
        print(f"warning: missing vectors for {missing:,} of {len(filled):,} rows", flush=True)
        reps = reps[:, filled, :]
        df_tok = df_tok.reset_index(drop=True).loc[filled].reset_index(drop=True)

    class_values = df_tok["class"].astype(str).to_numpy()
    layers = np.arange(reps.shape[0])
    out_dir = args.out_root / FEATURE_DIR[args.feature] / args.word_rep_mode / args.model
    table_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        is_fast = metric in fast_metrics
        compute = fast_metrics[metric] if is_fast else heavy_metrics[metric]
        print(f"computing {metric} ({'fast' if is_fast else 'heavy'})", flush=True)
        stats: dict[str, dict[str, np.ndarray | int]] = {}
        for klass in sorted(pd.unique(class_values), key=numeric_sort_key):
            indices = np.where(class_values == klass)[0]
            if len(indices) < 3:
                print(f"skip class={klass}: only {len(indices)} rows", flush=True)
                continue
            mean, low, high = bootstrap_layers(reps, indices, metric, compute, is_fast, args)
            stats[klass] = {
                "mean": mean,
                "ci_low": low,
                "ci_high": high,
                "n_tokens": int(len(indices)),
            }
            gc.collect()

        suffix = FEATURE_SUFFIX[args.feature]
        filename = f"{metric}_{args.model}_{suffix}.png"
        csv_name = f"{metric}_{args.model}_{suffix}.csv"
        write_metric_csv(stats, metric, class_col, resolved_model, args, table_dir / csv_name)
        plot_metric(stats, layers, metric, args, out_dir / filename)
        print(f"saved {out_dir / filename}", flush=True)
        print(f"saved {table_dir / csv_name}", flush=True)
        del stats
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    del reps
    gc.collect()
    print("done", flush=True)


if __name__ == "__main__":
    main()
