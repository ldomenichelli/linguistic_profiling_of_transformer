#!/usr/bin/env python3
"""Build the project website from committed results and sentence annotations."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GROUPS = ROOT / "plots_main/plots_features/feature_contrasts/tables"
REPO = "https://github.com/ldomenichelli/linguistic_profiling_of_transformer"
COLUMNS = ["tokens", "index", "length", "pos", "head_dist", "relation_type", "ariety"]
COMPARISONS = {
    "pos_open_closed": ("Part of speech", "Open and closed word classes"),
    "length_long_short": ("Word length", "Long and short words"),
    "index_late_early": ("Sentence position", "Late and early positions"),
    "arity_high_low": ("Dependents: high vs low", "Words with many or few dependents"),
    "arity_mid_low": ("Dependents: mid vs low", "Words with mid or low numbers of dependents"),
    "head_dist_head_after_before": ("Head direction", "Words before and after their dependency head"),
    "relation_type_content_function": ("Dependency relation", "Content and function relations"),
    "verbal_valency_mid_low": ("Verbal valency", "Verbs with mid or low numbers of dependents"),
}
METRICS = {"isoscore": "iso", "LID": "lid", "NLID": "nlid"}
CORPORA = {"en": ("English · EWT", "en_ewt", 10067, 194916), "it": ("Italian · ISDT", "it_isdt", 12112, 253771)}
ATTRIBUTION = {
    "en": {"upstream": "https://github.com/UniversalDependencies/UD_English-EWT", "terms": "https://github.com/UniversalDependencies/UD_English-EWT/blob/master/README.md#licensecopyright", "credit": "UD English EWT annotations © 2013–2021 Stanford University; annotations CC BY-SA 4.0. Underlying texts retain their original source terms."},
    "it": {"upstream": "https://github.com/UniversalDependencies/UD_Italian-ISDT", "terms": "https://github.com/UniversalDependencies/UD_Italian-ISDT/blob/master/LICENSE.txt", "credit": "UD Italian ISDT: Cristina Bosco, Simonetta Montemagni, Maria Simi; CC BY-NC-SA 3.0. Original resource: http://medialab.di.unipi.it/wiki/ISDT/"},
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def write_js(path, name, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("</", "<\\/")
    path.write_text(f"window.{name}={encoded};\n")


def number(value):
    if value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite source value: {value}")
    return result


def build_results():
    series, excluded, sources, exports = [], [], {}, []
    for path in sorted(GROUPS.glob("*_groups.csv")):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        first = rows[0]
        assert all(all(r[k] == first[k] for k in ["feature", "contrast", "model", "metric", "source_path"]) for r in rows)
        if first["metric"].startswith("legacy_pdf"):
            excluded.append({"path": str(path.relative_to(ROOT)), "reason": "digitized_from_pdf"})
            continue
        comparison = first["feature"] + "_" + first["contrast"]
        metric = METRICS[first["metric_label"]]
        model = "bert" if first["model_short"] == "bert" else "gpt2"
        assert comparison in COMPARISONS
        groups = {"low": "negative", "mid": "positive", "negative": "negative", "positive": "positive"}
        assert len(rows) == 26 and {(groups[r["group_key"]], int(r["layer"])) for r in rows} == {(g, n) for g in ["positive", "negative"] for n in range(13)}
        points = []
        for row in rows:
            point = {"group": groups[row["group_key"]], "sourceGroup": row["group_key"], "label": row["class_group"], "layer": int(row["layer"]), "mean": number(row["mean"]), "low": number(row["ci_low"]), "high": number(row["ci_high"]), "n": int(row["n_tokens"]), "intervalMethod": row["ci_method"]}
            assert point["mean"] is not None
            assert (point["low"] is None) == (point["high"] is None)
            if point["low"] is not None:
                assert point["low"] <= point["high"]
            points.append(point)
            exports.append({"comparison": comparison, "model": first["model"], "measure": metric, "estimator": first["metric"], "group": row["class_group"], "layer": row["layer"], "mean": row["mean"], "ci_low": row["ci_low"], "ci_high": row["ci_high"], "n_tokens": row["n_tokens"], "interval_method": row["ci_method"], "aggregation": row["aggregation"], "source": str(path.relative_to(ROOT))})
        target = DOCS / "data/results" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        original = first["source_path"]
        methods = sorted({row["ci_method"] for row in rows})
        entry = {"id": f"{comparison}.{model}.{metric}", "comparison": comparison, "model": model, "metric": metric, "estimator": first["metric"], "points": points, "source": str(path.relative_to(ROOT)), "csv": str(target.relative_to(DOCS)), "sourcePath": original, "excludesFirst": "no_index" in original or "no_idx0" in original, "intervalMethods": methods, "aggregation": first["aggregation"]}
        series.append(entry)
        sources[str(path.relative_to(ROOT))] = sha(path)
    assert len(series) == 43 and len({s["id"] for s in series}) == 43
    assert len(excluded) == 5
    target = DOCS / "data/layer-comparisons.csv"
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exports[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(exports)
    return series, excluded, sources


def build_corpora():
    metadata = []
    for key, (label, stem, expected_sentences, expected_tokens) in CORPORA.items():
        source = ROOT / "code/data" / (stem + "-ud-train_sentences.csv")
        records, tokens, pos = [], 0, set()
        with source.open(newline="") as stream:
            for row in csv.DictReader(stream):
                values = [ast.literal_eval(row[column]) for column in COLUMNS]
                assert all(isinstance(v, list) and len(v) == len(values[0]) for v in values)
                assert all(isinstance(i, int) and i >= 1 for i in values[1]), row["sentence_id"]
                assert all(isinstance(w, str) for w in values[0])
                records.append([row["sentence_id"], *values])
                tokens += len(values[0])
                pos.update(values[3])
        assert len(records) == expected_sentences and tokens == expected_tokens
        assert len({r[0] for r in records}) == len(records)
        chunk = {"id": key, "columns": ["sentence_id", *COLUMNS], "attribution": ATTRIBUTION[key], "records": records}
        target = DOCS / "assets/corpora" / (key + ".js")
        write_js(target, "CORPUS_" + key.upper(), chunk)
        metadata.append({"id": key, "label": label, "sentences": len(records), "tokens": tokens, "pos": sorted(pos), "path": str(target.relative_to(DOCS)), "source": str(source.relative_to(ROOT)), "sourceSha256": sha(source), "sha256": sha(target), "attribution": ATTRIBUTION[key]})
    return metadata


def build_figures():
    figures = []
    previous_path = DOCS / "assets/site-data.json"
    previous = {f["id"]: f for f in json.loads(previous_path.read_text()).get("figures", [])} if previous_path.exists() else {}
    names = {"pos": "Part of speech", "length": "Word length", "index": "Sentence position", "arity": "Dependents", "head": "Head direction", "relation": "Dependency relation"}
    choices = [(f"all_{suffix}", f"All tokens · {label}", "overview", "both", metric, ROOT / "plots_main/plots_all" / f"all_{suffix}.pdf") for suffix, label, metric in [("iso", "IsoScore", "iso"), ("LID", "Linear ID", "lid"), ("NLID", "Nonlinear ID", "nlid")]]
    for model in ["bert", "gpt"]:
        for feature, label in names.items():
            for metric, suffix in [("iso", "isoscore"), ("lid", "LID"), ("nlid", "NLID")]:
                name = f"{model}_{feature}_{suffix}"
                choices.append((name, label + " · " + ("BERT" if model == "bert" else "GPT-2") + " · " + {"iso": "IsoScore", "lid": "Linear ID", "nlid": "Nonlinear ID"}[metric], feature, "bert" if model == "bert" else "gpt2", metric, ROOT / "plots_main/plots_features" / (name + ".pdf")))
        for metric, suffix in [("iso", "iso"), ("lid", "LID"), ("nlid", "NLID")]:
            name = f"{model}_arity_{suffix}_VERB"
            choices.append((name, "Verbal valency · " + ("BERT" if model == "bert" else "GPT-2") + " · " + {"iso": "IsoScore", "lid": "Linear ID", "nlid": "Nonlinear ID"}[metric], "verbal_valency", "bert" if model == "bert" else "gpt2", metric, ROOT / "plots_main/plots_features" / (name + ".pdf")))
    for ident, title, feature, model, metric, source in choices:
        assert source.is_file(), source
        pdf = DOCS / "figures" / (ident + ".pdf")
        png = pdf.with_suffix(".png")
        pdf.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        reuse = png.exists() and previous.get(ident, {}).get("sourceSha256") == digest and previous.get(ident, {}).get("pngSha256") == sha(png)
        shutil.copy2(source, pdf)
        if not reuse:
            with tempfile.TemporaryDirectory(prefix="transformer-figure-") as tmp:
                output = Path(tmp) / "preview"
                subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-scale-to", "1400", "-png", str(source), str(output)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                shutil.copy2(output.with_suffix(".png"), png)
        figures.append({"id": ident, "title": title, "feature": feature, "model": model, "metric": metric, "pdf": str(pdf.relative_to(DOCS)), "png": str(png.relative_to(DOCS)), "source": str(source.relative_to(ROOT)), "sourceSha256": digest, "pngSha256": sha(png)})
    return figures


def main():
    series, excluded, sources = build_results()
    corpora = build_corpora()
    figures = build_figures()
    payload = {"title": "Linguistic Profiling of Transformer Embedding Geometry", "repository": REPO, "comparisons": [{"id": key, "label": value[0], "description": value[1]} for key, value in COMPARISONS.items()], "series": series, "corpora": corpora, "figures": figures, "sources": sources, "excluded": excluded}
    write_json(DOCS / "assets/site-data.json", payload)
    write_js(DOCS / "assets/site-data.js", "SITE_DATA", payload)
    (DOCS / ".nojekyll").touch()
    print(f"Built {len(series)} result series, {len(figures)} figures, and {sum(c['sentences'] for c in corpora):,} annotated sentences.")


if __name__ == "__main__":
    main()
