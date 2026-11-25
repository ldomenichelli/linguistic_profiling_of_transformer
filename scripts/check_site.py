#!/usr/bin/env python3
"""Check the website's numeric exports, sentence annotations, and local assets."""
import ast
import csv
import hashlib
import json
import math
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_js(path, name):
    text = path.read_text()
    prefix = f"window.{name}="
    assert text.startswith(prefix) and text.endswith(";\n")
    return json.loads(text[len(prefix):-2])


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths, self.ids = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        self.paths.extend(attrs[key] for key in ["src", "href"] if key in attrs)


def main():
    payload = json.loads((DOCS / "assets/site-data.json").read_text())
    assert payload == read_js(DOCS / "assets/site-data.js", "SITE_DATA")
    assert len(payload["series"]) == 43 and len(payload["excluded"]) == 5
    exported = list(csv.DictReader((DOCS / "data/layer-comparisons.csv").open()))
    assert len(exported) == 43 * 26
    for series in payload["series"]:
        source = ROOT / series["source"]
        assert digest(source) == payload["sources"][series["source"]]
        assert source.read_bytes() == (DOCS / series["csv"]).read_bytes()
        original = list(csv.DictReader(source.open()))
        assert len(series["points"]) == len(original)
        for point, row in zip(series["points"], original):
            assert point["label"] == row["class_group"] and point["layer"] == int(row["layer"])
            for key, column in [("mean", "mean"), ("low", "ci_low"), ("high", "ci_high")]:
                assert point[key] == (float(row[column]) if row[column] else None)
            assert point["n"] == int(row["n_tokens"])
            assert point["intervalMethod"] == row["ci_method"] and point["sourceGroup"] == row["group_key"]
        matching = [r for r in exported if r["source"] == series["source"]]
        assert len(matching) == 26
        for out, row in zip(matching, original):
            assert all(out[k] == row[k] for k in ["layer", "mean", "ci_low", "ci_high", "n_tokens"])
        assert not series["estimator"].startswith("legacy_pdf")
    token_count = 0
    for corpus in payload["corpora"]:
        source = ROOT / corpus["source"]
        chunk = DOCS / corpus["path"]
        assert digest(source) == corpus["sourceSha256"] and digest(chunk) == corpus["sha256"]
        records = read_js(chunk, "CORPUS_" + corpus["id"].upper())["records"]
        rows = list(csv.DictReader(source.open()))
        assert len(records) == len(rows) == corpus["sentences"]
        for record, row in zip(records, rows):
            assert record[0] == row["sentence_id"]
            for value, key in zip(record[1:], ["tokens", "index", "length", "pos", "head_dist", "relation_type", "ariety"]):
                assert value == ast.literal_eval(row[key])
            token_count += len(record[1])
        assert sum(len(r[1]) for r in records) == corpus["tokens"]
    for figure in payload["figures"]:
        pdf, png = DOCS / figure["pdf"], DOCS / figure["png"]
        assert digest(pdf) == digest(ROOT / figure["source"]) == figure["sourceSha256"]
        assert digest(png) == figure["pngSha256"]
        assert pdf.read_bytes().startswith(b"%PDF-") and png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    pca = json.loads((DOCS / "assets/pca/manifest.json").read_text())
    assert {v["id"] for v in pca["views"]} == {"bert", "gpt2", "bert-pos"}
    for view in pca["views"]:
        assert len(view["layers"]) == 13
        assert view["points"] == sum(len(g["indices"]) for g in view["groups"])
        for group in view["groups"]:
            assert group["indices"] == sorted(set(group["indices"]))
            assert len(group["indices"]) <= view["capPerTrace"] and max(group["indices"]) < group["originalPoints"]
        for layer, entry in enumerate(view["layers"]):
            path = DOCS / entry["path"]
            assert digest(path) == entry["sha256"]
            data = json.loads(path.read_text())
            assert data["view"] == view["id"] and data["layer"] == layer
            assert len(data["traces"]) == len(view["groups"])
            for trace, group in zip(data["traces"], view["groups"]):
                assert trace["name"] == group["name"] and trace["color"] == group["color"]
                assert all(len(trace[key]) == len(group["indices"]) for key in ["text", "x", "y", "z"])
                assert all(math.isfinite(v) for key in ["x", "y", "z"] for v in trace[key])
    parser = Links()
    parser.feed((DOCS / "index.html").read_text())
    assert len(parser.ids) == len(set(parser.ids))
    for url in parser.paths:
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc:
            continue
        if parsed.path:
            path = (DOCS / parsed.path).resolve()
            assert path.is_relative_to(DOCS.resolve()) and path.is_file(), url
        elif parsed.fragment:
            assert parsed.fragment in parser.ids, url
    assert not any("review" in p.name.lower() or p.suffix in {".tex", ".bib"} for p in DOCS.rglob("*") if p.is_file())
    assert not any(p.read_bytes().startswith(b"version https://git-lfs.github.com/spec") for p in DOCS.rglob("*") if p.is_file())
    print(f"Site checks passed: {len(payload['series'])} series, {token_count:,} tokens, {len(payload['figures'])} figures.")


if __name__ == "__main__":
    main()
