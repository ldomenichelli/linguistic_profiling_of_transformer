#!/usr/bin/env python3
"""Extract fixed display samples from the original PCA exports (requires Git LFS)."""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/pca"
VIEWS = [
    ("bert-pos", "BERT · part of speech", "plots_extra/pca3d_features/bert-base-uncased_pca3d_pos_classes.html", 350),
    ("bert", "BERT · all words", "plots_extra/pca3d_alltok/bert-base-uncased_pca3d_layers.html", 6000),
    ("gpt2", "GPT-2 · all words", "plots_extra/pca3d_alltok/gpt2_pca3d_layers.html", 6000),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_plot(path):
    text = path.read_text()
    start = text.index("[", text.index("Plotly.newPlot("))
    traces, end = json.JSONDecoder().raw_decode(text, start)
    start = end
    while text[start] in " ,\n":
        start += 1
    layout, _ = json.JSONDecoder().raw_decode(text, start)
    return traces, layout


def export_view(ident, label, source, cap, check=False):
    path = ROOT / source
    traces, layout = read_plot(path)
    steps = layout["sliders"][0]["steps"]
    assert [s["label"] for s in steps] == [str(i) for i in range(13)]
    blocks = [[t for t, visible in zip(traces, step["args"][0]["visible"]) if visible] for step in steps]
    first = blocks[0]
    selections = [sorted(random.Random(42).sample(range(len(t["x"])), min(cap, len(t["x"])))) for t in first]
    groups = [{"name": t["name"] if ident == "bert-pos" else "Words", "color": t["marker"]["color"], "originalPoints": len(t["x"]), "indices": indices} for t, indices in zip(first, selections)]
    layers = []
    for layer, block in enumerate(blocks):
        assert len(block) == len(first)
        selected = []
        for trace, original, group in zip(block, first, groups):
            assert trace["text"] == original["text"]
            assert all(len(trace[k]) == group["originalPoints"] for k in ["x", "y", "z"])
            selected.append({"name": group["name"], "color": group["color"], **{key: [trace[key][i] for i in group["indices"]] for key in ["text", "x", "y", "z"]}})
        data = {"view": ident, "layer": layer, "traces": selected}
        target = OUT / ident / f"{layer}.json"
        encoded = (json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
        if check:
            assert target.read_bytes() == encoded, target
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(encoded)
        layers.append({"path": str(target.relative_to(OUT.parent.parent)), "sha256": hashlib.sha256(encoded).hexdigest()})
    return {"id": ident, "label": label, "source": source, "sourceSha256": digest(path), "sampling": "fixed random sample without replacement, seed 42; identical point indices at every layer", "capPerTrace": cap, "points": sum(len(g["indices"]) for g in groups), "groups": groups, "layers": layers}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare the saved samples with the full source exports")
    args = parser.parse_args()
    manifest = {"views": [export_view(*v, check=args.check) for v in VIEWS]}
    target = OUT / "manifest.json"
    encoded = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n"
    if args.check:
        assert target.read_text() == encoded
    else:
        target.write_text(encoded)
    print("PCA samples verified against original coordinates." if args.check else "Prepared three PCA views across 13 layers.")


if __name__ == "__main__":
    main()
