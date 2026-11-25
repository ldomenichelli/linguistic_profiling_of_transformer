# Linguistic Profiling of Transformer Embedding Geometry

**Lucia Domenichelli · Dominique Brunato · Felice Dell’Orletta**

Companion materials for *Linguistic Profiling of Transformer Embedding Geometry*, CoNLL 2026.

We study how linguistic features of words relate to the geometry of representations in BERT and GPT-2, from the embedding layer through the final transformer layer.

## Explore the results

The [project website](https://ldomenichelli.github.io/linguistic_profiling_of_transformer/) brings together layer comparisons, figures, and a browser of English and Italian annotated texts.

Compare word groups, inspect the saved results, or select a word to view its linguistic features.

## Rebuild the website

```bash
make site
make check
make serve
```

Open <http://localhost:8766>. The build uses Python 3.10 or later; figure previews also require `pdftoppm`, available in Poppler. Notebook checks require `python3 -m pip install -r requirements-checks.txt`.

See the [reproduction guide](REPRODUCIBILITY.md) for the analysis environment and [publishing instructions](PUBLISHING.md) for GitHub Pages setup.

## Repository

| Path | Contents |
|---|---|
| `docs/` | Project website |
| `code/` | Analysis notebooks and preprocessed sentence data |
| `plots_main/` | Main figures and result tables |
| `plots_extra/` | Additional analyses and controls |
| `correlation/` | Correlation analyses |
| `no_index_1/` | Analyses excluding sentence-initial words |
| `dataset_statistics/` | Linguistic feature summaries |
| `scripts/` | Analysis, build, and validation scripts |

Citation details are in [CITATION.cff](CITATION.cff). The annotated texts retain their [original source terms](THIRD_PARTY.md).
