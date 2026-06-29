# Linguistic Profiling of Transformer Embedding Geometry

Companion repository for the CoNLL paper "Linguistic Profiling of Transformer
Embedding Geometry".

The project studies how linguistic properties are reflected in the geometry of
token representations across BERT and GPT-2 layers. It groups Universal
Dependencies tokens by sentence position, token length, part of speech,
dependency head distance, relation type, and verbal valency, then compares
isotropy, linear intrinsic dimensionality, and nonlinear intrinsic
dimensionality across layers.

## Artifact Scope

This repository includes:

- cleaned experiment notebooks in `code/`;
- preprocessed sentence-level CSV data in `code/data/`;
- scripts for lightweight dataset and notebook validation in `scripts/`;
- main-paper plots in `plots_main/`;
- supplementary plots, ablations, and robustness summaries in `plots_extra/`;
- correlation analyses in `correlation/`;
- no-first-token artifacts in `no_index_1/`.

The lightweight checks run on a laptop. Full representation extraction and
metric recomputation can require substantial time, RAM, disk space, and a CUDA
GPU.

## Setup

Create an environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Transformer model downloads and caches are handled by Hugging Face
`transformers`.

## Lightweight Checks

Before running the long notebooks, audit the preprocessed dataset and validate
the notebooks:

```bash
python3 scripts/audit_dataset.py
python3 scripts/validate_notebooks.py --execute-setup
```

`audit_dataset.py` checks sentence/token counts, token-feature alignment, and
distributions for index, length, POS, head distance, dependency relation, and
arity. `validate_notebooks.py` checks notebook schema, Python syntax, cleared
outputs, repo-relative path helpers, setup cells, and slow direct IsoScore
calls.

## Reproducing the Analyses

Run notebooks from the repository root or from inside `code/`. The main
analysis notebooks are:

| Notebook | Main experiment |
|---|---|
| `code/statistics.ipynb` | Dataset summaries for POS, index, length, head distance, arity, and relation type. |
| `code/pos.ipynb` | POS-conditioned geometry and all-token comparisons. |
| `code/index.ipynb` | Token-index conditioned geometry and PCA visualizations. |
| `code/length.ipynb` | Token-length conditioned geometry and PCA visualizations. |
| `code/head_dist.ipynb` | Dependency head-distance conditioned geometry and PCA visualizations. |
| `code/arity.ipynb` | Dependency arity conditioned geometry and PCA visualizations. |
| `code/relation.ipynb` | Dependency-relation conditioned geometry and PCA visualizations. |
| `code/correlation.ipynb` | Metric-to-metric Spearman correlation analysis. |
| `code/convergence.ipynb` | Subsample-size robustness and convergence analysis. |
| `code/outlier_dims.ipynb` | Outlier-dimension analyses for BERT and GPT-2. |
| `code/sanity_check.ipynb` | Synthetic-manifold metric sanity checks. |
| `code/plots.ipynb` | Figure aggregation and paper-style plotting utilities. |

The plotting and post-processing scripts in `scripts/` regenerate the derived
CSV summaries and paper figures stored under `plots_main/`, `plots_extra/`, and
`correlation/`.

## Interactive HTML Visualizations

The interactive PCA visualizations are Plotly HTML files tracked with Git LFS.
After cloning, fetch the LFS files:

```bash
git lfs install
git lfs pull
```

Serve the repository root locally:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/plots_extra/pca3d_alltok/bert-base-uncased_pca3d_layers.html
http://localhost:8000/plots_extra/pca3d_alltok/gpt2_pca3d_layers.html
http://localhost:8000/plots_extra/pca3d_features/bert-base-uncased_pca3d_pos_classes.html
```

The files are large and can take a little while to load. Once loaded, they are
interactive: drag to rotate, scroll to zoom, and use the Plotly legend/layer
controls.

GitHub's repository file view does not render these HTML files as interactive
pages. For interactive viewing, clone the repository and use the local server
above, or publish smaller HTML exports through a static site or release asset
page.

## Repository Layout

| Path | Contents |
|---|---|
| `code/` | Experiment, analysis, and figure-generation notebooks. |
| `code/data/` | Preprocessed sentence-level data used by the notebooks. |
| `dataset_statistics/` | Dataset statistics and distribution plots for linguistic features. |
| `plots_main/` | Main-paper figures. |
| `plots_extra/` | Supplementary plots, ablations, balancing controls, and robustness summaries. |
| `correlation/` | Metric-correlation analyses, tables, and heatmaps. |
| `outlier_dim/` | Outlier-dimension analyses. |
| `no_index_1/` | Artifacts from analyses excluding first sentence tokens. |
| `scripts/` | Validation, plotting, and post-processing scripts. |

## Citation

Please cite the CoNLL paper if you use this code or the derived artifacts:

```bibtex
@inproceedings{domenichelli-etal-2026-linguistic-profiling,
  title = {Linguistic Profiling of Transformer Embedding Geometry},
  author = {Domenichelli, Lucia and Brunato, Dominique and Dell'Orletta, Felice},
  booktitle = {Proceedings of the Conference on Computational Natural Language Learning (CoNLL)},
  year = {2026}
}
```
