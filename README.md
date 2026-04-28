# Linguistic Profiling of Transformer Embedding Geometry

Code and plots for the paper “Linguistic Profiling of Transformer Embedding Geometry”.
This repository contains cleaned notebooks for reproducing the full experimental workflow.

The notebooks in `code/` were ported from the full research folder and cleaned by stripping
execution outputs, making dataset paths robust, and keeping the full experiment structure.
They can be run from either the repository root or from inside `code/`.

## Setup

Install the Python dependencies with:

```bash
python3 -m pip install -r requirements.txt
```

The transformer model downloads/caches are handled by Hugging Face `transformers`. For the
large experiments, a CUDA GPU is strongly recommended.


## Repository layout

| Folder | What it contains |
|---|---|
| `code/` | Notebooks for experiments / analyses / figure generation. |
| `code/data/` | Preprocessed dataset used by notebooks. |
| `dataset_statistics/` | Dataset statistics + distribution plots for linguistic features. |
| `outlier_dim/` | Outlier-dimension analyses. |
| `correlation/` | Metric–metric correlation analyses (Spearman heatmaps + tables). |
| `3dpca/` | Interactive 3D PCA visualizations by layer/class. |
| `plots_main/` | Plots used in the main paper figures. |
| `plots_extra/` | Additional / ablation plots (balancing controls, extra metrics, etc.). |

## Experiment notebooks

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
| `code/convergence.ipynb` | Subsample-size robustness / convergence analysis. |
| `code/outlier_dims.ipynb` | Outlier-dimension analyses for BERT and GPT-2. |
| `code/sanity_check.ipynb` | Synthetic-manifold metric sanity checks. |
| `code/plots.ipynb` | Figure aggregation and paper-style plotting utilities. |

```text
├── dataset_statistics/
├── data/
├── plots_extra/            --> first index kept
│   ├── information_imbalance/
│   ├── baseline/           --> unpretrained BERT and GPT-2 models
│   ├── metrics/
│   │   ├── all_tokens/     
│   │   ├── features/
│   │   ├── fixed_freq/     --> features but fixed frequency per classes
│   │   └── fixed_type/     --> features but fixed #types per class
│   ├── pos_x_right_left/   --> fixed left or right of head on feature "pos"
│   ├── pos_x_head_dist/    --> fixed head distance on feature "pos"
│   ├──pca3d_alltok/        --> 3D interactive PCA on all tokens
│   └──pca3d_features/      --> 3D interactive PCA x feature
├── code/
├── plots_main/             --> first index removed
│   ├── plots_all/
│   └── plots_features/
├── correlation/
│   ├── plots/
│   └── tables/
└── outlier_dim/
