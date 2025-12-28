# Linguistic Profiling of Transformer Embedding Geometry

Code and plots for the paper “Linguistic Profiling of Transformer Embedding Geometry”.
This repository contains a lightweight subset of the full experimental codebase.


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
│   └── pos_x_head_dist/    --> fixed head distance on feature "pos"
├── code/
├── plots_main/             --> first index removed
│   ├── plots_all/
│   └── plots_features/
├── correlation/
│   ├── plots/
│   └── tables/
└── outlier_dim/
