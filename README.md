# Linguistic Profiling of Transformer Embedding Geometry

Welcome to the accompanying repository for the paper **“Linguistic Profiling of Transformer Embedding Geometry.”**

## Folder Guide

| Folder                    | Description |
|--------------------------|-------------|
| **3dpca/**               | Interactive 3D PCA visualizations for different layers and model. |
| **code/**                | Notebooks (experiments, analysis, and figure generation). |
| **data/**                | Dataset used in the study. |
| **dataset_statistics/**  | Dataset statistics, summaries, and descriptive analyses. |
| **correlation/**         | Correlation analyses between geometric measures and linguistic variables. |
| **outlier_dim/**         | Outlier-dimension analyses and related outputs. |
| **plots_main/**          | Main plots used in the paper’s figures. |
| **plots_extra/**         | Additional plots that complement the main figures. |


## Repository structure

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
