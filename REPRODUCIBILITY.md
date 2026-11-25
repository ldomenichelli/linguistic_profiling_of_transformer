# Reproducing the analyses

## Analysis environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The full environment includes PyTorch, Transformers, IsoScore, DADApy, scikit-dimension, and Jupyter. Representation extraction and geometry estimation can require a GPU, substantial memory, and model downloads. Building the website uses saved results and does not run the models.

| Notebook | Analysis |
|---|---|
| `code/pos.ipynb` | Part of speech |
| `code/index.ipynb` | Sentence position |
| `code/length.ipynb` | Word length |
| `code/head_dist.ipynb` | Dependency head distance |
| `code/arity.ipynb` | Number of dependents |
| `code/relation.ipynb` | Dependency relations |
| `code/statistics.ipynb` | Dataset summaries |
| `code/correlation.ipynb` | Relations between geometry measures |
| `code/convergence.ipynb` | Sample-size robustness |
| `code/outlier_dims.ipynb` | Outlier dimensions |
| `code/sanity_check.ipynb` | Synthetic-manifold checks |
| `code/plots.ipynb` | Figure preparation |

Run notebooks from the repository root or `code/`, after reviewing their model and data settings.

## Data and measures

The preprocessed English EWT CSV contains 10,067 sentences and 194,916 stored tokens; the Italian ISDT CSV contains 12,112 sentences and 253,771 tokens. These are the counts in this repository, not the size of the upstream treebanks. The sentence browser exposes both CSVs independently of the group-level geometry results.

Each row contains `sentence_id` and Python-list strings for `tokens`, `index`, `length`, `pos`, `head_dist`, `relation_type`, and `ariety`. The last spelling is inherited from the dataset. Indices are stored sentence positions and may have gaps; the website preserves them. Head distance is signed: positive values denote words before their head and negative values words after it. The browser retains all supplied values.

The geometry measures are IsoScore, linear intrinsic dimension using a 99% PCA variance threshold (`pca99` or `lpca99`), and nonlinear intrinsic dimension using GRIDE. The website retains the estimator identifier for each source table. Layer 0 is the embedding representation; layers 1–12 are transformer layers.

## Website result sources

The layer explorer reads the `*_groups.csv` files in `plots_main/plots_features/feature_contrasts/tables/`. It includes 43 numeric series across eight linguistic contrasts. Source CSVs are copied without changing their precision; a combined CSV also records estimator, interval method, and source path.

Five verbal-valency series marked `legacy_pdf_*` were reconstructed from plots. They are omitted from the numeric explorer; the saved figures remain in the gallery. BERT verbal-valency IsoScore is available from a numeric source.

Where a source row marks uncertainty as `no_source_uncertainty`, its repeated mean/bound values are not displayed as confidence intervals. This affects GPT-2 POS tables and some BERT arity values. Other intervals are taken directly from the exported tables. The source's aggregation and interval method are retained in the JSON metadata. Cohorts are not identical across every comparison: panels indicate exclusions of sentence-initial words.

The sentence browser shows annotations only. It does not assign a geometric score to individual words or sentences.

## Checks and website build

```bash
python3 -m pip install -r requirements-checks.txt
make site
make check
make serve
```

The builder uses Python’s standard library. To create or refresh PNG previews from figure PDFs, install Poppler (`sudo apt-get install poppler-utils` on Ubuntu). Unchanged previews are reused. The website is served at <http://localhost:8766>.

`make check` compares web exports with their numeric sources, validates both sentence corpora and figure copies, and checks the notebooks. To inspect dataset distributions or execute notebook setup cells in the full analysis environment:

```bash
make audit-data
python3 scripts/validate_notebooks.py --execute-setup
```

These checks do not rerun the representation extraction or geometry experiments.

## Large interactive PCA plots

The original Plotly HTML exports are stored through Git LFS. They are separate from the lightweight website:

```bash
git lfs install
git lfs pull
python3 -m http.server 8000
```

Open the HTML files under `plots_extra/pca3d_alltok/` or `plots_extra/pca3d_features/` through the local server. The site’s own HTML uses ordinary Git storage and does not require LFS downloads.
