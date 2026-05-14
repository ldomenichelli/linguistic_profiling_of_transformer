# No Index 1 Artifacts

This folder collects plots and bootstrap tables where the first token of each
sentence was excluded from the analysis.

In these experiments, `index=1` means the first word in the sentence in the
1-based linguistic index convention. In token-level data this corresponds to
`word_id == 0`.

## Contents

| Folder | Contents |
|---|---|
| `plots/` | Copied plot artifacts for no-`index=1` runs. |
| `tables/` | Copied bootstrap CSV tables for no-`index=1` runs. |

The copied artifacts preserve their legacy relative paths, for example:

```text
plots/gpt2_no_index/results_LENGTH_no_index/
plots/gpt2_no_index/results_HEADDIST_no_index/
tables/gpt2_no_index/tables_LENGTH_no_index/
tables/gpt2_no_index/tables_HEADDIST_no_index/
```

Source root:

```text
/home/ldomenichelli/linguistic_profiling_of_the_geometry_copy/geometric_profiling_of_a_neural_language_model
```

Current inventory:

```text
21 plot files
20 CSV table files
```
