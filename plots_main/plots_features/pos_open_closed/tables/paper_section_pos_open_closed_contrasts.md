# POS Open-vs-Closed Contrast

**Hypothesis.** Open-class POS (`ADJ`, `ADV`, `NOUN`, `PROPN`, `VERB`) have higher geometry scores than closed-class POS (`ADP`, `AUX`, `CCONJ`, `DET`, `PART`, `PRON`, `SCONJ`). The contrast is operationalized as a token-weighted open-class mean minus a token-weighted closed-class mean at each layer.

**Decision rule.** A layer-level contrast is marked as supporting the hypothesis when the approximate 95% confidence interval for open-minus-closed is strictly above zero. If the source POS table has zero-width intervals, the result is reported as an observed contrast without an uncertainty decision.

## Main Summary

| Model | Metric | Scope | Contrast | 95% CI | Support |
|---|---|---|---:|---:|---|
| bert-base-uncased | LID | last layer | 20.91 | [20.5, 21.32] | supports open greater than closed |
| bert-base-uncased | LID | layer mean | 138.7 | [138.4, 139.0] | supports open greater than closed |
| bert-base-uncased | NLID | last layer | 4.138 | [3.829, 4.448] | supports open greater than closed |
| bert-base-uncased | NLID | layer mean | 4.816 | [4.528, 5.104] | supports open greater than closed |
| bert-base-uncased | isoscore | last layer | 0.01456 | [0.01418, 0.01494] | supports open greater than closed |
| bert-base-uncased | isoscore | layer mean | 0.02683 | [0.02668, 0.02697] | supports open greater than closed |
| gpt2 | LID | last layer | 41.94 | [41.94, 41.94] | observed open greater than closed no ci |
| gpt2 | LID | layer mean | 113.3 | [113.3, 113.3] | observed open greater than closed no ci |
| gpt2 | NLID | last layer | 0.8847 | [0.8847, 0.8847] | observed open greater than closed no ci |
| gpt2 | NLID | layer mean | 13.25 | [13.25, 13.25] | observed open greater than closed no ci |
| gpt2 | isoscore | last layer | 7.97e-05 | [7.97e-05, 7.97e-05] | observed open greater than closed no ci |
| gpt2 | isoscore | layer mean | 7.26e-04 | [7.26e-04, 7.26e-04] | observed open greater than closed no ci |

## Manuscript-Ready Wording

To make the POS observation directly quantitative, we collapsed POS tags into two predefined groups: open-class tags (`ADJ`, `ADV`, `NOUN`, `PROPN`, `VERB`) and closed-class tags (`ADP`, `AUX`, `CCONJ`, `DET`, `PART`, `PRON`, `SCONJ`). For each model, metric, and layer, we computed the token-weighted mean score for each group and tested the directional contrast open minus closed. This turns the qualitative reading of the POS curves into a layer-wise hypothesis test against zero.

For BERT, the available POS-level bootstrap intervals support a positive open-minus-closed contrast in all reported layers and metric families. GPT-2 source tables contain zero-width intervals, so the GPT-2 rows should be described as observed positive contrasts unless those metric tables are regenerated with bootstrap replicates.
