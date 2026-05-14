# POS Open-vs-Closed Contrast

**Hypothesis.** Open-class POS (`ADJ`, `ADV`, `INTJ`, `NOUN`, `PROPN`, `VERB`) have higher geometry scores than closed-class POS (`ADP`, `AUX`, `CCONJ`, `DET`, `NUM`, `PART`, `PRON`, `SCONJ`). The contrast is operationalized as a token-weighted open-class mean minus a token-weighted closed-class mean at each layer.

**Decision rule.** A layer-level contrast is marked as supporting the hypothesis when the approximate 95% confidence interval for open-minus-closed is strictly above zero. If the source POS table has zero-width intervals, the result is reported as an observed contrast without an uncertainty decision.

## Main Summary

| Model | Metric | Scope | Contrast | 95% CI | Support |
|---|---|---|---:|---:|---|
| bert-base-uncased | LID | last layer | 21.12 | [20.55, 21.68] | supports open greater than closed |
| bert-base-uncased | LID | layer mean | 134.8 | [134.5, 135.1] | supports open greater than closed |
| bert-base-uncased | NLID | last layer | 5.332 | [4.942, 5.722] | supports open greater than closed |
| bert-base-uncased | NLID | layer mean | 7.428 | [7.254, 7.602] | supports open greater than closed |
| bert-base-uncased | isoscore | last layer | 0.08497 | [0.08207, 0.08787] | supports open greater than closed |
| bert-base-uncased | isoscore | layer mean | 0.1265 | [0.1259, 0.1271] | supports open greater than closed |
| gpt2 | LID | last layer | 67.98 | [62.32, 73.64] | supports open greater than closed |
| gpt2 | LID | layer mean | 149.7 | [146.7, 152.8] | supports open greater than closed |
| gpt2 | NLID | last layer | 1.423 | [1.097, 1.748] | supports open greater than closed |
| gpt2 | NLID | layer mean | 7.072 | [6.861, 7.282] | supports open greater than closed |
| gpt2 | isoscore | last layer | 1.15e-04 | [6.67e-05, 1.63e-04] | supports open greater than closed |
| gpt2 | isoscore | layer mean | 0.00318 | [0.002883, 0.003477] | supports open greater than closed |

## Manuscript-Ready Wording

To make the POS observation directly quantitative, we collapsed POS tags into two predefined groups: open-class tags (`ADJ`, `ADV`, `INTJ`, `NOUN`, `PROPN`, `VERB`) and closed-class tags (`ADP`, `AUX`, `CCONJ`, `DET`, `NUM`, `PART`, `PRON`, `SCONJ`). For each model, metric, and layer, we computed the token-weighted mean score for each group and tested the directional contrast open minus closed. This turns the qualitative reading of the POS curves into a layer-wise hypothesis test against zero.

For both models, the available POS-level intervals support a positive open-minus-closed contrast in all reported layers and metric families.
