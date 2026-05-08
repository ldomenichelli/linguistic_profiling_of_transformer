# All Feature Contrasts

Each qualitative feature observation is represented as a binary contrast. The positive direction is listed explicitly so the hypothesis can be accepted, rejected, or marked as inconclusive by checking whether the approximate 95% CI for the contrast excludes zero.

| Feature | Contrast | Model | Metric | Scope | Effect | 95% CI | Support |
|---|---|---|---|---|---:|---:|---|
| POS | Open classes - Closed classes | bert-base-uncased | LID | last layer | 20.91 | [20.5, 21.32] | supports positive group greater |
| POS | Open classes - Closed classes | bert-base-uncased | LID | layer mean | 138.7 | [138.4, 139.0] | supports positive group greater |
| POS | Open classes - Closed classes | bert-base-uncased | NLID | last layer | 4.138 | [3.829, 4.448] | supports positive group greater |
| POS | Open classes - Closed classes | bert-base-uncased | NLID | layer mean | 4.816 | [4.528, 5.104] | supports positive group greater |
| POS | Open classes - Closed classes | bert-base-uncased | isoscore | last layer | 0.01456 | [0.01418, 0.01494] | supports positive group greater |
| POS | Open classes - Closed classes | bert-base-uncased | isoscore | layer mean | 0.02683 | [0.02668, 0.02697] | supports positive group greater |
| POS | Open classes - Closed classes | gpt2 | LID | last layer | 41.94 | [41.94, 41.94] | observed positive group greater no ci |
| POS | Open classes - Closed classes | gpt2 | LID | layer mean | 113.3 | [113.3, 113.3] | observed positive group greater no ci |
| POS | Open classes - Closed classes | gpt2 | NLID | last layer | 0.8847 | [0.8847, 0.8847] | observed positive group greater no ci |
| POS | Open classes - Closed classes | gpt2 | NLID | layer mean | 13.25 | [13.25, 13.25] | observed positive group greater no ci |
| POS | Open classes - Closed classes | gpt2 | isoscore | last layer | 7.97e-05 | [7.97e-05, 7.97e-05] | observed positive group greater no ci |
| POS | Open classes - Closed classes | gpt2 | isoscore | layer mean | 7.26e-04 | [7.26e-04, 7.26e-04] | observed positive group greater no ci |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | bert-base-uncased | LID | last layer | 14.4 | [13.56, 15.23] | supports positive group greater |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | bert-base-uncased | LID | layer mean | 98.98 | [98.38, 99.57] | supports positive group greater |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | bert-base-uncased | NLID | last layer | 5.655 | [4.984, 6.325] | supports positive group greater |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | bert-base-uncased | NLID | layer mean | 3.907 | [3.706, 4.108] | supports positive group greater |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | bert-base-uncased | isoscore | last layer | 0.01706 | [0.01628, 0.01785] | supports positive group greater |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | bert-base-uncased | isoscore | layer mean | 0.03025 | [0.02997, 0.03053] | supports positive group greater |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | gpt2 | LID | last layer | 45.61 | [37.57, 53.65] | supports positive group greater |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | gpt2 | LID | layer mean | 96.37 | [95.49, 97.25] | supports positive group greater |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | gpt2 | NLID | last layer | 1.302 | [1.184, 1.42] | supports positive group greater |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | gpt2 | NLID | layer mean | 3.511 | [3.351, 3.671] | supports positive group greater |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | gpt2 | isoscore | last layer | -9.96e-05 | [-1.18e-04, -8.14e-05] | supports negative group greater |
| Token Length | Long tokens (length >= 7) - Short tokens (length <= 3) | gpt2 | isoscore | layer mean | 0.01312 | [0.01302, 0.01322] | supports positive group greater |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | bert-base-uncased | LID | last layer | 22.01 | [21.24, 22.77] | supports positive group greater |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | bert-base-uncased | LID | layer mean | 43.61 | [43.1, 44.13] | supports positive group greater |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | bert-base-uncased | NLID | last layer | 4.031 | [3.145, 4.916] | supports positive group greater |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | bert-base-uncased | NLID | layer mean | 8.122 | [7.833, 8.411] | supports positive group greater |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | bert-base-uncased | isoscore | last layer | -0.008627 | [-0.009032, -0.008223] | supports negative group greater |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | bert-base-uncased | isoscore | layer mean | 1.13e-04 | [-3.10e-05, 2.57e-04] | inconclusive |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | gpt2 | LID | last layer | 148.0 | [132.7, 163.3] | supports positive group greater |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | gpt2 | LID | layer mean | 234.4 | [233.2, 235.7] | supports positive group greater |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | gpt2 | NLID | last layer | 4.138 | [3.503, 4.774] | supports positive group greater |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | gpt2 | NLID | layer mean | 10.17 | [9.826, 10.51] | supports positive group greater |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | gpt2 | isoscore | last layer | 2.54e-04 | [2.40e-04, 2.68e-04] | supports positive group greater |
| Token Index | Late positions (index >= 8) - Early positions (index <= 3) | gpt2 | isoscore | layer mean | 0.004016 | [0.003924, 0.004108] | supports positive group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | bert-base-uncased | LID | last layer | -3.546 | [-3.966, -3.125] | supports negative group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | bert-base-uncased | LID | layer mean | 2.828 | [2.648, 3.008] | supports positive group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | bert-base-uncased | NLID | last layer | 5.883 | [5.392, 6.374] | supports positive group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | bert-base-uncased | NLID | layer mean | 3.555 | [3.255, 3.855] | supports positive group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | bert-base-uncased | isoscore | last layer | 0.01722 | [0.01661, 0.01784] | supports positive group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | bert-base-uncased | isoscore | layer mean | 0.006874 | [0.006653, 0.007095] | supports positive group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | gpt2 | LID | last layer | -30.55 | [-38.06, -23.04] | supports negative group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | gpt2 | LID | layer mean | -5.932 | [-6.566, -5.297] | supports negative group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | gpt2 | NLID | last layer | 1.12 | [0.9929, 1.247] | supports positive group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | gpt2 | NLID | layer mean | 2.322 | [2.109, 2.536] | supports positive group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | gpt2 | isoscore | last layer | -1.70e-04 | [-1.90e-04, -1.51e-04] | supports negative group greater |
| Dependency Arity | Mid arity (2-4 dependents) - Low arity (<= 1) | gpt2 | isoscore | layer mean | 0.003118 | [0.003022, 0.003214] | supports positive group greater |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | bert-base-uncased | LID | last layer | 11.91 | [11.15, 12.66] | supports positive group greater |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | bert-base-uncased | LID | layer mean | -15.47 | [-15.98, -14.96] | supports negative group greater |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | bert-base-uncased | NLID | last layer | -0.9674 | [-1.576, -0.3589] | supports negative group greater |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | bert-base-uncased | NLID | layer mean | -5.308 | [-5.554, -5.061] | supports negative group greater |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | bert-base-uncased | isoscore | last layer | 0.007948 | [0.007406, 0.008491] | supports positive group greater |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | bert-base-uncased | isoscore | layer mean | -3.48e-05 | [-1.74e-04, 1.05e-04] | inconclusive |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | gpt2 | LID | last layer | 76.12 | [71.27, 80.97] | supports positive group greater |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | gpt2 | LID | layer mean | 2.984 | [2.568, 3.4] | supports positive group greater |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | gpt2 | NLID | last layer | 0.04033 | [-0.05337, 0.134] | inconclusive |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | gpt2 | NLID | layer mean | -2.919 | [-3.058, -2.78] | supports negative group greater |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | gpt2 | isoscore | last layer | 1.23e-04 | [1.09e-04, 1.36e-04] | supports positive group greater |
| Head Distance | Before Head (dist > 0) - After Head (dist < 0) | gpt2 | isoscore | layer mean | 0.004928 | [0.004854, 0.005001] | supports positive group greater |
| Relation Type | Content relations - Function relations | bert-base-uncased | LID | last layer | 0.1394 | [-0.4937, 0.7724] | inconclusive |
| Relation Type | Content relations - Function relations | bert-base-uncased | LID | layer mean | 112.1 | [111.7, 112.5] | supports positive group greater |
| Relation Type | Content relations - Function relations | bert-base-uncased | NLID | last layer | -1.4 | [-1.867, -0.9325] | supports negative group greater |
| Relation Type | Content relations - Function relations | bert-base-uncased | NLID | layer mean | 0.02792 | [-0.1425, 0.1984] | inconclusive |
| Relation Type | Content relations - Function relations | bert-base-uncased | isoscore | last layer | 0.004184 | [0.003693, 0.004674] | supports positive group greater |
| Relation Type | Content relations - Function relations | bert-base-uncased | isoscore | layer mean | 0.01308 | [0.01294, 0.01321] | supports positive group greater |
| Relation Type | Content relations - Function relations | gpt2 | LID | last layer | -74.81 | [-84.65, -64.98] | supports negative group greater |
| Relation Type | Content relations - Function relations | gpt2 | LID | layer mean | 104.4 | [103.5, 105.4] | supports positive group greater |
| Relation Type | Content relations - Function relations | gpt2 | NLID | last layer | -0.486 | [-0.6114, -0.3607] | supports negative group greater |
| Relation Type | Content relations - Function relations | gpt2 | NLID | layer mean | 0.3156 | [0.198, 0.4331] | supports positive group greater |
| Relation Type | Content relations - Function relations | gpt2 | isoscore | last layer | -3.48e-04 | [-3.74e-04, -3.22e-04] | supports negative group greater |
| Relation Type | Content relations - Function relations | gpt2 | isoscore | layer mean | 0.006578 | [0.006503, 0.006653] | supports positive group greater |

## Hypotheses

- **POS**: Open-class POS have higher geometry scores than closed-class POS.
- **Token Length**: Long tokens have higher geometry scores than short tokens.
- **Token Index**: Late sentence positions differ from early positions.
- **Dependency Arity**: Mid-arity tokens (2-4 dependents) occupy higher-dimensional and more isotropic subspaces than low-arity tokens.
- **Head Distance**: Tokens before their syntactic head differ from tokens after their syntactic head.
- **Relation Type**: Content-bearing dependency relations have higher geometry scores than function relations.

## Notes

- Layer-mean intervals are descriptive summaries across correlated layers, not independent-layer tests.
- Intervals are approximated from the saved class-level CIs because the original bootstrap replicates are not stored in these CSVs.
- Middle or neutral classes are excluded from each binary contrast, for example medium token lengths, root-distance 0, punctuation, and relation labels outside the content/function sets.
- The arity contrast uses the available all-token arity tables. The manuscript wording should say `predicate` only if predicate-conditioned arity tables are regenerated; otherwise describe it as mid-arity tokens.
- Rows marked `no ci` come from source tables with zero-width intervals and should be described as observed contrasts rather than CI-supported effects.
