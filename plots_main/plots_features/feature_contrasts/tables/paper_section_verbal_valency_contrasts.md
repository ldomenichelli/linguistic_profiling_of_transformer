# Verbal Valency Contrast

Verbal valency is defined over `VERB` tokens only, using the number of syntactic dependents attached to the verb. The contrast is mid-valency verbs (`2-4` dependents) minus low-valency verbs (`0-1` dependents).

| Model | Metric | Scope | Effect | 95% CI | Support |
|---|---|---|---:|---:|---|
| bert-base-uncased | LID | last layer | 134.9 | [131.4, 138.4] | supports mid valency greater |
| bert-base-uncased | LID | layer mean | 136.6 | [135.4, 137.8] | supports mid valency greater |
| bert-base-uncased | NLID | last layer | 0.9297 | [-0.543, 2.402] | inconclusive |
| bert-base-uncased | NLID | layer mean | -1.333 | [-1.737, -0.9285] | supports low valency greater |
| bert-base-uncased | isoscore | last layer | 0.01742 | [0.01189, 0.02295] | supports mid valency greater |
| bert-base-uncased | isoscore | layer mean | 0.02121 | [0.01974, 0.02268] | supports mid valency greater |
| gpt2 | LID | last layer | 30.31 | [17.38, 43.24] | supports mid valency greater |
| gpt2 | LID | layer mean | 113.7 | [112.1, 115.2] | supports mid valency greater |
| gpt2 | NLID | last layer | 2.461 | [2.213, 2.709] | supports mid valency greater |
| gpt2 | NLID | layer mean | -3.059 | [-3.467, -2.65] | supports low valency greater |
| gpt2 | isoscore | last layer | -9.92e-05 | [-2.53e-04, 5.41e-05] | inconclusive |
| gpt2 | isoscore | layer mean | 0.004116 | [0.003173, 0.005059] | supports mid valency greater |
