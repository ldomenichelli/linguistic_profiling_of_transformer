# Verbal Valency Contrast

Verbal valency is defined over `VERB` tokens only, using the number of syntactic dependents attached to the verb. The contrast is mid-valency verbs (`2-4` dependents) minus low-valency verbs (`0-1` dependents).

| Model | Metric | Scope | Effect | 95% CI | Support |
|---|---|---|---:|---:|---|
| bert-base-uncased | isoscore | last layer | 0.01742 | [0.01189, 0.02295] | supports mid valency greater |
| bert-base-uncased | isoscore | layer mean | 0.02121 | [0.01974, 0.02268] | supports mid valency greater |

## Missing Tables

- bert-base-uncased LID: no verb-only bootstrap table found
- bert-base-uncased NLID: no verb-only bootstrap table found
- gpt2 isoscore: no verb-only bootstrap table found
- gpt2 LID: no verb-only bootstrap table found
- gpt2 NLID: no verb-only bootstrap table found
