# Reviewer 2 Revision Notes

Reviewer concern: the work currently reads as an exploratory intrinsic-dimensionality survey of BERT/GPT-2 on EWT, with crowded plots and qualitative interpretations that should be turned into clearer quantitative contrasts.

## Main Revision Strategy

1. Reframe novelty as linguistic conditioning plus quantitative contrasts, not as "first ID study of BERT/GPT".
   - Safer claim: prior work studies global/contextual embedding geometry and ID; this paper contributes a feature-conditioned profiling framework that compares encoder/decoder representations across surface, lexical, and syntactic token classes with multiple geometry families.
   - Also clarify scope: the analysis estimates the geometry of BERT/GPT-2 representations under EWT stimuli, not an unconditional property of either the model or the corpus alone.

2. Replace or demote crowded 13-line plots.
   - Keep full class-wise curves in the appendix.
   - In the main paper, use hypothesis-driven contrasts:
     - POS: open vs closed classes.
     - Head distance: head-before vs head-after groups.
     - Corpus-control evidence: feature-controlled correlations with token/type counts.
     - POS-conditioned head-distance control, especially nouns.

3. Convert qualitative claims into accept/reject-style statements.
   - Example hypothesis: open-class POS classes have higher isotropy, linear ID, and nonlinear ID than closed-class POS classes.
   - Example hypothesis: lexical diversity, rather than raw class frequency, is associated with higher geometric richness.
   - Example hypothesis: signed head-distance effects are not reducible to POS mixture, because they persist or change in POS-conditioned analyses.

## Quantitative Results Already Available

Open-vs-closed POS tables:
`plots_main/plots_features/pos_open_closed/tables/*_pos_open_closed_*.csv`

Layer-final open minus closed POS contrasts:

| Model | Metric | Open | Closed | Difference |
|---|---:|---:|---:|---:|
| BERT | IsoScore | 0.03342 | 0.01886 | +0.01456 |
| BERT | Linear ID | 680.0 | 659.1 | +20.91 |
| BERT | Nonlinear ID | 23.45 | 19.31 | +4.14 |
| GPT-2 | IsoScore | 0.001676 | 0.001596 | +0.000080 |
| GPT-2 | Linear ID | 108.8 | 66.81 | +41.94 |
| GPT-2 | Nonlinear ID | 4.505 | 3.621 | +0.885 |

Layer-mean open minus closed POS contrasts:

| Model | Metric | Difference |
|---|---:|---:|
| BERT | IsoScore | +0.02683 |
| BERT | Linear ID | +138.7 |
| BERT | Nonlinear ID | +4.816 |
| GPT-2 | IsoScore | +0.000726 |
| GPT-2 | Linear ID | +113.3 |
| GPT-2 | Nonlinear ID | +13.25 |

Feature-controlled corpus-statistic correlations:
`correlation/class_frequency_type_correlations/paper_section_feature_controlled_pooled_correlations.md`

After within-feature standardization, raw token frequency is weak and non-significant:

| Model | Metric family | rho with tokens | p |
|---|---|---:|---:|
| BERT | IsoScore | -0.152 | 0.245 |
| BERT | Linear ID | 0.136 | 0.372 |
| BERT | Nonlinear ID | -0.054 | 0.685 |
| GPT-2 | IsoScore | -0.181 | 0.166 |
| GPT-2 | Linear ID | -0.103 | 0.567 |
| GPT-2 | Nonlinear ID | -0.060 | 0.649 |

Unique type count is strongly positive:

| Model | Metric family | rho with types | p |
|---|---|---:|---:|
| BERT | IsoScore | 0.599 | < 0.001 |
| BERT | Linear ID | 0.860 | < 0.001 |
| BERT | Nonlinear ID | 0.550 | < 0.001 |
| GPT-2 | IsoScore | 0.482 | < 0.001 |
| GPT-2 | Linear ID | 0.745 | < 0.001 |
| GPT-2 | Nonlinear ID | 0.626 | < 0.001 |

BERT POS-conditioned signed head-distance controls:
`tables_POS_HEAD_DIST/bert/first/all/*/headdist_*`

For nouns, positive head distance means the head follows the noun; negative head distance means the head precedes the noun. The noun-only contrast shows that nouns whose heads precede them are higher in IsoScore and Linear ID than nouns whose heads follow them:

| POS | Metric | Last-layer positive minus negative | Layer-mean positive minus negative |
|---|---|---:|---:|
| NOUN | IsoScore | -0.02066 | -0.02394 |
| NOUN | Linear ID | -88.3 | -89.76 |
| NOUN | Nonlinear ID | -0.770 | +4.039 |

This is useful because it directly answers the reviewer's corpus/model concern: the aggregate head-distance pattern is at least partly shaped by POS composition, and the POS-conditioned control lets us separate that mixture from within-category effects.

Concrete linguistic examples from EWT:

- Head precedes noun: "the preacher at the mosque" gives `mosque -> preacher`, with the noun following its head.
- Head follows noun: "American forces killed..." gives `forces -> killed`, with the noun preceding its head.
- Compound-like pre-head noun: "investment firm" gives `investment -> firm`.
- Post-head nominal dependent: "head of an investment firm" gives `firm -> head`.

## Draft Rebuttal Language

We thank the reviewer for pointing out that several claims were expressed as qualitative readings of dense layer-wise plots. We have revised the paper to make the central observations more explicitly quantitative and hypothesis-driven. In particular, we added compact open-vs-closed POS contrasts, signed head-distance contrasts, and feature-controlled correlations with corpus statistics. The full class-wise curves are retained as diagnostic material, while the main text now reports focused contrasts that directly test the interpretations discussed in the paper.

Regarding novelty, we agree that intrinsic dimensionality of Transformer representations has been studied before, and we have softened claims that could suggest otherwise. The contribution of the revised paper is not the introduction of ID analysis itself, but a linguistically conditioned profiling framework: we compare encoder and decoder representations across token classes defined by POS, surface position, word length, dependency distance, and arity, and we combine isotropy, linear ID, and nonlinear ID to show which linguistic partitions induce different geometric signatures.

To address the concern that some effects may reflect the evaluation corpus rather than the model alone, we now state the scope more carefully: the analysis characterizes model representations elicited by the EWT corpus. We added feature-controlled corpus-statistic analyses showing that, after controlling for feature identity, raw token frequency is not significantly correlated with the geometric metrics, whereas lexical type count is strongly and positively associated with IsoScore, Linear ID, and Nonlinear ID in both BERT and GPT-2. We also added POS-conditioned head-distance analyses; for example, the noun-only BERT control shows that signed head-distance effects are not simply a by-product of mixing nouns with function words, although the direction and magnitude of the effect vary by POS and metric.

## Manuscript Edits To Make

- Remove the stray draft note in the introduction: "Da FARE: effetti del corpus o dei modelli? 2) more quantitative?"
- Add a paragraph in the introduction or limitations: "We analyze the geometry of model representations under EWT stimuli; corpus distribution and model geometry are therefore not fully separable in this observational design."
- Move full POS line plots with many colors to appendix; keep open/closed POS contrast in the main text.
- Add exact numerical contrasts in the POS section, using the tables above.
- Add corpus-control paragraph or figure based on `Feature_controlled_correlations`.
- Add concrete EWT examples when discussing head-distance interpretations.
