# Matched Head-Direction Control

The comparison fixes POS and compares `Before Head` (`head_dist > 0`) against `After Head` (`head_dist < 0`) after coarsened exact matching on corpus covariates.

| Metric | Scope | Before - After | 95% CI | Support |
|---|---|---:|---:|---|
| gride | last layer | 0.4256 | [-1.525, 2.376] | inconclusive |
| gride | layer mean | -1.627 | [-2.912, -0.3422] | after greater |
| iso | last layer | -0.01514 | [-0.02303, -0.007238] | after greater |
| iso | layer mean | -0.007386 | [-0.01107, -0.003705] | after greater |
| lpca99 | last layer | 2.48 | [-11.3, 16.26] | inconclusive |
| lpca99 | layer mean | 1.398 | [-2.466, 5.261] | inconclusive |

## Files

- categorical_balance: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/bert/first/tables/categorical_balance.csv`
- corpus_summary: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/bert/first/tables/corpus_summary.csv`
- histogram_last_pdf: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/bert/first/matched_noun_head_direction_histogram_last.pdf`
- histogram_mean_pdf: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/bert/first/matched_noun_head_direction_histogram_mean.pdf`
- matched_contrast: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/bert/first/tables/matched_before_minus_after_contrast.csv`
- matched_metrics: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/bert/first/tables/matched_head_direction_metrics.csv`
- matched_strata: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/bert/first/tables/matched_strata.csv`
- matched_summary: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/bert/first/tables/matched_before_minus_after_summary.csv`
- matched_tokens: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/bert/first/tables/matched_before_after_tokens.csv`
- numeric_balance: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/bert/first/tables/numeric_balance.csv`
- raw_tokens: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/bert/first/tables/raw_before_after_tokens.csv`
