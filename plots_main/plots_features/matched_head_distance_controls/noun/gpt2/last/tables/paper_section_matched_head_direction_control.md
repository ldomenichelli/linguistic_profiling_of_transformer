# Matched Head-Direction Control

The comparison fixes POS and compares `Before Head` (`head_dist > 0`) against `After Head` (`head_dist < 0`) after coarsened exact matching on corpus covariates.

| Metric | Scope | Before - After | 95% CI | Support |
|---|---|---:|---:|---|
| gride | last layer | -1.08 | [-1.388, -0.772] | after greater |
| gride | layer mean | -1.678 | [-2.418, -0.9382] | after greater |
| iso | last layer | -0.0003239 | [-0.0004701, -0.0001777] | after greater |
| iso | layer mean | -0.01215 | [-0.01442, -0.009878] | after greater |
| lpca99 | last layer | -34.54 | [-49.3, -19.78] | after greater |
| lpca99 | layer mean | -3.019 | [-6.511, 0.4734] | inconclusive |

## Files

- categorical_balance: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/gpt2/last/tables/categorical_balance.csv`
- corpus_summary: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/gpt2/last/tables/corpus_summary.csv`
- histogram_last_pdf: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/gpt2/last/matched_noun_head_direction_histogram_last.pdf`
- histogram_mean_pdf: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/gpt2/last/matched_noun_head_direction_histogram_mean.pdf`
- matched_contrast: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/gpt2/last/tables/matched_before_minus_after_contrast.csv`
- matched_metrics: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/gpt2/last/tables/matched_head_direction_metrics.csv`
- matched_strata: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/gpt2/last/tables/matched_strata.csv`
- matched_summary: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/gpt2/last/tables/matched_before_minus_after_summary.csv`
- matched_tokens: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/gpt2/last/tables/matched_before_after_tokens.csv`
- numeric_balance: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/gpt2/last/tables/numeric_balance.csv`
- raw_tokens: `/home/ldomenichelli/linguistic_profiling_of_transformer/plots_main/plots_features/matched_head_distance_controls/noun/gpt2/last/tables/raw_before_after_tokens.csv`
