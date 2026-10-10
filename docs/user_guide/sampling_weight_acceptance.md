# Verify sampling and sample weights

Use a controlled comparison to check that a configured weight column actually
reaches fitting and remains aligned after resampling. Start with
[Weighted Training & Support](weighted_training.md) for the supported models,
weight validation and `synthetic_weight` policies. The procedure below is useful
for your own fitted recipe and data; a different prediction alone is not proof
that the intended rows received the intended weights.

## Set up a comparable pair

Keep the Delta source version, record keys, split seed, preprocessing, estimator
parameters and sampling seed identical. Fit one candidate with its weight column
and one with `weight_column=None`. Keep `class_weight` unchanged; using `None`
is useful when isolating sample weights. Use variable weights within classes so
that the comparison does more than accept a uniform vector.

In generated projects, put shared source and split settings in
`config/training.yml` under `defaults`, and model-specific settings under the
selected `models` entry. Keep the weight column and record keys out of model
features. Sampling belongs inside training folds; never resample the final
holdout or scoring requests.

| Resampling | Weight behavior to verify |
| --- | --- |
| None | Each retained training row keeps its original weight |
| Undersampling | Selected rows retain their aligned original weights |
| SMOTE with `synthetic_weight="class_mean"` | Original weights are retained; synthetic rows receive the original training class mean |
| SMOTE without sample weights | No sample-weight vector is passed merely because sampling is enabled |

Check the supported combinations in the weighted training guide before selecting
a strategy. Evaluation and threshold selection remain unweighted.

## Check the full path

1. Pin a source version and compare training/holdout membership by stable keys.
2. In a controlled validation harness, capture the actual estimator `fit` inputs
   and delegate to the unchanged estimator. Compare feature, target and weight
   lengths and row alignment after sampling.
3. Independently reconstruct the sampler and fit the same estimator using those
   inputs. Compare fitted parameters where meaningful, predictions and class
   probabilities on the same untouched holdout.
4. Save and reload the candidate through the actual artifact or registry path.
   Score using feature columns only and compare with the direct fitted model.
5. If the workflow publishes to Delta, read back predictions by key, inspect the
   concrete model version and receipt, and verify an unchanged-source replay.

Record your source version, recipe, dependency versions, sampled membership,
fit-input evidence and comparison results with the candidate. Avoid putting raw
feature or weight values in public reports. MLflow tracking records only the
artifacts your workflow explicitly logs; a weight parameter in metadata does
not prove the estimator used it.

## Interpret differences

Weights can change coefficients and probabilities without improving holdout
quality. Use the held-out metric appropriate to the task and keep comparison
populations identical. Small numerical differences can also come from dtype,
serialization or estimator parallelism; use a justified tolerance and inspect
large discrepancies rather than increasing it to hide a failure.

For job execution, candidate registration and publication, see the
[Databricks Python SDK](databricks_sdk.md) and
[Bundle operator walkthrough](databricks_bundle_walkthrough.md). Validate CV,
search, promotion, concurrent writers and serving separately when your production
workflow uses them.
