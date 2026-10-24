# Voting and stacking ensembles

This page covers ensemble candidates in a generated Databricks Bundle's
`config/training.yml`: the voting and stacking `model.params` fields, nested
member search axes, worked voting, stacking and calibrated examples, and the
setup questions and defaults for each slot. Read it when one model candidate
should combine several learners. The worked examples append a `models` block to
the [common setup header](databricks_tuning.md#common-setup-for-the-worked-examples)
in the tuning guide.

[Back to Databricks Bundle tuning](databricks_tuning.md)

An ensemble is one model candidate containing several learners. In contrast,
competition evaluates separate candidates and selects one winner. Choose
`voting_regressor`, `voting_classifier`, `stacking_regressor` or
`stacking_classifier`; see their [model cards](databricks_model_catalog.md) for
task-specific member menus and defaults.

| `model.params` setting | Type/default | Effect |
| --- | --- | --- |
| `base_estimators` | List of distinct member names; model-specific defaults | Choose member keys such as `ridge`, `lasso`, `random_forest`, not automatically the top-level model IDs. Unknown, duplicate and empty selections fail. |
| `base_estimator_params` | Mapping, `{}` | Fixed constructor overrides keyed by selected member name. Unknown members/parameters fail. |
| `weights` | Voting only; null means equal | Ordered list with one finite nonnegative value per member, or a member-name mapping with omitted selected members defaulting to 1. At least one value must be positive. These weight model votes, not training rows. |
| `voting` | Classification voting only, `soft` | `soft` averages probabilities; every member needs probability output. `hard` votes on class labels and has no ordinary `predict_proba` output. |
| `final_estimator` | Stacking only; `ridge` for regression, `logistic_regression` for classification | A supported member key for the learner trained on base-model out-of-fold predictions. |
| `final_estimator_params` | Stacking only; mapping, `{}` | Fixed final-learner constructor overrides. |
| `cv` | Stacking only; integer `5`, 2..20 | Internal out-of-fold predictions for the final learner. `prefit` is not admitted. |
| `passthrough` | Stacking only; boolean `false` | Also give original input features to the final learner. |
| `calibrate_base_models` | Boolean, `false` | Classification only: wrap each base classifier in probability calibration. |
| `calibration_method` | String, `sigmoid` | `sigmoid` or `isotonic`; isotonic needs enough calibration data. |
| `calibration_cv` | Integer, `3`; 2..20 | Internal calibration folds, separate from shared search CV and stacking CV. |
| `tune_base_models` | Boolean; false without tuning, inserted true by a tuning wrapper if absent | With automatic spaces, include selected members' search axes. Explicit true requires a tuning wrapper. A manually supplied space owns its own axes. |
| `n_jobs` | Integer, `1` for Bundle search | Keep ensemble and member workers at one during search. |

Voting rejects stacking-only fields; stacking rejects `weights` and `voting`.
Classification calibration controls are different from the downstream
`decision_threshold` policy. Row `weight_column` is a third, independent concept;
all selected members and the final learner must meet the sample-weight contract.

For manual spaces, a member parameter is named `<member>__<parameter>`, and a
stacking final-learner parameter is `final_estimator__<parameter>`. Calibrated
members insert `estimator__`: for example
`logistic_regression__estimator__C`. Fixed overrides remain fixed in automatic
spaces, and conflicting manual axes fail. Do not search structural member lists,
weights arrays, or estimator objects as scalar axes.

## Voting example

Append this alternative `models` block to the common regression header. It
searches ridge while keeping lasso's alpha fixed. To fit the same ensemble once,
delete the whole `tuning` mapping.

<!-- tuning-example:voting -->
```yaml
models:
  revenue:
    model:
      type: voting_regressor
      params:
        base_estimators: [ridge, lasso]
        base_estimator_params:
          lasso: {alpha: 0.01}
        weights: {ridge: 2, lasso: 1}
    tuning:
      strategy: grid
      metric: rmse
      max_candidates: 2
      search_space: {ridge__alpha: [0.1, 1.0]}
```

## Stacking example

This alternative searches both a member and the final estimator. Its internal
`cv: 2` happens inside each outer search fit, so it costs more than a single
ridge grid even though there are only four candidate combinations.

<!-- tuning-example:stacking -->
```yaml
models:
  revenue:
    model:
      type: stacking_regressor
      params:
        base_estimators: [ridge, lasso]
        base_estimator_params:
          lasso: {alpha: 0.01}
        final_estimator: ridge
        cv: 2
        passthrough: false
    tuning:
      strategy: grid
      metric: rmse
      max_candidates: 4
      search_space:
        ridge__alpha: [0.1, 1.0]
        final_estimator__alpha: [0.1, 1.0]
```

## Calibrated classification example

This block overrides the header's task/target/metric for a classification model.
The table must have a binary `churn` label and enough examples per class for
both search and calibration folds. Update the example model's quality threshold
to your domain's accuracy gate before deployment.

<!-- tuning-example:calibrated -->
```yaml
models:
  churn:
    task: classification
    target_column: churn
    metric: heldout_accuracy
    quality_threshold: 0.7
    stratify: true
    model_name: "{catalog}.{metadata_schema}.churn_model{resource_suffix}"
    model:
      type: voting_classifier
      params:
        base_estimators: [logistic_regression, decision_tree]
        base_estimator_params:
          decision_tree: {max_depth: 2, random_state: 42}
        voting: soft
        calibrate_base_models: true
        calibration_method: sigmoid
        calibration_cv: 2
    tuning:
      strategy: grid
      metric: accuracy
      max_candidates: 2
      search_space:
        logistic_regression__estimator__C: [0.1, 1.0]
```

## Spark scoring and serving endpoints

Voting, stacking and calibrated ensembles can score on Spark workers
(`batch_compute: spark`) and behind serving endpoints. Training first checks
the fitted ensemble as one model: every member, every calibration estimator and
the stacking final learner must be a supported built-in model, and the
ensemble's own predictions must give the same results in batches of any size.
Spark and online-feature projects run this check automatically; an
endpoint-only project enables it with `defaults.probe_pipeline_for_serving: true`.
The check, its evidence and the full model list are described in
[Score on Spark workers](databricks_spark_scoring.md#built-in-recipes-and-model-evidence).

A linear final learner such as the default `logistic_regression` can differ in
the last floating-point digits between batch sizes. The check allows such
[rounding differences](databricks_spark_scoring.md#when-a-built-in-step-or-model-is-rejected)
in probabilities but never a changed label. Voting, including soft voting, and
stacking with a `decision_tree` final learner passed the same checks on real
data.

## Setup questions and defaults for each slot

Detailed setup repeats the ensemble question group for the selected single
model, each competition candidate, and each multi-target branch. It asks for
the member count (default two), each distinct member and the stacking final
estimator where applicable. Regression offers up to 14 member positions and
classification up to 13; installed optional packages and weighted-mode
eligibility still constrain the choices. Selecting two members exposes two
member questions, rather than silently filling every available position.

The wizard does not ask for the settings below. Each slot writes their defaults
into `model.params` in `config/training.yml`; edit them there, or set them in an
answer file before generation:

| Setting | Answer-file suffix | Default |
| --- | --- | --- |
| Voting weight of member `K` | `weight_K` | `1` |
| Classification voting mode | `voting` | `soft` |
| Stacking folds | `cv` | **3** (**5** when omitted from hand-written YAML) |
| Stacking `passthrough` | `passthrough` | `false` |
| Calibrate classification members | `calibrate` (YAML `calibrate_base_models`) | `false` |
| Calibration folds | `calibration_cv` | **3** |

Calibration method defaults to `sigmoid`; edit YAML to select `isotonic`.

The answer-file names follow these patterns, with `N` the candidate/branch
number, `task` equal to `regression` or `classification`, and `K` the member
position:

| Scope | Answer-file prefix | Common suffixes |
| --- | --- | --- |
| Single model | `single_ensemble_<task>_` | `base_count`, `base_K`, `weight_K`, `final` |
| Competition slot | `competition_ensemble_N_<task>_` | Same, isolated to that candidate. |
| Multi-target branch | `branch_N_ensemble_<task>_` | Same, isolated to that branch. |

For the shared slot suffixes `voting`, `cv`, `passthrough`, `calibrate` and
`calibration_cv`, omit the task portion of the prefix. Examples are
`single_ensemble_cv`, `competition_ensemble_2_calibrate` and
`branch_1_ensemble_passthrough`.

For example, `competition_ensemble_2_regression_base_1: "ridge"` selects the
first member of candidate two; its `weight_1` key applies only to voting.
Setup writes these choices into that model's `params`; the editable YAML
is the place for detailed member overrides, calibration and manual nested axes.
Inspect the emitted entry because setup keeps these defaults without asking.
Shared competition search settings and per-branch settings still obey the same
accepted tuning fields and budgets in the
[tuning guide](databricks_tuning.md#competition-budget).
