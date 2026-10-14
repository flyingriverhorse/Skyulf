# Configure classification decision thresholds

## When you need it

Use a decision threshold when a classifier's saved class decision should differ
from its native prediction. This affects class labels, not probability calibration.
It is independent of promotion `quality_threshold` and of business output bands.
New projects leave this policy off.

## What to edit

Merge this fragment into an existing classification `config/training.yml`, keeping
its model, inputs, source, split and other settings. A named model can override
`defaults.decision_threshold`:

```yaml
defaults:
  decision_threshold:
    mode: auto
    metric: balanced_accuracy
    validation_fraction: 0.2
    random_state: 42
```

New projects keep decision thresholds **off**. Configure `decision_threshold`
on a named model entry in `config/training.yml`. These Python mappings illustrate
the corresponding Core pipeline value; use the equivalent YAML mapping in a
Bundle model entry:

```python
# Native classifier decisions; no calibration partition is reserved.
DECISION_THRESHOLD = {"mode": "off"}

# Binary: predict "churn" when its probability is >= 0.7, including exact ties.
DECISION_THRESHOLD = {"mode": "manual", "positive_class": "churn", "value": 0.7}

# Multiclass: exactly one class wins argmax(probability / class value).
DECISION_THRESHOLD = {
    "mode": "manual",
    "thresholds": [
        {"class": "low", "value": 0.5},
        {"class": "medium", "value": 0.3},
        {"class": "high", "value": 0.2},
    ],
}

# Binary or multiclass: select thresholds on a reserved part of training data.
DECISION_THRESHOLD = {
    "mode": "auto",
    "metric": "balanced_accuracy",
    "validation_fraction": 0.2,
    "random_state": 42,
}
```

Use original target labels, preserving their types (e.g. integer `1` versus
string `"1"`), even when preprocessing encodes the target. The fitted encoder
chain maps these labels to the model's probability columns. Saved local models
return original labels and record them in `manifest.classes`; evaluation uses
the same mapping. This covers LabelEncoder and OrdinalEncoder, including explicit
category order and repeated encoding around resampling. Feature-only encoders
do not change target labels. Multiclass entries must cover every class exactly once, with
positive finite values; they are relative decision weights, not minimum confidence
requirements. The result always has one class, even if all probabilities are low.
Binary manual cutoffs permit both endpoints, zero and one.

Automatic mode reserves 20% of the training partition by default, **before fitting
preprocessing or selecting model parameters**. It fits the model on the remaining
rows, selects thresholds on calibration rows, and saves that fitted model without
refitting on the calibration population. Final holdout labels never select an
estimator, preprocessing state or threshold. Random calibration is stratified;
group CV keeps groups separate; temporal workflows use a chronological calibration
tail and honor the configured CV gap. Every class must occur in both partitions,
otherwise training fails with an actionable error. Small datasets can therefore
require a different split or manual mode.

For random row splitting, `validation_fraction: 0.25` fits the saved model on
approximately 75% of the training rows and uses the remainder only to choose
its decision threshold. With group splitting, the fraction selects **groups**,
not rows: unequal group sizes can produce very different row proportions.
Whole groups stay together; inspect `fitting_rows` and `calibration_rows` in
the threshold evidence for the actual populations.
There is no final full-training refit in this mode. `off` and `manual` do not
reserve this extra calibration population. Full-data refitting is a different
policy and can change the probabilities to which a selected threshold applies.

Supported automatic objectives are `balanced_accuracy`, binary `f1`, `f1_macro`,
`f1_weighted`, and `matthews_corrcoef`. For binary selection, `positive_class` is
optional and defaults to the fitted model's second class. The existing heldout
binary precision/recall/F1 and probability metrics retain that second-class
reporting convention, even when a different label owns the decision cutoff.
Sample weights affect estimator fitting, but calibration objectives and their
reported scores remain **unweighted**, as do the other evaluation metrics.
`f1_weighted` means class-support averaging, not row-weighted scoring. Automatic
threshold selection has no row-cost/sample-weight objective option;
it should not be interpreted as optimizing weighted business costs.
Thresholds change class predictions; they do not calibrate or alter probabilities.
An improved calibration score does not guarantee a better independent holdout score.

CV and competition independently refit the complete model-and-threshold policy
inside every outer training fold. Parameter searches also run inside those folds;
their extra work counts toward `competition_max_trials`. The original model-search
scores in `tuning.json` describe base-model selection, while `cross_validation.json`
and `competition_evaluation.json` evaluate the final decision policy.
`decision_threshold.json` records fitting/calibration counts, selected values,
the positive class and calibration scores. Model reload, registry evaluation,
batch prediction preserve the saved decision rule. For an additional standalone
inference bundle export, pass `use_tuned_thresholds=True` to `build_bundle`.
That separate portable format retains its existing supported-node restrictions;
target-encoder pipelines use the full local artifact saved by the Bundle.

Prediction columns retain stable positional names: `probability_0` belongs to
`manifest.classes[0]`, `probability_1` to `manifest.classes[1]`, and so on.
For example, `manifest.classes = ["no", "yes"]` maps `probability_1` to `"yes"`.
This mapping uses the original labels even when the target was encoded. Do not
infer a class from its spelling or assume the positive class always has index 1
when explicitly configuring a different `positive_class`.

Regression and classifiers without `predict_proba` cannot enable thresholds.
The separate nested-CV binary search `modeling.tune_threshold` is accepted; do
not enable it alongside `manual` or `auto`. An explicit `off` policy leaves that
independently requested search intact. Promotion `quality_threshold` remains a separate
acceptance gate.


## The command

From the generated project root:

```powershell
python src/tools/smoke.py
python src/tools/preview.py --action train
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run train -t test --profile YOUR_PROFILE
```

Use your configured target/profile. Preview checks configuration; only fitting
can verify class support and the actual calibration population. Inspect the
new saved version before [selecting it for scoring](databricks_scoring.md).

## What you should see

Open the training run's **Artifacts > decision_threshold.json** for selected
values, positive class, fitting/calibration counts and calibration scores.
`cross_validation.json` and `competition_evaluation.json` evaluate the complete
policy; `tuning.json` describes base-model selection. See
[training reports](databricks_training_reports.md) for the job tasks and MLflow
locations, and [search and cross-validation](databricks_tuning.md) for fold controls.
Check the saved `manifest.classes` before interpreting `probability_0`,
`probability_1` or later columns. Saved predictions use the selected rule and
original label types after reload. Holdout metrics remain independent evidence.

## Common errors and fixes

| Symptom | Check and fix |
| --- | --- |
| Too few classes in fit/calibration | Supply more representative data, change the split, or choose manual/off mode. Both partitions must contain every class. |
| Manual class is not found | Use the original target label and type, and cover every multiclass label exactly once. |
| An auto threshold consumes more rows than expected | The calibration population is reserved before fitting. With groups, validation_fraction counts groups, not rows. Inspect the saved counts. |
| Probability values did not change | Expected: thresholds change class decisions, not probability estimates. |
| Regression or a classifier without predict_proba is rejected | Keep threshold mode off or use a supported probabilistic classifier. |
| `tune_threshold` conflicts with manual/auto | Choose one policy; explicit off leaves a separately requested nested search intact. |
| Better calibration score, worse holdout score | Calibration optimization is not a promise of better independent model quality. Retain the holdout gate. |
