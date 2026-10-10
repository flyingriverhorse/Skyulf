# Preprocessing diagnostics and context

Use preprocessing diagnostics to check whether a saved transformation behaves
consistently when the same sample is repeated, reordered, or split into batches.
Install `skyulf-core` with the dependencies used to fit the model. The
[complete runnable example](#complete-runnable-example) trains a tiny model only
to make a saved artifact available; diagnosing your own artifact does not retrain it.
The existing step still owns its calculation. Training calls `fit`; inference
calls `apply` with the information saved during training.

## What does context mean?

Context describes **the other input rows needed during inference**. It does not
describe how much data training needed, choose an engine, or change the formula.

| Context | Information required at inference | Example | Current probe behavior |
| --- | --- | --- | --- |
| `row` | This row and saved training state | Saved mean, scaler, category mapping | Compare full, repeated, chunked and reversed requests |
| `group` | All relevant rows in the same group | Subtract this request's customer-group mean | Report `requires_context`; do not run independent chunks |
| `window` | Neighboring or earlier rows, usually in a defined order | Lag or rolling mean | Report `requires_context`; do not invent history |
| `global` | The complete intended input population | Choose duplicates across the whole request | Report `requires_context` if the step is active |
| `unknown` | Not yet declared for this implementation/configuration | An undeclared custom callback | Try sample comparisons; retain `unknown` even when they pass |

These declarations currently guide an explicit diagnostic. They do not create a
Spark grouping/shuffle, sort the data, fetch older records or configure serving.
Training can opt into the diagnostic below. Prediction does not run it.

### Why is a fitted GroupImputer a row operation?

Suppose training learned these income replacement values:

| Segment | Saved mean income |
| --- | ---: |
| A | 200 |
| B | 1200 |

For a new row `{segment: A, income: null}`, `GroupImputer.apply` looks up the
saved A value, 200. That remains 200 whether the request contains one row or a
thousand rows. The training-time aggregation is already finished. Its inference
context is therefore `row`, even though the step's name contains "Group".
Unseen groups use the fitted fallback according to the node's saved policy.

Compare that with a custom function computing
`df.groupby("customer_id")["amount"].transform("mean")` on the incoming request.
For one customer with amounts 10 and 30, the complete group gives 20 for both
rows. Separating those rows into singleton requests gives 10 and 30. This is
`group` context: the request must contain the intended complete groups. Saving
the function in a model does not save future customer transactions.

### Window and global examples

A rolling mean over 10, 20, 30 with a two-row window is 10, 15, 25 when the first
window accepts one row. Independently processing chunks `[10, 20]` and `[30]`
loses the previous value for the last chunk. Declaring `window` makes this
requirement visible. Carry-history mode additionally needs an explicit
continuation session; restarting each request from training history is not
equivalent to a continuous stream. The current probe does not run that session.

### Predictions with continued history

Use the existing carry-history modes of `LagFeatures` and `RollingAggregate`
through `score_pipeline_with_history`. Supply the actual request frame in
the saved engine/schema. Each step retains its own transformed tail; subsequent
requests receive the returned JSON-compatible state:

```python
from skyulf.inference.pipeline_scoring import score_pipeline_with_history

first = score_pipeline_with_history(first_batch, artifact)
second = score_pipeline_with_history(
    next_batch, artifact, history_state=first.history
)
predictions = second.frame
next_history = second.history
```

The first call starts from the saved training seed. For a complete initial
history replay, use `bootstrap_history=True` on that first call to exclude the
seed; do not combine it with `history_state`. Calls neither fetch older rows nor
change the artifact. Saved state belongs to its model digest and engine. Existing
ordering, tie-breaker, per-group limits and late/repeated-row checks still apply.
Empty requests preserve validated history and return nullable prediction columns.
Models without carry steps return `history=None` and reject supplied history.

Persist predictions and returned history atomically. Serialize calls sharing a
history stream or use compare-and-swap in your storage so two writers cannot
overwrite each other's progress. Reusing an old detached state can replay an old
request; this API cannot detect storage races. Databricks incremental scoring
already owns its durable history/receipt path and uses the same session factory.

For custom `group`, `window` or `global` callbacks, pass the complete intended
request, including required context rows, through whole-frame scoring. The wrapper
preserves that request as one frame; it cannot infer group completeness, build
custom history, or admit independent Spark partitions. Built-in carry history
does not provide state for arbitrary custom callbacks.

Deduplication needs all rows in the intended deduplication population; duplicates
in different partitions cannot be discovered independently. Its declared context
is `global`. However, ordinary row-preserving prediction skips the built-in
`Deduplicate` step. The probe follows that same rule and reports
`action: skip_preserve_rows`, `status: skipped`; it does not remove predictions.

## Where do I declare context?

Built-in declarations belong to the existing Skyulf preprocessing implementation.
You do not add a context setting to every built-in YAML entry. Some built-in
configurations/engines are not reviewed yet and will report `unknown`.

For a custom function, declare its actual behavior in the existing factory in
`src/features/preprocessing.py`. For example, replace the shipped `log_feature`
factory with this version; keep its existing `log1p_value` calculation:

```python
def log_feature(column):
    """Add a per-row logarithm with no request-time population statistics."""
    return column_step(
        f"log_{column}",
        log1p_value,
        output=f"log_{column}",
        params={"column": column},
        inference_context="row",
    )
```

In a generated Bundle, select it normally in `config/preprocessing.yml`:

```yaml
version: 1
recipes:
  default:
    - custom: preprocessing.log_feature
      params: {column: income}
```

The YAML selects the factory; the returned step contains the context. It is not
a second YAML definition of the calculation. For `fitted_step`, use the same
keyword when `apply(df, state)` only reads saved state. Do not label a callback
`row` if it recalculates statistics from `df` during prediction.

`filter_step` also accepts this diagnostic keyword. Its pre-split contract still
requires fixed, row-local eligibility rules; `group`/`window`/`global` declarations
do not make context-dependent pre-split filters valid. The preprocessing probe
does not inspect the separate project scoring/pre-split eligibility policy.

For an advanced custom class, add an optional static method to its existing
Applier class, alongside its existing `apply`:

```python
from skyulf.core.capabilities import ExecutionCapability

# Inside your existing Applier class:
@staticmethod
def inference_capability(state, *, engine):
    """Describe only engines and modes supported by this saved implementation."""
    if engine not in ("pandas", "polars"):
        return None
    return ExecutionCapability(engine, "apply", "local", "preserve", "row")
```

Use that example only if the class actually supports both engines and preserves
rows. Otherwise narrow it. Omitting a declaration keeps context `unknown`.
Editing the factory/class affects newly trained artifacts; an existing model
continues using its captured source. A declaration is a claim to test, not a
Spark/REST admission certificate.

## What does `report = probe_fitted_preprocessing(artifact, sample)` do?

It means: **take this already fitted pipeline, try its saved preprocessing on
these example inputs in several ways, and return what happened**.

| Name | What you provide or receive |
| --- | --- |
| `artifact` | A `FittedPipelineArtifact`: the loaded pipeline plus its manifest and saved fitted state. It is not a model-name string, endpoint or unfitted estimator. |
| `sample` | A small pandas or Polars DataFrame matching the saved model input columns, order and dtypes. It normally excludes the target and unrelated record keys. |
| `report` | A Python dictionary containing step names, context, hashes, checks, statuses and failure reasons. It is not predictions or a trained model. |

The artifact is loaded with `load_pipeline(path)` from an existing fitted
pipeline artifact directory, such as one created by `save_pipeline` or
`fit_workflow`. The loader verifies saved payload/package checks. Do not
pass `models:/...`, an endpoint name, or the result of a generic pyfunc loader
directly to this API; those are different objects.

If you already have the artifact, use a matching sample directly. If the model
was trained on merged features, sample those features. The probe does not turn a
raw source table into merged features or rerun upstream joins.

### Complete runnable example

This standalone example creates a tiny fitted artifact so the origin of every
variable is clear. In a real project, replace the training section with loading
your existing artifact; you do not retrain to run the probe.

```python
import json
from pathlib import Path

import pandas as pd

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline

training = pd.DataFrame({
    "segment": ["A", "A", "B", "B", "A", "B"],
    "income": [100.0, 300.0, 1000.0, 1400.0, 200.0, 1200.0],
    "target": [1.0, 3.0, 10.0, 14.0, 2.0, 12.0],
})
pipeline = SkyulfPipeline({
    "preprocessing": [
        {"name": "fill_income", "transformer": "GroupImputer",
         "params": {"columns": ["income"], "group_by": "segment", "strategy": "mean"}},
        {"name": "encode_segment", "transformer": "OneHotEncoder",
         "params": {"columns": ["segment"], "handle_unknown": "ignore", "max_categories": None}},
    ],
    "modeling": {"type": "linear_regression"},
})
pipeline.fit(
    SplitDataset(train=training, test=training.iloc[:0]),
    target_column="target",
)

# Training is finished. Save once; all following checks use this fitted state.
artifact_path = Path("artifacts/context_demo")
save_pipeline(pipeline, artifact_path)
artifact = load_pipeline(artifact_path)

# These are model inputs, including missing income and a previously unseen segment.
sample = pd.DataFrame({
    "segment": ["A", "B", "NEW"],
    "income": [None, 1600.0, None],
})
report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 2))

print(report["status"])
for step in report["steps"]:
    print(step["name"], step.get("context"), step["status"])

# Optional: persist diagnostic evidence locally. No prediction table is written.
Path("preprocessing_report.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
```

Expected summary:

```text
passed
fill_income row passed
encode_segment row passed
```

The sample contains three rows. `chunks:1` makes three separate one-row apply
calls, and `chunks:2` makes one two-row call and one one-row call. Each result is
compared with the corresponding rows from the full-sample result. The learned
segment means and categories remain the ones from `training`.

## How to read the report

### Optional training report

In generated `config/training.yml`, enable the diagnostic explicitly:

```yaml
defaults:
  preprocessing_probe: true
```

The default is `false`. Single-model, competition and multi-target training use the
same saved-model check. Each fit checks its saved and reloaded artifact using at most the first 256 holdout input rows and
uses the probe's 8 MiB frame limit; it does not log the sample values or refit.
The run records `preprocessing_probe.json`, displayed under **Preprocessing
diagnostics** in the training report. The SDK equivalent is
`TrainingSpec(..., preprocessing_probe=True)`, passed to `train_candidate`.
To enable it on an existing specification:

```python
from dataclasses import replace

spec = replace(spec, preprocessing_probe=True)
```

Open the MLflow training run's **Artifacts** tab and select
`preprocessing_probe.json`. In competition and multi-target layouts, inspect
the individual candidate/branch run that fitted the pipeline.

`failed`, `requires_context` and empty-holdout `not_run` remain diagnostic
outcomes. They do not block promotion or change the data identity, split, model,
thresholds or aliases. An MLflow write error follows the existing training
failure path. Custom callbacks remain trusted code with possible external side
effects; the row/byte limits are not a sandbox or execution timeout.

### Status fields

| Field or status | Meaning and next action |
| --- | --- |
| Top-level `status: passed` | All required sample checks and the final feature schema matched. Inspect context and empty-input support too. |
| `checks` | `full`, `repeat`, `chunks:N`, `reverse` and `empty` name the attempted strategy. |
| `failed` + `output_mismatch` | Inspect the named step for batch statistics, order, dtype changes or numerical rounding. |
| `state_mutation` / `input_mutation` | The step changed its saved state or received input during apply. Keep learned state read-only and return transformed data separately. |
| `apply_error` + `error_type` | The existing apply raised. Reproduce the named step locally with trusted data; sample-bearing exception messages are omitted from the report. |
| `invalid_step_contract` | Saved identity/configuration/validation could not be inspected. Check the artifact and the optional validator; do not infer numerical parity. |
| `requires_context` | Supply a separately designed group/history execution path; changing the label to `row` does not resolve the dependency. |
| `skipped` | The normal prediction chain skips this fitted training step. No apply test ran for it. |
| `not_run` | An earlier step failed or required context, so no result is claimed for this step. |
| `empty` check: `not_supported` | Empty input raised. Other checks can pass; this does not promise support for empty requests. |
| `state_validation: node_owned` | The preprocessing owner inspected its saved state. This can be a pandas/Polars diagnostic contract; it does not mean worker admission. |
| `state_validation: unavailable` | No applicable node-owned validator exists for this fitted state. Empirical checks may still pass. |
| `admission: diagnostic_only` | The report grants no distributed or endpoint eligibility. |

Validation of the initial artifact, sample schema and limits can raise an
exception **before** a report exists. For example, wrong columns or more than
`max_rows` are caller errors. Step execution failures normally appear in the
report. The default limits are 256 rows and 8 MiB per frame; the probe rejects
oversized samples instead of silently truncating them. These limits do not cap
the project's full Spark scoring population.

Choose a nonempty sample with realistic values, nulls, unseen categories and
varied group keys. Only bounded pandas/Polars frames with immutable scalar cells
are supported; nested mutable cells are rejected. Comparisons are exact. A
floating-point difference near machine precision may need investigation but
does not by itself demonstrate refitting. No tolerance setting is currently
exposed. Passing finite samples cannot establish behavior for every future row.

A `row` declaration describes the data the operation needs. It does not promise
that every engine/configuration passes exact chunk comparisons. For example,
pandas can choose different output dtypes for integer casts, clipped integer
values or numeric replacements across chunks. The report must retain that
failure even though neither operation needs other rows to calculate its values.

### Practical built-in boundaries

| Step | Meaning for saved inference |
| --- | --- |
| `Casting` | Saved categorical vocabularies can be reused per row. Legacy pandas category inference and coercive fallback conversions can depend on the complete request; the diagnostic reports that context. |
| `GeneralBinning`, `KBinsDiscretizer`, `Winsorize` | Apply reuses training bins/limits. Integer pandas clipping may still fail exact chunk dtype comparisons; use representative model-input dtypes. |
| `ManualBounds` | This is a row-local filter, not an automatically skipped step. In-bound prediction succeeds; a request that would lose rows is rejected to preserve prediction alignment. |
| `DropMissingRows` | Native apply filters rows, but ordinary prediction skips it. Use an imputer for missing scoring values; skipping does not guarantee the model accepts nulls. |
| `Oversampling`, `Undersampling` | These balance training classes using global context and are skipped during prediction. Mixed effects, replacement sampling and arbitrary callbacks can remain `unknown`. |
| `DatasetProfile`, `DataSnapshot` | Fit saves training reports; apply passes prediction data through unchanged. The report or snapshot is not recomputed per request. |
| `TrainTestSplitter`, `Split`, `feature_target_split` | These are training structure helpers, absent from saved `fitted_steps` and from the probe report. No prediction-time splitting occurs. The train/test splitter deliberately has no feature-frame context declaration. |
| `DateFeatures` | New UTC-aware fitted settings can be inspected per row. Legacy non-UTC settings can stay `unknown`; existing generated-name collisions are not renamed by the probe. |
| `PolynomialFeatures` | Apply reconstructs combinatorial terms from saved configuration and feature shape. Its internal sklearn `fit` does not learn means or categories. Empty/null inputs retain their existing errors. |
| `TextCleaning`, `InvalidValueReplacement` | Fixed operations reuse saved settings. Null-only chunks, regex support and dtype changes are engine-specific, so include those cases in your sample. |
| `DummyEncoder`, `LabelEncoder`, `OrdinalEncoder`, `WOEEncoder`, `HashEncoder` | Current category keys reuse deterministic saved settings. Legacy pandas datetime rendering may depend on the batch; refit the whole pipeline to update that contract. HashEncoder empty output can still fail strict schema comparisons. |
| `TargetEncoder`, `WOEEncoder` | Prediction reuses the saved full-data mapping. TargetEncoder's out-of-fold training values deliberately differ from its inference values; WOE requires a binary training target. |
| `KNNImputer`, `IterativeImputer` | Neighbors and prediction estimators come from training. IterativeImputer with active rounds needs request context because a completely null request takes a different initial-fill path. Custom/stochastic modes can stay `unknown`. |
| `PowerTransformer`, power rules in `GeneralTransformation` | Lambdas and scaler values are saved. Existing error handling returns a complete column/frame unchanged after one invalid value, so active power transforms report `global`. |
| `FeatureGeneration`, `FeatureMath`, `FeatureGenerationNode` | Fitted group aggregates are saved lookups, so inference does not need other group rows. Pandas similarity rendering can depend on neighboring datetimes and reports `global`; legacy unpinned similarity stays `unknown`. |
| `ModelBasedSelection`, `feature_selection` | Prediction uses saved selected columns. The facade delegates inspection to its fitted concrete selector; it does not select features again. |
| `IQR`, `ZScore`, `EllipticEnvelope` | Saved limits or detector models filter rows. Normal model prediction rejects a request that would lose rows; these steps are not automatically skipped. |
| `count_vectorizer`, `tfidf_vectorizer`, `hashing_vectorizer`, `tokenizer` | Reuse learned vocabulary/IDF or saved analyzer settings. Outputs are dense. Custom callbacks, native estimator cache mutations and empty-output dtypes need their own diagnostic evidence. |
| `H3Index` | Saved coordinate names and resolution drive the existing H3 calculation. Install and pin the optional `h3` dependency in the scoring environment; the built-in artifact does not automatically capture that package pin. Pandas empty-output dtype differences remain visible. |

`global` can therefore describe error handling as well as population statistics.
For Box-Cox, the valid row in `[2, -1]` can remain untransformed when the negative
neighbor makes the whole request fail, whereas `[2]` alone is transformed. The
probe reports `requires_context`; it does not silently replace this behavior or
claim singleton requests are equivalent. Correct invalid inputs and preserve
request boundaries when using these modes.

These pandas/Polars declarations do not make additional steps eligible for Spark workers
or serving endpoints. Saved-state validation also does not certify every value
for artifact serialization: for example, native Decimal settings may be accepted
by an applier but remain unsupported by sealed-model hashing.

The same boundary applies to temporal scalar values saved in snapshots: the
semantic hash rejects unsupported dates, times, durations and periods rather
than silently treating different values as equal. Native passthrough apply and
binary-payload checks are separate from diagnostic state hashing. NumPy temporal
arrays retain their supported array encoding; no new scalar codec is implied.

A skipped fitted step has no apply checks and reports
`state_validation: unavailable`. An unrecorded split marker has no report entry
at all. Neither result is a successful test of running that training operation
on individual prediction rows.

Built-in owners can inspect saved pandas/Polars state through `validate_inference_state`.
The probe calls that hook on a detached artifact before the existing `apply`;
malformed state produces `invalid_step_contract`. Learned arrays, fixed bin
edges and selected columns are checked, not learned again. This hook is separate
from the stricter fitted-state/configuration contracts used for Spark workers.
Your custom preprocessing still runs through its existing saved apply function;
there is no second implementation of its transformation to maintain.

Run this after training/loading and when changing custom preprocessing or its
dependencies, or enable the optional training report above. It is not a separate
Bundle task, production monitor, accuracy
evaluation, drift check or performance-loss policy. It does not call the model's
prediction method. It tests detached preprocessing state; module globals and
external services accessed by trusted custom code are not isolated.

See [preprocessing placement](preprocessing_placement.md), the
[Databricks Python SDK](databricks_local_sdk.md), and
[Bundle distributed inference](databricks_bundle.md#distributed-inference-settings)
for their separate execution requirements.

## Runnable history continuation

This example fits a lag feature on ordered observations, reloads the artifact,
and scores two new requests. `history_mode="carry"` saves the training tail;
passing the returned history advances that tail without changing the artifact.
The example uses one stream. For multiple entities, set `group_by` and supply
ordered observations for each entity.

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.pipeline_scoring import score_pipeline_with_history
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline

observations = pd.DataFrame({
    "time": range(10),
    "value": np.arange(1.0, 11.0),
    "target": np.arange(1.0, 11.0) * 3,
})
pipeline = SkyulfPipeline({
    "preprocessing": [
        {"name": "lag", "transformer": "LagFeatures", "params": {
            "columns": ["value"], "lags": [1], "sort_by": "time",
            "history_mode": "carry",
        }},
        {"name": "fill", "transformer": "SimpleImputer", "params": {
            "columns": ["value_lag_1"], "strategy": "mean",
        }},
        {"name": "drop_clock", "transformer": "DropMissingColumns", "params": {
            "columns": ["time"], "missing_threshold": None,
        }},
    ],
    "modeling": {"type": "linear_regression"},
})
pipeline.fit(
    SplitDataset(train=observations.iloc[:8], test=observations.iloc[8:]),
    target_column="target",
)
artifact_path = Path("artifacts/history_demo")
save_pipeline(pipeline, artifact_path)
artifact = load_pipeline(artifact_path)
requests = pd.DataFrame({"time": [10, 11], "value": [11.0, 12.0]})

first = score_pipeline_with_history(requests.iloc[:1], artifact)
# JSON round-trip represents transporting the proposed state to another call.
state = json.loads(json.dumps(first.history))
second = score_pipeline_with_history(
    requests.iloc[1:], load_pipeline(artifact_path), history_state=state,
)
expected = score_pipeline_with_history(requests, artifact)
np.testing.assert_allclose(
    pd.concat([first.frame, second.frame])["prediction"],
    expected.frame["prediction"],
)
assert second.history == expected.history

# Independent row/chunk comparisons cannot stand in for this history session.
context_report = probe_fitted_preprocessing(artifact, requests)
assert context_report["status"] == "requires_context"
assert context_report["steps"][1]["status"] == "not_run"
```

After each successful request, commit its predictions and proposed history in
one transaction. A later request must read that committed history. If either
write fails, retain the previous state and retry the same request from it.
Serialize requests for a stream or use compare-and-swap to reject a stale writer;
the function cannot detect another process publishing competing state.

Use `bootstrap_history=True` only when the first request is a complete initial
observation snapshot and should replace the training seed. It cannot accompany
`history_state`. Preserve the configured ordering, unique entity/time keys and
history limits; late or repeated rows are rejected. Complete-group custom
callbacks still require complete groups in each frame, and custom windows still
require their own caller-supplied history. This built-in continuation does not
make those arbitrary callbacks stateful.
