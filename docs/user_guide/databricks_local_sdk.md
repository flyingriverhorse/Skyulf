# Local-engine SDK for Databricks jobs

The first local workflow runs pandas or Polars feature engineering and a fitted
scikit-learn model **inside one Python process**. `runtime="databricks"` means
that process may be a Databricks job task; it does not switch inference to Spark.
The same code also runs with `runtime="local"`. No Bundle project or
`databricks.yml` is needed to use this SDK.

SM-25 provides configuration, artifact selection and preflight. It accepts a
caller-owned, bounded pandas/Polars frame and returns predictions. SM-24a adds
an explicit, versioned UC Delta source reader for selected-period local batches.
SM-15L publishes an explicitly selected period to UC Delta. The separate
incremental runner below discovers new source inserts automatically.

## Train a candidate when labels are ready

`train_local_candidate` reads a **specific Delta table version**, filters a
bounded event window in Spark when temporal splitting is selected. Random
splitting uses Core `DataSplitter` on stable record-key order and needs no dates.
Optional availability filtering excludes unknown results and those after the
independent `result_cutoff` from both fit and holdout.
For temporal splitting, the feature/model pipeline fits only on events before
`holdout_start`; later eligible events are reserved for evaluation. The fit engine can be pandas or
Polars. The run logs the Skyulf config, source and code identities, fitted model,
held-out metrics and `candidate_comparison.json`. It registers a new concrete
model version but never assigns `@challenger` or `@champion`.

```python
from datetime import UTC, datetime
from skyulf.integrations.databricks import LocalTrainingSpec, train_local_candidate

spec = LocalTrainingSpec(
    table="catalog.schema.labeled_events", version=12,
    split_strategy="temporal", filter_unavailable_results=True,
    result_cutoff=datetime(2026, 3, 15, tzinfo=UTC),
    start=datetime(2026, 1, 1, tzinfo=UTC),
    holdout_start=datetime(2026, 2, 1, tzinfo=UTC),
    cutoff=datetime(2026, 3, 1, tzinfo=UTC),
    event_column="event_time", result_available_at_column="label_available_at",
    record_key_columns=("event_id",), input_columns=("feature_a", "feature_b"),
    target_column="target", max_rows=10_000, max_bytes=32_000_000,
)
result = train_local_candidate(
    spark, spec, pipeline_config,
    model_name="catalog.schema.customer_model",
    tracking_uri="databricks", registry_uri="databricks-uc",
    experiment_name="/Users/me/customer-model", run_name="candidate-2026-03",
    artifact_path="/tmp/customer-model", metric="heldout_rmse",
    min_improvement=0.05, engine="polars",
    champion_version="7",  # Pin this concrete version before the job starts.
)
print(result.model_version, result.comparison.eligible)
```

The example uses native Spark timestamp instants; their stored instants are
retained independently of the Spark session timezone. A
missing or later label is not a training label, even if its target value is
already present in the pinned table. The source must have unique nonnull row
keys and distinct feature, target and timestamp columns. A nonempty training
set and holdout of at least two rows each are required. The row and byte limits
bound decoded rows and the local frame, not Spark's wire transfer size. The
caller supplies a concrete `champion_version`; omit it for the first candidate.
Review the comparison report, then use the separate guarded staging/promotion
operations if a human approves a change. If fit, packaging, registration or
comparison fails, no alias moves; a registered but unapproved candidate may
remain for inspection. A scheduling policy is separate from this service.

For string, local-clock or date-only source columns, supply independent rules:

```python
from dataclasses import replace
from skyulf.integrations.databricks import TrainingDateSpec

spec = replace(
    spec,
    event_time_parsing=TrainingDateSpec(
        format="%d/%m/%Y %H:%M:%S", timezone="Europe/Copenhagen"
    ),
    result_time_parsing=TrainingDateSpec(format="%Y-%m-%dT%H:%M:%S%z"),
)
```

Leave the format unset for native timestamp columns. Naive local timestamps
need an explicit source timezone; dates additionally require
`date_only="midnight"`. The default rejects date-only inputs. Ambiguous and
nonexistent local times are rejected, and strings are never guessed.
See the [source-date contract](databricks_bundle.md#source-date-formats-and-timezones)
for examples, boundaries, supported directives and distributed validation cost.
`candidate_training_spec.json` retains both parsing rules for approval replay.

For a table without dates, create the same specification with no time fields:

```python
spec = LocalTrainingSpec(
    table="catalog.schema.labeled_customers", version=12,
    record_key_columns=("customer_id",), input_columns=("income", "age"),
    target_column="target", max_rows=10_000, max_bytes=32_000_000,
    split_strategy="random", test_size=0.2, random_state=42, stratify=False,
)
```

Pass this spec to the same `train_local_candidate` call above with either fit
engine. All targets must be known. Enable `filter_unavailable_results` and set
`result_available_at_column` plus `result_cutoff` if outcomes arrive later;
random splitting still needs no observation timestamp. The returned candidate
and saved spec record holdout membership evidence for approval replay.
See the [split/availability matrix](databricks_bundle.md#choose-the-evaluation-split-and-result-availability)
for the four supported combinations.

## Fit and score a small batch

Candidate training also accepts `cv=LocalCVSpec(enabled=True, folds=3)` from
`skyulf.integrations.databricks.local_cv`. It reuses Core CV on training rows,
refits preprocessing per fold and logs a separate report to the candidate's
MLflow run. Use `method="stratified_k_fold"` for classification; time-series CV
requires an explicit event window and `shuffle=False`. The saved final pipeline
and its protected holdout are independent of the fold fits.

`LocalTrainingSpec(training_sample_rows=10_000, training_sample_seed=42, ...)`
opts into seeded Spark-side selection before local transfer. The limit includes
training and final holdout rows and cannot exceed `max_rows`. Sample membership
is pinned for approval replay. Leaving it null preserves overflow-fail behavior.
See [Bundle CV, sampling and selection](databricks_bundle.md#optional-basic-model-cross-validation)
for full contracts and the Bundle's `train` calendar settings. The Bundle uses
the same action for manual and scheduled runs: null source version resolves
latest once, explicit version pins the snapshot, and rolling windows derive at
invocation. The direct `train_local_candidate` API still takes a concrete
`LocalTrainingSpec` with its source version and any active date boundaries.

Install the same `skyulf-core`, pandas, Polars and scikit-learn versions in the
training and scoring environments. The local artifact records those versions,
the fit engine and the raw/feature schemas. It contains trusted Python pickle;
only load packages from your own controlled producer.

```python
import pandas as pd

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import save_local_pipeline
from skyulf.integrations.databricks import (
    InputSource, LocalWorkflowConfig, ModelSelection, OutputSink,
    prepare_local_workflow,
)
from skyulf.pipeline import SkyulfPipeline

train = pd.DataFrame({
    "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "target": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
})
pipeline = SkyulfPipeline({
    "preprocessing": [],
    "modeling": {"type": "linear_regression"},
})
pipeline.fit(
    SplitDataset(train=train.iloc[:5], test=train.iloc[5:]),
    target_column="target",
)
save_local_pipeline(pipeline, "/tmp/customer-model")

config = LocalWorkflowConfig(
    runtime="databricks",  # or "local" on a workstation
    engine="pandas",       # must match the recorded fit engine
    source=InputSource(kind="caller_frame", max_rows=10_000, max_bytes=32_000_000),
    model=ModelSelection(kind="local_pipeline", path="/tmp/customer-model"),
    sink=OutputSink(kind="return_frame"),
)
prepared = prepare_local_workflow(config)
query = pd.DataFrame({"x": [7.0, 8.0]})
predictions = prepared.predict(query)
assert prepared.preflight.feature_order == ("x",)
print(predictions)
```

The example uses a path within one process. A Databricks job's temporary path
is not a cross-job model store. For separate training and scoring jobs, log the
artifact with `log_local_model`, register that run model explicitly, then select
the registered version in scoring. An alias is read once and converted to a
concrete `models:/name/version` reference; later alias changes cannot alter the
already prepared predictor.

```python
from skyulf.integrations.databricks import ModelSelection

selection = ModelSelection(
    kind="local_pipeline",
    name="catalog.schema.customer_model",
    alias="champion",  # or version="7"; never both
    tracking_uri=tracking_uri,
    registry_uri=registry_uri,
)
config = config.model_copy(update={"model": selection})
prepared = prepare_local_workflow(config)  # read-only registry resolution and load
print(prepared.preflight.model_version, prepared.preflight.model_digest)
```

Obtain `tracking_uri` and `registry_uri` from your job's non-secret settings;
authenticate through the runtime's credential provider. Do not put tokens or
passwords into config or store URIs. Registering or changing an alias is a
separate, explicit operation. The local SDK never promotes a model.

## Preflight and custom pipelines

`LocalWorkflowConfig` is frozen, serializable and has no implicit source
snapshot, period or promotion decision. `preflight_local(config, artifact=...)`
performs offline checks without a registry call, package load or job submission.
It returns issues with a code, category and suggested fix. For a registry
selection, offline preflight reports `model_unresolved`; call
`prepare_local_workflow` when read-only registry access is allowed. That step
loads the pinned package and performs full preflight before returning a
predictor. `local_issues`, `remote_issues` and `remote_checked` keep the two
stages visible. It may deserialize trusted pickle, but creates no cloud resource.
If representative rows are available before submitting a job, pass
`probe_frame=sample` to `preflight_local` or `prepare_local_workflow`. The probe
runs the actual fitted FE and model and reports a `prediction_probe_failed`
issue for an incompatible sample. A passing sample is evidence for that sample,
not a guarantee about every later batch.

```python
from skyulf.inference.local_pipeline import load_local_pipeline
from skyulf.integrations.databricks import preflight_local

artifact = load_local_pipeline("/tmp/customer-model")
report = preflight_local(config.model_copy(update={
    "model": ModelSelection(kind="local_pipeline", path="/tmp/customer-model")
}), artifact=artifact)
for issue in report.issues:
    print(issue.category, issue.code, issue.fix)
```

For custom feature engineering, edit the normal `SkyulfPipeline` preprocessing
configuration, fit it on training data, save it, and compare predictions before
and after loading. The SDK has **no separate node or model allowlist**: it
restores the fitted `SkyulfPipeline` and calls its existing prediction path.
Integration examples cover imputation, scaling, encoding, binning, linear and
logistic regression, and a MinMaxScaler/random-forest combination on both local
engines. Other library combinations remain governed by their own fitted
inference behavior and should receive replay tests before production use.
The package is not eligible for Spark-native, Spark-worker or row-local HTTP
execution. The separate `portable_bundle` selection uses
`InferenceBundle` metadata and its own `predict_local` path for supported
portable FE; selecting the wrong package kind fails preflight.

For example, an advanced pipeline may fit explicit binning followed by encoding
before it is packaged; its saved edges and category positions are then replayed
by the local artifact:

```python
pipeline = SkyulfPipeline({
    "preprocessing": [
        {"name": "bands", "transformer": "CustomBinning", "params": {
            "columns": ["x"], "bins": [0.0, 2.0, 4.0, 6.0],
            "output_suffix": "_band",
        }},
        {"name": "encode_bands", "transformer": "OneHotEncoder", "params": {
            "columns": ["x_band"], "drop_original": True,
            "handle_unknown": "ignore",
        }},
    ],
    "modeling": {"type": "linear_regression"},
})
# Fit with SplitDataset, then save_local_pipeline as in the first example.
```

`InputSource` defaults to 100,000 rows and 128 MiB. Set tighter limits for each
job. `PreparedLocalWorkflow.predict` checks both before running FE and the model.
It preserves the caller's input column order and rejects schema mismatches; it
does not silently collect a Spark DataFrame. On Databricks, supply either a
bounded caller-owned frame or the explicit pinned UC source adapter below.

## Evaluate the held-out split in the model run

Keep labeled test rows outside the fit split. After fitting and loading the
artifact, `evaluate_local_holdout` calls the saved pipeline's normal prediction
path on those raw test features. It returns `heldout_mae`, `heldout_rmse` and
`heldout_r2` for regression; classification returns `heldout_accuracy` and
`heldout_f1_weighted`, plus `heldout_f1` for binary models (the model's second
class is positive). A negative R2 is a valid poor-generalization result, not a
logging failure. The caller chooses the split and logs metrics to the same
MLflow run ID that contains the model artifact:

```python
import mlflow

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks import evaluate_local_holdout, fit_local_workflow
from skyulf.integrations.mlflow.local_model import log_local_model

# `frame` is a bounded, labeled pandas or Polars frame with 1,000 rows.
training, heldout = frame[:800], frame[800:]
artifact_path = "/tmp/customer-model"
artifact = fit_local_workflow(
    {"preprocessing": preprocessing, "modeling": {"type": model_type}},
    SplitDataset(train=training, test=heldout),
    target_column="target",
    artifact_path=artifact_path,
    max_rows=1000,
    max_bytes=4_000_000,
)
metrics = evaluate_local_holdout(artifact, heldout, target_column="target")
with mlflow.start_run(run_name="customer-model") as run:
    mlflow.log_param("heldout_rows", len(heldout))
    mlflow.log_metrics(metrics)
    model_uri = log_local_model(
        artifact_path, run_id=run.info.run_id,
        artifact_path="model", tracking_uri="databricks",
    )
```

The held-out labels are used only for evaluation. Monthly scoring reads raw
features without labels and applies the already fitted FE and model. The
SM-24a live validation (`initiatives/spark_and_mlflow/10-sm24a-heldout-metrics-report.md`)
uses five model configurations with 800 training and 200 held-out rows each.

## Fit and score a pinned UC source

`fit_local_workflow` requires an explicit `SplitDataset`; it bounds the total
rows and in-memory bytes before fitting and saves the same trusted artifact as
`save_local_pipeline`. Tracking and registration remain opt-in calls through
`track_run`, `log_local_model` and `register_model`. They do not move an alias.

```python
from datetime import UTC, datetime

from skyulf.integrations.databricks import (
    InputSource, LocalSourceSpec, LocalWorkflowConfig, ModelSelection,
    OutputSink, prepare_local_workflow, score_local_source,
)

table = "catalog.schema.new_entities"
config = LocalWorkflowConfig(
    runtime="databricks",
    engine="polars",  # must match the registered artifact's fit engine
    source=InputSource(kind="uc_table", table=table, version=12,
                       max_rows=10_000, max_bytes=32_000_000),
    model=ModelSelection(kind="local_pipeline",
                         name="catalog.schema.customer_model", version="7",
                         tracking_uri="databricks", registry_uri="databricks-uc"),
    sink=OutputSink(kind="return_frame"),
)
prepared = prepare_local_workflow(config)  # resolves a concrete model version
period = LocalSourceSpec(
    table=table, version=12, record_key_columns=("entity_id",),
    input_columns=("amount", "city"),  # exact saved raw input order
    period_start=datetime(2026, 1, 1, tzinfo=UTC),
    period_end=datetime(2026, 2, 1, tzinfo=UTC),
    max_rows=10_000, max_bytes=32_000_000,
)
result = score_local_source(spark, period, prepared)
print(result.predictions, result.diagnostics)
```

The reader applies the pinned Delta version, half-open UTC period filter,
column projection, deterministic row-key order and `max_rows + 1` limit before
iterating on the driver. It stops when decoded serialized rows or the resulting
local frame exceed `max_bytes`, and rejects duplicate row keys. This byte guard
measures decoded payload and local memory, not Spark's private wire framing;
very wide individual rows should therefore be excluded or handled by a
separately proven paged transport. The selected source and its limits must match
the prepared workflow. The result keeps row keys outside the model input and
records the source version, period, model digest and concrete model version.
No prediction table is created here. Whole-frame FE retains local pandas or
Polars behavior even though Spark performs the bounded UC read.

## Publish one local-scored month to a UC Delta table

`run_local_batch` keeps feature engineering and model prediction in the saved
pandas or Polars engine. Spark reads the pinned UC source and converts only the
bounded final prediction rows to a DataFrame with the target's explicit schema.
The guarded Delta writer then replaces exactly the requested half-open period.
A second month therefore leaves the first month's predictions and run metadata
unchanged; a backfill must explicitly name the earlier period.

Provision the source, prediction target and admission control table before the
job. The source and target must be different Delta tables. The target must have
the row-key columns with the same Spark types as the source (`long` or
`string`), a `timestamp` period column, the saved model's output columns at
their declared types, and three string columns:
`run_id`, `model_name` and `model_version`.
Here `run_id` identifies the prediction publication, not the MLflow training run;
the incremental writer derives it from the source/model snapshot. Existing test
targets with `__skyulf_*` columns retain their historical schema; create a
new target with these names for the updated writer.

The control table has exactly one row, `target_id STRING` equal to the target
Delta table ID and nullable `owner STRING` initially null. Every publisher
of this target must use that same control table. The job identity needs read
access to the source/model and write access to the target/control tables.

```python
from datetime import UTC, datetime, timedelta
from importlib.metadata import version

from skyulf.integrations.databricks import (
    BatchSpec, InputSource, LocalSourceSpec, LocalWorkflowConfig,
    ModelSelection, OutputSink, prepare_local_workflow, run_local_batch,
)
from skyulf.integrations.databricks.delta_admission import DeltaTableAdmission

source_table = "catalog.schema.scoring_source"
target_table = "catalog.schema.predictions"
control_table = "catalog.schema.prediction_admission"
source_version = 12
source = LocalSourceSpec(
    table=source_table,
    version=source_version,
    period_start=datetime(2026, 1, 1, tzinfo=UTC),
    period_end=datetime(2026, 2, 1, tzinfo=UTC),
    record_key_columns=("entity_id",),
    input_columns=("amount", "city"),
    max_rows=10_000,
    max_bytes=32_000_000,
)
config = LocalWorkflowConfig(
    runtime="databricks",
    engine="polars",
    source=InputSource(
        kind="uc_table", table=source_table, version=source_version,
        max_rows=source.max_rows, max_bytes=source.max_bytes,
    ),
    model=ModelSelection(
        kind="local_pipeline", name="catalog.schema.customer_model", version="7",
        tracking_uri="databricks", registry_uri="databricks-uc",
    ),
    sink=OutputSink(kind="uc_delta", table=target_table),
)
prepared = prepare_local_workflow(config)
spec = BatchSpec(
    period_start=source.period_start,
    period_end=source.period_end,
    as_of=datetime.now(UTC) + timedelta(minutes=2),
    record_key_columns=source.record_key_columns,
    output_table=target_table,
    model_name=config.model.name,
    model_version=prepared.preflight.model_version,
    model_digest=prepared.preflight.model_digest,
    source_version=source_version,
    code_version=version("skyulf-core"),
    run_id="january-2026-v1",
    expected_target_version=0,  # read and pin before submission
    mode="local_pipeline",
)
result = run_local_batch(
    spark, source, prepared, spec,
    admission=DeltaTableAdmission(spark, control_table),
)
print(result.commit_version, result.replayed, result.manifest)
```

Use a new logical run ID and the current target version for the next month's
job. Retry an uncertain job with the *identical* spec, including its original
run ID and expected target version; a successful replay returns the existing
receipt. A new run with a stale expected version fails instead of overwriting
newer results. An empty period fails unless `allow_empty=True` is explicit.
The source snapshot must have been committed by `as_of`. The reader and
result both enforce the configured local row and memory budgets; this does not
measure Spark's private wire bytes. The control-row admission protocol protects
participating writers only; do not let other jobs bypass it or manually clear
an owner while a writer may still be active.


## Score newly inserted rows automatically

`run_incremental_local_batch` selects records by **Delta source commits**, not
by calendar date. On the first run it scores the existing bounded snapshot. On
later runs it reads only inserts after the source version recorded in the last
prediction-table commit. It appends predictions without replacing an older
period. A source `event_time` column is optional; if you want to carry it into
the target, set `period_column="event_time"` in the stable job configuration.
It never filters rows by that column, including late arrivals.

Provision a distinct, initially empty Delta prediction table with the
globally unique key, saved model output columns and
`run_id`, `model_name`, `model_version` string
columns. If carrying an event timestamp, add a matching Spark `timestamp`
column to both tables. Enable Delta Change Data Feed on the source **before**
any changes that the job must consume. Provision the same shared admission
control table used by the explicit-period writer; all target writers must use
that admission authority. The job identity needs source, model, target and
control-table permissions.

```python
from skyulf.integrations.databricks import (
    InputSource, LocalWorkflowConfig, ModelSelection, OutputSink,
    prepare_local_workflow, run_incremental_local_batch,
)
from skyulf.integrations.databricks.delta_admission import DeltaTableAdmission

config = LocalWorkflowConfig(
    runtime="databricks",
    engine="polars",
    source=InputSource(
        kind="uc_table", table="catalog.schema.new_records",
        read_mode="incremental", max_rows=10_000, max_bytes=32_000_000,
    ),
    model=ModelSelection(
        kind="local_pipeline", name="catalog.schema.customer_model", version="7",
        tracking_uri="databricks", registry_uri="databricks-uc",
    ),
    sink=OutputSink(kind="uc_delta", table="catalog.schema.predictions"),
)
prepared = prepare_local_workflow(config)
result = run_incremental_local_batch(
    spark, prepared, record_key_columns=("event_id",),
    admission=DeltaTableAdmission(spark, "catalog.schema.prediction_admission"),
)
print(result.input_count, result.commit_version, result.noop)
```

Keep this configuration for scheduled runs; do not supply a period or source
version on each invocation. A repeat run with no new source inserts makes no
target write. If an append succeeds but its acknowledgement is lost, the next
run reads its committed receipt and continues after that source version.
Source updates and deletes, duplicate keys, a missing or expired change feed,
and an unrecognized target commit fail without advancing the watermark.
Key uniqueness must hold across all runs, not only within one batch. Oversized
increments fail under the same local row/decoded-memory limits; increase the
budget deliberately or handle such a source with a separate distributed path.
The target receipt is in the same Delta commit as the predictions. This
protects participating writers only; changes outside the admission protocol
must be handled explicitly.

## Lag and rolling history across batches

`LagFeatures` and `RollingAggregate` default to `history_mode="batch"`: only
the supplied frame participates. Set `history_mode="carry"` in a preprocessing
step to save a bounded training tail per entity. Put this step **after splitting**.
For example:

```python
{"name": "recent_value", "transformer": "RollingAggregate",
 "params": {"columns": ["value"], "window": 5,
            "sort_by": "observation_time", "group_by": ["entity"],
            "history_mode": "carry",
            "history_max_rows": 1000, "history_max_bytes": 48000}}
```

Declare the observed value, clock and entity in `input_columns`. For the
Databricks source contract, this clock must be distinct from the job's
`event_column` and record keys. Drop or encode non-model columns after the
temporal step. Inputs must be available at prediction time; target-history
forecasting is not supported.
The clock must be numeric or a typed datetime; parse JSON/string timestamps in
an earlier preprocessing step before using them as temporal ordering keys.

- The artifact seed stays fixed. Lag 3 retains three prior rows per entity;
  rolling window 5 retains four. Window 1 retains one ordering marker.
- Training never reads its own saved tail. Temporal holdout and each CV fold
  use only their own training history. Carry mode requires Time Series CV
  (or nested Time Series); skipped gap rows do not enter history.
- Missing or tied entity/time keys and observations at or before the saved
  entity time are rejected. New entities start with empty history. Returned
  rows keep request order. `drop_na` is disallowed; use an imputer instead.
- Each chained temporal step stores its own input context. Context rows do
  not enter subsequent imputer/scaler/model fitting or returned predictions.
- Limits apply across all entities per step: by default 10,000 rows and 1 MiB.
  Exceeding a limit fails explicitly; inactive entities are not silently evicted.

**Incremental Databricks scoring:** the initial complete source snapshot
reconstructs history from that snapshot, without prepending the training seed.
Later increments read `temporal_history` from the last prediction receipt.
The new context, source watermark and predictions share one Delta commit.
A failed write leaves the prior committed context; a retry reads the actual
receipt, and a no-op makes no write. Receipts have a 64 KiB history budget.
A changed model or a prior receipt without history requires a fresh target;
contexts are never silently mixed across models. This is bounded local batch
execution, not Spark worker or streaming state.

**Period scoring:** `run_local_batch(..., history_state=...)` accepts an explicit
earlier context. A successful result's manifest contains the next context.
Period replacement does not automatically choose another period's history.
Retry identity includes the supplied context, so conflicting retries fail.

**Core and backend:** ordinary prediction uses the immutable artifact seed.
For successive Core calls, wrap prediction in
`TemporalHistorySession(immutable_model_id, previous_state)` from
`skyulf.preprocessing.time_series.history`, then persist `session.state`
alongside successful predictions. The backend `/deployment/predict` accepts
`continue_history=true` for the first request and `history_state` thereafter;
its response returns the next state. The caller owns durable storage and
serialization of those requests. The backend does not keep a hidden mutable
history in its model cache. Repeating a request with the same input state is
deterministic. An HTTP error returns no next state.

## Real-data end-to-end example

`skyulf-core/examples/databricks_local_real_taxi_job.py` is a one-time
Databricks notebook using the public `samples.nyctaxi.trips` dataset. It
materializes a bounded copy in an isolated Unity Catalog schema, trains a
Skyulf `SimpleImputer` -> `StandardScaler` -> `OneHotEncoder` ->
`random_forest_regressor` pipeline, saves its full artifact, logs held-out
MAE/RMSE/R2 in MLflow and registers a concrete UC model version. Separate
`score_initial` and `score_append` jobs then call the incremental runner on
200 existing and 100 subsequently inserted trips. Both persist keyed Delta
predictions; the second job checks prior rows and a no-op replay. See the
SM-15I real NYC taxi live report under `initiatives/spark_and_mlflow/` for
run IDs and measured results.

The example's trip duration and dropoff ZIP are known only after a trip, so
it demonstrates retrospective batch fare estimation. Its hard-coded schema,
experiment and workspace folder are test resources; select your own names
before using it elsewhere. The example is not a scheduled Bundle job.

## Saved scoring policies and project assets

Generated projects separate `src/features/pre_split.py`, `preprocessing.py`
and `scoring.py`. Preprocessing fits transformations inside each training fold.
Scoring has a three-way mode and an independent target-filter switch:

```python
SCORING_MODE = "pre_split"           # "pre_split", "custom", or "combined"
SKIP_TARGET_PRE_SPLIT_STEPS = False # fail if reuse requires the actual target
```

In `pre_split` or `combined`, set the second switch to `True` to skip target-reading filters while keeping
other pre-split steps. This never changes the predicted target. For example,
"price is present" is a training-label check; "floor_area is present" can also
be useful when predicting an unknown price. A mixed target/feature filter is
skipped whole, because deleting one field would change its meaning. Fixed edits
are projected onto feature columns. Preview records skipped names and reused steps.

Reused rules use existing Core implementations and survivor guards on a copy.
Accepted original rows go to the model, so its saved normalization runs once.
Filter dependencies must exist in scoring `input_columns`; include a field there
and remove it from model features in preprocessing if necessary. Deduplication
checks the current batch only. Empty recipes in pre_split mode preserve ordinary output.
These choices are saved per model version; changing a local switch does not
change the behavior of an already registered model.

Use `SCORING_MODE="custom"` for custom rules alone, or `"combined"` to run
pre-split first and custom eligibility only on its survivors. First pre-split
exclusion reasons are retained. Custom callbacks receive original accepted inputs;
fixed model transformations still run once. All-excluded batches skip custom
callbacks and the model; combined mode with no pre-split steps still uses custom
rules. Both modes expose the same separate custom sections:

- `eligibility`: checks **before prediction**. A null reason accepts a row;
  a text reason excludes it. The template shows both required-field and finite
  inclusive numeric-range checks, using `feature_value` as an editable example.
- `outputs`: rules **after prediction**. Add fields such as a prediction band.
  They do not change model predictions or select training data.

`build_eligibility_rules()` and `build_output_rules()` configure the callable
paths and parameters; `custom/scoring_custom.py` implements the functions under
BEFORE/AFTER headings. Both lists start empty; the sample dictionaries in
`scoring.py` are commented out. Uncomment and adapt a desired example explicitly.
Selecting custom or combined mode alone does not activate sample rules.
The complete dictionary form
below is also supported (returning `None` explicitly disables all scoring rules):

```python
# src/features/scoring.py

def build_scoring():
    return {
        "eligibility": [{
            "name": "observed_fields", "version": "1",
            "function": "custom.scoring_custom.require_observed_values",
            "params": {"columns": ["feature_value"]},
        }],
        "outputs": [{
            "name": "risk_band", "version": "1",
            "function": "custom.scoring_custom.prediction_band",
            "params": {"column": "prediction", "thresholds": [10.0, 50.0],
                       "labels": ["low", "medium", "high"], "output": "band"},
            "columns": [{"name": "band", "dtype": "string"}],
        }],
    }
```

For classification, the band rule can use `probability_0`, `probability_1`, etc.;
class order comes from the saved artifact manifest. Callbacks are ordinary trusted
project functions saved with the model. Eligibility receives `(frame, params)`
and returns a Series of nullable reason strings: null means eligible. The first
non-null reason wins in configured order. Output callbacks receive
`(frame, predictions, params)` and return exactly their declared columns, types,
rows and index. Supported output types: `float64`, `int64`, `string`, `bool`.
Callbacks receive defensive pandas frames with a RangeIndex for both engines;
Core preprocessing and model prediction retain the recorded engine. Rules cannot
overwrite raw inputs, estimates, keys or publication metadata. Keep callbacks
deterministic; use saved assets instead of mutable files, clocks or network data.

Every source key has an output row. With a scoring policy enabled, that row has
`scoring_status` (`predicted`/`excluded`) and nullable `exclusion_reason`.
Excluded rows have null predictions and business outputs. Receipts include
`predicted_count` and `excluded_count`; `output_count` counts all outcome rows.
An all-excluded increment is a successful atomic publication that advances the
source watermark. Failed callbacks/writes advance nothing. Replays cannot publish
duplicate keys. Existing targets require the exact output schema, so introduce a
new scoring schema using a fresh target. Training, CV and holdout comparison still
use the raw predictor and their independently defined population.

Eligibility precedes lag/rolling processing. Excluded observations never enter
continuation history. All-excluded batches retain the previous history unchanged.
The saved input schema still applies to excluded rows; normalize input types before
scoring rather than relying on eligibility to reinterpret an incompatible schema.

### Deliver lookup files and external Python dependencies

`src/features/assets.json` includes detailed `_help` instructions and a `files`
list, initially empty. Create your data file and add its feature-root-relative
path, for example `"files": ["assets/bands.json"]`. The original plain-list format
`["assets/bands.json"]` also works. Help text is never embedded as model data.
Assets can support pre-split, preprocessing or scoring; they do not automatically
add a pipeline step. Inside saved code, read a declared file as bytes:

```python
from skyulf.inference.project_package import read_project_asset

payload = read_project_asset(__package__, "assets/bands.json")
```

Code, encoded file contents and pins share the bounded 64 KiB source snapshot.
Paths must remain inside the feature package and asset symlinks are rejected.
Undeclared assets, large external model downloads and files outside the package
are not embedded. Replacing a local lookup file cannot change an existing model.

Declare exact `distribution==version` pins in `src/features/requirements.txt`.
Training/load verifies installed versions before executing package code or
unpickling the model. The artifact manifest records the pins; MLflow includes them
in its saved environment. Provision the same dependencies in the training and
scoring job environments; prediction never installs packages. Ranges, URLs,
recursive requirement files, options, extras and environment markers are rejected.

## Train independent target models in one job

Choose `training_layout=multi_target` when generating a project to use the existing
`train` job for several named training branches. Configure branches in
`src/modeling/branches.py`. The default `single_model` layout retains the ordinary
training and lifecycle graph. Multi-target setup asks shared source, key, limit,
compute and training schedule questions; target/model/search/CV/split/quality
settings belong in branches.py.

A branch has its own target, input columns, preprocessing package, estimator,
tuning/CV settings, quality metric and registered model name. For example, a
customer dataset can train a purchase classifier, a spending regressor and a
visit-count ensemble. Each model remains useful independently. Different targets
have different metrics; their scores are not ranked against each other.

The coordinator resolves one Delta table version before training starts. Branches
read that same immutable snapshot sequentially within their individual row/byte
budgets. Missing labels are excluded separately per target before sampling and
splitting. A missing spending label cannot remove an otherwise labeled purchase
example. All target columns are forbidden as model inputs to prevent cross-target
leakage. The existing training service fits learned preprocessing inside training
folds and keeps the final holdout separate.

MLflow records a parent run and a linked run for each trained branch. The parent
saves the exact branch plan and progress; child runs retain their fitted pipeline,
CV/tuning evidence, holdout metrics, dependency pins and immutable model version.
With a model set enabled, every candidate is compared with its corresponding
component in the one champion set pinned before training. Independent component
aliases cannot change that baseline. Train-only branches retain their own pinned champions.

All branches are required. If one fails, execution stops and the parent is failed;
earlier immutable candidate versions remain available for inspection. A complete
result is written only after all branches succeed. Replaying a saved plan keeps
the source version and split/sample settings; it is a new attempt and can create
new model versions. It is not an exactly-once registration retry.

Individual branches require `promotion_policy=manual_approval` and
`score_handoff=disabled`; activation is controlled by the complete set's separate
`promotion_policy`. Newly generated projects enable
`src/modeling/model_set.py`: its `build_model_set()` factory declares the set's
registered model name, prediction table, publication settings and shared rule path. Returning
`None` disables set packaging; older projects without this file remain train only.
The project still has two jobs. Training registers the complete set candidate;
training nominates the complete set as `challenger`. Automatic set activation can
move the set champion, while component aliases stay unchanged. A later candidate
moves the displaced contender to `previous_challenger`. The multi-target score job uses
`score_models.py` to select one complete saved set.

### Activate and score a coherent model set

The set copies each component's exact fitted artifact and records its concrete
model name, version and digest. It also captures `src/features/`, the combined rule
configuration and typed record keys. Scoring and approval load these saved assets;
editing project files cannot change an existing set. Packaging changed rules
creates a new set candidate and preserves earlier registered packages.

Each newly registered Bundle set version exposes `model_set_model_count` and
`model_set_<branch>_name`, `model_set_<branch>_version`, `model_set_<branch>_type`
tags for quick inspection. The type identifies the selected estimator or ensemble,
including when tuning was used. Long qualified names continue in `_name_2`,
`_name_3`, etc. These tags are display metadata; the saved manifest remains the
execution authority. Existing registered versions are not automatically backfilled.

In `features/scoring.py`, `build_model_rules()` configures each model's eligibility
and output rules, while `build_combined_rules()` configures calculations across
model results. Both stages execute during multi-model scoring. The latter's
default empty list adds no combined business rules. Each
component retains its namespaced predictions and exclusion outcomes. Optional
rules declare a named function, rule version, parameters, output column types and
`required_components`. Functions receive `(inputs, predictions, params)` and
return a pandas DataFrame containing exactly their declared output columns. A
profit rule can require revenue and cost without depending on an unrelated churn
component. Excluded required predictions make that rule ineligible; missing
predictions are never substituted with zero.

`inputs` contains only the set's declared keys and component input columns.
Declare rule Python dependencies as exact pins in
`src/features/requirements.txt`; the saved set includes them in its MLflow
requirements and rejects pins that conflict with a component.
Legacy projects using `composition_config` and `src/composition/` remain readable;
existing saved artifacts keep their original source without migration.

### Choose output storage and consumer views

Multi-target initialization asks `model_set_name`, `model_set_output_mode` and a custom physical
table name. The generated `modeling/model_set.py` contains the editable
`publication` settings. Modes are:

| Mode | Stored values | Consumer access |
| --- | --- | --- |
| `all` | Every model output and combined rule result | One prediction table |
| `combined_only` | Combined results, keys, rule outcomes and set provenance | One prediction table |
| `separate_views` | All results, written once | Selected model views and a combined-result view |

Leaving names blank uses `<project_name>_set` for the registered model set,
`<project_name>_set_scores` for the physical result table,
`<project_name>_predictions_<branch>` for model views, and
`<project_name>_business_results` for the combined view. Custom names replace
these defaults. Registry names use the metadata schema; tables/views use the
output schema. The deployment suffix is added to both.

`combined_only` still computes all models and their scoring policies. It requires
saved combined rules; an empty rule list fails explicitly. Changing this storage
schema requires a new compatible table. Model output values are not stored in
this mode, even though needed for the combined calculations.

Only `separate_views` asks for `model_view_prefix` and `combined_view_name`.
Blank names use project defaults. Explicit names gain the active target catalog,
output schema and resource suffix. Fine-tune names and selection in the factory:

```python
"publication": {
    "mode": "separate_views",
    "model_views": {
        "revenue": "{catalog}.{output_schema}.revenue_predictions{resource_suffix}",
        "cost": "{catalog}.{output_schema}.cost_predictions{resource_suffix}",
    },
    "combined_view": "{catalog}.{output_schema}.profit_results{resource_suffix}",
}
```

`model_views=None` selects every saved branch using `model_view_template`;
an explicit mapping selects only listed branches; `{}` selects none.
Combined views exist only if the saved set has combined rules. Every view keeps
record keys, relevant output/status fields and set provenance. Branch output
names retain their prefix. These are views of one Delta table, not independently
written prediction copies. Setup verifies or creates all views before the data
write. A setup failure can leave views of empty/previous data, but no new subset
of predictions is published. Unrelated objects are never overwritten. Existing
view definitions must match their recorded projection; use new names to change
the source or columns. Removing configuration does not delete catalog objects.

### Per-component quality gates and automatic set activation

Bundle initialization asks `model_set_promotion_policy` for multi-target projects.
The default is `manual_approval`; `automatic` validates and activates a passing
complete set after training. Existing projects can select this in the dictionary
returned by `src/modeling/model_set.py`:

```python
"promotion_policy": "automatic",
```

In each branch's `workflow` dictionary in `src/modeling/branches.py`, define its
own task-appropriate metric and limits. For example, a revenue regressor can use:

```python
"metric": "heldout_rmse",
"quality_threshold": 10.0,       # RMSE must be <= 10.
"quality_gates": {"heldout_r2": 0.8},  # R2 must also be >= 0.8.
"min_improvement": 0.0,          # Replacements must strictly improve RMSE.
```

A classifier can instead use `metric="heldout_f1"` and `quality_threshold=0.8`.
Ensembles use the same task-specific policy. These are illustrative limits, not
recommended thresholds for every dataset. Every branch must have a primary
`quality_threshold` for activation; automatic mode checks its presence before training.

The first set must pass all absolute gates. A replacement must additionally
improve **every** component over its counterpart in the pinned champion set by
`min_improvement`; ties fail even when that value is zero. All models are
evaluated on their saved heldout rows, separately for each target. Policies,
comparison digests and the expected set champion are frozen into the package.
Changing project files cannot relax an existing candidate's gates. Changed
policies require training a new candidate. Set replacements require the same
branch names and registered component names; use a new set for a changed layout.

An automatic quality failure leaves the candidate available and the champion
unchanged. The notebook result and `model_set_quality/.../decision.json` identify
every failed component and its metrics. Execution, missing evidence and stale
champion errors fail the job. Successful approval also requires each saved model
and business rule to execute on representative scoring input. Scoring remains a
separate job after activation.

Manual approval enforces the same gates. After reviewing evidence, run the training job with
`lifecycle_action=approve`, the set's `candidate_version`, and
`expected_champion_version` (`none` for first activation). Approval runs every
saved component and rule against one bounded scoring-source snapshot and requires
each to produce at least one result, then rechecks saved holdout quality under the
same alias admission. Save the returned promotion receipt. To restore its complete prior set,
use `lifecycle_action=rollback`, `promotion_receipt_json` and the expected current
champion version. Approval clears the promoted set's `challenger` alias and saves
the former champion as `previous_champion`. `previous_challenger` records displaced
candidates, not the former champion.

To explicitly reject the current candidate, run the same training job with
`lifecycle_action=reject`, `candidate_version`, `expected_champion_version`, and a
nonempty `rejection_reason` (at most 256 UTF-8 bytes). Rejection uses the frozen set
identity and controlled nomination receipt; it does not run training, scoring, or
require passing quality gates. The rejected version remains `challenger` for
inspection until a newer candidate replaces it. Its `approval_status=rejected`
and `approval_reason` tags block later approval. An identical repeat returns the
original rejection receipt; changed candidates, champions or reasons fail.

Rollback restores the whole previous set while preserving an unrelated current
challenger and its history. A stale or manually changed alias blocks the operation.

Previously saved sets remain scoreable. Sets without saved quality evidence must
be retrained and packaged before approval through this updated Bundle flow.
The low-level SDK retains explicit functional-only approval for legacy packages;
new quality-bound packages require controlled nomination and the quality validator.
Rollback verifies the
recorded successful evidence and restores the complete prior set.

Run the score job after approval. It resolves the set champion once, or uses an
explicit `score_model_version`, and joins component outcomes by unique non-null
integer, string or boolean keys. One final Delta publication contains the complete
selected result and set provenance. A component or rule exception fails the job
before a new data commit; previous successful output remains intact.
Repeated committed source versions are no-ops; model changes follow the configured
append or full rebuild policy. Both jobs use `max_concurrent_runs=1` and require
exclusive ownership of set alias changes and prediction-table writes.

Append mode preserves older rows and their original set identities. Full rebuild
atomically replaces the compatible output table when the selected set changes,
even without new input. A transition involving temporal carry history requires
full rebuild so lag/rolling state is reconstructed from the complete snapshot.
Schema changes require a separate compatible output table. Source updates and
deletes cannot be consumed as incremental inserts.

For example, inserting a new `id=102` after scoring `id=101` appends a new
prediction. Updating a feature or deleting the already-scored `id=101` instead
fails the next incremental batch: there is no update/delete reconciliation of
previous predictions under the default `source_change_policy="reject"`.
For model-set scoring, select `rebuild_on_change` in Bundle setup or the
`modeling/model_set.py` factory to reuse the complete snapshot rebuild when
updates/deletes are observed, even without a model-set change. The selected set
rescores ALL current rows, recomputes model/combined rules and resets temporal
history. Deleted records disappear. This can replace earlier predictions from
older sets; new inserts alone still append. No model training is involved.

`model_change_mode` independently controls what happens when the model selection
changes. Source-triggered rebuilding respects the full snapshot row/byte limits
and commits results/history once after successful computation. Empty snapshots
clear the table and history. A failed calculation, exceeded budget, missing CDF
history or permission error preserves prior output. Only recognized readable
source changes trigger recovery. Receipts record `source_change_policy`,
`source_rebuilt` (source-triggered rebuild) and `write_mode`.
This setting applies to the model-set adapter; the standalone single-model
incremental scorer still rejects source updates/deletes.


### Select preprocessing and pre-split recipes independently

Keep custom implementations in `src/features/custom/`. In `preprocessing.py`,
`build_preprocessing(recipe="default")` selects an ordered Core/custom step list.
`pre_split.py` independently exposes `build_pre_split_steps(recipe="default")`.
Select the two names at branch level (beside its `workflow` overlay):

```python
"preprocessing_recipe": "example_frequency",
"pre_split_recipe": "example_complete_inputs",
```

Another branch can select `example_imputer` with `none`, while a third selects
`example_imputer_frequency` with the same `example_complete_inputs`. No feature-package copy is needed.
The shipped starters use `feature_value` and `category`; adapt their columns or
add your own recipe function to the relevant builder's mapping. `example_frequency`
uses only the custom encoder, `example_imputer` uses only the Core mean imputer, and
`example_imputer_frequency` runs imputation before encoding. `example_complete_inputs` requires at least
one of those inputs; `none` produces an empty list. Both default recipes remain
empty until explicitly configured.

Keep `workflow.pipeline.preprocessing` empty in branches.py: the selected Python
builder supplies the steps. `features_path` still selects the whole package and
defaults to `../features`. All branches may share it while selecting different
recipes. Learned values remain separate per model/fold. Pre-split scoring reuse
uses the selected filter list; target-dependent skip policy still applies.

Selectors are optional. Omitting one calls that builder without arguments,
preserving older project factories. An explicit name requires a builder accepting
`recipe=...`; missing or misspelled names fail before data reads. The saved code
binds selected names so fresh-process model loading and training-plan replay use
the exact original recipe even after the editable files change.
