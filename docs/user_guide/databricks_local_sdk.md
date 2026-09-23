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

## Fit and score a small batch

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
    table=table, version=12, row_keys=("entity_id",),
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
    row_keys=("entity_id",),
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
    row_keys=source.row_keys,
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
    spark, prepared, row_keys=("event_id",),
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
