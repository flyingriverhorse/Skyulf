# Track a training run with MLflow

Use this guide to fit a tiny regression model and save its measured holdout
metrics, parameters and chosen configuration to a local MLflow experiment.
The result is an inspectable run ID; packaging and registration are separate
[next steps](mlflow_models.md).

## Install and choose a store

Use Python 3.12 or newer and a virtual environment:

```bash
uv pip install "skyulf-core[mlflow]"
```

For repeatable jobs, pin the published Skyulf release and its resolved dependencies
in your environment lock, and use the same versions when loading models.
Repository development examples require the matching checkout/build when they
use APIs newer than your installed release. Do not infer release availability
from the version printed by an editable development install.

## Fit, evaluate and log real results

Run this entire block in an empty working directory. It creates `mlflow.db`
and the artifact storage managed by MLflow; keep both to inspect the run later.
No Databricks account or Spark session is needed.

<!-- skyulf-example: native tracking -->
```python
from pathlib import Path

import pandas as pd
from mlflow import MlflowClient

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.pipeline_evaluation import evaluate_holdout
from skyulf.integrations.mlflow import TrackingConfig, track_run
from skyulf.pipeline import SkyulfPipeline

frame = pd.DataFrame({
    "x": [1., 2., 3., 4., 5., 6., 7., 8.],
    "target": [2., 4., 6., 8., 10., 12., 14., 16.],
})
training, heldout = frame.iloc[:6].copy(), frame.iloc[6:].copy()
pipeline_config = {
    "preprocessing": [],
    "modeling": {"type": "linear_regression"},
}
tracking_uri = "sqlite:///" + (Path.cwd() / "mlflow.db").as_posix()
config = TrackingConfig(
    enabled=True, tracking_uri=tracking_uri,
    experiment_name="revenue-demo", failure_policy="raise",
)
with track_run(config, run_name="linear-baseline") as run:
    pipeline = SkyulfPipeline(pipeline_config)
    pipeline.fit(SplitDataset(train=training, test=heldout), target_column="target")
    save_pipeline(pipeline, "artifacts/revenue")
    metrics = evaluate_holdout(
        load_pipeline("artifacts/revenue"), heldout, target_column="target"
    )
    run.log_params({"model": "linear_regression", "training_rows": len(training)})
    run.log_metrics(metrics)
    run.set_tags({"dataset_id": "synthetic-v1/holdout-rows-7-8"})
    run.log_config(pipeline_config)
    run_id = run.run_id

stored = MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
assert stored.info.status == "FINISHED"
assert stored.data.metrics["heldout_rmse"] < 1e-10
assert "config_sha256" in stored.data.params
print(run_id, stored.data.metrics)
```

Expect a concrete run ID and `heldout_rmse` close to zero, alongside the other
available regression metrics. These are measured values from the saved model,
not example numbers sent directly to the tracker. `save_pipeline` writes a local
artifact; this block uploads only the explicit tracking metadata and config.
Use [model packaging](mlflow_models.md) to upload a loadable model too.

Start the UI from the same directory, then open the printed URL:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Choose **Experiments > revenue-demo > linear-baseline**. Inspect **Metrics** for
`heldout_rmse`, **Parameters** for `training_rows` and `config_sha256`, and
**Artifacts > config.json** for the configuration. This manual example does not
populate **Inputs** because it does not call `log_inputs`.

## Turn tracking off or choose a failure policy

The default is `TrackingConfig(enabled=False)`. The handle exposes the same
methods but constructs no client and makes no tracking network calls:

<!-- skyulf-example: native tracking -->
```python
from skyulf.integrations.mlflow import TrackingConfig, track_run

with track_run(TrackingConfig(), run_name="offline-fit") as offline:
    offline.log_metrics({"example_metric": 1.0})  # discarded
assert offline.run_id is None
```

| Choice | Default and effect |
| --- | --- |
| `enabled` | `False`; no tracking client or run is created. |
| `tracking_uri` | `None`; enabled tracking uses MLflow's configured store. Set it explicitly in jobs. |
| `experiment_name` | `None`; uses the default experiment. A supplied name is looked up or created. |
| `failure_policy` | `"raise"`; propagate tracking failures. `"warn"` preserves execution and records `run.tracking_error` for the caller to inspect. |

The adapter binds every call to its own run ID, leaves a caller-owned active
run open, and marks its run `FINISHED` on success or `FAILED` when the training
body raises. It does not write backend job metadata. Explicit model-packaging
failures still propagate; `warn` is not a model-publication retry policy.

`log_config` is explicit and records `config_sha256`. It does not automatically
serialize the whole pipeline, input rows, secrets or environment variables.
Review anything you pass to params, tags and config before logging it.

| Symptom | Action |
| --- | --- |
| No run appears | Confirm `enabled=True`, the tracking URI and selected experiment. |
| Authentication/store error | Configure access to the chosen store; use `warn` only if missing tracking is acceptable and record `tracking_error`. |
| Run is failed | Inspect the training exception; tracking does not make a failed fit successful. |
| No model in Artifacts | Call `log_pipeline_model` or `log_model`; fitting/tracking alone does not package it. |
| No datasets in Inputs | Manual tracking requires explicit `log_inputs`; candidate training logs them automatically as described below. |

## Training datasets in Inputs

Databricks candidate training records two native MLflow dataset inputs on the
run that produces the model: `training` and `holdout`. This happens automatically
for `train_candidate` and generated Bundle training jobs, with either pandas or
Polars. Calling `SkyulfPipeline.fit` alone does not create a tracking run.

Open the registered model version, follow its source run, then open **Inputs**.
For a model competition, each fitted candidate has its own inputs and the winning
model's parent run records the selected candidate's inputs. For multiple targets,
inspect the individual model's run; the orchestration parent does not fit a model.

| Evidence | Where to inspect it |
| --- | --- |
| Actual fitted and held-out row counts, schema and digest | The `training` and `holdout` dataset entries. Counts describe the bounded splits after source filtering and sampling, rather than the entire Delta table. |
| Original Delta table and pinned version, including version `0` | Each dataset's source metadata. Native Unity Catalog identity is included when captured inside Databricks. |
| Model feature names, target and optional weight column | Input tags `skyulf.feature_columns`, `skyulf.target_column` and `skyulf.weight_column`. A target or weight column is not an input feature merely because it appears in a dataset schema. |
| Split, sampling, source identity and feature lookup evidence | **Artifacts > dataset_inputs.json**. Each input's source metadata references this artifact and its SHA-256 digest. |

The dataset represents the frame passed to fitting or final evaluation, before
the pipeline's fitted preprocessing transforms it. Sample weights are supplied
separately to fitting; their column role and existing weight evidence remain in
the provenance. Cross-validation folds do not create a fictitious validation
split in this view.

Inputs are metadata, not an additional upload of every data row or a backup of
the source table. The native Delta source identifies the originating snapshot;
loading that source alone does not reconstruct filtering, sampling or split
membership. Keep the referenced Delta versions and the run's existing artifacts
for reproducibility. Later table updates do not rewrite previously logged inputs.

This describes the **Inputs entries**, not every artifact a workflow may save.
Candidate training with a feature-lookup binding separately saves enriched
holdout rows in **Artifacts > lifecycle/data/holdout.parquet** for comparison-bound
approval replay. Saved model state and explicitly enabled reports can also
contain data-derived values. Review artifact access/retention alongside source
permissions; an Inputs digest is not a promise that the run contains no data.

For custom tracked code, `run.log_inputs(dataset_inputs)` accepts native
`mlflow.entities.DatasetInput` records and writes to that handle's explicit run ID. It follows the
same disabled-tracking and failure-policy behavior as the other logging methods. See the native
[MLflow client API](https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.log_inputs)
for the dataset input format.

## Training preprocessing diagnostics

With `TrainingSpec(..., preprocessing_probe=True)` or `defaults.save_preprocessing_diagnostics:
true` in a generated Bundle, the run that fitted each pipeline saves
`preprocessing_probe.json` in its **Artifacts** tab. The fitting task's Output
(`train_and_tune` or `train_<name>`) shows the result; the final
`training_report` shows only a status column and link. The
[preprocessing diagnostics guide](preprocessing_context.md) explains the check
and every status.
