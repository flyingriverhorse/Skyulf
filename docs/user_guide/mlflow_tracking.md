# Optional MLflow tracking

Skyulf Core keeps experiment tracking separate from training and inference. The
base installation does not import or require MLflow. Install the optional
extra only for jobs that should create tracking runs:

```bash
uv pip install "skyulf-core[mlflow]"
# or, from this repository:
uv pip install -r requirements-mlflow.txt
```

Use `TrackingConfig(enabled=False)` for a local or Spark job that does not need
tracking. The returned handle has the same logging methods, but they are no-op
and do not construct a client or make a network call:

```python
from skyulf.integrations.mlflow import TrackingConfig, track_run

with track_run(TrackingConfig(), run_name="offline-fit") as run:
    # The training code remains unchanged when tracking is off.
    run.log_metrics({"rmse": 0.5})
```

Enable tracking with a tracking URI and, optionally, a named experiment. The
adapter uses a client-bound run ID instead of MLflow's process-global active
run. This makes concurrent jobs independent and leaves a caller-owned active
run open:

```python
config = TrackingConfig(
    enabled=True,
    tracking_uri="sqlite:///mlflow.db",
    experiment_name="skyulf-training",
)

with track_run(config, run_name="fit-2026-09") as run:
    run.log_params({"engine": "pandas", "model": "random_forest"})
    run.log_metrics({"rmse": 0.42})
    run.set_tags({"stage": "validation"})
    run.log_config({"features": ["age", "income"], "target": "label"})
```

`log_config` is explicit. Skyulf does not automatically serialize an entire
pipeline configuration, input rows, or environment variables because those
may contain sensitive data. The artifact receives a deterministic SHA-256
parameter named `config_sha256`.

The default `failure_policy="raise"` propagates tracking failures. Use
`failure_policy="warn"` when the model result must be preserved even if the
tracking service is unavailable; inspect `run.tracking_error` and propagate it
to your job metadata if the surrounding runner has such a field. The adapter
does not write backend job metadata itself. A successful context terminates its
own run as `FINISHED`; an exception in the training body terminates it as
`FAILED`.

This adapter creates tracking metadata. See [MLflow models](mlflow_models.md)
to package a fitted pipeline or portable inference bundle in the same run, and
[MLflow registry](mlflow_registry.md) to register and resolve model versions.
[Databricks publication](databricks_batch.md) has a separate data-write contract.

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

For custom tracked code, `run.log_inputs(dataset_inputs)` accepts native
`mlflow.entities.DatasetInput` records and writes to that handle's explicit run
ID. It follows the same disabled-tracking and failure-policy behavior as the
other logging methods. See the native
[MLflow client API](https://www.mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.log_inputs)
for the dataset input format.

## Training preprocessing diagnostics

Candidate training can log a saved-state diagnostic by setting
`TrainingSpec(..., preprocessing_probe=True)`, or `defaults.preprocessing_probe:
true` in a generated Bundle's `config/training.yml`. It defaults to off. The
training run's **Artifacts** tab contains `preprocessing_probe.json`; competition
and multi-target layouts record it on the run that actually fitted each pipeline.
The training report displays it under **Preprocessing diagnostics**.

The check uses the saved/reloaded artifact and at most the first 256 holdout rows,
with an 8 MiB input/output limit. It neither refits nor controls promotion.
See [preprocessing diagnostics and context](preprocessing_context.md) for the
runnable example and interpretation of `passed`, `failed`, `requires_context`,
`not_run` and unsupported empty input. MLflow artifact-write failures retain the
training workflow's failure policy.
