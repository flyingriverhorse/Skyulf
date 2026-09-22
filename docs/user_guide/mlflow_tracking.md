# Optional MLflow tracking

Skyulf Core keeps experiment tracking separate from training and inference. The
base installation does not import or require MLflow. Install the optional
extra only for jobs that should create tracking runs:

```bash
pip install skyulf-core[mlflow]
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

This integration creates tracking metadata only. Model packaging, registry or
Unity Catalog resolution, Delta batch publication, and Databricks deployment
are separate later stages of the 0.9.0 initiative.
