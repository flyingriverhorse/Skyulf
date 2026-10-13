# Increase serving traffic and promote a challenger

Use a daily rollout to introduce a registered pipeline version gradually. The
default policy increases challenger traffic by **10 percentage points after at
least 24 hours at each stage**. After a passing observation at 100% traffic, the
scheduled job automatically promotes that version to the `champion` alias.

This workflow requires an existing champion, a staged challenger with a saved
passing comparison, and two compatible serving packages. Train the challenger
with `promotion_policy: manual_approval` so training leaves the current champion
in place. The rollout owns the later automatic promotion. Both packages must
use the serving-compatible Skyulf runtime used by the controller.

Automatic promotion currently supports one fitted pipeline per version. It
rejects model sets before initialization. Spark-native models and preprocessing
that needs other request rows are outside the existing endpoint certificate.
Model-set traffic composition is available at the lower-level endpoint API, but
the generated daily job requires the single-pipeline approval workflow.

## Decisions and timing

| Observation | Action |
| --- | --- |
| Zero traffic, saved comparison and targeted requests pass | Advance when the initial minimum stage has elapsed |
| Enough fresh requests and mature labels; health and quality pass | Increase by one configured step |
| Too few requests or labels, stale evidence, or an incomplete stage | Hold the current allocation |
| Confirmed error-rate, latency, or quality regression | Return traffic to the retained champion and stop the rollout |
| 100% traffic completes its final passing stage | Recheck the saved heldout evidence and automatically promote |

A missed scheduled run does not cause several increases at once. Stage time
starts when the new routing configuration is verified as ready. Endpoint traffic
and registry aliases are separate: reaching 100% does not itself change an alias.
The final approval also checks that the original champion, challenger, comparison
digest and quality policy still match.

Starting with Core 0.9.4, `health.maximum_latency_ms` bounds **p99 model execution
time**, the duration at or below which approximately 99% of timed captured
requests fall. A single unusually slow request can exceed the threshold without
failing the stage. For example, p99 of 80 ms passes a 100 ms limit even if the
maximum is 900 ms; p99 of 120 ms fails the same limit and requests rollback.
This measures model execution, not total client/network response time.

If `latency_p99_ms` is absent or null, the check falls back to `latency_max_ms`
for older summaries. Recorded zero is valid. An invalid non-null p99, or no
usable latency in either field, holds the stage. The setting name and unit stay
unchanged, but its meaning changes from always checking the maximum; review
your limit before upgrading the controller. Error-rate, minimum request and
label counts, quality, stage duration and incomplete-log checks still apply.

## Prepare an isolated endpoint

Use a new A/B endpoint name. Keep an existing SQL function's pinned endpoint
separate; changing A/B traffic would change that function's prediction contract.

```python
from dataclasses import replace
from databricks.sdk import WorkspaceClient
from skyulf.integrations.databricks.serving import (
    PinnedEndpointSpec,
    build_rollout_endpoint,
    prepare_pinned_endpoint,
    rollout_endpoint_ready,
)

workspace = WorkspaceClient()
registry_options = {"tracking_uri": "databricks", "registry_uri": "databricks-uc"}
champion_spec = PinnedEndpointSpec(
    endpoint_name="risk-gradual-release",
    model_name="models.risk.customer_risk",
    model_version="7",
    logging_catalog="ops",
    logging_schema="serving",
    logging_table_prefix="risk_gradual_release",
)
challenger_spec = replace(champion_spec, model_version="8")
plan = build_rollout_endpoint(
    prepare_pinned_endpoint(champion_spec, **registry_options),
    prepare_pinned_endpoint(challenger_spec, **registry_options),
)
print(plan.config)

# Run creation once. The native create operation rejects an existing name.
created = workspace.api_client.do(
    "POST", "/api/2.0/serving-endpoints", body=plan.config
)
```

Creation is asynchronous. In a later cell, inspect the same endpoint:

```python
endpoint = workspace.api_client.do(
    "GET", f"/api/2.0/serving-endpoints/{champion_spec.endpoint_name}"
)
ready = rollout_endpoint_ready(endpoint, plan, challenger_percentage=0)
print(ready)
```

Continue only when `ready` is true. Both model versions must be loaded, routing
must be exactly 100/0, and the configured inference logging must be intact. Do
not repeat creation after a timeout without inspecting the endpoint first.

## Initialize durable rollout state

Create an explicit MLflow run for this rollout. Use the comparison digest and
quality settings from the challenger's saved training result. The approval
settings below are examples; their metric, threshold and improvement must match
that saved comparison.

```python
from mlflow.tracking import MlflowClient
from skyulf.integrations.databricks.data.admission import SingleWriterAdmission
from skyulf.integrations.databricks.serving import (
    DailyRolloutPolicy,
    MLflowRolloutStore,
    build_rollout_promotion,
    initialize_rollout,
)

registry = MlflowClient(**registry_options)
approval = {
    "model_name": champion_spec.model_name,
    "metric": "heldout_rmse",
    "min_improvement": 0.0,
    "quality_threshold": 5.0,
    "quality_gates": None,
    "max_rows": 10000,
    "max_input_mb": 64,
    **registry_options,
}
comparison_sha256 = "REPLACE_WITH_SAVED_COMPARISON_SHA256"
promotion = build_rollout_promotion(
    plan, approval, auto_promote=True, comparison_sha256=comparison_sha256
)
experiment_id = "REPLACE_WITH_EXISTING_EXPERIMENT_ID"
run = registry.create_run(experiment_id)
store = MLflowRolloutStore(registry, run.info.run_id)
result = initialize_rollout(
    workspace,
    plan,
    store=store,
    policy=DailyRolloutPolicy(),
    admission=SingleWriterAdmission(),
    rollout_id=run.info.run_id,
    promotion=promotion,
)
print(result)
print("rollout run_id:", run.info.run_id)
```

`SingleWriterAdmission` is an explicit ownership contract, not a distributed
lock. Restrict endpoint and alias mutation permissions to the controller's
identity, serialize all its writers, and keep `max_concurrent_runs: 1`. Another
job or UI writer must not modify those resources concurrently. Applications with
multiple writers must supply a shared admission provider to the SDK operations.
There is no native atomic compare-and-swap for an endpoint configuration.

## Enable the daily Bundle job

Generate the project with `include_daily_rollout: "true"` in the template init
configuration. This adds `resources/daily_rollout.job.yml`, its notebook,
`config/serving.yml`, and serving dependencies. Existing projects can regenerate
and review the added files. The job is initially `PAUSED`; its configuration is
initially disabled.

Fill `rollout` in `config/serving.yml`:

- Set `run_id` to the initialized rollout run and copy the comparison digest and
  exact `approval` settings used above.
- Supply representative `smoke_records` matching the admitted request schema.
  The job calls both named served versions directly, including the version with
  zero routed traffic.
- Configure minimum request and label counts, label coverage, maximum error
  rate and the latency limit (p99) in `health`.
- Configure `monitoring` for the concrete challenger version and endpoint.
  `source_table` and `prediction_table` select its inference payload table.
  Use `execution_engine: spark`, an existing `reference_namespace`, and a label
  table with a result-availability timestamp.
- Set `enabled: true` and `exclusive_writer: true` only after establishing the
  ownership contract. Deploy, test one run, and then unpause the schedule.

Serving labels join on `databricks_request_id` and `request_row_index`; retain
these identifiers when collecting outcomes. The adapter uses the actual served
model identity from telemetry and evaluates the current stage's time interval.
It does not recompute historical predictions. Late or missing outcomes hold the
rollout until enough mature labels are available. Quality thresholds come from
the saved comparison; the job does not accept a configurable PASS flag.

The job records the complete observation and verifies its MLflow artifact before
changing traffic. Its output includes `status`, `state.challenger_percentage`,
the decision and an observation digest. A successful final transition also
includes the guarded `promotion` receipt.

## Retry, stop and recovery

Before an endpoint update, the controller persists the exact intended request.
After a timeout or interruption, rerun the job to reconcile that request. A
reconciliation run does not immediately apply another traffic increase. The
stage clock advances only after the expected endpoint revision and routes are
verified.

If the request remains `PREPARED` while the endpoint still shows its old
configuration, the controller holds. It cannot tell whether an uncertain request
will still apply. Inspect the recorded intent and native endpoint operation
before resolving the rollout; do not edit receipt tags or blindly resend the
request. An unexpected endpoint ID, revision, model or routing setting stops
automation rather than overwriting another writer's change.

Pause the scheduled job to stop further decisions. A verified regression rollback
is terminal for that rollout; starting again requires a new explicit rollout.
After a successful promotion, both concrete served entities remain available.
Retiring the previous model or changing SQL pins is a separate operator action.

## Feature updates during a rollout

New feature-bound candidates save their enriched training-time holdout at
`lifecycle/data/holdout.parquet` in the candidate's MLflow run. The saved comparison
binds its hash, owner run, schema, row and memory limits, split membership and
model identity. Lifecycle training reuses its prepared holdout; SDK training
records the same bounded artifact when needed.

Final approval reloads those verified inputs and evaluates the fitted pipeline.
Feature table versions may advance while the rollout runs: current feature
values do not replace historical holdout values. The original source and feature
table identities must still match. Missing or altered artifacts, changed split
metadata, insufficient read budgets or a dropped and recreated table block approval.
Retain the candidate run and its artifacts throughout the rollout and approval.

Older candidates without this saved reference keep strict snapshot reconstruction.
If a feature table has advanced, approval explains that no frozen holdout is
available and remains blocked. Train a new candidate with the updated runtime;
do not edit comparison artifacts to bypass that check.

Online requests still use current published features and enforce their missing
value and freshness policy. Historical holdout success does not replace fresh
serving health, quality or final-stage observations. A shared feature failure can
affect both versions; returning traffic to the champion does not repair that data.
See [online features](databricks_online_features.md) for the serving freshness
contract and supported request shape.
