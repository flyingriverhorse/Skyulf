# Score saved models in this project

The score job loads saved preprocessing/rules and publishes keyed outcomes. It
runs independently from [training](TRAINING.md); changing saved-model rules needs
a new trained version.

## Files and selection

| File | Purpose |
| --- | --- |
| `config/inference.yml` | Source/destination, model selection, append/rebuild and recovery policies. |
| `src/features/scoring.py` | Legacy Python-only policy: build_model_rules returns None, preserving prediction outputs without status fields. |
| `resources/score.job.yml` | Deployed score graph; recovery/handoff must match configuration. |
| `deployment/variables.yml`, `deployment/targets.yml` | Schedule, compute, retries and notifications. |

This project scores one model. `pinned_version` uses `model_version`;
`champion` resolves the controlled alias once. `score_model_version` overrides
either for one run. A concrete pinned candidate need not be champion.


## Commands and receipt

```powershell
python src/tools/preview.py --action score
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run score -t test --profile YOUR_PROFILE
```

Preview checks declarations; it does not load cloud models or prove permissions.
Open **score > Output**; this graph has no `scoring_report` task.
Check concrete model/set and source versions, input/output counts, destination
and Delta receipt. Excluded rows retain keys, null predictions and reasons.
Unchanged replay is a no-op; failed rules/writes do not advance the watermark.

## Critical rules

- This example deliberately saves no scoring policy. The caller validates inputs;
  saved preprocessing and estimator validation still apply.
- `on_model_change` controls a changed model; model-set `on_source_change`
  separately controls UPDATE/DELETE. Ordinary single-model insert scoring rejects them.
- CDF-expiry recovery is opt-in and requires matching graph/notebooks/wheel.
  It does not recover arbitrary permissions, schema or source-identity failures.
- Keep `workflow_contract: "3"` and deployed handoff/recovery parameters aligned.
  Retry scoring independently after promotion; do not repair lifecycle actions.
- This project uses `batch_compute: single_node`.
  Local means whole-frame scoring on selected compute; size each initial/rebuild
  population against `max_rows` and `max_input_mb`.


## Detailed instructions

- [Eligibility, selectors, receipts and publication](https://www.skyulf.com/manual/user_guide/databricks_scoring.html)
- [Distributed inference and worker limits](https://www.skyulf.com/manual/user_guide/databricks_scoring.html#distributed-inference-settings)
- [Monitoring enrollment and references](https://www.skyulf.com/manual/user_guide/databricks_monitoring_enrollment.html)
- [Deployment schedules, time limits and recovery](https://www.skyulf.com/manual/user_guide/databricks_deployment.html)

## Version note

Public links show the latest manual; [README.md](README.md) explains how to match
them to this project's pinned Core release.

## Legacy policy compatibility

This example intentionally has no config/scoring.yml. Its exported
build_model_rules() returns None: training filters do not run during scoring,
and endpoint outputs keep their original prediction schema. The caller validates
request values. Existing saved models replay these captured Python hooks.

New projects use config/scoring.yml. To migrate this example, remove the Python
builder definitions and exports before adding YAML. An empty custom policy adds
scoring_status and exclusion_reason, so review the endpoint schema before
publishing a new version. For row eligibility with saved filters on Spark and
endpoints, use the taxi_fare_robust example.
