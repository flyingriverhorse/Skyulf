# Score saved models in this project


The score job loads saved preprocessing/rules and publishes keyed outcomes. It
runs independently from [training](TRAINING.md); changing saved-model rules needs
a new trained version.

## Files and selection

| File | Purpose |
| --- | --- |
| `config/inference.yml` | Source/destination, model selection, append/rebuild and recovery policies. |
| `config/scoring.yml` | Eligibility, output and combined rules, captured by the next fit. |
| `src/features/scoring.py` | Optional custom callbacks selected from scoring.yml. |
| `resources/score.job.yml` | Deployed score graph; recovery/handoff must match configuration. |
| `deployment/variables.yml`, `deployment/targets.yml` | Schedule, compute, retries and notifications. |

This project scores a complete model set. Ordinary runs resolve its champion;
`score_model_version` selects a concrete set version. Configure publication under
`inference.yml` > `model_set`; top-level single-model selectors do not select it.


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

- Default `mode: pre_split` in scoring.yml reuses saved eligibility. Target-reading
  filters need an explicit `skip_target_pre_split_steps: true` before training,
  or a deliberate custom mode. Commented sample rules are inactive.
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

## Choose the saved scoring policy

Edit `config/scoring.yml` before training a new version. `mode: pre_split`
reuses saved pre-split eligibility; `custom` runs its eligibility and output
lists; `combined` runs pre-split first and keeps its first exclusion reason.
The lists start empty. Set `skip_target_pre_split_steps: true` only when
pre_split/combined should omit filters reading the unavailable target.

Built-in eligibility uses `check: required_values` or `check: value_range`;
outputs use `rule: prediction_band`. Each rule has a name, optional string
version (default `"1"`) and params. Outputs declare columns and dtypes.
`combined` is the separate list of cross-model rules: select `operation: weighted_sum` or `operation: weighted_mean` or `custom: scoring.<function>` and declare
required_components. The top-level `mode: combined` concerns eligibility,
not cross-model arithmetic.

Select optional Python helpers with `custom: scoring.<function>`; never add a
Python scoring builder alongside scoring.yml. Older Python-only projects keep
their builders when no scoring.yml exists. Saved versions replay their captured
rules even after editable YAML or Python changes. Spark/endpoint lint, row and
partition probes still apply to custom functions and reused pre-split filters.
