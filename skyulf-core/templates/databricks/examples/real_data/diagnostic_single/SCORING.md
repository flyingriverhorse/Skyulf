# Score saved models in this project

The score job loads saved preprocessing/rules and publishes keyed outcomes. It
runs independently from [training](TRAINING.md); changing saved-model rules needs
a new trained version.

## Files and selection

| File | Purpose |
| --- | --- |
| `config/inference.yml` | Source/destination, model selection, append/rebuild and recovery policies. |
| `src/features/scoring.py` | Eligibility, business outputs and their callbacks, captured by the next fit. |
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

- Default `SCORING_MODE="pre_split"` reuses saved eligibility. Target-reading
  filters need an explicit `SKIP_TARGET_PRE_SPLIT_STEPS=True` before training,
  or a deliberate custom mode. Commented sample rules are inactive.
- `model_change_mode` controls a changed model; model-set `source_change_policy`
  separately controls UPDATE/DELETE. Ordinary single-model insert scoring rejects them.
- CDF-expiry recovery is opt-in and requires matching graph/notebooks/wheel.
  It does not recover arbitrary permissions, schema or source-identity failures.
- Keep `workflow_contract: "3"` and deployed handoff/recovery parameters aligned.
  Retry scoring independently after promotion; do not repair lifecycle actions.
- This project uses `inference_mode=spark`.
  Spark requires admitted pandas-fitted artifacts and exact worker dependencies.
  The generated environment manager is `local`; prediction batches default
  to 10000 rows. Arbitrary callbacks and temporal carry are not admitted.
  Training lints and probes saved eligibility/output rules; check them first with
  `python src/tools/preview.py --action train` (see https://www.skyulf.com/manual/user_guide/databricks_spark_scoring.html#when-the-probe-fails).


## Detailed instructions

- [Eligibility, selectors, receipts and publication](https://www.skyulf.com/manual/user_guide/databricks_scoring.html)
- [Distributed inference and worker limits](https://www.skyulf.com/manual/user_guide/databricks_scoring.html#distributed-inference-settings)
- [Monitoring enrollment and references](https://www.skyulf.com/manual/user_guide/databricks_monitoring_enrollment.html)
- [Deployment schedules, time limits and recovery](https://www.skyulf.com/manual/user_guide/databricks_deployment.html)

## Version note

Public links show the latest manual; [README.md](README.md) explains how to match
them to this project's pinned Core release.
