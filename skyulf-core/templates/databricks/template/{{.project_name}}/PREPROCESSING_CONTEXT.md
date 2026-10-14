# Check saved preprocessing in this project

Use diagnostics when saved preprocessing may behave differently for one row,
reordered input or worker-sized batches. A probe applies saved state; it does
not retrain, measure model quality or grant Spark/REST eligibility.

## Files and option

Set this field in the existing `config/training.yml` defaults:

```yaml
defaults:
  preprocessing_probe: true
```

It defaults to false. Each fitting run probes the saved/reloaded artifact on at
most 256 holdout rows with an 8 MiB input/output bound and logs
`preprocessing_probe.json`. For custom behavior, declare truthful context in
the factory or Applier class; do not relabel a history-dependent calculation
as row-local to bypass admission.

## Commands and results

Replace `test` with your target (`test_development` for personal targets) and `YOUR_PROFILE`
with that target's authenticated profile.

```powershell
python src/tools/smoke.py
python src/tools/preview.py --action train
databricks bundle run train -t test --profile YOUR_PROFILE
```

Use your configured target/profile and deploy changed configuration first.
Preview does not run the probe. After training, open **train_and_tune > Output >
Preprocessing diagnostics**, or **train_<name>** for competition/model-set branches.
The same report is in that fitting run's MLflow **Artifacts > preprocessing_probe.json**.

| Context/result | Meaning |
| --- | --- |
| `row` | Applying fitted state needs only each input row; saved GroupImputer means can be row-local. |
| `group`, `window`, `global` | Request-time behavior needs the declared population or history. |
| `passed` | This exact artifact/sample passed the reported comparisons. |
| `failed` | Inspect the first failure or redacted error_type; correct the step/input contract. |
| `requires_context` | Independent row/chunk checks cannot represent required context. |
| `not_run` | An earlier boundary prevented this step; top-level empty_holdout means no sample. |
| `skipped` | The normal prediction path omits this fitted training step. |
| Empty check `not_supported` | Empty input raised; other checks can still pass. |
| `feature_schema` | passed / mismatch / not_run: final saved model feature-schema comparison. |

For native local checks, load the existing artifact and call
`probe_fitted_preprocessing(artifact, sample)`. For admitted lag/rolling
continuation use `score_pipeline_with_history` and carry its returned history
into the next ordered request. Commit predictions and proposed history together;
a failed write must not advance stored history. Do not silently treat
independent request chunks as a continuous history session.

## Detailed instructions

- [Runnable fit/save/load/probe and history examples](https://www.skyulf.com/manual/user_guide/preprocessing_context.html)
- [Context declarations and complete status definitions](https://www.skyulf.com/manual/user_guide/preprocessing_context.html#how-to-read-the-report)
- [Choose and configure preprocessing steps](https://www.skyulf.com/manual/user_guide/preprocessing_recipes.html)
- [Training report locations](https://www.skyulf.com/manual/user_guide/databricks_training_reports.html)
- [Separate distributed scoring requirements](https://www.skyulf.com/manual/user_guide/databricks_scoring.html#distributed-inference-settings)

## Version note

Public links show the latest manual; [README.md](README.md) explains how to match
them to this project's pinned Core release.
