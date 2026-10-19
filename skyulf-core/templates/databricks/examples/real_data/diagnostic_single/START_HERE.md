# Start here

Read this page first. It takes **diagnostic_single** from setup to a trained
model and a scored table; the other Markdown files answer later questions.

This project trains one model for one target and registers it as a candidate.
Training runs **pandas** in one bounded Python process.
Batch scoring uses **spark** batch compute.
Run every command beside `databricks.yml`.

## Before the first run

Setup writes placeholders where it cannot know your workspace. Replace them and
check that the data exists before you deploy:

| Where | What to set |
| --- | --- |
| `deployment/targets.yml` | `host` under `test`: replace `https://replace-test.example.invalid` with your workspace URL. |
| `deployment/variables.yml` | `test_catalog`, `test_input_schema`, `test_output_schema` and `test_metadata_schema` (`REPLACE_TEST_*`). |
| Commands below | `YOUR_PROFILE`: an authenticated Databricks CLI profile for that workspace. |
| Unity Catalog | The catalog, schemas and source table must exist. The table needs the record key, input and target columns from setup, and Change Data Feed on before the first score. |
| `config/training.yml` | A numeric `quality_threshold`, if you will approve the first champion or promote automatically. Set it before training; a trained version keeps its policy. |
| `deployment/variables.yml` (optional) | Monitoring runs only when `monitoring_enabled` is `"true"` and `monitoring_catalog`/`monitoring_schema` name the store. A catalog answer defaults the shared schema to `skyulf_monitoring`; a blank catalog disables monitoring. Use the same store in every project so all models appear in one dashboard; only one project creates the dashboard, the others connect with `src/tools/configure_monitoring_dashboard.py --dashboard-url`. |

For the shared `test` target, check that `test_root_path` in `deployment/variables.yml`
is a workspace folder your team can write to. The local checks below do not flag
these placeholders.

## First train and score

Replace `YOUR_PROFILE` in every command.

1. **Check locally.** Smoke and preview read configuration only, not source data
   or workspace access:

   ```powershell
   uv pip install -r deployment/train-requirements.txt
   python src/tools/smoke.py
   python src/tools/preview.py --action train
   ```

2. **Validate and deploy:**

   ```powershell
   databricks bundle validate --strict -t test --profile YOUR_PROFILE
   databricks bundle deploy -t test --profile YOUR_PROFILE
   ```

3. **Train:**

   ```powershell
   databricks bundle run -t test --profile YOUR_PROFILE train
   ```

4. **Read the report.** Open `train_and_tune` and `training_report`. Registration alone does not
   activate a candidate.
   Start with **Where to look** in `training_report` > Output.

5. **Approve.** Copy the **Available action: approve** block from the report (the first
   approval uses expected champion `none`)
   into the train job's **Run now with different parameters**; see
   [operator actions](https://www.skyulf.com/manual/user_guide/databricks_lifecycle.html#operator-actions-through-run-now).

6. **Score:**

   ```powershell
   databricks bundle run -t test --profile YOUR_PROFILE score
   ```

   Scoring uses the pinned `model_version` in `config/inference.yml` (`"1"` for
   a new model) and needs no approval. Change it to score a later version.
   Check the committed output, then `monitoring_report` and `evaluate_retraining`.
   Before monitoring an unactivated pinned candidate for the first time, prepare its
   [reference](https://www.skyulf.com/manual/user_guide/databricks_monitoring_enrollment.html).

## When you need it

| Question | Read |
| --- | --- |
| Which file owns a setting? | [README.md](README.md) |
| How do I change the model, search, CV or quality gates? | [TRAINING.md](TRAINING.md) |
| How do I add cleanup, filters or fitted steps? | [PREPROCESSING.md](PREPROCESSING.md) |
| How do I write my own steps or scoring rules? | [src/features/README.md](src/features/README.md) |
| Does a step need groups or history, and how do I probe it? | [PREPROCESSING_CONTEXT.md](PREPROCESSING_CONTEXT.md) |
| How do I build feature tables before training? | [FEATURE_TABLES.md](FEATURE_TABLES.md) |
| How do I choose the scored version or add business rules? | [SCORING.md](SCORING.md) |
| How do I serve the model or call it from SQL? | [SERVING.md](SERVING.md) |
| How do I set targets, identities, compute and schedules? | [deployment/README.md](deployment/README.md) |
| How do I turn on monitoring and read the dashboard? | [src/monitoring/README.md](src/monitoring/README.md) |
| How do I preview or refresh the training graph locally? | [src/tools/README.md](src/tools/README.md) |

## If something fails

Start with the first failing task and record its run ID, target and
model version. A monitoring failure does not undo committed predictions.

- [Exact validation errors and fixes](https://www.skyulf.com/manual/user_guide/databricks_troubleshooting.html)
- [Create and preview a project](https://www.skyulf.com/manual/user_guide/databricks_project_setup.html)
- [Complete worked project](https://www.skyulf.com/manual/user_guide/databricks_bundle_walkthrough.html)

## Upgrading this project

This template pins `skyulf-core[mlflow]==0.9.5` in `deployment/requirements.txt`;
public guides describe the current revision. Before changing the pin, follow
[Upgrade an existing project](https://www.skyulf.com/manual/user_guide/databricks_project_upgrade.html)
and **Upgrade steps for generated Databricks projects** in the
[release notes](https://github.com/flyingriverhorse/Skyulf/blob/master/changelog/0.9.x.md).
