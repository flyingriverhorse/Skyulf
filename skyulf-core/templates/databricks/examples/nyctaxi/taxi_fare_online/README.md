# taxi_fare_online

Start here to train your first model and inspect its batch predictions.
Use [GUIDE.md](GUIDE.md) when changing settings or operating the project;
these are the only two project guides. Run the first experiment in a development/test target.

This project trains one model for one target and registers it as a candidate.
Skyulf 0.9.6 Bundle: **regression** with **pandas** training, **batch** prediction delivery and **serverless** compute.
Training runs in one bounded Python process.
Batch scoring uses **single_node** batch compute.
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
| `deployment/variables.yml` and `deployment/targets.yml` (optional) | Monitoring runs only when `monitoring_enabled` is `"true"`. It writes to `monitoring_catalog`.`monitoring_schema` (variables.yml); the catalog defaults to each target's own catalog and the schema to `skyulf_monitoring`. Use the same schema in every project so all models appear in one dashboard. `deployment/targets.yml` holds each target's dashboard and the development targets' monitoring settings; only one project creates a dashboard, the others connect to it there. |

For the shared `test` target, check that `test_root_path` in `deployment/variables.yml`
is a workspace folder your team can write to. The local checks below do not flag
these placeholders.

Source arrangement: **single_table**. Initialization does not create source
tables. For existing tables, use the [source checks](GUIDE.md#source-data) after
installing dependencies in step 1. Each record key must identify one observation;
keep the actual target out of model inputs. Use the same guide if tables do not
exist yet or outcomes arrive separately.

Before training, choose `initial_scoring` in `config/inference.yml`:
`full_snapshot` includes existing rows; `from_now` records a zero-row checkpoint
and scores later inserts. Training may automatically start scoring, and this
choice cannot change on an existing prediction target.

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

   Scoring resolves the champion alias, so a version must be promoted first.
   Check the committed output.

## Learn by making one change

After the first successful run, use this short exercise:

1. Write down the saved model version, training run and holdout result.
2. Find one prediction and explain its inputs and model version. A registered
   candidate, an approved champion and a live endpoint are separate states.
3. Keep the training snapshot and split fixed; change one supported model
   parameter or preprocessing step using the [file map](GUIDE.md#file-map).
4. Preview, deploy and train again. Compare the new report; the old saved
   artifact remains unchanged. Editing YAML does not change an existing model.
5. If monitoring is enabled, distinguish drift, label coverage and measured
   performance using [monitoring](GUIDE.md#monitoring).

Use the [reference guide](GUIDE.md) for the next setting you need. The
[worked walkthrough](https://www.skyulf.com/manual/user_guide/databricks_bundle_walkthrough.html)
provides a small practice dataset.

## If something fails

Start with the first failing task and record its run ID, target and
model version. A monitoring failure does not undo committed predictions.

- [Exact validation errors and fixes](https://www.skyulf.com/manual/user_guide/databricks_troubleshooting.html)
- [Create and preview a project](https://www.skyulf.com/manual/user_guide/databricks_project_setup.html)
- [Complete worked project](https://www.skyulf.com/manual/user_guide/databricks_bundle_walkthrough.html)

## Upgrading this project

This template pins `skyulf-core[mlflow]==0.9.6` in `deployment/requirements.txt`;
public guides describe the current revision. Before changing the pin, follow
[Upgrade an existing project](https://www.skyulf.com/manual/user_guide/databricks_project_upgrade.html)
and **Upgrade steps for generated Databricks projects** in the
[release notes](https://github.com/flyingriverhorse/Skyulf/blob/master/changelog/0.9.x.md).

## Legacy Python-only scoring policy

This example deliberately keeps its original Python scoring hooks and no
config/scoring.yml. build_model_rules() returns None, preserving the endpoint's
prediction schema without scoring-status columns. New template projects use
scoring.yml; this example also verifies older Python-only projects still work.
To migrate, remove the Python builder definitions and exports before adding YAML;
an empty custom policy adds scoring_status and exclusion_reason, so review the
endpoint schema first. See [Batch predictions](GUIDE.md#batch-predictions).
