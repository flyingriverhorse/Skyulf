# bike_demand_set

Start here to train your first model and inspect its batch predictions.
Use [GUIDE.md](GUIDE.md) when changing settings or operating the project;
these are the only two project guides. Run the first experiment in a development/test target.

This project trains one model per target and packages them as one model set that is
approved and scored together.
Skyulf 0.9.6 Bundle: **independent target models** with **pandas** training, **batch** prediction delivery and **serverless** compute.
Training runs in one bounded Python process.
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
| Unity Catalog grants | The job identity reads the inputs, writes the output schema and registers models in the metadata schema; see [required access](https://www.skyulf.com/manual/user_guide/databricks_deployment.html#data-model-and-experiment-access). |
| Unity Catalog | The catalog, schemas and source table must exist. The table needs the record key, input and target columns from setup, and Change Data Feed on before the first score. |
| `config/training.yml` | A numeric `quality_threshold` in every `models` entry, if you will approve the first champion or promote automatically. Set it before training; a trained version keeps its policy. |
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
and scores later inserts. In this project scoring starts only when you run it; this
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

4. **Read the report.** Open `train_<name>`, `training_report` (`models_report.py`) and `model_decision`;
   review every component before activating the set.
   Start with **Where to look** in `training_report` > Output.

5. **Approve.** Copy the **Available action: approve** block from the report (the first
   approval uses expected champion `none`)
   into the train job's **Run now with different parameters**; see
   [operator actions](https://www.skyulf.com/manual/user_guide/databricks_lifecycle.html#operator-actions-through-run-now).

6. **Score:**

   ```powershell
   databricks bundle run -t test --profile YOUR_PROFILE score
   ```

   Ordinary runs score the approved set champion; `score_model_version` selects
   a concrete set version for one run.
   Check the committed output, then `monitoring_report` and `evaluate_retraining`.
   Before monitoring an unactivated pinned candidate for the first time, prepare its
   [reference](https://www.skyulf.com/manual/user_guide/databricks_monitoring_enrollment.html).

## Learn by making one change

After the first successful run, change one setting and compare two fits of the same data:

1. **Record the first fit.** In `training_report` > **Where to look**, note the
   version (`<model> vN`), the holdout metric and the **Training data** column's
   `Delta version`.
2. **Fix the data.** In `config/training.yml` defaults, set `training_table_version`
   to that Delta version, so the second fit reads the same rows. The split seed
   (`random_state`) already stays the same.
3. **Change one setting.** For a tuned model, raise the search size under the
   model's `tuning`, for example:

   ```yaml
   # before
       tuning:
         n_trials: 3
   # after
       tuning:
         n_trials: 10
   ```

   For a fixed model, change one value under its `model` > `params` instead.
   `python src/tools/preview.py --action train` shows the new value before you deploy.
4. **Deploy and train again.** A new version `vN+1` is registered; `vN` and its
   files do not change.
5. **Compare.** Put both training reports side by side: same source version and
   holdout rows, so a metric difference comes from your change. More trials do
   not guarantee a better score; keep whichever version is better for your goal.
   Score the new version for one run with `--params score_model_version=N+1` and
   compare one record's prediction from both versions.
6. If monitoring is enabled, distinguish drift, label coverage and measured
   performance using [monitoring](GUIDE.md#monitoring).

Use the [reference guide](GUIDE.md) for the next setting you need. The
[worked walkthrough](https://www.skyulf.com/manual/user_guide/databricks_bundle_walkthrough.html)
provides a small practice dataset.

## If something fails

Start with the first failing task and record its run ID, target and
model set version. A monitoring failure does not undo committed predictions.

- [Exact validation errors and fixes](https://www.skyulf.com/manual/user_guide/databricks_troubleshooting.html)
- [Create and preview a project](https://www.skyulf.com/manual/user_guide/databricks_project_setup.html)
- [Complete worked project](https://www.skyulf.com/manual/user_guide/databricks_bundle_walkthrough.html)

## Upgrading this project

This template pins `skyulf-core[mlflow]==0.9.6` in `deployment/requirements.txt`;
public guides describe the current revision. Before changing the pin, follow
[Upgrade an existing project](https://www.skyulf.com/manual/user_guide/databricks_project_upgrade.html)
and **Upgrade steps for generated Databricks projects** in the
[release notes](https://github.com/flyingriverhorse/Skyulf/blob/master/changelog/0.9.x.md).
