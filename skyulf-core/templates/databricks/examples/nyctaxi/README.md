# NYC taxi example projects

Four projects generated from this template with `databricks bundle init` on the
`samples.nyctaxi.trips` dataset. Each folder is ordinary template output for its
answer file, without the template's teaching samples it does not use, plus the
files an author writes: recipes, scoring rules and custom steps.
`build_example_project.py --check` keeps them identical to the current template.

| Project | Layout | What it shows |
|---|---|---|
| `taxi_fare/` | single model, regression on `fare_amount` | Spark inference with a partition-safe recipe (`spark_safe`), Optuna tuning, 3-fold CV, automatic promotion behind an RMSE bound, score after promotion, evaluation charts, SHAP, monitoring |
| `taxi_fare_online/` | single model, regression on `fare_amount` | An online serving endpoint whose model looks up pickup-zone features (`zone_features`) from an online feature store at request time; manual approval, feature publication and endpoint setup jobs |
| `taxi_high_fare/` | model competition, classification on `high_fare` | Polars engine, four candidates (logistic regression, XGBoost, soft voting, stacking), nested stratified CV, random search with threshold tuning, manual approval, custom pre-split and preprocessing steps in `src/features/trip_rules.py`, `custom` scoring with an input rule and a `fare_band` output |
| `taxi_trips/` | model set: `fare` regression and `long_trip` classification | Per-model pre-split and preprocessing recipes, grid and halving search, manual set approval, `combined` scoring with an `expected_long_trip_fare` rule that uses both models |

Pick the closest one: Spark scoring, `taxi_fare`; online endpoint,
`taxi_fare_online`; several candidate algorithms, `taxi_high_fare`; several
targets approved together, `taxi_trips`. Copying one is the fastest start; the
[project setup guide](https://www.skyulf.com/manual/user_guide/databricks_project_setup.html#start-from-the-closest-example)
shows the same path before the setup wizard.

[FEATURES.md](FEATURES.md) explains each project's filters, fitted steps, lookups
and scoring rules with flow diagrams.

`taxi_fare`, `taxi_high_fare` and `taxi_trips` were trained, approved, scored and
monitored on Databricks. The answer files next to the folders (`taxi_fare.json`
and so on) record every setup answer that differs from the defaults.

## Run one

1. Copy the project folder out of the repository.
2. Create its source tables: open `load_nyctaxi.sql` in the Databricks SQL
   editor, set `catalog` to `workspace` (or your catalog) and `schema` to the
   project name, and run it. It creates `trips_train`, `trips_score`,
   `trips_labels` with January trips and `zone_features`, and never overwrites
   an existing table.
3. In `deployment/targets.yml` replace `https://replace-test.example.invalid`
   with your workspace URL. If your catalog is not `workspace`, change
   `test_catalog` and the monitoring catalog in `deployment/variables.yml`.
4. Deploy and train:

   ```bash
   databricks bundle deploy -t test --profile <profile>
   databricks bundle run train -t test --profile <profile>
   ```

5. `taxi_fare` promotes and scores by itself. For the other projects, copy the
   approve parameters from the final training report and run the train job with
   them; scoring follows the approval (`taxi_fare_online` has no batch scoring).

To try retraining, uncomment the February block at the end of `load_nyctaxi.sql`,
run it, and train again. The new candidate is compared with the current
champion on the new holdout.

## Serve `taxi_fare_online`

The online model needs its zone features in an online store at request time.

1. Provision an online store once (Databricks Lakebase; billable), for example
   `features.create_online_store(name="taxi-features", capacity="CU_1")` with
   the Feature Engineering client, as in the online features guide.
2. In `config/serving.yml`, under `online_publication`, set `enabled: true`,
   `exclusive_writer: true` and `source_table: workspace.taxi_fare_online.zone_features`,
   then run the `online_publication` job to sync the table into the store.
3. After approving a model version, fill `online_serving`: `enabled: true`,
   `exclusive_writer: true`, an `endpoint_name`, the `model_name` and the
   concrete approved `model_version`, and the publication's source table, its
   table id, online table and `online_store` name. Run `online_serving_setup`.
4. Refresh `zone_features` (and its publication) within `max_age_seconds` in
   `config/training.yml` (7 days); older features are rejected as stale.

The generated `SERVING.md` and `deployment/README.md` in each project explain
the remaining fields, targets, permissions and Core version pins.

## Keeping them current

After a template change, run from the repository root:

```bash
python skyulf-core/templates/databricks/build_example_project.py          # refresh
python skyulf-core/templates/databricks/build_example_project.py --check  # CI gate
```

Refreshing keeps each project's own files and fails when the generated
`training.yml` no longer contains a line its documented edits expect.
