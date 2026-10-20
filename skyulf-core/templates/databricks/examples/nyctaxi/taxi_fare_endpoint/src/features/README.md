# src/features: where feature code lives and how YAML selects it

This folder holds every piece of Python a model needs besides the estimator.
Nothing here runs on its own. A config file names a function in this folder,
and the training or scoring job calls it at a fixed point in the flow.

## One table to remember

| Folder or file | What it holds | Selected from | Runs when | Learns from data | Saved in the model |
| --- | --- | --- | --- | --- | --- |
| `groups/*.py` | Spark functions that build shared feature tables | `config/features.yml` (`transform: src/features/groups/<file>.py:<function>`) | The separate `features` job, before any training | No | No; the merged Delta table is the input |
| `pre_split.py` | Row filters: keep or drop a training row | `config/pre_split.yml` (`custom: pre_split.<factory>`) | Training job, before the train/holdout split | Never | Yes; `config/scoring.yml` can replay them |
| `preprocessing.py` | Fitted steps: learn on training rows, replay later | `config/preprocessing.yml` (`custom: preprocessing.<factory>`) | Training job, after the split, inside every CV fold | Yes; the learned dict is saved | Yes; replayed at prediction time. Spark and endpoints require admission evidence for eligible recipes, as explained below |
| `scoring.py` | Optional custom eligibility, output and cross-model callbacks | `config/scoring.yml` (`custom: scoring.<function>`) | Scoring; Spark and endpoints require the checks below | No | Yes; the rules travel with the model version |
| Any other `<file>.py` | Your own factories, selected as `<file>.<factory>` | Same YAML files as above | Same as the factory kind | Depends on the factory | Yes |
| `assets.json` + `assets/` | Small lookup files your code reads | Listed in `assets.json`, read with `read_project_asset` | Whenever a step reads them | No | Yes; edit means a new training run |
| `requirements.txt` | Exact pins your code imports | Installed by `deployment/requirements.txt` | Job start | No | Pins are recorded |

`groups/` is the odd one out. It produces input tables with Spark before a model
exists, so it is excluded from the model snapshot and must not be imported from
the other files. Everything else is pandas code that is copied into the model
version at training time; editing it later changes only new versions.

## How a YAML line reaches a function

```text
config/training.yml
  defaults:            (or one entry under models:)
    pre_split_recipe: default            # name of a list in config/pre_split.yml
    preprocessing_recipe: zones_and_trips # name of a list in config/preprocessing.yml
        |
        v
config/preprocessing.yml
  recipes:
    zones_and_trips:
      - custom: trip_rules.clip_outliers  # <file>.<factory> inside src/features
        params: {column: duration_min}    # keyword arguments of the factory
      - name: scale                       # a built-in step: name + transformer + params
        transformer: StandardScaler
        params: {columns: [trip_distance]}
        |
        v
src/features/trip_rules.py
  def clip_outliers(column):              # the factory; returns a step
      return fitted_step("clip_duration_min", learn_bounds, clip_values,
                         output=column, replace=True, params={"column": column})
```

Rules of the chain:

- A recipe name that is missing from `training.yml` means `default`. Each YAML
  file always has a `default` list; an empty list is allowed.
- `custom:` is `<module>.<factory>`. The module is any `.py` file directly under
  `src/features` (`pre_split`, `preprocessing`, `scoring`, `trip_rules`, ...).
  The file name is free; the examples keep filters in `pre_split.py` and
  fitted steps in `preprocessing.py` only for readability.
- `params` are passed to the factory as keyword arguments. The factory decides
  what the step's own `params` dict looks like.
- A built-in step has no `custom:`; it names a registered transformer. The
  public [preprocessing recipes catalog](https://www.skyulf.com/manual/user_guide/preprocessing_recipes.html)
  lists them with their parameters.
- Competition candidates share one pre-split recipe; a candidate may override
  `preprocessing_recipe`. A model set (`multi_target`) names both recipes per
  model entry, so each model can filter and preprocess differently.

## The three factory kinds

| Factory | Your functions | Use it for |
| --- | --- | --- |
| `filter_step(name, fn, columns=[...], params={...})` | `fn(df, params) -> True/False per row` | Row eligibility. Reads only the row. Never changes values. |
| `column_step(name, fn, output="new_col", params={...})` | `fn(df, params) -> one value per row` | A derived column with nothing to learn, such as `log(1 + x)`. |
| `fitted_step(name, learn, apply, output=..., replace=..., params={...})` | `learn(df, y, params) -> dict` on training rows, `apply(df, state, params) -> column(s)` everywhere | Anything learned: means, quantiles, category shares, mappings. |

`fitted_step` is the one that makes training and scoring agree: `learn` runs on
the training rows of each fold, the returned dict is saved with the model, and
`apply` receives exactly that dict on holdout rows and on batch scoring rows.
Write `learn` so the dict is plain JSON: strings, numbers, lists, no NaN.

Spark scoring (`batch_compute: spark`) and serving endpoints score independent
batches. They admit reviewed built-ins and models, plus eligible built-in
recipes, models, `column_step`/`fitted_step` functions and scoring rules that
passed the training checks. Reused pre-split filters count as scoring rules;
training-only filters do not run at prediction time.

Spark and online-feature projects run admission checks automatically, including
when `defaults.allow_tested_custom_steps` is `false`. Set it to `true` in
`config/training.yml` for an endpoint-only project. Training lints custom code
and probes the fitted recipe on bounded holdout rows before sealing evidence
with its saved code and state. The separate diagnostics report
(`save_preprocessing_diagnostics`) does not grant admission. An `inference_context="row"` declaration is
optional; `group`, `window` and `global` declarations remain unsupported here.

Do not read `df.index` or keep mutable caches in module globals: each worker
gets its own process and batches. Passing checks are evidence for the fitted
recipe and tested rows, not a guarantee for every future input. See
[Score on Spark workers](https://www.skyulf.com/manual/user_guide/databricks_spark_scoring.html)
for supported settings, minimum probe rows, exclusions and troubleshooting.
Calculations that need other rows can instead run in a `groups/` Spark producer
before the model.

## Prefer a built-in when it expresses the same rule

The custom factories below teach how to package your own code. New projects can
use the following registered steps without maintaining those factories. These
alternatives require the development changes for 0.9.5; use a wheel built from
this checkout until that release is published, then change the requirements pin.
Changing a recipe requires retraining and checking its new output schema.

| Demonstration | Built-in alternative | Check before replacing it |
| --- | --- | --- |
| `pre_split.value_range` | `ManualBounds` with `bounds: {column: {lower: minimum, upper: maximum}}` | Bounds are inclusive. ManualBounds retains missing values; precede it with DropMissingRows when the custom filter rejects missing values. |
| `trip_rules.known_zips` | `DropMissingRows` with `subset: [pickup_zip, dropoff_zip]` | This rejects nulls, not arbitrary invalid or unknown zone identifiers. |
| `trip_rules.clip_outliers` / diagnostic `clip_measurements` | `Winsorize`, columns selected explicitly, percentiles 1 and 99 | Fit on training rows. Use float measurements; cast integer inputs explicitly if fractional limits are needed. |
| Bike `weather_imputation` | `Casting` to float64, then `SimpleImputer` with median | Casting preserves the custom step's always-float64 output even if a training column has no missing values. |
| Air-quality `missing_marker` | `ValueReplacement` with an explicit column list and `mapping: {-200: null}`, then SimpleImputer | Float columns stay numeric. Do this before fitting imputation, clipping or scaling statistics. |
| Taxi `log_feature` | `ClipValues` with lower 0, then `SimpleTransformation` with method log | The built-in computes log1p but replaces the input; the custom factory adds log_column and keeps the original. Keep the factory if both columns are needed. |
| `frequency_encoding` | `FrequencyEncoder` with `mode: frequency` | Both use the total training row count. The built-in treats null as a learned category; the demonstration always maps null to zero. |
| `rare_categories` | `RareCategoryGrouping` with min_count 1 and the same min_share | The built-in groups rare or unseen nulls too; the demonstration always preserves null. |
| Bike `clock_cycle` | `DateFeatures` with `features: [hour_sin, hour_cos]` on a timestamp | The example's hr is already a number, not a timestamp. Keep its numeric factory or select the real timestamp; generated names also differ. |
| Air-quality `temperature_anomaly` | `FeatureMath` group_agg with method deviation | Saves group means and a global fallback; name the output to match downstream columns. |

For a non-null zone policy, these are complete alternatives for the two worked
recipes. Paste the first into `config/pre_split.yml`, the second into
`config/preprocessing.yml`, then select them in `config/training.yml`.

```yaml
recipes:
  builtin_eligibility:
    - name: required_trip_fields
      transformer: DropMissingRows
      params: {subset: [trip_distance, duration_min, pickup_zip, dropoff_zip]}
    - name: plausible_trip
      transformer: ManualBounds
      params:
        bounds:
          trip_distance: {lower: 0.01, upper: 100}
          duration_min: {lower: 1, upper: 240}
```

```yaml
recipes:
  builtin_zones_and_trips:
    - name: zone_shares
      transformer: FrequencyEncoder
      params: {columns: [pickup_zip, dropoff_zip], mode: frequency}
    - name: float_measurements
      transformer: Casting
      params: {columns: [trip_distance, duration_min], target_type: float64}
    - name: duration_limits
      transformer: Winsorize
      params: {columns: [duration_min], lower_percentile: 1, upper_percentile: 99}
    - name: fill_numeric
      transformer: SimpleImputer
      params: {columns: [trip_distance, duration_min], strategy: median}
    - name: scale
      transformer: StandardScaler
      params: {columns: [pickup_hour, pickup_dow, trip_distance, duration_min]}
```

Frequency learning sees only rows remaining after the eligibility filter. A
scoring policy that permits null zones still differs from the custom factory;
choose that behavior deliberately. Model versions already trained with custom
code continue replaying their saved code until you train a replacement.

## Worked example: `taxi_high_fare`

The `examples/nyctaxi/taxi_high_fare` project classifies whether a trip's fare
is above 15. Follow one row through its files.

**1. `config/training.yml`** names the columns and the preprocessing recipe. The
pre-split recipe is not named, so `default` is used.

```yaml
defaults:
  input_columns: ["pickup_hour", "pickup_dow", "trip_distance", "duration_min", "pickup_zip", "dropoff_zip"]
  target_column: "high_fare"
  preprocessing_recipe: zones_and_trips
```

**2. `config/pre_split.yml`**, list `default`, decides which rows may train.
Two filters come from the shipped `pre_split.py`, one from the project's own
`trip_rules.py`:

```yaml
recipes:
  default:
    - custom: pre_split.value_range
      params: {column: trip_distance, minimum: 0.01, maximum: 100}
    - custom: pre_split.value_range
      params: {column: duration_min, minimum: 1, maximum: 240}
    - custom: trip_rules.known_zips
      params: {columns: [pickup_zip, dropoff_zip]}
```

`trip_rules.known_zips(columns=[...])` wraps `zips_known(df, params)` with
`filter_step`. A row with a missing zone is dropped before the split. Nothing is
learned, so the filter can later be replayed on scoring rows unchanged.

**3. `config/preprocessing.yml`**, list `zones_and_trips`, runs after the split
on training rows and is saved with the model:

```yaml
recipes:
  zones_and_trips:
    - custom: preprocessing.frequency_encoding
      params: {columns: [pickup_zip, dropoff_zip]}
    - custom: trip_rules.clip_outliers
      params: {column: duration_min}
    - name: fill_numeric
      transformer: SimpleImputer
      params: {columns: [trip_distance, duration_min], strategy: median}
    - name: scale
      transformer: StandardScaler
      params: {columns: [pickup_hour, pickup_dow, trip_distance, duration_min]}
```

`trip_rules.clip_outliers("duration_min")` is a `fitted_step`: `learn_bounds`
reads the 1st and 99th percentile of `duration_min` on the training fold and
returns `{"low": ..., "high": ...}`; `clip_values` clips every later batch with
those two saved numbers. The two built-ins after it need no Python at all.

**4. `config/scoring.yml`** decides what happens at batch scoring time.
This project sets `mode: custom`: the pre-split filters are not
replayed; instead one eligibility rule (`check: value_range` on
`trip_distance`) marks implausible rows with a reason and an empty prediction,
and one output rule (`rule: prediction_band` on `probability_1`) adds a
`fare_band` column. Both built-ins are selected directly from YAML; custom
callbacks use `custom: scoring.<function>` instead.

So `trip_rules.py` is used twice, by two different config files, and nowhere
else: once as a filter before the split, once as a fitted step after it. The
module name only has to match the prefix in `custom:`.

**Compare the other examples.** `taxi_fare` uses only built-ins (`spark_safe`
recipe) because Spark scoring replays steps per partition and keeps no scoring
rules. `taxi_trips` names a different pre-split and preprocessing recipe per
model entry and uses `mode: combined` with
`skip_target_pre_split_steps: true`, so the saved filters are replayed except the
one that reads the target, then a combined rule multiplies the two models'
results. `taxi_fare_online` adds a `feature_lookup` in `training.yml` whose
columns arrive from an online store at request time; its recipe uses built-in
steps only, because endpoints run the same admission as Spark scoring. `FEATURES.md` next to the example projects in the
Skyulf repository (`skyulf-core/templates/databricks/examples/nyctaxi/`) draws
each flow.

## Where each piece runs

```text
features job (optional, Spark)      groups/*.py  ->  merged Delta table = training_table
train job                           pre_split recipe -> split -> preprocessing recipe fitted
                                    per CV fold and on the final training rows -> model
score job (batch, pandas/polars)    scoring.yml eligibility -> saved preprocessing -> predict
                                    -> scoring.yml outputs -> prediction table
score job (Spark inference)         admitted saved eligibility -> built-in/probed row steps
                                    -> predict -> admitted saved outputs, per partition
serving endpoint                    optional feature lookup -> admitted saved eligibility
                                    -> saved steps -> predict -> saved outputs; same admission
```

## Adding your own module

1. Create `src/features/my_rules.py` with top-level `def` functions and one
   factory per step. Import `filter_step`, `column_step` or `fitted_step` from
   `skyulf.preprocessing`. Use relative imports for helpers inside this folder;
   never import from `groups/`.
2. Reference it from a recipe as `custom: my_rules.<factory>` with `params`
   matching the factory's keyword arguments, then name that recipe in
   `config/training.yml`.
3. Data files go under `src/features/assets/`, listed in `assets.json`, read with
   `read_project_asset(__package__, "assets/<file>")`. Extra packages go in
   `requirements.txt` with exact pins.
4. Check before deploying, from the project root:

   ```bash
   python src/tools/preview.py --action train
   python src/tools/preview.py --action score
   python src/tools/smoke.py
   ```

   `preview` resolves every recipe and rule and prints the fitted plan;
   `smoke` compiles the package the way the train job will load it.

## Read next

- [PREPROCESSING.md](../../PREPROCESSING.md): files to edit and critical rules
  for recipes in this project.
- [groups/README.md](groups/README.md) and [FEATURE_TABLES.md](../../FEATURE_TABLES.md):
  Spark feature groups.
- [SCORING.md](../../SCORING.md): scoring modes and the prediction table.
- Public guides: [custom steps](https://www.skyulf.com/manual/user_guide/databricks_custom_steps.html),
  [feature tables](https://www.skyulf.com/manual/user_guide/databricks_features.html),
  [scoring](https://www.skyulf.com/manual/user_guide/databricks_scoring.html),
  [online features](https://www.skyulf.com/manual/user_guide/databricks_online_features.html).
