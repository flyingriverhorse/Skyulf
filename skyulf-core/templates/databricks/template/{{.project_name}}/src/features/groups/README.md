# Spark feature groups

This folder ships two editable Spark examples. They do nothing until selected
in [config/features.yml](../../../config/features.yml). That file contains a
complete commented configuration; replace its active `version: 1` / `groups: {}`
lines with the uncommented example, and use your actual catalog/schema names.

| File | Expected source columns | Features returned |
| --- | --- | --- |
| [company.py](company.py) | `company_id`, `observed_at`, `employee_count` | `employee_count` |
| [activity.py](activity.py) | `company_id`, `observed_at`, `amount` | `monthly_amount`, `transaction_count` |

Each output also includes `company_id` and `observed_at`. Keep their types equal
to the base table. `activity.py` aggregates multiple transaction rows at an
already defined monthly availability date; it does not infer calendar windows
from raw event timestamps. Read its docstring before adapting it.

## How YAML selects a function

This is the `company` entry **inside** `features.yml` -> `groups`:

```yaml
company:
  source_table: workspace.demo.company_source
  output_table: workspace.demo.company_features
  transform: src/features/groups/company.py:compute_features
  columns: [employee_count]
  lookup: exact
  allow_missing: false
```

The part before `:` is a file path relative to the project root. The part after
it is the function to call. Group names, file names and function names do not
have to match. One function can create several feature columns; you do not need
one file per column or per model. Group functions accept only `frame`; this
configuration has no `params` field. Keep fixed business constants in the module.

The feature job reads the pinned `source_table` version, calls your function,
validates its result, and writes `output_table`. Do not call `spark.table`,
`toPandas`, `collect` or a table writer inside these example functions. Return a
Spark DataFrame with exactly the configured keys, timestamp and feature columns.
No labels belong in group outputs. Your source may contain multiple rows per
key/time when the function aggregates them; the returned frame must be unique.

## Base table, merge and training

The existing `base_table` supplies the observation rows and optional labels.
It has no group function. For the shipped configuration:

```text
company_observations (company_id, observed_at, churn)
  + company_features (employee_count, exact key/time match)
  + activity_features (monthly_amount, transaction_count, latest past match)
  = company_merged (all base columns plus the three features)
```

For company `1`, suppose activity has two amounts `10` and `20` available on
January 31, one amount `40` available on February 28, and a future amount `9999`
available on March 31. Company snapshots match each observation date exactly.
The merged result is:

| company_id | observed_at | churn | employee_count | monthly_amount | transaction_count |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-02-01 | 0 | 10 | 30 | 2 |
| 1 | 2026-03-01 | 1 | 12 | 40 | 1 |

The future amount is not joined into either observation. `asof` chooses the
latest available group row; it does not sum all prior months or fill gaps with
zero. All-null amounts remain null, while transaction count includes those rows.

Run these commands from the generated project root after editing the config:

```powershell
python src/tools/refresh_feature_graph.py
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run features -t test --profile YOUR_PROFILE
```

Initialization pins inputs, the two group tasks can run in parallel, and the
merge waits for both. Set `config/training.yml` -> `defaults.training_table`
and `config/inference.yml` -> `score_source_table` to the merged table, then run
the downstream jobs. Choose `employee_count`, `monthly_amount` and
`transaction_count` as model input columns; keep `churn` as the training target.
Use `[company_id, observed_at]` as record keys so different months stay distinct.
For independent-target projects, update each model's overriding input settings.

To add another domain, create e.g. `balances.py`, define `compute_features(frame)`
and add `groups.balances` with its input, output and new column names. Refresh the
feature graph and redeploy. To rerun one existing group, set the job parameter
`selected_groups` to its name; unselected outputs must already exist. See
[FEATURE_TABLES.md](../../../FEATURE_TABLES.md) for joins, reuse and repair limits.

Use this stage for shared aggregations and fixed calculations. Learned imputers,
scalers and encoders belong in `config/preprocessing.yml`; their custom functions
belong in `../preprocessing.py`. They learn within training folds and replay saved state
during inference. Apply a transformation in one stage, not both.

This folder is excluded from model source snapshots. Do not import it from model
recipes or add it to `../__init__.py`. No `__init__.py` is required here. The feature
job hashes each configured transform file; it does not freeze helper modules.
Empty `groups: {}` keeps feature production disabled and creates no extra job.
