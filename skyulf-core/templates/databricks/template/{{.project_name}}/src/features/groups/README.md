# Spark feature groups in this project

These examples run only when selected in
[config/features.yml](../../../config/features.yml). Empty `groups: {}` creates
no feature job. Replace the disabled configuration with the adapted commented
example; use actual catalog/schema names and one owner per output table.

## Files and function contract

| File | Source columns | Returned features |
| --- | --- | --- |
| [company.py](company.py) | company_id, observed_at, employee_count | employee_count |
| [activity.py](activity.py) | company_id, observed_at, amount | monthly_amount, transaction_count |

Each function receives only its pinned source Spark `frame`; there is no YAML
`params` field. Return unique keys/time plus exactly the declared feature columns.
`transform: src/features/groups/company.py:compute_features` selects the file
and callable; the group/file/function names need not match. One function can
produce many columns. Keep reads, collection and writes outside the function.

`activity.py` assumes an existing availability date; it does not infer monthly
windows from raw events. `asof` chooses the latest available row at or before
each observation, not a cumulative sum. Keep company/time as the feature grain;
use a separate stable observation_id BIGINT as prediction identity, not a timestamp.

## Commands and result

From the project root, using your configured target and profile:

Replace `test` with your target (`test_development` for personal targets) and `YOUR_PROFILE`
with that target's authenticated profile.

```powershell
python src/tools/refresh_feature_graph.py
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run features -t test --profile YOUR_PROFILE
```

Inspect `initialize_features`, each `feature_<group>` receipt and `merge_features`.
Point training_table and score_source_table at the merged output before running
downstream jobs. Keep labels and record keys outside model inputs. Adding a new
group requires its module/configuration, graph refresh and redeployment.
`selected_groups` chooses groups to recompute; reused outputs must already exist.

Use upstream groups for fixed shared calculations. Learned transformations
belong in preprocessing.yml inside training folds. This folder is excluded
from model source snapshots; do not import it from model recipes or add it to
features/__init__.py. No groups/__init__.py is needed; only each configured
transform file is hashed, not undeclared helpers.

## Detailed instructions and version

[Feature tables](../../../FEATURE_TABLES.md) lists project entry points.
The [public feature guide](https://www.skyulf.com/manual/user_guide/databricks_features.html)
contains complete SQL/YAML, transform examples, field contracts, join results,
reuse and repair. Public links show the latest manual; [README.md](../../../README.md)
explains how to match them to this project's pinned Core release.
