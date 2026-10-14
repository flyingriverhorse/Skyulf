# Prepare optional feature tables for this project

Train/score directly from a ready Delta table, or run an upstream feature job
when inputs need shared Spark joins or fixed calculations. `groups: {}` in
`config/features.yml` disables the feature job and merged table.

## Files to edit

| File | Purpose |
| --- | --- |
| `config/features.yml` | Base observations, group inputs/outputs, join keys/time, declared features. |
| `src/features/groups/company.py`, `activity.py` | Editable Spark transform examples; receives the pinned source frame. |
| `config/training.yml` | Point training_table and input_columns at prepared data. |
| `config/inference.yml` | Point score_source_table at corresponding prepared observations. |
| `config/preprocessing.yml` | Learned imputers/scalers/encoders fitted only inside training folds. |

Each group transform returns exactly keys, timestamp and declared feature
columns. One function can compute several columns. Feature names must be unique
across the base/groups. Names in features.yml are exact three-part tables;
deployment placeholders/suffixes are not expanded there.

## Commands and results

Replace `test` with your target (`test_development` for personal targets) and `YOUR_PROFILE`
with that target's authenticated profile.

```powershell
python src/tools/refresh_feature_graph.py
python src/tools/smoke.py
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run features -t test --profile YOUR_PROFILE
```

Use your configured target/profile. Refresh after changing features or inherited
compute/identity/timeout controls. The enabled graph has `initialize_features`,
`feature_<group>` tasks and `merge_features`; inspect task receipts for versions,
counts and reuse. Run downstream training/scoring separately after production.
`selected_groups=*` computes all groups; names select a subset; empty reuses all
stored groups and only merges. Reused outputs must already exist.

## Critical rules

- Keep feature join grain separate from prediction identity. Use a stable unique
  nonnull supported record key such as observation_id BIGINT, not a timestamp.
- Keys/time must be typed consistently, unique and nonnull. `exact` joins time;
  `asof` chooses the latest feature available at or before the observation and
  has no implicit maximum age. Make historical availability explicit upstream.
- Outputs must differ from all inputs/outputs and have one owning producer.
  Writes upsert changed keys with CDF; they do not propagate deletes or evolve
  schemas automatically. Missing matches fail unless allow_missing is explicit.
- Keep learned population statistics in model preprocessing. Root `groups/`
  is excluded from saved model code; do not import it from model callbacks.
- Repair failed groups and dependent merge only with unchanged code/config and
  original pinned inputs. Otherwise start a new run.
- Native `feature_lookup` is a separate optional route requiring actual UC
  keys/lineage and stable feature versions; ordinary Delta outputs do not create
  those constraints or publish an online store automatically.

## Detailed instructions

- [Feature groups, complete SQL/YAML, transforms and repair](https://www.skyulf.com/manual/user_guide/databricks_features.html)
- [Native UC batch feature lookup](https://www.skyulf.com/manual/user_guide/databricks_features.html#bind-existing-uc-features-instead-of-materializing-a-merge)
- [Spark worker admission and model-set composition](https://www.skyulf.com/manual/user_guide/databricks_scoring.html#distributed-inference-settings)
- [Online feature publication and freshness](https://www.skyulf.com/manual/user_guide/databricks_online_features.html)
- [Scoring source corrections and output policy](https://www.skyulf.com/manual/user_guide/databricks_scoring.html#source-changes-are-separate-from-model-changes)

## Version note

Public links show the latest manual; [README.md](README.md) explains how to match
them to this project's pinned Core release.
