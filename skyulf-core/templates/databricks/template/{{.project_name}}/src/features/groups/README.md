# Spark feature groups

Place optional Spark table-producing functions here, for example `company.py`
or `activity.py`. Connect each function in `config/features.yml`:

```yaml
transform: src/features/groups/company.py:compute_features
```

One function can create several feature columns. It receives the group's Spark
source DataFrame and returns keys, timestamp and the configured feature columns.
The feature job writes its Delta output and merges it with the base observations.
The base input table must already exist. See [FEATURE_TABLES.md](../../../FEATURE_TABLES.md)
for complete configuration, examples, graph generation and execution instructions.

Use this stage for shared aggregations and fixed calculations. Learned imputers,
scalers and encoders belong in `../preprocessing.py`; custom model steps belong
in `../custom/`. They learn within training folds and replay their saved state
during inference. Apply a transformation in one stage, not both.

This folder is excluded from model source snapshots. Do not import it from model
recipes or add it to `../__init__.py`. No `__init__.py` is required here. The feature
job hashes each configured transform file; it does not freeze helper modules.
Empty `groups: {}` keeps feature production disabled and creates no extra job.
