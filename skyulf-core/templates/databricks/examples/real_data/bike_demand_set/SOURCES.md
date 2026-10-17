# Source tables

Selected arrangement: **single_table**. No source table is created or
modified by initialization or deployment.

**Existing tables:** set their names in the project configuration and run the
read-only `preview.py --action sources` check below. You do not need a CREATE
TABLE script. `include_source_sql` defaults to `"false"`.

**No source tables yet:** answer `true` to `include_source_sql` in the setup wizard,
or add `"include_source_sql": "true"` to your JSON answer file, to generate an
editable starter. This does not run SQL.
This project has no starter SQL file. If needed later, generate a separate
project folder with `"include_source_sql": "true"` and copy its `config/sources.sql`;
do not overwrite your configured project. Replace all namespace and type
placeholders before running the script yourself.


| Role | Binding |
| --- | --- |
| Training features | `config/training.yml`: `training_table` |
| Scoring input | `config/inference.yml`: `score_source_table` |
| Actual outcomes | `deployment/variables.yml`: `monitoring_label_table` and `monitoring_result_available_at_column` |
| Training outcomes | Target columns already present in the training table |

Training expects actual target values in its source. A monitoring outcome
binding measures performance but does not automatically join labels into training.
To use separate outcomes for training, explicitly configure `training_label_table`,
`training_label_version: null`, `filter_unavailable_results: true`, and the UTC
`result_available_at_column` in each applicable training entry. The configured
training input must include the new feature rows; a separate scoring table is
never silently appended to training.

Your answer file explicitly overrides the scoring source with
`observations_score`. This legacy override is preserved: the effective
input arrangement has separate training and scoring tables. Quick setup offers
`separate_all` when starting a project with that arrangement.

Record keys identify one observation, not merely a customer. They must be
non-null, unique within each input/outcome table, and have identical types.
Each outcome row contains the key, actual target(s), and the UTC timestamp at
which the outcome became available. Never copy a prediction into an actual
target. Missing or late outcomes remain unavailable; they are not evidence of
good performance. Duplicate outcomes are rejected; resolve corrections upstream.

Enable Delta change data feed before scoring. Scoring inputs are append-only:
do not append a second row with an existing key to deliver its late label.
Write that label to the separate outcomes table. Source updates/deletes and
model-version changes follow the explicit policies in `config/inference.yml`;
this arrangement adds no automatic rescore policy.

Choose `initial_scoring` in `config/inference.yml` before the first score:

- `full_snapshot` (default) processes existing source rows, including training
  history if inputs are shared. This is not independent production performance.
- `from_now` records the current Delta version with zero predictions. Later
  runs score only new inserts. Model changes and recovery retain this boundary;
  normal rebuilds use saved prediction keys and recent CDF. CDF-expiry recovery
  needs the original Delta snapshot to exclude its keys. If that snapshot is no
  longer readable, recovery stops instead of scoring historical rows.

The policy is fixed once recorded in a prediction target. Use a new target to
start a separate scoring history. Record keys must be immutable and never reused.

Use `python src/tools/preview.py --action sources --profile YOUR_PROFILE
--warehouse-id YOUR_EXISTING_WAREHOUSE --catalog YOUR_CATALOG
--input-schema YOUR_SCHEMA --output-schema YOUR_SCHEMA
--metadata-schema YOUR_SCHEMA` from the project root. This executes read-only
metadata and key checks; uniqueness checks can scan the full source. It checks
current tables, not future inserts, permissions of another run-as identity,
or whether a model's preprocessing supports every value. No job is started.

For a repeatable append example, see the
[air-quality walkthrough](https://www.skyulf.com/manual/user_guide/databricks_training_data.html).
