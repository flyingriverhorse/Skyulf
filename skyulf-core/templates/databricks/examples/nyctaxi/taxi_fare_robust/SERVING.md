# Serve a registered model

Use REST or a SQL function to query a concrete registered version. Optional
online lookup, feature publication and daily rollout are independent selections.
Training and endpoint-only use do not require a batch prediction table.

## Files to edit

| File | Purpose |
| --- | --- |
| `deployment/requirements.txt` | Core runtime matching the saved model producer |
| `deployment/serving-requirements.txt` (when generated) | Optional serving dependencies |
| `config/serving.yml` (when generated) | Endpoint/publication identities and rollout settings |
| Your notebook | Concrete `PinnedEndpointSpec` and new SQL function name |

## Commands

Run from the project root after completing the optional job configuration:

Replace `test` with your target (`test_development` for personal targets) and `YOUR_PROFILE`
with that target's authenticated profile.

```bash
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
# Only if include_online_serving was selected:
databricks bundle run online_serving_setup -t test --profile YOUR_PROFILE
# Only if include_online_publication was selected:
databricks bundle run online_publication -t test --profile YOUR_PROFILE --params publication_action=start
databricks bundle run online_publication -t test --profile YOUR_PROFILE --params publication_action=status
# Only after explicit rollout initialization, if include_daily_rollout was selected:
databricks bundle run daily_rollout -t test --profile YOUR_PROFILE
```

REST/SQL endpoint creation is a separate notebook operation. Follow the
[endpoint and SQL commands](https://www.skyulf.com/manual/user_guide/databricks_serving.html#the-command),
then inspect readiness before querying. The endpoint setup job reports its real
`input_schema` and typed `request_example`; use every required field.

## Rules to keep

- Pin a numeric model version and matching producer runtime. Admission is for
  certified packages; updating requirements does not upgrade an old saved model.
- Set `enabled` and `exclusive_writer` only after completing identities and
  serializing the resource's writers. A submission is not readiness or feature freshness.
- Keep a daily A/B endpoint separate from one used by a pinned SQL function.
  Publication defaults to `TRIGGERED` with its hourly schedule paused;
  `CONTINUOUS` uses explicit controls without a periodic schedule.

## Detailed instructions

- [Endpoint setup, REST, SQL and permissions](https://www.skyulf.com/manual/user_guide/databricks_serving.html)
- [Feature lookup, freshness and publication controls](https://www.skyulf.com/manual/user_guide/databricks_online_features.html)
- [Daily traffic increases, holds and rollback](https://www.skyulf.com/manual/user_guide/databricks_serving_rollout.html)
- [Serving observations and labels](https://www.skyulf.com/manual/user_guide/databricks_serving_monitoring.html)

Version: `deployment/requirements.txt` pins `skyulf-core[mlflow]==0.9.4`.
Public links show the latest documentation; compare them with that pin before
using newer settings. Rollout latency checks p99 model execution time when it is
recorded and falls back to the maximum latency otherwise.
