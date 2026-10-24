# Lakeflow: train from a streaming table

`bronze_observations` trains a regression model straight from a Lakeflow
**streaming table**, without a separate job that copies bronze data into a
regular table first. The raw table has no ready target and only a `DATE`
observation column, so one small function derives both.

```text
bronze_observations (streaming table)
  └─ features job: copy once per run, pin, prepare(), publish ─▶ observations (regular Delta, CDF on)
                                                                  ├─▶ train
                                                                  └─▶ score job (new rows via CDF)
bronze_observations ─▶ ai_query / serving SQL function   (scores the streaming table directly)
```

| File | What it shows |
| --- | --- |
| `config/features.yml` | `base_table` is the streaming table; `base_transform` names `prepare()`; `groups: {}`. |
| `src/features/groups/base.py` | `prepare()` derives `age` from `birth_date` and a `yyyy-MM-dd` text key `observed_on` next to the `DATE` grain `observed_date`. |
| `config/training.yml` | Trains from `observations` with `record_key_columns: [entity_id, observed_on]`; `skip_missing_targets: true` leaves out rows without a birth date. |
| `config/preprocessing.yml` | `region_code` is a nullable integer code; Casting writes `"81"` (not `"81.0"`) so the fixed one-hot categories match. |

`groups` is empty because every input is already in the bronze table. Add a
group only to join features computed from another table; its sources may be
streaming tables too.

## Run it

1. Create the demo data, a raw table and a streaming table over it, with
   [setup.sql](setup.sql) on a SQL warehouse.
2. In `bronze_observations/deployment/targets.yml`, replace
   `https://replace-test.example.invalid` with your workspace URL.
3. Deploy and run, in order:

   ```powershell
   databricks bundle deploy -t test --profile <profile>
   databricks bundle run features -t test --profile <profile>
   databricks bundle run train -t test --profile <profile>
   databricks bundle run score -t test --profile <profile>
   ```

The features job copies the streaming table into `skyulf_src_<hash>` in the
output schema, pins that copy for every task, publishes `observations` and drops
the copy. Rerun it before training or scoring when new bronze rows should be
included. Set `source_snapshot: fail` in `config/features.yml` to forbid copying.

Batch scoring reads `observations`, which keeps Change Data Feed on. To score the
streaming table itself, call the model through `ai_query` or the serving SQL
function (enable an endpoint), passing the raw input columns
`region_code, employee_count, revenue`.
