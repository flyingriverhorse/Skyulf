# Select training rows, time windows and holdout

## When you need it

Choose the population and evaluation split before training. Snapshot version, observation window,
label availability, sample, split and job schedule are separate choices. These rules apply to
single-model, competition and model-set projects.

## What to edit

### Prepare the source table

Skyulf reads existing tables; it never creates or repairs them. Before the first
training run, make sure the training and scoring sources meet these rules:

| Requirement | Why |
| --- | --- |
| Unity Catalog Delta table | Training pins a Delta snapshot version; scoring reads Delta changes. |
| Change Data Feed on the scoring source, before new rows arrive | Incremental scoring reads new rows from the change feed. The first score processes the rows that already exist. [Scoring](databricks_scoring.md) explains how updates and deletes are handled. |
| Record key columns of type `STRING` or `BIGINT`, nonnull and unique | Keys identify each prediction. Date or timestamp keys are not supported. |
| Keys kept out of `input_columns` | Keys are copied to the output, not learned from. |
| A nonnull target for every eligible training row | Skyulf does not create labels. Use [result availability](#record-identity-and-result-availability) for delayed outcomes. |
| The same key and input columns, with the same types, in training and scoring sources | The saved pipeline expects the training input schema. The scoring source needs no target. |
| Date columns stored consistently | Declare their type and parsing as described in [source date formats](#source-date-formats-and-timezones). |

To enable Change Data Feed on an existing table you own:

```sql
ALTER TABLE my_catalog.my_schema.customer_observations
SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
```

For a worked example table, see the
[end-to-end guide](databricks_bundle_walkthrough.md).

### Choose a table layout

Training, scoring and monitoring each read a table you name: `training_table`
in `config/training.yml`, the scoring source in `config/inference.yml`, and
`monitoring_label_table` in `deployment/targets.yml`. They may be the same table
or different ones. Choose by how your actual outcomes arrive:

| Layout | Use it when | Configure |
| --- | --- | --- |
| **One append-only table** | Each row is complete when it is inserted (inputs, target and the time the outcome became known) and is never updated afterwards. | Point all three settings at the same table. Training selects its window with `start`/`cutoff` and keeps rows whose outcome is available; scoring reads newly inserted rows. |
| **Measurements plus a separate outcomes table** | The outcome is known only later. New measurements are inserted without a target; outcomes are inserted into a second table keyed by the same record key. | Training and scoring read the measurements table (or two tables, as in the examples); `monitoring_label_table` points at the outcomes table. Neither table is ever updated. |
| **A raw table plus derived tables or views** | Your raw source receives updates or deletes, such as late corrections or outcomes written into existing rows. | Keep the raw table for your own processes, and give Skyulf insert-only derived tables: one for training, one for scoring inputs, one for outcomes. |

Three behaviours decide which layout works:

- **Incremental scoring accepts inserts only.** For a single model or a
  competition, an UPDATE or DELETE in the scoring source fails the score job
  with *Source updates and deletes require an explicit rescore policy*. Model
  sets alone can opt into `source_change_policy: rebuild_on_change`; see
  [source changes](databricks_scoring.md#source-changes-are-separate-from-model-changes).
  Writing an outcome into an existing scoring row is an update, so use a
  separate outcomes table instead.
- **The first score covers every existing row.** When training and scoring
  share one table, the first score run predicts the historical rows too,
  including the rows the model was trained on. Later runs score only new inserts.
  Start scoring from a separate table, or accept and filter that first population.
- **Monitoring needs to know when an outcome became available.** The outcomes
  table must contain the saved record keys, the target and a UTC timestamp for
  when the actual value was known (`monitoring_result_available_at_column`); see
  [delayed labels](databricks_monitoring_labels.md).

[Tables in your workspace](databricks_workspace_tables.md) shows these tables next
to the ones the jobs create. The prepared examples use the second layout: `observations_train`,
`observations_score` and `labels`, with no column ever updated. When new data
arrives, insert it: new measurements into the scoring source, measured outcomes
into the outcomes table, and newly labelled rows into the training table before
retraining.

### Local input limits

`max_rows` bounds the rows read into the local process before train/test splitting.
Without sampling it applies after any observation-window selection, but before
result-availability filtering. An overflow fails; the reader never silently
truncates data. Explicit sampling is available as described below.

`max_input_mb` is required; setup writes `64`. One unit is 1 MiB (1,048,576
bytes). It bounds
measured decoded input/serialized row sizes and local frames; it does not cap
Spark's scan, total process RAM, preprocessing expansion or model training RAM.
The existing scoring services also check bounded prediction frames. Row count
alone cannot predict size: wide numeric tables and long strings cost more.

```yaml
max_rows: 100000
max_input_mb: 256
```

These are illustrative limits, not a claim that every 100,000-row dataset fits
256 MiB. Training, scoring and approval convert this value to bytes internally.
Approval may tighten but never relax the saved training budget. The lower-level
Python SDK accepts `max_bytes`; Bundle settings use only `max_input_mb`. In
an answer file use `"max_input_mb": "64"`; the generated `config/training.yml`
stores the number `64`.

### Record identity and result availability

Keys must be non-null `STRING` or `BIGINT` values and uniquely identify records
across the initial data and later inserts. Initialization accepts
`customer_id, observation_id` and emits an ordered array. For repeat observations
of a customer, choose a composite key before deployment. Keep keys out of
`input_columns`; the Bundle does not create IDs or turn them into model features.

| Reference | Role |
| --- | --- |
| `training_table` | Existing labeled source, pinned to a Delta snapshot for training |
| `score_source_table` | Existing CDF-enabled source for predictions |
| `prediction_table` | Output table, or active view in full-rebuild mode |
| `model_name` | Unity Catalog registered model, not a table |

Training and scoring default to the same existing source. Separate them when
labeled examples and new prediction requests have different lifecycles. The
initial score processes existing source rows, including historical labeled rows
if you retain that default. A prediction keeps its keys, so it can be queried:

```sql
SELECT customer_id, prediction, model_version
FROM catalog.schema.predictions
WHERE customer_id = 'C123';
```

`config/training.yml` uses the following names:

```yaml
record_key_columns: [customer_id, observation_id]
event_column: observation_date
target_column: claim_amount
result_available_at_column: result_confirmed_at
```

The key columns identify a source record and are copied to prediction output.
They do not create an ID, become features, or implicitly define CV groups.
`event_column` names the source observation timestamp; `start`, `holdout_start`
and `cutoff` are boundaries applied to that column. `result_available_at_column`
names each row's actual result-availability timestamp. Different rows can have
different availability dates. With `filter_unavailable_results: true`, exclude
unknown results and results later than the independent `result_cutoff`. Skyulf does not invent
those timestamps or fill them with the job's execution time.

Use `record_key_columns` and `result_available_at_column` in answer files.
The default example identity column is `entity_id`. Date columns you do not
answer are omitted from the generated YAML, which means null. Initialization
does not create columns or generate timestamps. The same names are used
throughout the library and saved training settings.

### Choose the evaluation split and result availability

The default `split_strategy: "random"` needs only existing keys, features and a
nonnull target. `training_version` pins the source Delta snapshot; it is unrelated
to dates. Leave it null to resolve latest once at run start, or set a nonnegative
integer to pin that version. This example shows the training-related
part of the generated `config/training.yml` defaults. Generation omits keys
whose value is null, such as `event_column`, `start` or `result_cutoff`:

```yaml
defaults:
  training_version: 12
  record_key_columns: [customer_id]
  input_columns: [income, age]
  target_column: claim_amount
  split_strategy: random
  training_window_mode: full_snapshot
  test_size: 0.2
  random_state: 42
  stratify: false
  filter_unavailable_results: false
```

| Split | Availability filter | Required date settings |
| --- | --- | --- |
| Random | Disabled | None; all supplied targets must be known. |
| Random | Enabled | `result_available_at_column`; explicit `result_cutoff` or invocation time minus lag; source parsing rules when needed. |
| Temporal | Disabled | `event_column`; `start`, `holdout_start`, `cutoff` for `fixed_window`, or `rolling_calendar` or `rolling_days` settings; source parsing rules when needed. |
| Temporal | Enabled | Both sets above; observation and result cutoffs are independent. |

For random splitting with delayed outcomes, keep the event fields null and set
`filter_unavailable_results: true`, `result_available_at_column: "confirmed_at"`
and an aware `result_cutoff`, for example `2026-09-15T00:00:00+00:00`, or leave
it null to derive invocation time minus `result_availability_lag_hours`. Each row
is eligible only if its own result date is known and no later than that instant.
A future snapshot/cutoff can admit results that became available later. Null
availability is excluded; a known, eligible result with a null target fails.
With filtering disabled, any null target fails: Skyulf does not manufacture labels.

Core `DataSplitter` selects the random holdout after sorting by the complete
record key. The same snapshot, keys, seed and split settings retain membership
when input row order changes. `stratify: true` is for classification; insufficient
class counts or an impossible partition fail instead of silently disabling it.
A different snapshot may produce different membership even with the same seed.
The final holdout never fits preprocessing or the model.

Saved training settings include the split policy and a holdout-key digest.
`holdout_membership.json` records the key-column names, count and digest without
labels. Approval replays the saved snapshot and verifies that membership before
comparison; changing today's workflow cannot silently replace the evaluation set.
See [training CV](databricks_tuning.md#cv-nested-cv-and-total-work)
and the independent source-window controls below.

Inactive fields must stay null/default: full-snapshot selection rejects event
fields; random splitting rejects `holdout_start`. A random split can use an
explicit event window without becoming a temporal split. A result column with
availability filtering disabled is rejected. Missing temporal dates never switch
training to random.
Initialization hides irrelevant prompts but preserves explicitly supplied values
so validation can identify conflicts. The examples folder contains
`date-free-init.example.json`, `random-delayed-results-init.example.json` and
`temporal-delayed-results-init.example.json`. Replace their table/feature names
and snapshot before using them. Their date examples assume native timestamp
instants; edit parsing rules for other source types. In answer-file JSON,
`test_size`, `random_state`, `stratify` and `filter_unavailable_results` are
strings; the generated YAML stores numbers and booleans.

<a id="optional-basic-model-cross-validation"></a>

### Explicit training sampling

```yaml
training_sample_rows: 10000
training_sample_seed: 42
max_rows: 10000
max_input_mb: 64
```

`training_sample_rows: null` keeps the full selected input and fails on budget
overflow. When enabled, Spark selects up to the requested number of eligible
records using a seeded SHA-256 ordering of complete record keys. Selection occurs
before local transfer, after window and result-availability filtering. Changing
partition layout does not change the sample. A different seed or source snapshot
may change it. The requested count must be 4 through `max_rows`.

The count includes the final holdout: a 10,000-row sample with `test_size: 0.2`
produces 8,000 training and 2,000 test rows. Sampling is without replacement and
is not class-balanced; split/CV class-count guards still apply. For temporal
splits it samples within the chosen window; both partitions must remain usable.
It never samples inference rows. Record keys must be nonnull and unique across
the selected source, and eligible targets must be nonnull before sampling.

`training_selection.json` records source/eligible/selected counts, seed, algorithm
and membership digest. Saved training evidence pins both sample and holdout
membership; approval replays those choices. Row/memory budgets remain enforced.
Sampling bounds transfer, not Spark scans: validation, counts and hash ordering
can scan the selected source several times.

### Source windows are independent of splitting and scheduling

| `training_window_mode` | Source selection | Settings |
| --- | --- | --- |
| `full_snapshot` | All eligible rows from the pinned snapshot | Random split; event fields, lookback and window timezone null. |
| `fixed_window` | Explicit `[start, cutoff)` observations | Event column and boundaries; random or temporal split; lookback and window timezone null. |
| `rolling_calendar` | Completed calendar months, derived at every `train` invocation | Event column, explicit `window_timezone` and `monthly_lookback_months`; random or temporal split. |
| `rolling_days` | Preceding elapsed UTC days, ending at invocation time | Event column, `lookback_days`; temporal uses `holdout_days`, random uses `test_size`. Calendar-only fields stay null. |

With `training_window_mode: auto`, setup writes `full_snapshot` for random
projects and an editable `rolling_calendar` with four months and
`window_timezone: "UTC"` for temporal projects. Every `train`
invocation resolves latest once when `training_version` is null or missing;
an explicit version is honored regardless of trigger. `rolling_calendar` and
`rolling_days` derive new observation boundaries, including for manual runs. Fixed
windows stay fixed even if a job runs again a month later.

Sampling fields participate in the saved dataset identity even when sampling
is disabled.

For example, `window_timezone: "Europe/Vilnius"` uses Vilnius month boundaries,
including the correct seasonal UTC offsets. Four months includes the temporal
holdout month: three months for fit, the last completed month for final test.
Random splitting instead divides the selected four-month data by `test_size`
and leaves `holdout_start` null. Temporal lookback is 2-120 months; random is
1-120. Source parsing timezone, window timezone and cron timezone serve different
purposes. Neither the cron day nor the source timestamp format determines the
calendar implicitly. When availability filtering is enabled, an explicit
`result_cutoff` is honored; null derives invocation time minus
`result_availability_lag_hours`, equally for manual and scheduled runs.

```mermaid
flowchart LR
    Source["Pinned Delta source"] --> Window["Full, fixed or rolling selection"]
    Window --> Sample["Optional eligible-row sample on Spark"]
    Sample --> Split["Random or temporal outer split"]
    Split --> Train["Training rows"]
    Split --> Test["Protected final holdout"]
    Train --> CV["Optional CV and search with fold-local preprocessing"]
    CV --> Fit["Final pipeline fit with fixed or tuned parameters"]
    Train --> Fit
    CV --> Report["MLflow CV report"]
    Fit --> Evaluation["Final evaluation and champion comparison"]
    Test --> Evaluation
```

### Source date formats and timezones

Column names and time boundaries serve different purposes. `event_column`
selects the source column; `event_time_parsing` explains its values. The
`start`, `holdout_start` and `cutoff` boundaries are timezone-aware instants.
They do not need to use the same UTC offset as source values.

Edit these settings in the generated `config/training.yml` defaults. Calendar
and day-window fields are omitted because `fixed_window` does not use them:

```yaml
defaults:
  split_strategy: temporal
  training_window_mode: fixed_window
  filter_unavailable_results: true
  event_column: observation_date
  event_time_parsing: {format: "%d/%m/%Y %H:%M:%S", timezone: Europe/Copenhagen, date_only: reject}
  result_available_at_column: result_confirmed_at
  result_time_parsing: {format: "%Y-%m-%dT%H:%M:%S%z", timezone: null, date_only: reject}
  start: "2026-06-01T00:00:00+00:00"
  holdout_start: "2026-08-01T00:00:00+00:00"
  cutoff: "2026-09-01T00:00:00+00:00"
  result_cutoff: "2026-09-15T00:00:00+00:00"
```

In this example `01/08/2026 02:00:00` in Copenhagen and
`2026-08-01T03:00:00+03:00` represent the same instant: August 1 at 00:00 UTC.
An event at that instant belongs to the holdout. The window includes `start`
and excludes `cutoff`; the holdout includes `holdout_start`.

| Source values | Parsing settings |
| --- | --- |
| Native Spark `TIMESTAMP` instants | Keep defaults: `format: null`, `timezone: null`, `date_only: "reject"`; the stored instant is retained. |
| Native `TIMESTAMP_NTZ` or naive local datetimes | Set the source IANA `timezone`; leave `format: null`. |
| Strings containing UTC offsets | Set an explicit `format` with `%z`; leave `timezone: null`. |
| Strings containing local clock times | Set an explicit `format` and source `timezone`. |
| Native `DATE` or date-only strings | Set `date_only: "midnight"` and source `timezone`; strings also need a date `format`. |

Formats use Python numeric directives: `%Y`, `%m`, `%d`, `%H`, `%M`, `%S`,
`%f` and `%z`, with literal separators. A complete four-digit year, month and
day are required. They are not Spark/Java patterns such as `yyyy-MM-dd`.
The same parser handles formatted source values and local validation, so
pandas and Polars training receive the same instants. Precision is limited to
microseconds; submicrosecond pandas timestamps are rejected. Local month names,
missing years and guessed day/month order are not supported. An explicit
`%d/%m/%Y` resolves day/month order; a string without a format does not.

Date-only values are not automatically interpreted as UTC. For example,
`2026-08-01` with `date_only: "midnight"` and `Europe/Copenhagen` means
July 31 at 22:00 UTC, before the holdout boundary above. Ambiguous or nonexistent
local times at daylight-saving transitions fail; supply an offset-bearing
source timestamp to make the intended instant explicit.

Source dates are normalized in Spark before filtering. Invalid/null event
values fail validation even if a window filter would otherwise hide them.
A null result-availability value remains unavailable, while a malformed value
fails. Normalized instants cross the Spark/Python boundary as integer
microseconds; session and process timezone settings do not reinterpret them.
The local row/byte limits still apply after filtering. Date validation may scan
the pinned source snapshot in Spark; these limits bound driver materialization,
not the amount of distributed source validation.

Keep driver, worker and replay environments' timezone data aligned. Saved rules
record zone identifiers, not a copy of the IANA timezone database.

Parsing rules are saved with the candidate's training specification and dataset identity. Approval
replays those saved rules. The job's cron timezone only controls when it runs; it does not define
the source timestamps' timezone. For Python APIs, use `TrainingDateSpec` in the
[Python SDK training guide](databricks_sdk.md#train-a-candidate-when-labels-are-ready). The
supported format vocabulary is a subset of Python's
[strptime directives](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes);
source timezone rules use [IANA zoneinfo](https://docs.python.org/3/library/zoneinfo.html).

`training_version` defaults to `null`, resolving latest once at the start of
each `train` invocation. Set an explicit version for a repeatable snapshot.
Date boundaries default to null and are required only for `fixed_window`;
`rolling_calendar` and `rolling_days` derive them at invocation. Scoring and saved-evidence actions
do not require a new training window.

## The command

Edit `config/training.yml` defaults or the selected model entry, then inspect the resolved request:

```powershell
python src/tools/preview.py --action train
databricks bundle validate --strict -t test --profile YOUR_PROFILE
```

Preview validates configuration, not real source values. After
[deployment](databricks_deployment.md), run `train`; it pins the source and checks actual keys,
dates, labels and budgets before fitting.

## What you should see

Inspect `training_data.json`, `training_selection.json` when sampling is configured, and
`holdout_membership.json` in the model-producing MLflow run. They identify the source and saved
selection; [training reports](databricks_training_reports.md) explain the run graph and Inputs view.

Saved-artifact holdout evaluation uses sequential joblib prediction so parallel
Random Forest tree summation does not introduce last-bit metric differences into
exact approval evidence. This affects evaluation only: model hyperparameters and
normal training/batch inference parallelism are retained. Arbitrary numerical
runtimes may still be nondeterministic and must pass replay checks.

## Common errors and fixes

| Symptom | Fix |
| --- | --- |
| Row/byte budget exceeded | Choose an explicit sample or a smaller source window; there is no silent truncation. |
| Temporal boundaries are missing | Supply a fixed window or valid `rolling_calendar` or `rolling_days` settings; training does not switch to random. |
| Null labels remain | Enable a valid availability filter if labels are delayed; eligible rows still need a real target. |
| A date parses in the wrong timezone | Declare its source format and zone; cron timezone only schedules the job. |
| Approval cannot replay a candidate | Restore access to its saved source version and evidence; today's configuration cannot replace its holdout. |

For CV and search on the selected training rows, see [tuning](databricks_tuning.md). For delayed
labels arriving after predictions, see [monitoring outcomes](databricks_monitoring_labels.md).
