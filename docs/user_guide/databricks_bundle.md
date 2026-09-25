# Local-engine Databricks Bundle

The custom Skyulf template generates one editable Bundle with `dev`, `test`,
`syst` and `prod` targets. It fits and predicts with pandas or Polars. Spark
reads bounded Unity Catalog Delta rows and publishes predictions; local
feature engineering and model prediction are not distributed Spark work.

Initialize a project from a Skyulf checkout:

```powershell
databricks bundle init skyulf-core/templates/databricks --output-dir ./generated
```

The short path asks for project name, engine, one existing source row-key
column, model-change and model-selection modes, optional retraining mode and cron, serverless
or policy-backed job compute and the existing `dev`
catalog/schema. Serverless is the default. Reviewable
noninteractive examples are in `skyulf-core/templates/databricks/examples/`. The generated
project has its own `databricks.yml`; Skyulf's root has no Bundle config.

`dev` uses the selected CLI profile's workspace host. `test`, `syst` and
`prod` each have a different placeholder host and catalog in the generated
file. Edit each target's host, catalog and input/output/metadata schema
bindings, then validate with its designated profile. No Danske, Danica or
personal-workspace value is built into those three targets. Policy-backed
compute uses the policy name, runtime, node type and cost tag supplied at
initialization. Serverless compute needs none of those fields.

For a company policy requiring `PayingRegNo`, select that value as
`cost_tag_key` and supply the approved registration number as `cost_tag_value`.
The `paying-reg-no-init.example.json` example demonstrates both settings;
the generic default remains `CostCenter`. This tag applies to policy-cluster
compute. Selecting it does not assign a model risk category.

Initialization also asks for an optional `risk_category`, such as `Low`,
`Medium`, or `High`. Leave it blank to omit the tag. The value remains editable
in `config/workflow.json` and is recorded on future training runs and model
versions; it does not change promotion gates or relabel existing versions.

## Where the workflow lives

The generated `src/workflow.py` reads job widgets and JSON, calls the installed
library, and serializes its result. Reusable behavior lives in
`skyulf.integrations.databricks.local_workflow`: `resolve_target_config` binds
target names and `run_action` executes train, train_monthly or score.
`prediction_output` validates and creates output tables and safely switches
full-rebuild views. Both reuse the existing training, inference and registry
services. Importing them does not create a Spark session or cloud resource.

Edit the business pipeline in `config/workflow.json`. Model/preprocessing
choices remain project configuration; common workflow fixes ship in the
Skyulf wheel instead of requiring edits to every generated notebook.

### Independent library policies (SM-32 in progress)

Direct `run_action` callers can now separate these two decisions:

| `score_model_selection` | `promotion_policy` | Behavior |
| --- | --- | --- |
| `pinned_version` | `manual_approval` | Register/evaluate the contender; score keeps the configured version |
| `pinned_version` | `automatic` | Apply promotion gates; score still keeps the configured version |
| `champion` | `manual_approval` | Register/evaluate the contender; score follows the existing champion |
| `champion` | `automatic` | Apply promotion gates; the next score follows the resulting champion |

For example, a library configuration containing `model_version: "2"`,
`score_model_selection: "pinned_version"` and `promotion_policy: "automatic"`
continues scoring with v2 after a successful promotion of v4. Selecting
`champion` instead resolves the controlled alias once per score run. A missing
champion fails before output preparation; there is no fallback to latest.

Both new fields are required together. Remove `model_selection_mode` when
using them; mixed or partial configurations fail before training or scoring.
Legacy `auto_champion` retains champion/automatic behavior, and legacy
`pinned_version` retains pinned/manual behavior, with a deprecation warning.
Automatic promotion still requires `quality_threshold`, including when
scoring is pinned. Scoring never updates the caller's configured version.

This first slice is available at the library boundary. The generated Bundle
still uses its existing `model_selection_mode` and job graph. Do not migrate
only its JSON: the lifecycle actions and matching job handoff must be delivered
together. `manual_approval` currently means no automatic promotion; approve,
reject and rollback are separate operator actions. The library now supports
`approve`, `reject` and `rollback`. `previous_challenger` and the matching Bundle
choices/action handoff remain open in SM-32.
No new cloud deployment is implied by these library changes.

### Approve an existing candidate without training

Use an explicit manual policy and the comparison returned by the earlier
training job. Review that saved result before passing its version and digest:

```python
import hashlib
import json
from dataclasses import asdict

from skyulf.integrations.databricks.local_workflow import run_action

# candidate is the result of an earlier training job; no training runs here.
comparison = asdict(candidate.comparison)
comparison_sha256 = hashlib.sha256(
    json.dumps(comparison, sort_keys=True, allow_nan=False).encode()
).hexdigest()
receipt = run_action(
    spark,
    config,  # promotion_policy="manual_approval" and both independent fields
    "approve",
    candidate_version=candidate.model_version,
    comparison_sha256=comparison_sha256,
    expected_champion_version=candidate.comparison.champion_version,
)
```

Approval downloads `candidate_comparison.json` and `candidate_training_spec.json`
from the run associated with that registered version. It verifies the digest,
current metric policy, expected champion and controlled challenger receipt.
It then reads the original Delta version and evaluation window and rechecks
quality using the existing Core comparison/promotion services. The original
engine is preserved; current read budgets may tighten the saved limits.

The action does not train, register or upload a model, rewrite `model_version`,
or publish predictions. A later score run follows the configured selector.
Retrying an unchanged successful approval returns its original committed
receipt; changed champion, pending writes or incompatible evidence fail.
Training runs created before the saved specification was introduced cannot
use this action without explicitly supplying that missing provenance through
a separately reviewed migration; there is no fallback to current training dates.
Use the same externally serialized lifecycle writer as training. Bundle widgets
and automatic score handoff for operator actions are still pending.

### Reject a candidate or roll back a promotion

To decline the candidate above, use the same version and evidence digest with
an explicit manual policy. This is an alternative to approving that candidate:

```python
decision = run_action(
    spark,
    config,
    "reject",
    candidate_version=candidate.model_version,
    comparison_sha256=comparison_sha256,
    expected_champion_version=candidate.comparison.champion_version,
    rejection_reason="Business review requires another candidate",
)
```

Rejection retains both aliases and the evaluation metrics/status. It records
`approval_status=rejected` and the human-readable `approval_reason` separately.
The reason must be nonempty and at most 256 UTF-8 bytes. A rejected version
cannot be implicitly approved, restaged, renominated or made the first champion.
There is no reopen action in this slice; a later training run creates a new candidate.
Neither rejection nor rollback reads training rows, fits or registers a model.

To undo an earlier completed promotion, pass that promotion's receipt:

```python
reversal = run_action(
    spark,
    config,
    "rollback",
    promotion_receipt=receipt,  # kind="promotion", not first-champion initialization
    expected_champion_version=receipt.new_version,
)
```

Rollback is available under either promotion policy. It restores the previous
champion only when the saved transition still matches controlled registry state.
A separately verified challenger is retained. Repeating an unchanged rejection
or rollback returns the same committed receipt; incompatible evidence,
conflicting alias state and unresolved writes stop the action.

All lifecycle actions must use the same externally serialized writer as training.
No additional control table is created. These actions do not launch score or
rewrite its version pin. A later `champion` score follows the restored alias;
a `pinned_version` score continues using its configured version. Rollback does
not undo predictions already written; scoring's model-change policy still applies.

## What is created

`bundle deploy` creates only two jobs and uploads their code:

| Job | Purpose | UC objects created when run |
| --- | --- | --- |
| `train` | Fit one candidate and log held-out metrics | One registered model/version |
| `score` | Score initial and later CDF rows | Prediction output and rows |

The default project has no schedule. Choosing `monthly_paused` at initialization
adds a paused schedule to the existing `train` job, without adding a third
job or running it at deployment. The Quartz cron and timezone are selected
at initialization and remain editable Bundle variables. No endpoint or Unity Catalog table is
created by deployment alone.
`pinned_version` keeps manual model selection. `auto_champion` uses the same
train job to compare candidate and champion on a pinned temporal holdout. The
selected `metric`, `min_improvement`, and absolute `quality_threshold` stay
editable in `config/workflow.json`. A first champion requires a numeric
absolute threshold because no prior version exists for comparison. Later
versions must pass that threshold and improve on champion by the chosen
minimum. A passing candidate is staged and promoted through checked registry
receipts; an ineligible candidate leaves champion unchanged. The train job
then calls the existing score job. No third job or control table is added.

In both modes, registration nominates `@challenger` before comparison. A tied
or worse candidate retains that alias with `validation_status=rejected` and
a reason; comparison errors retain it with `validation_status=error`.
New contenders replace the pointer without deleting earlier version evidence.
Manual mode records these results without promotion or changing pinned scoring.
Training resolves the current champion; a stale explicit `champion_version`
fails before fit. Generic Core training remains alias-free unless a caller
supplies the explicit `on_registered` lifecycle callback.

Training runs and model versions share readable metadata: `train_data_version`,
`test_data_version`, `train_data_destination`, `test_data_destination`,
`model_type`, `candidate_date_tag`, and `engine`. Optional `risk_category` is
editable in workflow.json. Since fit and evaluation split one Delta snapshot,
their table names and versions match; `train_start`, `test_start`, and
`data_end` explain the split. The exact dataset identity is saved in
`training_data.json`. Run tags use `task=training`.

Validation reasons use plain English. Technical event tags retain unique IDs
and evidence hashes for rollback/recovery; their JSON fields are `action`,
`from_version`, `proof`, `parent`, `state`, and `previous`. `from_version`
identifies the version previously held by the affected alias, not necessarily
the previous champion. These are audit records,
not settings to edit. Earlier compact receipts remain supported.

In automatic mode, score resolves `@champion` once to a concrete version per
run. In both modes, only the serialized train job identity may write this
model's aliases; other alias-write grants must be removed before training.
Alias promotion and Delta scoring are separate transactions. If scoring fails
after promotion, the last successful prediction output remains and the score
job must be retried. Unknown alias outcomes need receipt reconciliation.
Skyulf marks an alias transition as pending before writing it; automatic
training and scoring stop until that pending event is reconciled. An existing
champion alias set outside this controlled lifecycle also needs reconciliation
before automatic mode can use it.

For example, a regression project can select its gate in the generated config:

```json
{
  "engine": "polars",
  "model_selection_mode": "auto_champion",
  "metric": "heldout_rmse",
  "min_improvement": 0.1,
  "quality_threshold": 5.0,
  "model_change_mode": "incremental_append"
}
```

These example values mean RMSE must be at most `5.0`; a later candidate must
reduce champion's RMSE by at least `0.1` on the same holdout. Improvement is
an absolute metric difference, not a percentage. Skyulf derives the direction
from the metric: RMSE/MAE/log loss are minimized, while R²/accuracy/F1 are
maximized. Use a metric supported by the configured model task. The first
comparison reports `reason=no_champion` and `eligible=false` because no
baseline exists; successful bootstrap is recorded separately as
`alias_change.kind=initial` after the absolute gate passes.

If promotion succeeds but scoring fails, the train job reports a failed
dependent score task. Correct the scoring problem and run `score` again;
retraining is unnecessary. Full-rebuild output continues to expose the last
successfully activated generation during this recovery.

The four relevant Unity Catalog names have different roles:

| Name | Meaning |
| --- | --- |
| `training_table` | Existing labeled input table, pinned to a Delta version for training |
| `score_source_table` | Existing CDF-enabled source of rows to predict |
| `prediction_table` | The output table in append mode, or the stable active view in full-rebuild mode |
| `model_name` | Registered Unity Catalog model, not a table |

The first two references point to **the same existing table by default**.
Neither reference creates a table. Separate them only when labeled training
data and new scoring data have different lifecycles. If they stay together,
the first `score` run predicts all existing rows, including historical labeled
rows; review whether that is intended for your use case.

In `full_rebuild`, a version-specific physical table records the model name,
version and artifact digest. Scoring rejects an existing table at that name
when its model provenance differs. A new model with the same numeric version
needs a different logical prediction name or a new model version.

The initialization question `row_key` defaults to `entity_id`, but you can
choose an existing `customer_id` column. It becomes `row_keys` in the generated
configuration, and the same column appears in the prediction table. It must
be non-null, `STRING` or `BIGINT`, and unique across the initial data and all
later inserts. For multiple predictions per customer, edit the generated
configuration to a composite key such as
`"row_keys": ["customer_id", "observation_id"]` before deployment. The Bundle
does not create an ID in the source. Keep keys out of `input_columns`. With
`customer_id` in the output, a query
can find a customer's predictions:

```sql
SELECT customer_id, prediction, model_version
FROM catalog.schema.predictions
WHERE customer_id = 'C123';
```

## First run

Build and place the matching Skyulf wheel in the generated project's `dist/`,
then edit `config/workflow.json` for real source columns, preprocessing, model,
temporal split and size limits. The JSON values are an example, not a dataset.
Enable Change Data Feed on the scoring source before later inserts arrive.

```powershell
databricks bundle validate --strict -t dev --profile <profile>
databricks bundle deploy -t dev --profile <profile>
databricks bundle run train -t dev --profile <profile>
```

In manual mode, inspect the registered model version, put that concrete value
in `model_version`, redeploy the changed JSON, then run `score`. In automatic
mode, inspect the train and dependent score task results. The first score
rejects a missing source, disabled CDF, unsuitable row keys, a model output
mismatch or an existing target schema mismatch before creating prediction
output. It checks initial row count against `max_rows`; each score
also checks decoded transfer bytes against `max_bytes`. Existing tables are
never overwritten. The first score processes the current source
snapshot; later runs process only new inserts since the committed Delta
receipt. A repeat without new rows is a no-op. No monthly date or source
version is entered for each run.

In manual mode, the first candidate does not become champion automatically;
scoring stays pinned to its configured version. At initialization, choose
`incremental_append` to keep
v1 predictions and score only later source inserts with pinned v2. Choose
`full_rebuild` to write a new physical `<prediction_table>_v2` generation,
switch the stable `prediction_table` view after a successful complete score,
and append future inserts to v2. The previous generation remains available;
a failed rebuild leaves the active view unchanged. Full mode needs view
creation/ownership privileges and cannot silently turn an existing append-mode
physical table into a view. `max_concurrent_runs: 1` serializes the generated score job, but
does not coordinate other jobs. Grant prediction-table writes only to this
job's identity. For multiple publishers, use Skyulf Core's shared admission
provider. Online endpoints and Spark-native FE/model execution remain
separate work.

The optional monthly `train` schedule defaults to 03:00 UTC on day three and
starts paused. Edit its Bundle cron and timezone variables for the desired
monthly run time. After configuring real labeled data and verifying a manual
run, unpause it deliberately. Each run pins the source's latest Delta version and uses the
first day of the current UTC month as the label cutoff. The preceding month is
holdout; `monthly_lookback_months` (default four) controls the full window.
Only labels available by the cutoff are eligible. `@champion` is resolved to a
concrete version for comparison if present. Manual mode leaves selection
unchanged; automatic mode applies its metric gates and invokes score. The source version is
pinned at run start, so `label_at` must faithfully record availability.

The older SM-20a personal serverless rehearsal passed, but its jobs and test
schemas were removed at the user's request. The subsequent clean generic
`dev` rehearsal trained a Polars model from 600 real taxi rows, wrote 600
initial and 50 later predictions to one table, and replayed without another
commit. The later two-job design passed a separate personal serverless
rehearsal: 650 existing source rows, one later insert, and a no-op replay
left one prediction table with 651 rows. The `test`, `syst` and `prod`
placeholders have not been deployed in a company workspace.
