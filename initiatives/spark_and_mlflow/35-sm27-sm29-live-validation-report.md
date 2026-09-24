# SM-27/SM-29 live validation

Date: 2026-09-24. Profile: `skyulf`; personal workspace; `dev` only.
Implementation commit: `68f64935`; approved rehearsal plan: `556b63f3`.
Status: **PASS for the selected bounded Polars/serverless workflow**.

This is a functional lifecycle gate, not certification of production access
isolation. Both jobs ran as the same human owner. An exclusive production
alias writer remains a deployment precondition; concurrent train runs were
not exercised. Live contention evidence covers score requests only.

Databricks CLI: 1.17.0. Jobs used serverless environment 4 with
`mlflow==3.16.1` and the built `skyulf-core==0.9.0` wheel. Wheel SHA-256:
`ceb609354c9ab4fca0fee5bd7c740f68958d6dce5d0b48a8aeceabe622925414`.
The local two-engine precheck used Python 3.12.10, pandas 2.3.2,
Polars 1.44.1, MLflow 3.16.1 and scikit-learn 1.8.0. Those local package
versions are not a measurement of the remote serverless environment.

## Observed behavior

The generated Bundle deployed exactly two persistent jobs, train
`1106811466161550` and score `793298964293930`. Both have concurrency one,
queueing enabled and a 900-second test timeout. Deployment created no schema,
table or model: CLI schema lookups before and immediately after deployment
reported the schema absent. Those observations were recorded in the session
transcript, not archived as separate API responses. The separate ephemeral
setup run then created
`workspace.skyulf_sm27_sm29_20260924.source`, a CDF-enabled Delta table.

The test used 120 fit rows and 40 temporal holdout rows from source version
zero, a Skyulf `StandardScaler` and `linear_regression`, fitted with Polars.
The deterministic target was `10 + 4 * feature_value`. v1 omitted the model
intercept; v2/v3 included it. The heldout RMSE gate was `500.0` and minimum
improvement `0.1`, deliberately chosen for this controlled synthetic test,
not as production recommendations. All six regression metrics were logged
in each candidate's MLflow training run.

| Model | Heldout RMSE | Selection result |
| --- | ---: | --- |
| v1 | 148.375 | Passed the first-model absolute gate; initialized champion |
| v2 | 3.1776437161565094e-15 | Improved by more than 0.1; staged, promoted and retained v1 as previous champion |
| v3 | 3.1776437161565094e-15 | Equal performance; rejected with `insufficient_improvement` |

To test recovery, v2 training intentionally configured full rebuild against
the existing physical append-table name. Promotion succeeded; scoring
rejected the table/view conflict before writing. The independent audit found
both original outputs still at 160 v1 rows and Delta commit one, while
champion was v2. Correcting only the scoring target and rerunning score
created the complete v2 generation, then switched the view. No retraining or
alias rollback was needed. The two FAILED run entries below are this expected
negative test, including the serverless retry of its score task.

After ten new source inserts, the final independent audit and local assertions
verified:

| Object | Final state |
| --- | --- |
| `source` | 170 rows |
| `predictions_append` | 170 unique keys: 160 v1 predictions plus ten v2 predictions; Delta commit two |
| `predictions_full_v1` | Original 160 v1 rows, sum and Delta commit unchanged |
| `predictions_full_v2` | All 170 rows predicted with v2; Delta commit two |
| `predictions_full` | Stable view selecting `_v2`, 170 unique keys, original SELECT grant preserved |
| `model` | `champion=2`, `previous_champion=1`, no challenger or pending alias event |

The v2 table/view predictions matched the deterministic target to an absolute
error below `1e-8`. Physical generation properties matched the corresponding
registered artifact digests. Initial, challenger and promotion receipts were
committed. Rejected v3 triggered scoring with the unchanged champion; it was
a no-op. Two additional simultaneous score requests completed serially; the
second was observed queued. Neither added a Delta commit.

## Live run evidence

| Purpose | Run | Result |
| --- | --- | --- |
| Create the isolated 160-row CDF source | [994289042356535](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/435128515818943/run/994289042356535) | SUCCESS |
| First champion and automatic score handoff | [734876178121464](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/1106811466161550/run/734876178121464) | SUCCESS |
| Write 160 v1 append predictions | [1077735115421974](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/793298964293930/run/1077735115421974) | SUCCESS |
| Create 160-row v1 generation and view | [469943013477365](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/793298964293930/run/469943013477365) | SUCCESS |
| Promote v2; expected dependent scoring failure | [965294795399239](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/1106811466161550/run/965294795399239) | FAILED |
| Reject full-rebuild target using a physical table name | [619154417488](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/793298964293930/run/619154417488) | FAILED |
| Verify v1 outputs survived while champion became v2 | [868165136083032](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/490825661254282/run/868165136083032) | SUCCESS |
| Deny alias write by the read-only principal | [513771267496014](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/363313111655727/run/513771267496014) | SUCCESS |
| Retry score and activate complete 160-row v2 generation | [66863176859812](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/793298964293930/run/66863176859812) | SUCCESS |
| Append ten new source rows and audit | [1116599148703461](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/692533573335489/run/1116599148703461) | SUCCESS |
| Append only ten v2 predictions to the mixed-version target | [340142551545850](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/793298964293930/run/340142551545850) | SUCCESS |
| Append ten rows to the active v2 generation | [746243399428598](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/793298964293930/run/746243399428598) | SUCCESS |
| Reject equal-quality v3 and retain champion v2 | [185130188901742](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/1106811466161550/run/185130188901742) | SUCCESS |
| Automatic no-op score after rejected candidate | [225986416024048](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/793298964293930/run/225986416024048) | SUCCESS |
| First concurrent no-op score request | [980883336179936](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/793298964293930/run/980883336179936) | SUCCESS |
| Second score request waits then completes | [680565312006506](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/793298964293930/run/680565312006506) | SUCCESS |
| Verify final rows, predictions, aliases, metrics and receipts | [36123647896664](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/91117816108119/run/36123647896664) | SUCCESS |

## Permission and runtime boundary

The existing restricted principal `skyulf-sm16-restricted-20260922` received
only schema traversal, model EXECUTE and test-notebook read access. It read
champion v2, received `PERMISSION_DENIED` when trying to assign v1, and still
read v2 afterward. Its SELECT grant on the stable view survived `ALTER VIEW`.
This verifies the selected restricted identity, not every possible workspace
administrator. The Bundle's table-free alias coordination still requires an
exclusive serialized writer identity in each production deployment. UC alias
management requires model ownership; model execution alone does not permit
alias changes. See the official [model lifecycle permissions](https://docs.databricks.com/gcp/en/machine-learning/manage-model-lifecycle).

This rehearsal used Polars for remote FE/model execution and Spark for UC
table I/O. The same deterministic pipeline was checked locally with pandas
and Polars and produced identical v1/v2/v3 metrics. This does not establish
all models, policy clusters, company targets, Spark-native FE, streaming,
source updates/deletes, or multi-writer publication. The template's existing
explicit bounds and insert-only source contract remain in force.

No production resource was changed. The approved test schema, model, source,
two prediction outputs (append table and full view with retained generations),
two persistent Bundle jobs and audit notebooks remain for user inspection.
The train job has no active schedule. Temporary setup/audit/permission runs
did not add persistent Bundle jobs.

## Verification and corrections

Raw run outputs, assertions, wheel and the generated Bundle are retained
locally under `initiatives/spark_and_mlflow/rehearsals/sm29_20260924/`
(ignored by Git). The commands below show the original execution paths
before that local archive move.

Commands used from the repository root or generated project as appropriate:

```text
uv build --wheel --no-build-isolation --out-dir .tmp-sm29-live/wheel skyulf-core
databricks bundle init skyulf-core/templates/databricks --config-file .tmp-sm29-live/init.json --output-dir .tmp-sm29-live/generated
databricks bundle validate --strict -t dev --profile skyulf
databricks bundle deploy -t dev --profile skyulf
databricks bundle run train --no-wait -t dev --profile skyulf
databricks bundle run score --no-wait -t dev --profile skyulf
databricks jobs submit --no-wait --json @<scoped-request.json> --profile skyulf -o json
.venv/Scripts/python.exe .tmp-sm29-live/verify_local.py
.venv/Scripts/python.exe .tmp-sm29-live/validate_evidence.py
.venv/Scripts/mkdocs.exe build --strict --site-dir .tmp-sm29-live/site
```

- The implementation's 94-test local gate and signed commit passed before
  deployment; no library or generated-workflow fix was needed during this
  live rehearsal.
- The current wheel built successfully; strict generated-Bundle validation
  passed. English usage docs now explain absolute metric differences, first
  champion bootstrap evidence and scoring-only recovery; strict MkDocs passed.
- Live evidence assertions checked counts, uniqueness, metric gates, model
  versions, aliases, committed receipts, generation digests, prediction values,
  failure preservation, retries, no-op commits and view grants.
- Test tooling corrections: PowerShell UTF-8 BOM initially prevented a local
  submit JSON from parsing (no remote run was created); the audit changed from
  search-result tags to concrete `get_model_version(...).tags`; the poller
  learned to collect failed `INTERNAL_ERROR` run outputs. These affected test
  reporting only and were corrected before the final audit.
