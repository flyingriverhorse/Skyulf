# SM-41 live CDF recovery verification

Date: 2026-09-30. **The approved optional CDF-recovery slice passed live verification.**
This report extends [the local implementation record](135-sm41-cdf-recovery-plan.md).
That record's original no-live-execution boundary describes the earlier delivery;
the verified live work below is subsequent evidence, not a retroactive local claim.

## Scope and isolation

Profile `skyulf`, existing workspace `https://dbc-45604623-c18b.cloud.databricks.com`.
The rehearsal uses the isolated synthetic schema `workspace.sm41_live_20260930`
and workspace files under the corresponding personal rehearsal directory. Six
generated projects cover single-model, competition and multi-target scoring with
pandas and Polars. Execution uses the existing personal owner/admin identity;
this is not service-principal, separated-writer or restricted-account acceptance.

The score role uses concrete saved model/set version 1. The competition projects
reuse seeded saved models: these runs exercise their generated scoring path, not
competition training, candidate selection or alias promotion. The optional CDF
recovery flag is enabled in the main generated jobs and disabled in dedicated
negative probes. Main generated jobs use the visible graph:

`score -> recovery_needed -> recover_predictions -> scoring_report`

The final report joins normal and recovery branches using `NONE_FAILED`. Job
`SUCCESS` alone is never acceptance: task outcomes, notebook results, exact
receipts and independent persisted-output checks must agree.

Evidence files are in [the rehearsal directory](../../temp_test_artifacts_sm52/sm41-live/).
The recorded wheel is `skyulf_core-0.9.1-py3-none-any.whl`, with 302 packaged
Python modules and SHA256
`e310e3678e4c53e910f9b1a60a80f7c36289b8d59728d3a059c148e9bf5ceb1e`.
The upload helper checks packaged source bytes against the working tree; see
`wheel.json` and `driver.py`. Changes after a wheel build require a fresh comparison.

## Completed verification

| Evidence | Verified result | Boundary |
| --- | --- | --- |
| `local_retention_runtime.xml` | 364 passed, zero failures/errors/skips, 89.490 seconds | Local regression tests; not cloud acceptance |
| `result-run_cloud_tests-run_cloud_tests.json` | `exit_code=0`, exactly 20 cases, no case failures/errors/skips; 20 passed in 1373.92 seconds | CDF-expiry trigger injected; actual Spark/Delta reads, inference and writes |
| Six normal generated score jobs | 120 predictions each, target Delta v1, source watermark 0 | Normal branch and final report; no expiry in these runs |
| Four corrected negative probes | Two disabled-recovery failures and two recovery-budget failures at the intended tasks | Single/pandas and multi-target/Polars, not every matrix combination |
| `result-refresh_expiry-refresh_expiry.json` | All six targets retain their baseline row hash, receipt and Delta v1; 123 source rows materialized | Independent readback after negative probes and before successful recovery |

The 20-case cloud harness covers single-model and model-set paths on both engines.
It uses real partitioned Delta targets, saved temporal-carry models and consumer
views. Assertions cover complete and empty replacement, exact recovery replay,
incremental continuation, row/byte budget failures, changed model pins, real
source/target replacement and foreign target commits. The precommit failure is
deliberately injected immediately before the Delta writer; it is not a real
storage failure or uncertain network outcome. CDF expiry is injected only at
the CDF selection boundary. These cases do not establish real expiry detection.

The serverless runtime does not support the requested dynamic partition session
configuration. Real removal of absent partitions was tested; execution under a
dynamic session default must not be claimed. Local writer-option regressions
separately pin `partitionOverwriteMode=static` and `overwriteSchema=false`.

The harness notebook exits with a JSON result even when pytest fails. Its task
state therefore cannot replace checking `exit_code`, the expected case count and
each case's failure/error/skip list. The recorded result passes those checks.

### Normal generated-job matrix

| Layout | Engine | Run ID | Persisted baseline |
| --- | --- | --- | --- |
| Single model | pandas | 928350780532328 | 120 rows; Delta v1 |
| Single model | Polars | 745966009534591 | 120 rows; Delta v1 |
| Competition score | pandas | 287390279694642 | 120 rows; Delta v1 |
| Competition score | Polars | 348281546373192 | 120 rows; Delta v1 |
| Multi-target | pandas | 80737199892197 | 120 rows; Delta v1 |
| Multi-target | Polars | 484138357023269 | 120 rows; Delta v1 |

Every run has successful `score`, `recovery_needed` and `scoring_report` tasks,
condition outcome `false`, and `recover_predictions=EXCLUDED`. Twelve API
output/result pairs match without truncation or notebook errors. Six exported
HTML notebook results match the final report JSON and contain actual HTML display
output. Model-set reports intentionally omit continuation state (`set_history`)
from compact task values; other result fields agree with the originating task.

`baseline.json` initially matched the successful `prepare_expiry` notebook result;
its grants were then refreshed from `refresh_expiry` after the explicit test grants.
Its six receipts, counts and versions match the corresponding score results. The
independent baseline reader checked keys 0-119 and predicted values against the
synthetic data formula, then appended keys 120-122 to the source.

### Real retention-blocked CDF evidence

Initial VACUUM probes did **not** demonstrate physical CDF-file deletion:
`result-probe_after_vacuum-probe_after_vacuum.json` and the subsequent materializing
probe still read the older CDF rows. No physical-deletion acceptance is claimed.

The later actual source-retention probe produced the structured condition
`DELTA_UNSUPPORTED_TIME_TRAVEL_BEYOND_DELETED_FILE_RETENTION_DURATION` during
the CDF read. `activate_expiry` records source upper version 2, insert age
341.948653 seconds and `cdf_expired=true`. The classifier was updated for this
observed exact Delta condition, retaining the CDF-read-only boundary and negative
guards. This is real retention-policy rejection, not an injected error or proof
that VACUUM removed files.

The initial pinned-snapshot count alone was insufficient file-read evidence.
The subsequent `refresh_expiry` result materializes all 123 current source rows,
records source upper version 3, and again confirms the same real CDF condition.
Source version 1 is outside the configured retention window while the current
snapshot remains readable. A later recorded extension uses 15-minute retention
to preserve a usable current snapshot during longer-running tasks while the old
CDF remained outside retention. Final readback confirms restoration to seven days.

### Negative branch evidence

| Probe | Run ID | Actual failure and branch evidence |
| --- | --- | --- |
| Recovery disabled, single/pandas | 876435422096423 | `score` fails with `CdfRecoveryRequired`; no automatic replacement |
| Recovery disabled, multi-target/Polars | 36245692306523 | `score` fails with `CdfRecoveryRequired`; no automatic replacement |
| Recovery row budget, single/pandas | 83774124754372 | `score=SUCCESS`, condition `true`, `recover_predictions` fails with `Source increment exceeds max_rows`; report `UPSTREAM_FAILED` |
| Recovery row budget, multi-target/Polars | 597716018757823 | Same intended recovery-budget failure and blocked final report |

The independent refresh readback verifies unchanged rows, receipt and Delta v1
for all six prediction targets. Earlier multi-target probes with labels ending
`_fixture_path` failed on model-set fixture discovery. Those failures are excluded
from the four accepted negative cases and do not prove CDF or budget behavior.

### Grants and identity evidence

Original baseline `SHOW GRANTS` lists were empty. Comparing those lists alone
would not prove preservation of nonempty permissions. `explicit-grants.json`
records an explicit `SELECT` grant added to each of the six synthetic outputs
for its existing owner only. The refresh readback confirms those six nonempty
grants; it does not add access for another account. Post-recovery and final
readbacks both preserve these grants against the refreshed baseline. The explicit
test grants were then removed, restoring the original grant state.

Delta table IDs and Unity Catalog table IDs are separate recorded identifiers;
compare each with its corresponding prior value. Preserved grants for the owner
do not prove execution authorization for another user/service principal, denied
alias mutation, or separated writer permissions.

## Completed generated-job matrix

All six layouts/engines passed all four stages: initial 120-row score (Delta v1),
actual-retention recovery of 123 rows (v2), one-row incremental continuation (v3),
and an ordinary zero-row no-op retaining v3. Final outputs contain 124 unique
keys and independently verified predictions. Each final report matches the actual
selected branch; the recovery report explicitly displays `CDF recovery completed`.

| Layout | Engine | Actual-expiry recovery | Following increment | No-op |
| --- | --- | --- | --- | --- |
| Single | pandas | [661503575163371](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/109949428351639/run/661503575163371) | [1113008678919123](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/109949428351639/run/1113008678919123) | [681003772575048](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/109949428351639/run/681003772575048) |
| Single | polars | [773182350541296](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/38629030775060/run/773182350541296) | [411883326085765](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/38629030775060/run/411883326085765) | [795131814990145](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/38629030775060/run/795131814990145) |
| Competition score | pandas | [264809466773408](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/388993183101569/run/264809466773408) | [402202462887822](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/388993183101569/run/402202462887822) | [401032980929345](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/388993183101569/run/401032980929345) |
| Competition score | polars | [55499332644478](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/531079641254062/run/55499332644478) | [776830450576812](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/531079641254062/run/776830450576812) | [214444143695905](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/531079641254062/run/214444143695905) |
| Multi-target | pandas | [974984593330624](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/870359689132542/run/974984593330624) | [873655239286755](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/870359689132542/run/873655239286755) | [323081171643445](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/870359689132542/run/323081171643445) |
| Multi-target | polars | [769909931071266](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/184200348224463/run/769909931071266) | [879620819408017](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/184200348224463/run/879620819408017) | [844157538407923](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/184200348224463/run/844157538407923) |

`verified-normal.json`, `verified-recovery.json`, `verified-incremental.json` and
`verified-noop.json` pin task outcomes, counts, commit versions and report equality.
Every recovery receipt exactly matches the independently reconstructed request:
model name/version/digest, source and target identities, target v1, prior source
watermark 0 and the predecessor's source upper version. Five jobs pin source v3;
multi-target/pandas pins v4 after a metadata-only retention extension. All resume
at source v7 after the one new insert. The 123-row source contents stay unchanged
during the recovery cohort despite those metadata commits.

Independent run `31821177546610` verifies all recovered rows, unchanged Delta IDs,
nonempty SELECT grants and byte-equivalent prior prediction snapshots at v1.
It restores source retention and appends the following record. Final readback run
`1048082544452238` verifies 124 correct unique predictions per output, Delta v3,
the one-row incremental receipt, unchanged IDs/grants and seven-day source retention.
Ordinary no-op does not substitute for exact recovery replay; exact request replay
is independently covered for all four engine/artifact pairs by the 20-case harness.

### Full-rebuild publication and empty source

Four additional production `full_rebuild` publication cases **passed** in run
`902434798161618` (154.79 seconds), with `passed=true`, four successful case
records and no cleanup issues. Each case writes two predictions, recovers an
empty source, then publishes one new CDF row: target versions 1 -> 2 -> 3 and
output counts 2 -> 0 -> 1. Source/target identities, consumer view definitions,
model generation properties, prediction values and old history remain valid.
Single models exercise the real versioned-generation view activation path;
model sets exercise their supported `separate_views` projection over a stable
physical output table. The CDF boundary is explicitly injected in these four
cases; actual inference, Spark/Delta I/O, production actions and cleanup are real.
Exact recovery-request replay is covered by the separate 20-case harness.

## Final state and validation

`final-cleanup.json` verifies all 12 generated train/score jobs are PAUSED, retain
their original generated task graphs, and have no active runs. No owned ephemeral
test run remains active. Six explicit owner SELECT grants were removed after their
preservation was verified; original grants, Unity Catalog table IDs and owners
are restored/preserved. The synthetic schema, six prediction outputs, source,
pinned model artifacts and run history remain available for inspection. UUID
tables/views created by the 20-case and four-case harnesses were cleaned by their
ownership-checked finalizers. No unrelated workspace objects were removed.

The final working tree matches all 302 Python modules in the cloud-tested wheel.
Post-fix local verification: 364 tests passed; full repository Ruff, CI format scope
(1,171 files), full CI Ty scope and backend/Core Lizard CCN <= 10 passed. Earlier
template/CLI verification remains recorded in report135. No frontend or dependency
changes were made in this live-verification follow-up. No commit was requested.

SM-41 remains PARTIAL for explicit source-replacement recovery and generation
retention/cleanup. Physical CDF-file deletion after VACUUM, service-principal
separation, classic compute and additional company workspaces are not established
by these personal serverless results. Existing permission/network/schema failures
remain excluded from automatic recovery; the live negative cases above do not
claim separate-account authorization testing. The actual classifier addition is
the exact structured retention condition observed here, restricted to CDF reads.

## Requested commit verification

The user subsequently requested committing this slice. Fresh verification passed
428 runtime/template tests and all 143 opt-in CLI generation/deployment tests
(571 distinct cases). The CLI cases were initially skipped until explicitly
enabled; none remained unverified in that selected set. Integration collection
passed with 2,537 cases. Repository Ruff, CI format scope (1,171 files), full CI Ty,
backend/Core Lizard CCN <= 10, generated schema freshness and staged pre-commit
hooks passed. The full backend/Core test suites and cloud jobs were not rerun for
this commit; the live acceptance above remains the cloud evidence.

An initial sandboxed pytest attempt could not access the Windows temporary
directory. The rerun with the required local access passed. Existing sklearn and
legacy-selection warnings remain; the CLI run also reported a non-fatal pytest
cache-write warning. Unrelated historical queue deletions and temporary review
files are outside the requested SM-41 commit.
