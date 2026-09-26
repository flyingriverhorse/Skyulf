# Current handoff: SM-33H3 DONE; SM-34 READY

Post-acceptance refactor: `local_pre_split.fixed_columns` delegates to per-node
validators; `FIXED_TYPES` derives from the same admission rule table. This is
a behavior-preserving source refactor after the live run recorded below. The
live wheel digest identifies the pre-refactor implementation, not this newer
source. No additional Databricks run or deployment was made for the refactor.
Verification: 145 affected tests passed; 7,510 deterministic valid/invalid
configurations matched the previous return values/order and exact exceptions.
Full ty, scoped Ruff/format and diff checks passed. Ruff McCabe complexity for
`fixed_columns` fell from 49 to 4; its length fell from 214 to 18 lines.

Updated 2026-09-26. Requested prior work was committed as `55b31ca9` with DCO,
227 passing tests and applicable hooks; no push. New H2 changes are uncommitted.

H2 saves versioned `training_filter_evidence.json`, bound to the saved comparison:
recipe/source identity, ordered pre-filter/survivor/train/holdout membership and
counts. Automatic staging and manual lifecycle actions verify the registered
artifact's engine/source and saved recipe. Approval replays the pinned snapshot;
current editable Python cannot change it. Legacy evidence retains its old IDs.
No new Delta control table; training eligibility remains separate from scoring.

Final changed-path suite: 144 passed (five existing policy deprecation warnings).
The review found and fixed a JSON-null receipt bypass: two real MLflow tests
failed before the fix and passed after; 102 related lifecycle tests then passed.
Broader earlier suite: 248 passed, overlapping and before final automatic-path
tightening. Fresh-process MLflow tests cover both engines, CV, targetless
prediction and evidence/engine tampering. Parent local SQLite harness exercised
three versions per engine, bootstrap/promotion/rejection/rollback; it did not
exercise Delta prediction writes. All 15 real WSL Spark/Delta training-date tests,
56 CLI generation tests, full ty, scoped Ruff/format, strict docs and strict dev
Bundle validation with the current wheel passed.

Scoped round-two review approved the null-receipt fix and exact output checks.
The explicitly authorized [live run 926706369150614](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/886322042039901/run/926706369150614)
succeeded in 701.797 seconds; all three tasks passed. Both engines trained
three versions with CV/MLflow, approved v1/v2, rejected v3 and rolled back to v1.
The separate score task wrote 240 then 3 predictions per output; exact key and
prior-prediction identity stayed intact. A further run was a no-op at Delta
version 2. Local artifact and MLflow predictions matched without target/age.

Resources remain in `workspace.skyulf_lifecycle_test` with prefix
`sm33h2_20260926_r1_`: one source, two prediction tables and two three-version
models. Outputs have 243 rows each; champion v1 is intentional rollback state.
No new schema or persistent job was created. Report 62 has exact names, metrics,
run IDs and validation limits. The ignored rehearsal folder retains raw results.

Wheel SHA-256: `d5982ff99ebfc8e9e5e572be05ae1b60a3fd88d974084943cf8336945cba186d`.
Path: `.cache/sm33h2-cli-final/test_cli_emits_independent_pol0/output/`
`sm33_generated/dist/skyulf_core-0.9.0-py3-none-any.whl`.
H3 local implementation, scoped review and approved personal serverless
acceptance passed. Run 848857785722024 succeeded in 437.664 seconds, all three
tasks SUCCESS. SM-34 is READY. Matrix63 covers 62 non-model IDs,
58 calculators and 80 mode rows with bounded evidence and explicit limitations.
Fixed pre-split normalization, deterministic Deduplicate, opt-in custom filters,
raw-feature once-only replay, target semantic checks and saved custom approval
are implemented. Generated Python examples and the walkthrough are updated.

Final affected suites: 289 passed; CLI generation: 56 passed. Full ty, scoped
Ruff/format, strict docs and strict generated dev Bundle validation passed.
Thirty new ordinary-route tests cover 11 node recipes on both engines, empty
vectorizer schema/dtypes and executable custom-filter examples. Prior 30 node
fixtures passed all 60 runs. TF-IDF/hashing empty test-partition failures were
reproduced and fixed in the shared Core helper. Optional H3Index/sentence-model
packaging, temporal history/CV policy and custom value normalization remain
explicit limitations; see report62 and matrix63, not a blanket all-mode claim.

H3 wheel: `.cache/sm33h3-cli-final/test_cli_emits_independent_pol0/output/`
`sm33_generated/dist/skyulf_core-0.9.0-py3-none-any.whl`.
SHA-256: `ad08f6d91e3f329eb03294e36ed7d2004019fe14bfcca13cbda51aa196369c24`.
All 248 Python files matched the current source. The earlier H2 digest above
identifies only the already-completed H2 run.

H3 live: both engines retained 47/240 rows (35 train, 12 holdout), logged 30
identical metrics, and approved champion v1 using saved custom source in a
separate task. Each score wrote 240 then 3 rows; original predictions stayed
unchanged; noop=true left Delta version 2 unchanged. Direct local/MLflow
predictions matched without target/filter-only inputs. Report62 and rehearsal
README contain exact resources, run IDs, metrics and limits.

Next: SM-34 independent score/train schedules and training windows. Resources
remain in workspace.skyulf_lifecycle_test with sm33h3_20260926_r1_ prefix:
one 243-row source, two 243-row prediction outputs, two one-version models.
No retry, new schema or persistent job. H2/H3 changes are uncommitted; no push.
Leave unrelated `.tmp-review-model/` untouched. Matrix/rehearsal files match
ignore rules and need explicit inclusion in a future requested delivery commit.

## Previous handoff: SM-33H1 committed; SM-33H2 IN PROGRESS

Updated 2026-09-26. Prior temporal guards and cleanup plan committed as
`522c6e82` with DCO and passing hooks; no push. H1 and the H3/custom-step plan
were committed as `55b31ca9`; 227 tests, strict docs and all applicable hooks
passed. Continuation is H2; its new changes are not part of that commit.

Generated src/preprocessing.py now has optional build_pre_split_steps() beside
build_preprocessing(). Initial training-only filters are existing Core
DropMissingRows and ManualBounds. Learned/unknown steps are rejected; both
engines preserve keys, order and target pairing. Filter-only columns are read
without becoming model features. Limits/sample membership precede cleanup;
filters never refill a sample. Invalid keys remain errors before filtering.

Saved training spec carries the recipe; pre_split_filters.json logs per-step
counts. Automatic comparison and manual approval replay the same filtered split.
This minimum replay is H1; full code/survivor integrity, fresh-process MLflow and
live acceptance remain H2. Scoring never applies target-based training filters.

Final affected group: 71 passed; CLI generation: 56 passed; real local WSL Delta:
1 passed. Earlier affected group: 202 passed. Broader Bundle/runtime/CV had 128
passes and four stale notebook fixtures; repaired notebook module: 16 passed,
included in the final 71. Ruff/format, full ty, strict docs and generated dev
Bundle validation passed. Scoped review approved all fixes. No live job/deploy.

Next: SM-33H2, then newly requested SM-33H3, then SM-34. H3 routes every existing
node/mode to the correct pre-split or ordinary preprocessing phase, reusing Core
and the same Python file. It includes fixed normalization, existing dedup and
artifact/inference replay; learned nodes remain after split and resampling stays
train-only. No new generic row-filter node, group split or data-quality threshold
system is requested now; those suggestions are parked. Report 62 has remaining
acceptance tasks. Keep unrelated .tmp-review-model/ untouched. New Core test
module must be included at the next requested commit.
Follow-up planning adds opt-in custom pre-split logic to H3 in the same Python
file, reusing saved source with explicit row effects and replay checks. H1 still
rejects custom steps; this extension has not been implemented.

## Previous handoff: Core temporal guards verified; SM-33H1 READY

Updated 2026-09-26. At the user's request, audited lag/rolling before starting
the pre-split cleanup tasks. Fixed missing declared sort columns silently using
input order, invalid direct lag shifts, and current-target rolling leaking the
same row's answer. Core enforces the target rule before fit for all split
placements and direct calculator target context; backend admission reuses it.

Pandas/Polars grouped causal prefixes and batch-local history behavior are pinned.
Artifacts retain configuration only: automatic training-history retrieval, result
availability, forecast horizons and context trimming are NOT delivered. Those
remain SM-36a; do not admit lag/rolling into the initial pre-split cleanup recipe.

Validation: 1,981 Core tests and 1,731 backend tests passed; scoped Ruff, repository
ty and strict MkDocs passed. Regression reproduction was red before the fix
(17 Core failures, 3 backend failures). Documentation is in
`docs/user_guide/preprocessing_placement.md`; report 62 records scope and evidence.

Changes are uncommitted. Next: SM-33H1, then SM-33H2, then SM-34. Report 62 is
ignored by Git: force-add only that report at the next requested commit.
No cloud execution or deployment. Leave `.tmp-review-model/` untouched.

## Previous handoff: pre-split cleanup audit complete; SM-33H1 READY

Updated 2026-09-26. Baseline SM-33F/G committed as `8a88b2b2` with DCO and
passing hooks. User asked to inspect existing pre-split nodes and leakage guards
and add concrete tasks. This turn changes plans/queue only, not runtime.

Report 62 records the operation-level matrix, reuse paths and acceptance steps.
The existing Core classifier already distinguishes fixed vs learned modes and
is shared by backend DAG checks. 940 Core and 109 backend tests passed; real
DropMissingRows -> ManualBounds execution preserved keys/labels on both engines.
Deduplicate is currently classified learned/blocked before split. Lag/rolling
being flagged not learned does not establish point-in-time safety.

Next: SM-33H1 (initial row-filter recipe in the same preprocessing.py using Core),
then SM-33H2 (saved cleanup/holdout evidence and lifecycle replay), then SM-34.
Do not call a no-split advisory a rejection gate. Unknown custom code remains
blocked before split; no unrestricted frame callback. Shared feature transforms,
duplicate/group policy, scoring exclusions and temporal context remain SM-36a.

Plan changes are uncommitted. Report 62 is under the ignored initiatives tree;
force-add only that report at the next requested commit. No cloud run/deployment.
Leave `.tmp-review-model/` untouched.

## Previous handoff: SM-33G answer-driven setup verified

Updated 2026-09-26. User approved conditional questions without reading data.
Initializer now asks how a used date column is stored: timestamp, local timestamp,
date or text. Text asks whether it includes an offset, a local clock or only a day.
Only applicable format/timezone/date-only questions follow. Disabled observation
or result-date use hides the whole branch. Existing CV, scheduling and policy
compute visibility was tested. No new runtime config or data-reading logic.

37 template/preview tests and 56 real CLI generation tests passed; scoped Ruff,
ty, strict MkDocs and strict dev Bundle validation passed. Reports are in the
SM-33G section of report 61; details in the Bundle guide and generated README.
The user requested one SM-33F/G delivery commit, including report 61 and these
handoff/queue updates. Fresh tests and commit hooks gate that commit. Changelog
line endings were normalized to the repository's existing LF rule; no global
Git settings were changed. No deployment or cloud execution was performed.
Commit verification: 243 local tests passed (one optional PySpark skip), 56 CLI
generation tests passed, and all applicable pre-commit hooks passed.
SM-34 remains the next READY task. Exclude unrelated `.tmp-review-model/`.

## Previous handoff: SM-33F initializer usability verified

Updated 2026-09-26. User-requested SM-33D/E commit completed as `516b3f86`
with DCO sign-off and all applicable hooks passed; no push. The new usability
changes below are uncommitted, separately from that requested baseline commit.

SM-33F accepts comma-separated key/feature names, task-specific model/metric
menus and a chosen prediction output name. Source/output tables, dates, snapshots,
availability, CV and scheduling have concrete explanations. Preprocessing moves
to generated src/preprocessing.py: standard Core steps plus self-contained custom
Calculator/Applier classes. Training snapshots code with learned state; local and
MLflow loading restore it, with fold-local CV and model-version isolation. Keep
the JSON preprocessing list empty; model settings remain in config/workflow.json.
`scheduled` enables the selected cron after deployment, including dev. Runtime
`train_monthly` selects data automatically; it does not force monthly frequency.
See [report 61](61-bundle-initializer-usability.md) for input changes and evidence.

Validation: 56 actual CLI generation tests, 15 custom Python tests, 201 integration
regressions (one optional PySpark test skipped) and 11 existing local MLflow/
preview tests passed. Full ty, scoped Ruff/format, strict MkDocs and strict generated dev
validation passed. The current wheel contains both new source-loading modules;
resolved six-month cron is UNPAUSED and sync includes the Python recipe. Review
findings about regression wording and stale JSON instructions were fixed. No live
deployment or cloud resource change. Existing user-generated projects were not edited.

Next READY remains SM-34 (independent score/train scheduling and further window
controls). SM-33F brings forward only SM-36a's self-contained Python path;
dependency shipping, sibling modules, eligibility/output rules remain open.
At the next requested commit include updated reports 37/58 and force-add report 61; keep
ignored rehearsal outputs and unrelated `.tmp-review-model/`
out of the commit. Reports 59/60 were included in `516b3f86`.

## Historical snapshot: SM-33E acceptance before commit

# Current handoff: SM-33E live acceptance passed

Updated 2026-09-26. Baseline `fe3897c4`, branch `090`; SM-33D and SM-33E are
uncommitted (SM-33D staged, most SM-33E changes unstaged). No push or commit in
this continuation. Leave unrelated `.tmp-review-model/` untouched.

SM-33E adds guided Bundle setup, Core registry model/preprocessor discovery,
offline preview, optional CV/sampling/windows/date parsing and English examples.
Custom ordered preprocessing and estimator parameters remain editable in
workflow.json. Advanced search and custom/multiple-model scenarios remain later.
See [report 60](60-sm33e-guided-setup-and-live-validation.md).

Local checks: 164 focused native integration tests, 48 actual CLI generation
tests, and 110 evaluation/lifecycle regression tests passed (one native Spark
module skipped). Suites overlap. Full ty, scoped Ruff/format, strict MkDocs,
wheel build and strict generated dev validation passed; review findings fixed.

Live personal serverless acceptance passed for pandas classification and Polars
temporal regression: imputation/scaling, Random Forest, three-fold CV, MLflow
metrics/artifacts, saved-evidence approval, 240 initial predictions, three new
rows and no-op with target Delta version unchanged at 2. Final read-only audit
`626395227690495` and local saved-response assertions passed. Live fixes address
parallel forest evaluation sums and serverless automatic retries; report 60
records failures and limitations. A timestamp-preserving test harness copy was
also corrected, with remote config read-back added before model-specific runs.

Existing schema `workspace.skyulf_lifecycle_test`; source `sm33e_source_r1`.
Models `sm33e_pandas_classification_r1` champion v1 and
`sm33e_polars_regression_r1` champion v3, each with `_predictions` output.
Existing train job `155738051514173` and score job `684955889505992` now point to
the pandas configuration. Both are idle after acceptance; schedules stay
paused/absent. No new persistent Bundle jobs/schema or destructive cleanup.
Helpers/evidence: `rehearsals/sm33e_live/`; current deployed project remains
`rehearsals/sm30_live/generated/skyulf_lifecycle` with its deployment identity.

Next READY: **SM-34**, independent paused train/score schedules and explicit
window controls. Preserve SM-33D's independent full/fixed/rolling selection and
timezone contracts; cron must not choose a split/window implicitly. Company
production acceptance remains SM-43b. At the next requested commit include all
intended SM-33D/E files and force-add exactly reports 59 and 60 (initiative ignore
rules); do not add generated rehearsal projects, response dumps or wheels.

## Historical snapshot: SM-33D local verification

# Current handoff: SM-33D locally verified

Baseline `fe3897c4` is committed on branch `090`; SM-33D changes are uncommitted.
Implemented optional Core CV with fold-local FE, separate MLflow CV evidence,
seeded Spark-side training sampling with pinned membership, and independent
full/fixed/rolling windows with explicit business timezone. Both local engines
are covered. See [validation and boundaries](59-sm33d-cv-sampling-and-window-validation.md).

Verification: 274 native integration tests + 41 actual CLI generation tests +
14 real local WSL Spark/Delta tests passed. Full ty, scoped Ruff/format, strict
MkDocs, new wheel build and strict generated dev Bundle validation passed.
Independent review found no actionable issues. No cloud job/model/table/alias
was changed; no deployment or push occurred.

Next READY: SM-33E guided basics/preprocessing/model/CV initialization and
combined personal-serverless acceptance. SM-34 remains WAIT. SM-36 remains the
advanced tuning/search integration; current CV evaluates fixed parameters only.
Prior experimental evidence must be recreated for the new dataset identity.

At the next requested commit include new `local_cv.py`, CV/window tests and
force-add exactly report 59 (initiatives ignores new files). Leave unrelated
`.tmp-review-model/` untouched. The current turn did not request another commit.

## Historical snapshot: SM-33C delivery

# Current handoff: SM-33C locally verified

Commit preparation (2026-09-25): the user requested committing SM-33C, the
input-budget follow-up, and the subsequent CV/modular-setup plans together.
Fresh checks: 172 training/approval/workflow/template/lifecycle tests passed;
strict MkDocs passed. Reports 57 and 58 belong in this commit. No cloud deployment
or push is included. The uncommitted references below describe pre-commit work.

SM-33B committed as `7cec33a7` on branch `090`, signed with hooks passed; no push.
SM-33C is DONE locally and remains uncommitted. The follow-up input-budget
rename is also verified and uncommitted: Bundle JSON/init use `max_input_mb`
(default 64 MiB); existing SDK byte contracts stay internal. Training, scoring
and approval share the conversion; old Bundle `max_bytes` is rejected.
Follow-up verification: 211 integration/runtime/template tests + 41 real CLI
generation tests passed, full ty, scoped Ruff/format, strict docs, wheel build
and strict generated dev Bundle validation passed. No live deployment.
Explicit training subsampling is now included in SM-33D; `max_rows` still
fails on overflow, and scoring does not sample. See
[date-free training validation](57-sm33c-date-free-training-validation.md).

New initialization defaults to random splitting, no invented date columns,
Core `DataSplitter` with stable record-key ordering, seed/proportion and optional
classification stratification. Temporal splitting remains explicit. Independent
result filtering uses `filter_unavailable_results`, `result_available_at_column`
and `result_cutoff`. Inactive fields must be null/default; no compatibility layer.
Monthly random reads the latest bounded snapshot with no lookback. Monthly
result cutoff is invocation time; temporal observation windows remain UTC months.

Saved source/split/parsing settings and holdout membership are replayed for
approval. Membership is bound to the comparison dataset identity after review
found and corrected an initial omission. Read budgets remain outside identity.
Three initializer examples and English guides/Mermaid explain the four combinations.

Validation: 165 native integration tests, 54 real CLI/template tests and 11
real Delta tests passed (230 total). Full ty, scoped Ruff/format, strict docs,
current wheel build, strict generated dev Bundle validation and independent
review passed. Native tests use `.cache/sm33c-final` as basetemp to avoid Windows
Temp ACL issues. Delta runs through `.cache/sm15-linux-run.sh` under WSL.

Planning follow-up: inspected `mlmodeltesting` source read-only (no pickle loading
or execution). It scores four separate targets from shared features, then applies
color rules; its monthly job does not train. Added SM-36a custom project FE/code
packaging, SM-36b multiple training branches, SM-36c model-set scoring/lifecycle,
and SM-44 LATER company migration after the generic Bundle gate. See report 58.
These are planned only. User further requested progressive Bundle setup for
basics, preprocessing, model, optional CV/search and opt-in multi-model scenarios.
Recorded conditional prompts, Core schema reuse, editable config, execution
preview and phased delivery across SM-33E/36/36a/b/c/42 in report 58 and the queue.
At the next commit, force-add exactly report 58 too.

Core CV/tuning audit: ordinary-model optional CV routes to
`StatefulEstimator.cross_validate` with fold-local preprocessing; advanced search
routes to the existing pipeline `hyperparameter_tuner` and `TuningConfig`.
User confirmed ordinary defaults/overrides versus advanced search-space/budget
controls. Report 58 records shared CV mapping, temporal metadata, final-holdout
isolation, validator gaps and the diagnostic-only standalone nested CV caveat.
The four focused Core CV/leakage/time-series suites passed (96 tests). This was
inspection and planning; no Bundle CV/tuning runtime support was added here.

Canvas follow-up: source tracing confirmed supervised Basic also uses the tuner
internally with a single fixed candidate; Advanced searches multiple candidates.
Both can run a later CV report. Existing Canvas `cv_*` fields affect search and
post-fit evaluation. Report 58 now distinguishes this from proposed Bundle
routing and requires explicit search/report semantics, separate seeds and gated
decision-threshold tuning. No Canvas/runtime behavior was changed.

Next READY: SM-33D Core CV, explicit training sampling and selection/window configuration.
Then SM-33E combined live acceptance. SM-34 remains WAIT. No cloud resources
were changed; current remote jobs still use the previous SM-33 wheel/config.
Old experimental evidence must be recreated when doing combined acceptance.

At the next requested commit, include new date-free training test, three
initializer examples, and force-add exactly initiative report 57. Leave unrelated
`.tmp-review-model/` untouched. Do not push without a request.

## Historical snapshot: SM-33B delivery

# Current handoff: SM-33B locally verified

SM-33A committed as `fa7a1171` on branch `090`, signed with hooks passed;
no push. SM-33B is DONE locally and remains uncommitted. See
[date validation evidence](56-sm33b-training-date-validation.md).

Added `TrainingDateSpec` with independent `event_time_parsing` and
`result_time_parsing` (format/timezone/date_only). Source types determine native
vs string handling. Strict shared parsing, DST validation before source filters,
integer-microsecond transport, and saved parsing rules for approval are verified.
Generated JSON exposes editable native defaults; conditional prompts remain E.
No compatibility adapter was reintroduced. English docs include examples.

Validation: 161 CLI/template/lifecycle/workflow/runtime regressions; final
72 focused date/training/config tests; 7 updated real MLflow approval tests with
nondefault source formats on pandas/Polars; 9 real Delta date tests via WSL.
Full ty, scoped Ruff, strict docs and independent final review passed. Windows
Delta lacks Hadoop support: use `.cache/sm15-linux-run.sh` separately from plain
Spark fixtures. Keep timezone data aligned across execution/replay environments.

Next READY: SM-33C date-free/random training and optional result availability.
Then SM-33D Core CV and SM-33E combined live acceptance. SM-34 stays WAIT.
No Databricks deployment or cleanup happened; remote jobs retain the prior
SM-33 wheel/config. New projects/models will be created for combined acceptance.

At the next requested commit, include new `training_dates.py`, two date test
files and force-add exact initiative report 56. Leave unrelated
`.tmp-review-model/` untouched. Do not push without a request.

## Historical snapshot: SM-33A delivery

# Current handoff: SM-33A direct field rename

SM-33 was committed as `0c0fd17f` on branch `090`, with sign-off and hooks.
No push. SM-33A through SM-33E now precede SM-34; see
[the training contract plan](54-training-data-contract-plan.md).

The user explicitly rejected compatibility for the two renamed fields because
this is pre-production. Removed the uncommitted `_workflow_fields.py` adapter
and its compatibility-only tests. `record_key_columns` and
`result_available_at_column` now run directly through Core frame contracts,
Databricks training/scoring, saved evidence, template configuration and examples.
Old initializer fields are removed. Recreate earlier experimental projects and
models when carrying out live acceptance; do not reintroduce aliases.

SM-33A is committed as `fa7a1171` (signed, hooks passed; no push). Verification:
503 integration/contract tests, 106 Spark checks and 53 Delta cases passed
(52 in the suite plus one corrected fixture case on focused rerun). Full ty,
scoped Ruff, strict docs and real CLI generation passed. See
[SM-33A evidence](55-sm33a-field-naming-validation.md).
Current dated training requirements still apply; SM-33B parsing/timezones is ACTIVE,
then SM-33C date-free training, SM-33D CV and SM-33E combined live acceptance.
SM-34 stays WAIT. No live deployment or cloud deletion happened in this slice;
existing Databricks jobs still use the committed SM-33 wheel/config.

Leave unrelated `.tmp-review-model/` untouched. Force-add exact new initiative
files at the next requested commit because `initiatives/` ignores new files.

## Historical snapshot: SM-33 delivery

# Session handoff - 2026-09-25

## Current snapshot after SM-33

SM-33 is DONE for its documented local/personal-serverless scope; SM-34 is READY.
Changes are uncommitted on `090`, based on SM-32 commit `5ba388ec`.
See [SM-33 evidence and boundaries](53-sm33-config-and-runtime-validation.md).

Delivered: versioned offline workflow validation, explicit legacy migration,
task/source/column/composite-key initialization, and a run-only
`score_model_version` parameter. Shared reports display the selected model
separately from previous-write provenance. Generated manual snapshots are unset.

Verification: 178 affected regression tests, 40 template/real CLI checks and one
real Delta append/replay test passed. Full repository ty, scoped Ruff and strict
docs passed. The existing personal dev Bundle was strictly validated and updated
once to the SM-33 R1 wheel. Score runs `518924307260524` (v1 override) and
`440928047261182` (empty override -> champion v5) succeeded without redeployment
between runs. Both were no-ops at Delta version 2. Run `805365754123156` is an
intentional negative test: v0 was rejected before Core dispatch.

Existing train/score IDs, sources, outputs, and aliases are unchanged.
Champion is v5; previous_champion v1; previous_challenger v4 (live-verified).
No new persistent jobs, schemas or model versions were created. Each job remains
bounded to 900 seconds. Personal inference verification does not imply company
production readiness. Next: SM-34 schedules/windows, then the remaining local
Bundle improvement queue; broad Spark expansion remains parked.

When committing, include new `workflow_config.py`, its tests and the report
above. `initiatives/` ignores new files, so force-add that exact report if needed.
Leave unrelated `.tmp-review-model/` untouched. Do not push without a request.

## Historical snapshot after SM-32 delivery

SM-32 is DONE for its documented local/personal-serverless scope; SM-33 is READY.
The delivery includes checked challenger history, independent Bundle policies,
serialized operator actions, score parameter-pushdown handling, automatic proof
lookup, readable reports and the English operator walkthrough.

The user approved Polars v5 in run `24807365707427`; its score child
`379146043586828` succeeded with no new data. Current model:
`workspace.skyulf_lifecycle_test.sm32_model_polars`; champion v5,
previous_champion v1, previous_challenger v4. The former practice candidate
has already been approved; do not treat the older report's v1/v5 state as current.

Existing Bundle: Polars/manual_approval/champion/after_alias_change/incremental_append.
Two persistent jobs remain: train `155738051514173`, score `684955889505992`.
Schedules are inactive; both jobs use 900-second timeouts and the R5 wheel.
Existing source and prediction resources are retained.

R4 proved separate persisted HTML and JSON notebook cells. R5 makes rollback
optional in the report, distinguishes required/restore versions and collapses
the receipt. Requested R5 live acceptance: parent `769913416634457`;
HTML assertions passed for task `1092954776398504`; child score
`690155923861600` succeeded with a no-op and Delta version 2. All 144 affected
tests and strict docs passed before commit. Final run/check results
are in [the operator follow-up](52-sm32-operator-output-and-evidence.md).
The broader SM-32 live evidence is [report 51](51-sm32-live-validation-report.md).
The user guide is `docs/user_guide/databricks_bundle_walkthrough.md`.

The SM-32 delivery commit follows `6a7f9e95`; use git log for its final hash.
Reports 47 through 52 belong to that delivery. SM-33/37/43 retain
config/identity/company gates; do not claim company production readiness.

## Historical implementation context through SM-32 local preparation

The user approved the [local Bundle improvement program](37-local-bundle-improvement-program.md)
and explicitly requested [independent scoring selection and promotion policy](38-model-selection-and-approval-design.md).
**Earlier baseline: SM-30, company tags and SM-31 committed as `fcfcee31`.**
The requested commit passed 116 tests and all applicable hooks (including ty).
SM-32 now separates score_model_selection and promotion_policy at the library
run_action boundary. All four combinations passed on pandas and Polars;
78 relevant tests include real local MLflow lifecycle artifacts. These library
changes have not been deployed. See
[the progress and next implementation steps](43-sm32-policy-separation-progress.md).
Update: 36 Core-owned test files have moved from root tests into Core; see
[the relocation inventory](44-core-test-relocation.md). 224 cases passed before
and after relocation; Core collected 10,900 tests without collisions, and
the relocated cases plus workflow tests passed 281 tests after type narrowing.
Library approve now loads saved candidate evidence and rechecks it without fit
or registration; see [manual approval progress](45-sm32-manual-approval-progress.md).
Final combined regression suite passed 321 tests; full repository ty and
scoped Ruff passed. Commit `94b28320` includes the SM-32 policy/approval slice and
the Core test relocation; the complete SM-32 task remains open.
Commit `6a7f9e95` implements library reject/rollback and safe retries;
see [its evidence and limits](46-sm32-reject-rollback-progress.md).
The combined affected suites passed 145 tests, scoped Ruff lint/format and full
repository ty passed, and strict MkDocs built successfully. The progress
report is included alongside this implementation.
The subsequent uncommitted slice implements previous_challenger history;
see [its evidence and boundaries](47-sm32-challenger-history-progress.md).
Its combined affected suites passed 155 tests; Ruff, full ty and strict MkDocs
passed. Include the new progress report explicitly at commit time because
the repository's broad initiative ignore rule applies to new files.
Matching Bundle choices and serialized operator actions/handoff are now
implemented locally; see [the plan](48-sm32-bundle-actions-plan.md) and
[validation](49-sm32-bundle-actions-validation.md). That local-only slice
was followed by the completed live rehearsal in report 51. Combined affected regression: 188 passed; real CLI
generation: 16 passed; all eight serverless configurations passed strict dev
validation. Wheel import/source check, Ruff, full ty and strict MkDocs passed.
Reports 47, 48 and 49 need explicit inclusion at commit time.
Latest follow-up: Bundle initialization now asks for optional risk_category,
PayingRegNo has an explicit policy-compute example, and new receipts use
from_version while reading older formats. 44 relevant tests and real CLI
generation/strict serverless validation passed. No live resources changed.
The requested previous_challenger replacement-history alias now exists in
Core; its live Databricks validation subsequently passed in report 51.
The user requested readable/company-compatible tags and a clean live reset
before SM-31. [The clean rehearsal](41-company-tags-and-clean-live-validation.md)
passed all four tasks in run 607409241163605. [SM-31 evidence](42-sm31-refactor-plan-and-evidence.md)
records 110 passing tests, Ruff, scoped ty, strict MkDocs, installed-wheel imports,
and a successful deployed score run 383572337148835. These changes are in `fcfcee31`.

The old four test jobs and two schemas were deleted as requested. Retain the
new workspace.skyulf_lifecycle_test schema and exactly two skyulf_lifecycle
jobs (train 155738051514173, score 684955889505992). One source and two engine
prediction tables each contain 170 rows. Main model_polars has champion=2,
challenger=3 rejected. model_pandas has champion=1 and challenger=5 error after
the explicit rollback test. Both scorers replayed without new commits.
No schedule is active. That deployed notebook delegates to Core local_workflow.py
and prediction_output.py. The new, locally verified template uses separate
seven-line lifecycle and score entrypoints calling job_runtime. Keep business
config in the project.

SM-30 changes challenger semantics: a trained/registered contender such as
tied v3 remains challenger, with evaluation status/reason visible; only
promotion depends on passing quality gates. Newer contenders replace that
pointer while retaining history. Rollback now preserves a separate contender
with a verified receipt. Real local MLflow tests cover pandas and Polars,
including manual nomination, tied candidates and comparison errors.

After SM-32 live acceptance, expose validated runtime/configuration choices
and independent schedules, improve metrics/search, identities/operations,
packaging, generated CI, data recovery and documentation. SM-43a verifies the
combined personal-workspace workflow; SM-43b separately gates company use.
The explicitly requested serving, ai_query, A/B and offline/online feature
tasks follow SM-43a; see [their contract](39-serving-and-feature-lookup-delivery-plan.md).
Broad Spark and monitoring also remain later; SM-18 and
continuous streaming remain parked. Keep two jobs and no default control tables.

Preserve the earlier uncommitted two-literal test correction in
`skyulf-core/tests/integrations/test_mlflow_promotion.py`: the actual tag is
`pending_alias_event`. The preceding review ran 58 Bundle/promotion tests
successfully; that evidence does not validate the newly planned semantics.
The notes below describe historical resources that the user subsequently
requested to delete. Use the new inventory above for the current workspace.
SM-43a still owns combined acceptance for the remaining improvement program.

## Current state after SM-27/SM-29

The combined personal-workspace live rehearsal passed; SM-27 and SM-29 are
DONE for the documented bounded Polars/serverless workflow. v1 initialized
champion, improved v2 replaced it, and tied v3 was rejected. Append output
contains 160 v1 plus ten v2 predictions; full output exposes all 170 v2 rows
while retaining v1. Expected score failure, recovery, queued no-op runs,
committed alias receipts and restricted-principal write denial passed. See
[the final live report](35-sm27-sm29-live-validation-report.md).

Retain `workspace.skyulf_sm27_sm29_20260924`, the two `skyulf_sm29_verify`
jobs and their audit notebooks for user inspection. No schedule is active.
Production still requires exclusive alias-writer ownership and serialized
target publication. Company targets and policy compute remain unverified.
Spark expansion and endpoint tasks remain later; do not treat this selected
local-engine rehearsal as broad Spark coverage. The notes below preserve the
earlier local delivery state.

SM-29 now has an optional `auto_champion` Bundle selection. It asks for a
heldout metric, minimum improvement and absolute quality threshold, uses the
existing guarded alias receipts, and calls the serialized score job after
train. Manual selection remains the default. Generated manual/automatic
serverless projects passed strict CLI validation. This local phase was followed
by the completed live rehearsal above; see
[SM-29 validation](33-sm29-auto-champion-validation-report.md).

SM-27 implementation now exposes append-vs-full-rebuild model-change scoring
at Bundle initialization and editable paused-retraining cron/timezone
variables. Generated-project CLI validation passed, but an actual v1-to-v2
full-rebuild job had not run at that point. The live rehearsal above now verifies
the versioned physical tables and stable view; see
[SM-27 validation](30-sm27-model-change-validation-report.md).

The latest first Bundle has exactly two jobs (`train`, `score`), and its first
score creates only the prediction table; see
[SM-20S](26-sm20s-two-job-live-validation-report.md). SM-28b adds an optional
paused monthly schedule to that same `train` job. Manual and monthly generated
projects passed strict CLI validation; no monthly live job was deployed or
run. See [SM-28b validation](28-sm28b-monthly-retraining-validation-report.md).
The combined live SM-27/SM-29 verification of model selection and both scoring
modes is now complete. Company workspace targets
remain unconfigured and unverified. The remainder of this file is historical
context from earlier Bundle stages.

SM-00 through SM-16, SM-15L/15I and SM-24a/25/26 are complete for their
documented scopes. The selected Databricks serverless local monthly UC
publication passed with 80 pandas January and 80 Polars February predictions.
Latest user direction: **Do not require manual period/source-version
values for recurring jobs. SM-15I now provides automatic new-row scoring
for the first small-data pandas/Polars Databricks Bundle (SM-20a). Spark handles table
I/O; Spark FE/model execution follows the Bundle. SM-18 and streaming remain
parked.**
The revised order completed SM-22a/b/c comparison and controlled promotion,
then SM-28a label-aware candidate training before SM-20a. SM-28b adds the
optional monthly schedule after the Bundle; SM-27 remains later. Target release: 0.9.0. Branch: `090`.

SM-22a has a locally verified, read-only comparison API and passed its
isolated Databricks metrics and UC comparison gate; see the
[local report](16-sm22a-local-validation-report.md) and
[live report](17-sm22a-live-metrics-report.md). SM-22b has explicit
version-checked promotion and rollback with shared admission. Its isolated
UC promotion, conflict, rollback and restricted-principal denial passed; see
[live evidence](18-sm22b-live-validation-report.md). SM-22c added challenger
and previous-champion aliases; see [its report](19-sm22c-lifecycle-alias-validation-report.md).
SM-28a now trains from a pinned label-aware Delta snapshot and compares a
registered candidate without promotion; see [its local and live report](20-sm28a-label-aware-retraining-report.md).
SM-20a now generates and deploys the first local-engine Bundle. Its own jobs
passed Polars training, 2+2 incremental scoring/no-op, read-only comparison,
challenger staging and explicit promotion; see [the report](21-sm20a-local-bundle-validation-report.md).
The user requested a clean reset and a generic Bundle. The personal `skyulf`
workspace's ten SM-20a jobs and three Skyulf test schemas were deleted; post-
deletion job listing was empty and only default/system schemas remained. The
company-shaped SM-20P draft is superseded. SM-20R now generates one editable
`dev/test/syst/prod` project: serverless is the easy default, policy compute
and champion/challenger jobs are optional, and test/syst/prod have distinct
unconfigured host/catalog placeholders. Deployment itself creates no UC
tables. The clean personal test passed: one existing source, registered Polars
model version 1, one prediction table and one internal score-control table;
600 initial and 50 later predictions, followed by a no-op replay. The final
read-only check found 650 unique predictions. See [the live report](24-sm20r-clean-generic-bundle-validation-report.md)
and [the reset plan](23-sm20-reset-and-generic-bundle-plan.md). Those test
resources remain for inspection. Company targets are still placeholders.

## Starting point

- SM-00 through SM-16, SM-15L, SM-15I, SM-24a, SM-25, SM-26,
  SM-22a/b/c, SM-28a and SM-20a/20R are complete. Continue SM-28b from
  [the open queue](OPEN_QUEUE.md), using the generic Bundle and the
  [pre-Bundle lifecycle plan](06-prebundle-model-lifecycle-plan.md).
  SM-15I [live evidence](13-sm15i-live-validation-report.md) proves automatic
  80+80 insert-only scoring with no date column or per-run version input.
  A later [real NYC taxi rehearsal](15-sm15i-real-nyctaxi-live-report.md)
  trained a Skyulf model with held-out MLflow metrics, registered UC model
  version 1, then passed 200 initial + 100 new-row predictions and a no-op replay.
  SM-26 added local artifact and MLflow packaging without cloud execution. The
  SM-25 SDK adds immutable local workflow configuration, local/remote preflight,
  explicit path or pinned registry artifact selection and caller-frame limits.
  It has no separate FE node or model-family allowlist: the fitted local
  pipeline's own prediction contract governs them. A representative sample can
  be probed before job submission. SM-24a added versioned UC reads, bounded
  local training and monthly pandas/Polars scoring. Its
  [live report](08-sm24a-live-validation-report.md) records two isolated UC
  source tables in `workspace.skyulf_sm24a_20260923`, five cross-job models,
  replay/negative checks and the 62-ID preprocessing matrix. Local-result
  Delta writes were subsequently proven in SM-15L; no Bundle was created.
  The current `max_bytes` limit measures decoded payload and local frame
  memory, not exact Spark wire bytes; SM-24d tracks a hard transport budget.
  previous Spark-first direction is superseded. Preserve the later
  [NODE_SUPPORT.md](NODE_SUPPORT.md) inventory and
  [gap review](reports/2026-09-22-spark-databricks-gap-review.md).
  The review enumerated all 100 source registration IDs and added model training,
  broader model inference, configuration usability and parked platform follow-ups.
  It did not implement those features. Missing node/model coverage must remain
  open unless implemented or explicitly deferred by the user. Broad SM-17
  follows the first local Bundle. The current Spark integration uses compatible bundles;
  pandas/Polars -> Spark is not support for arbitrary local FE or model artifacts.
- SM-26 separates fitted pandas/Polars pipeline packaging from native Spark
  porting. The versioned trusted-pickle artifact records fit engine, schemas,
  model class, runtime versions and a checksum; the MLflow pyfunc restores it
  without refitting. Its declared scope is whole-frame local prediction;
  row-local HTTP and Spark modes fail a scope check. SM-25, SM-24a, SM-15L
  and SM-20a follow this artifact contract.
- The registry-bundle loader and first registry-to-Spark probe are implemented.
  Local validation passed 29 combined tests; the selected live platform gate is
  now complete. Tested 0.9.0 wheel hashes are recorded in the platform file.
- See [PLATFORM_VALIDATION.md](PLATFORM_VALIDATION.md). The user authorized the
  supplied workspace and `skyulf` profile; OAuth authentication is verified.
  No classic clusters exist. The user explicitly approved the isolated resources
  and serverless probe. Run `245612415275039` failed on restricted
  `spark.sql.caseSensitive` access after the UC/local round-trip. The fix passed
  75 real Spark tests. The user approved the corrected rerun; parent run
  `447606109645160` completed SUCCESS. The Polars-trained registered bundle
  produced the expected gold predictions in both Spark modes on serverless
  Spark Connect 4.2.0 / Python 3.12.3 / MLflow 3.16.1.
- Shared `DeltaTableAdmission` is implemented. Its local real Delta gate covers
  forced acquisition races, lost acknowledgements and public batch replay.
  Live monthly admission and independent-job contention passed.
- Checkpoint `311547fc` contains that code and the serverless identifier fix.
  Restricted service-principal run `2921374308246` passed model allow/deny,
  worker imports and 10k/50k synthetic parity. A subsequent Delta/alias run
  completed worker-content and alias prechecks, then failed on serverless
  `REFRESH TABLE`. Narrow refresh/cache fixes passed a combined 48-test Delta
  gate; live retry `783094949884769` passed monthly/alias/worker-content checks.
  Winner `973709879452231` committed after contender `404334394214907` verified
  held-owner rejection. The contender then failed a test-only assertion about
  the permission error code; final restricted run `977447944071613` passed the
  denial, unchanged data/version and released ownership checks. That failed run
  remains documented, not relabeled. The aggregate report was generated from
  actual results and checked against the r3 wheel; all 225 current package files
  match that wheel. See the platform evidence file for retained resources.
- Local-engine pandas/Polars UC Delta writing passed SM-15L after SM-24a.
  The current Spark `run_batch` performs inference itself and cannot be used
  for local predictions unchanged. Prefer a bounded local-result -> Spark
  DataFrame bridge into guarded Delta publication; SQL Connector is optional.
  The later local-first Bundle request supersedes the earlier one-day deferral.
  See the [SM-15L report](11-sm15l-live-validation-report.md) for pinned
  models, monthly UC rows, replay, stale rejection and local negative tests. A delta-rs filesystem test
  must not be presented as UC managed-table writer support; see the current
  platform document's external-client constraints.
- SM-15 baseline: `4a613cb5`. Its implementation and verification are in the
  delivery commit containing the original SM-15 handoff. Live evidence is SM-16.
- SM-13 starts from `67315253`; its implementation, guide and evidence are in
  the delivery commit containing this handoff.
- Previous guide/queue commit: `80dc9ee9` (SM-11 closure and SM-12 handoff).
- Read [OPEN_QUEUE.md](OPEN_QUEUE.md), [ARCHITECTURE.md](ARCHITECTURE.md) and
  the SM-16 section of [03-mlflow-batch-delivery-plan.md](03-mlflow-batch-delivery-plan.md).
- User-facing guide: [How inference works](../../docs/user_guide/inference_flow.md).

## Confirmed terminology and intended workflow

`local` describes single-machine execution, not the user's personal computer.
The runtime environment and the execution engine are independent choices.
Local pandas/Polars FE and sklearn training may run inside Databricks; a later
Spark inference job may also run inside Databricks.

The first deliverable is:

```text
Databricks training job: pandas/Polars FE + sklearn model
    -> Save fitted FE, model and input contract through MLflow
    -> Databricks scoring job: pandas/Polars local batch on bounded data
    -> Spark reads bounded UC source changes and publishes UC Delta predictions
       (SM-15I automatic incremental path)
    -> Compare candidate/champion, promote explicitly, train challenger (SM-22a/b, SM-28a)
    -> Generate and run the first local-engine Bundle (SM-20a)
```

SM-16 validated an earlier compatible Polars-trained -> Spark regression path
on selected serverless compute. It does not validate the new local UC sink or
generated Bundle. Spark becomes a later optional Bundle choice in SM-20b.
Endpoint and online-lookup resources are separate optional work.
Keep new documentation and diagram labels in English.

## Current execution boundaries

- `predict_local_pipeline` consumes the new fitted local artifact, while
  `predict_local` consumes the portable standalone bundle. The frontend does not
  call it yet. Frontend inference uses `POST /deployment/predict` and
  `DeploymentService`, which loads the existing artifact, applies FE, aligns
  model columns and predicts. The backend artifact bridge belongs to SM-18.
- `predict_spark(mode="native_features")` supports raw regression and
  classification bundles with supported native FE. Spark applies the saved
  rules; Python workers run the same fitted model on batches. There is no
  full-data driver collection.
- Portable FE currently supports SimpleImputer mean/constant, StandardScaler
  and an empty chain. Unsupported steps fail explicitly.
- Spark FE fit -> export -> restore -> Spark apply is available. This does
  not imply that end-to-end distributed model training is implemented.
- `mode="python_pipeline"` runs compatible fitted Python FE and model
  execution together inside workers, without refitting. It supports raw
  regression and classification, and rejects unsupported context-dependent,
  row-changing and non-portable FE operations.
- Classification output preserves manifest label types, class-ordered
  probability columns and saved threshold precedence in both Spark modes.

## Verification and workspace

Before parking, the English guide's example was checked with both pandas and
Polars training: local and Spark predictions matched `[400.0, 200.0]`. SM-10
passed its focused 14-test Spark lane; SM-11 classification and isolation
regressions passed in the corrective 63-test focused lane and the full Spark
gate passed 418 tests with 2 warnings. The standalone Spark
batch example completed in both modes with matching string labels and
probability columns.
The strict documentation build, four rendered diagrams and commit hooks passed.
A built 0.9.0 wheel imported the worker inference module in a separate process.
Synthetic local measurements recorded 10k/2 partitions at 3.929s and
50k/8 partitions at 8.089s with driver peak RSS 246.2/246.8 MB. These are local
evidence only; the later SM-16 worker-wheel evidence is recorded above. SM-12
also added optional MLflow tracking: the base environment keeps MLflow absent,
while an isolated MLflow 3.16.1 environment passed all 8 tracking tests for
explicit client-bound lifecycle, concurrency, caller-run preservation, config
digest/artifact logging, and warn-mode degradation.

SM-13 added MLflow pyfunc packaging around the immutable inference bundle.
The MLflow 3.16.1 lane passed 31 tests, including a separate Python `-I`
subprocess loading the final 0.9.0 wheel from site-packages. Consumer dependencies
were copied from the isolated MLflow environment; Skyulf was reinstalled from
the wheel. Base bundle/schema/integration tests passed 80 with 7 optional skips.
MLflow aligns named columns and safely casts compatible inputs before bundle
prediction; direct `predict_local` remains strict. Unsupported bundle integer
dtypes fail at packaging. Producer uv files and temporary paths are excluded.
Registry-to-G2 evidence continued through SM-14 and the completed SM-16 live
gate. Scheduling configuration, endpoints and templates remain later work.

SM-26 added `save_local_pipeline`/`load_local_pipeline`/`predict_local_pipeline`
and `log_local_model`. The pandas and Polars MLflow 3.16.1 lane proved
categorical encoding, null input, classification probabilities and tuned
decisions after load; an isolated subprocess also reloaded the logged model.
The changed 0.9.0 wheel was installed into the isolated environment and
`python -I` resolved the new module from site-packages.
Custom binning followed by encoding passed both local engines. The registry
adapter now resolves the local payload digest alongside the existing portable
bundle digest; no alias was promoted. This was local validation only. The
matching Skyulf wheel remains an explicit deployment dependency, and broader
model families still require parity evidence. See the MLflow model guide.

SM-14 added explicit MLflow registry publication and resolution. The adapter
publishes only `runs:/...` artifacts, leaves alias promotion explicit, resolves
an alias once to a concrete `models:/name/version` URI, and returns the packaged
signature and bundle digest. MLflow 3.16.1 local validation passed 10 tests,
including separate tracking/registry stores, missing/access/dependency failures,
alias movement and Unity Catalog name validation. SM-16 subsequently validated
the live UC-to-Spark workflow using the concrete registered bundle.

Pre-existing `.tmp-*` directories remain outside this change. Preserve them;
do not stage or delete them as part of the next task.

## SM-15 delivery and completed platform boundary

The monthly runner now reads a pinned Delta snapshot, checks its availability
at `as_of`, runs either Spark inference mode and atomically replaces one period
in a precreated target. Required provenance includes the model digest and
installed code version. The source producer still owns historical feature joins;
the cutoff does not establish their point-in-time correctness.

Local Linux validation passed **43 tests** on Spark **4.0.3** / Delta **4.0.0**:
18 real Delta cases plus 25 contract/admission cases. Base regression passed
105 with 26 optional skips. See the queue for exact commands and package versions.
The English [batch guide](../../docs/user_guide/databricks_batch.md) includes the
schema, code example, diagram, retry policy and limits.

`LocalTableLock` uses cross-process OS locks and requires a common directory
on one host. Both public entry points reject it on distributed Spark masters.
SM-16 added and validated shared Delta-table admission. Expiring leases cannot
protect this sink because it has no fencing-token mechanism. Every publisher
must use the same authority. Actual Delta/UC permission evidence is recorded
in the completed platform report.

Keep the current source snapshot and original spec available for retries. A
receipt returns the old committed version even after a newer recomputation,
without writing again. New computation needs a new run ID and reviewed target
version. History and transaction retention must cover the allowed retry window.

SM-16 is complete; see its evidence file for wheel checksums, approved namespace,
retained resources and finished runs. Start SM-26 without repeating these cloud
tests unless a new change requires them. Databricks CLI
1.17.0 is installed but absent from the current shell PATH; use its existing
WinGet executable or a refreshed shell. The user confirmed `databricks.yml` is
not needed in this repository; it is absent. DAB/templates and endpoints remain
later work. The user's subsequent explicit authorization covers the `skyulf`
profile and the named isolated serverless test resources.
