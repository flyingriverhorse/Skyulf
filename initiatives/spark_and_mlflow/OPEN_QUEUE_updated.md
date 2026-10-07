# Spark ve MLflow — Open Queue

> Active queue: this file (`OPEN_QUEUE_updated.md`), confirmed by the user on
> 2026-09-28. Use this task order and its added scopes for further work.
> Reference follow-up (2026-10-07): all 99 supplied documentation files reviewed;
> selected changes cover a layout-specific first-run guide, monitoring Run-as/ACL
> parity and independent runtime/budget settings, current-release packaging, and
> accurate online-monitoring documentation. PR196's remaining CI failures are
> tracked with the same batch in [Delivery179](179-reference-template-improvements.md).
> Fresh native testing remains user-deferred. SM-37/43b cloud identity acceptance
> and SM-40's broader generated-project CI remain open.
> Current sequence (2026-10-05): SM-23 batch scope closed with SM-23i Overview
> performance separation. SM-57 is delivered for the admitted pandas-worker scope,
> with all three serverless lifecycles, no-op parity and native refresh verified
> ([Delivery170](170-sm57-spark-pyfunc-delivery.md)). SM-58 scale/memory validation
> has passed its serverless matrix ([Validation171](171-sm58-scale-memory-validation.md)); classic-compute
> acceptance is blocked because this workspace supports only serverless.
> SM-23 online observability is functionally complete (2026-10-06): real HTTP
> parity, Spark telemetry/late labels, independent performance verdicts and native
> dashboard refresh passed ([Delivery177](177-sm23b-online-monitoring.md)). Full SM-19a load/cold-start gates remain separate; company/identity
> gates stay in SM-37/43b. Earlier dated/uncommitted notes below are historical;
> the shared dashboard now ships inside the existing project template, with automatic
> create/reuse storage and Overview summaries ([Delivery168](168-sm23k-monitoring-home-overview.md)).
> Overview now puts current drift/performance summaries before trends and explains
> monitoring registrations ([Delivery169](169-sm23l-overview-order.md)).
> SM-39 delivered (2026-10-04): centralized runtime requirements,
> automatic optional model/Optuna dependencies, checked wheel build and target
> compute overrides. Two clean installations and two real train-to-score jobs
> passed; 360 cloud predictions matched independent sklearn calculations.
> Content-tagged wheel redeployment preserved existing models and invalidated
> stale package cache identity. Company policy acceptance remains SM-43b;
> separate identities/concurrency remain SM-37. [Delivery157](157-sm39-runtime-packaging-delivery.md).
> Development monitoring isolation (2026-10-02): personal development
> targets bypass enrollment and the full monitoring/drift/retraining branch.
> Real temporal model-set train/scoring passed in both modes: dev automatic scoring
> wrote 20 predictions with zero central inventory/observation rows; normal scoring
> wrote 20 predictions and two drift observations. No-new-data retraining was skipped.
> Requested normal rerun `243687288401872` also passed: no new input meant no new
> predictions, the existing observation was reused, and retraining was skipped.
> All 251 affected tests passed, including 141 real CLI generation cases.
> The previously linked dashboard is currently TRASHED; dashboard access is a
> separate follow-up. [Delivery156](156-development-monitoring-isolation.md).
> Model-set policy follow-up (2026-10-01, uncommitted): one meaningful improvement
> now permits equal peers when no selected metric regresses and all absolute gates
> pass. Temporal set v3 automatically replaced v1 with improved revenue RMSE and
> tied risk accuracy; its scoring child wrote 20 verified v3 predictions and skipped
> redundant retraining. Pre-change audit matched all 320 Core files to the previous
> live wheel, 42 project files to saved source, and dashboard JSON to the index.
> [Delivery155](155-model-set-non-regressing-promotion.md).
> Temporal acceptance follow-up (2026-10-01, uncommitted evidence): full competition
> and model-set jobs passed with 90-day selection, 14-day temporal holdout and
> time-series CV. Recent labels correctly stayed in holdout; 120 late-arriving
> eligible rows triggered one automatic train per layout. Independent artifact
> replay proved actual 600 train / 240 holdout membership and post-drift ingestion.
> Repeated scoring preserved the original request; champion v1/challenger v2 remained
> under existing quality gates. 191 scoped tests passed; runtime code unchanged.
> [Delivery154](154-temporal-retraining-live-acceptance.md).
> SM-23c branch/coverage follow-up (2026-10-01, uncommitted): all generated score
> jobs now show an explicit retraining If/else. Competition and model-set positive
> drift-triggered training, no-new-data false branches, duplicate-request repairs
> and subsequent false branches passed live. Independent Delta/MLflow replay proved
> post-drift appended data entered the actual fits: 101 new train / 19 new holdout
> rows per model. Existing promotion gates correctly retained champions on ties.
> [Delivery153](153-retraining-branches-live-acceptance.md).
> SM-23c drift-trigger follow-up (2026-10-01, uncommitted): daily rolling windows,
> Bundle thresholds and optional `retrain_on_drift` implemented. Real unchanged
> data/healthy skips, positive train completion and duplicate-request repair
> verified. Final-wheel v2 scoring produced 120/120 predictions, detected drift
> and correctly skipped retraining on unchanged input. Larger
> performance-trigger scope remains partial. [Delivery152](152-drift-retraining-delivery.md).
> Evaluation-chart parameter headers fixed (2026-10-01, uncommitted): native
> images now live in parameter-free `Charts` children. Single, competition and
> model-set cloud publication passed: five children, 28 images, zero parameters;
> source training metadata/status preserved. All three demo Bundles now use the
> tested wheel. [Delivery150](150-evaluation-charts-child-runs.md).
> Latest SM-23a acceptance (2026-10-01, uncommitted): previous monitoring demo
> resources were deleted at the user's request. Fresh generated single-model,
> competition and multi-target train/score Bundles passed in three separate schemas.
> Multi-target components now enroll automatically before scoring. One shared
> dashboard shows 4 monitors and 5 observations (current: 3 healthy / 1 drift).
> The shifted single-model batch triggered monitoring automatically and retained
> its earlier healthy result. Current resource/run links and validation limits:
> [Delivery148](148-sm23a-full-generated-jobs.md). Subsequent `on_drift` work and
> newer observation counts are tracked in Delivery152 above.
> SM-36e is DONE (2026-09-30): optional SHAP prompts/dependencies and readable
> notebook/MLflow charts passed 193 local tests, five lifecycle tests, 12 CLI
> cases, three strict Bundle validations and 41 cloud contracts with zero skips.
> Real training and saved notebook charts passed; final-wheel report replay also
> passed. [Delivery121](121-sm36e-shap-delivery.md). SM-36d stays PARKED.
> Optional evaluation-chart follow-up delivered (2026-10-01, commit `158161ed`):
> default-off Bundle prompt and separate `generate_charts` task; regression,
> classification and model-specific diagnostics with distinct single/competition/
> model-set reporting. 219 affected tests, 129 real CLI generation cases, full
> analysis gates and six generated serverless jobs passed on pandas/Polars.
> All 68 native MLflow Image grid images were downloaded and verified. Two initial
> lifecycle task timeouts passed on fresh runs with unchanged code/limits; their
> cause remains unproven. All 12 test jobs are idle/manual-only. Policy-cluster
> generation is tested locally; live execution is serverless only.
> [Delivery137 and run links](137-optional-evaluation-charts.md).
> SM-23a batch monitoring and central AI/BI dashboard delivered and live-tested
> (2026-10-01, uncommitted). Two model schemas share one Delta inventory/history;
> delayed labels, drift, stale/no-data/failure states and retry dedup verified.
> Final Bundle job passed; dashboard redeploy retained both resource IDs.
> Independent producer repos now register directly in central Delta; no models.json.
> Explicit monitoring_catalog/schema select monitoring storage independently.
> 227 scoped tests, six real CLI generation cases and six strict Bundle validations
> passed. Two independent cloud scoring jobs, concurrent/repeated enrollment,
> file-free central observation and table-ID/data preservation passed on final code.
> Full analysis gates passed. Native InferenceLog, cost reports and serving
> monitoring remain outside this slice. [Delivery144](144-sm23a-central-inventory-delivery.md).
> SM-23a automatic-task follow-up delivered (2026-10-01, uncommitted): active
> models enroll before scoring through register_monitor; score jobs run a separate
> monitor_model task. Two cloud schemas passed never_observed enrollment and
> automatic post-score observation. Final-wheel pandas/Polars checks passed missing
> measurement recovery, repeat deduplication and atomic activation-order guards;
> pandas enrollment-only job repair also passed. 225 final runtime tests, 141 real
> CLI generation cases, six strict serverless validations and full analysis gates
> passed. Central periodic scheduling and automatic multi-target enrollment remain
> outside this follow-up. [Delivery145](145-sm23a-automatic-monitoring-tasks.md).
> SM-23a dashboard layout updated (2026-10-01, uncommitted): environment/project
> controls removed, metrics moved to columns, drift/quality/performance charts
> added. Seven live dataset queries and 52 metric-value preservation checks passed;
> shared inventory shows 4 drift rows instead of 28 and 4 performance rows instead
> of 24. Same dashboard ID republished. [Delivery146](146-sm23a-dashboard-layout.md).
> SM-23a approved simplification delivered (2026-10-01, uncommitted): drift and
> quality are tables, per-statistic limits are explicit, KS p-value is excluded
> from threshold comparison, and performance has dynamic metric columns, true
> outcome counts and one model/version trend. New Bundles default monitoring to
> enabled. 133 monitoring tests, 12 CLI cases, seven live queries and a final-wheel
> default-enabled score/monitor/repeat cloud run passed. [Delivery147](147-sm23a-approved-dashboard.md).
> SM-52 is DONE (2026-09-30): shared helpers use directly named definitions;
> compatibility aliases removed and active MLflow monkeypatch targets verified.
> User scope correction (2026-09-30): acceptance, monitoring and cost tools are
> removed and deferred to later work, including their dedicated code/tests and
> acceptance-only CI settings. Static smoke remains; SM-40 is PARTIAL, SM-23a LATER.
> [Delivery129](129-reference-followups-delivery.md) records the earlier checkpoint
> and this removal; its earlier full-suite totals predate the removal.
> Removal checks: 105 affected tests and 93 CLI generation tests passed;
> all 2,370 remaining integration cases collect; Ruff, full Ty and CCN 10 passed.
> SM-37 PARTIAL (2026-09-30): CI-independent deployment files, optional personal
> targets in test/syst/prod, shared/separate run identities and opt-in job ACLs
> delivered locally. 99 CLI generation cases, 20 strict loopback resolution cases
> and 89 local checks passed; 2,396 integration cases collect. Targets now use only
> test/test_development, syst/syst_development and prod/prod_development pairs;
> personal targets remain optional and every target includes permissions: [].
> Catalog/schema setup
> now generates editable bindings without interactive questions; personal jobs use
> dev_<user>_train/score, shared jobs retain <project>_train/score. The user subsequently
> authorized live testing: the earlier seven-target layout validated in the workspace;
> a fresh personal deployment passed 12 train tasks, automatic child score with
> 120 verified predictions, and repeat no-op with unchanged data/Delta version.
> The current paired layout was subsequently deployed for six isolated
> test_development scoring projects in SM-41; see report136.
> Separate-identity alias denial and concurrent lifecycle evidence remain open.
> [Plan130](130-sm37-deployment-plan.md), [Delivery131](131-sm37-local-deployment-delivery.md),
> [Live132](132-sm37-live-deployment-verification.md). Test jobs remain PAUSED.
> SM-37 template checkpoint committed as 65111c0b (DCO sign-off).
> SM-38 PARTIAL (2026-09-30): configurable job/task timeouts, empty-default
> health/email/webhook settings, opt-in score retries and source/model/count/no-op
> summaries delivered locally. Lifecycle retries remain forbidden. Final pre-commit
> verification passed 405 affected Python tests and 125 CLI cases; all 2,408
> integration cases collect. Ruff, format, full Ty, schema and complexity checks passed.
> Subsequent authorized live tests passed: 30-second task/job limits, one bounded
> retry, real score recovery after a post-commit failure and unchanged 120-row/Delta-v1
> readback. The user confirmed email arrival. Generated score HTML rendering was
> corrected and verified in actual notebook output.
> Test jobs are PAUSED/idle and the normal score notebook is restored. Optional
> webhook delivery remains open. Model-set live summaries were subsequently
> verified in [Live136](136-sm41-live-cdf-recovery.md).
> [Delivery133](133-sm38-operational-controls.md), [Live134](134-sm38-live-operations-verification.md).
> [Review126](126-sm52-reference-review.md) maps the supplied dbml template to
> current tasks and records additional acceptance proposals; quality gates stay parked.
> Approved SM-36e follow-up (2026-09-30): separate real training tasks and leaf
> SHAP reports are now in the Bundle template. Single/competition/multi-model
> live training, saved notebook output audits and all three score jobs passed
> (120 rows each). Single uses `validate_model`; multi exposes set registration,
> evaluation and policy/operator decision as real separate tasks. Champion and
> challenger aliases remain set-level. [Delivery122](122-training-and-shap-nodes.md).
> Multi now supports `score_handoff=after_alias_change`: live run 418921754428153
> passed all 14 tasks and automatically invoked score run 434291347583937,
> which committed 120 predictions. The wizard exposes the same handoff option.
> Final clean-schema acceptance: all three layouts and their automatic score
> children passed (120 rows each), seven SHAP reports/28 charts and independent
> Delta readback passed; 283 local tests passed. Active test namespace is now
> `workspace.skyulf_clean_20260930`. Both old test schemas were deleted after
> verification (147 tables/views, 119 models and associated functions).
> [Final acceptance123](123-clean-final-acceptance.md).
> Lifecycle replay124 passed: six pandas/Polars single/competition/multi scenarios,
> 380 distinct cloud contracts, nine real approve/reject/rollback jobs and six
> automatic score children. Full rebuild, append/no-op, quality rejection,
> rollback, multi source corrections and all three publication modes passed.
> Independent Delta readback verified model versions and changed predictions;
> both old schemas remain absent. [Evidence124](124-lifecycle-revalidation.md).
> User-requested cleanup125: only SHAP template notebooks pass displayHTML;
> normal nodes emit JSON. 66 affected tests and static gates passed. The current
> `workspace.skyulf_clean_20260930` schema is now empty (53 tables/views and 18
> models with their versions/associated functions removed); historical test
> evidence is retained locally. [Evidence125](125-json-notebooks-and-schema-cleanup.md).
> Latest delivery [113](113-model-set-quality-promotion.md) /
> [114](114-model-set-challenger-delivery.md): automatic/manual set quality gates,
> model inventory tags, challenger history and explicit rejection are verified.
> Databricks run `119140789004450`: all tasks SUCCESS; 226 contracts passed and
> real pandas/Polars mixed-model lifecycle/scoring checks passed. Six CLI-only
> cloud skips were checked locally. Included in the signed delivery based on
> `90808117`; SM-36d is PARKED by user. Tag inventory is in report113.
> SM-36a is DONE for its approved scope. Project packages/custom steps:
> `9cc81304`; temporal history: `84a7dcb4` ([report105](105-sm36a-temporal-history-delivery.md)).
> Keyed scoring outcomes, assets/pins and output rules are implemented and verified
> locally and on Databricks; commit `16e903c3` includes the final slice.
> [Final evidence](106-sm36a-project-delivery.md). SM-36b is DONE locally and on Databricks.
> [Evidence107](107-sm36b-training-branches.md). SM-36c is DONE; signed delivery commit requested.
> [Delivery109](109-sm36c-model-set-delivery.md): 114 final-wheel cloud tests plus real
> model-set lifecycle, Bundle pipeline and temporal/excluded Delta acceptance passed.
> Follow-up [110](110-model-set-output-plan.md): shared scoring rules and output
> selection implemented; local tests/CLI/static gates and live Delta acceptance passed.
> Follow-up [111](111-model-set-source-corrections.md): opt-in UPDATE/DELETE
> rebuild reuses batch scoring. Live source-correction acceptance passed in clean
> schema `workspace.skyulf_validation_20260929`; old resources preserved.
> [Acceptance112](112-clean-databricks-acceptance.md): 516 cloud contracts including
> 20 real Delta tests passed; four CLI-only skips passed locally. Model/strategy,
> CV/ensemble, nested policy and real UC/project pipelines verified. Nested-halving
> bug fixed; final 176-test replay with the g_score dependency fix passed on Databricks.
> Latest: SM-54 single-target model competition is DONE. Model-set
> nesting is outside this delivery; SM-36d remains PARKED. Both-engine live
> regression/classification lifecycle passed in run `837376719969902`; final
> wheel contracts 134/134 plus custom-filter/ensemble scoring passed in run
> `845512547977070`. [Delivery116](116-sm54-model-competition-delivery.md).
> Follow-up: Bundle now guides candidate count/model selection, shared search and
> per-candidate ensemble menus/settings, then generates editable candidates.py.
> 218 related tests, generated-recipe training and strict Bundle validation passed
> locally; this prompt follow-up did not run a new cloud job (report116).
> Model-owned settings follow-up: single models, candidates and generated branches
> now own ensemble/tuning settings and populated Core-derived search spaces.
> Legacy hooks remain supported. 273 local tests and three strict Bundle validations
> passed; no new cloud run. Uncommitted [evidence117](117-model-owned-settings-plan.md).
> Readable model layout: generated single_model.py, model_competition.py and
> multi_model.py now contain editable Python dictionaries; model_set.py retains
> joint promotion/publication policy. Compact schema preserves prompt semantics.
> Saved-action isolation and legacy compatibility are covered in
> [follow-up118](118-model-layout-plan.md): final 356-test regression passed,
> three strict Bundle validations and static gates passed at that checkpoint.
> Final live acceptance [119](119-model-layout-live-delivery.md) supersedes the
> earlier cloud-pending notes: 419 local tests, 511 installed-wheel cloud tests
> (zero skips), six strict CLI validations, and both-engine single/competition/
> multi-model scenarios passed. Regression/classification lifecycle, saved custom
> scoring, set quality rejection/promotion, and source UPDATE/DELETE rebuilds
> passed. Included in the signed SM-54/model-layout delivery; no push requested.
> [Recipe follow-up108](108-sm36b-named-recipes.md): independent preprocessing/pre-split selectors,
> inline asset help and shared-only setup; eight models passed the full cloud pipeline.
> This file replaces the removed historical `OPEN_QUEUE.md` as the active queue.

Updated: 2026-09-29 (dbml re-comparison queued; see report93). **SM-00 through SM-16, SM-15L/15I/24a/25/26, SM-22a/b/c/28a/28b, SM-20a/20R/20S and SM-27/29 complete for their documented scopes. Target: 0.9.0.**
The pre-Spark local Bundle lifecycle gate passed live: selectable rescoring,
automatic champion selection, failed-score recovery and serialized handoff.
This is functional completion for the selected workflow. Production must
still enforce one exclusive alias writer; both test jobs used a human owner,
and live contention was exercised for score requests, not concurrent trains.
See [the combined live evidence](35-sm27-sm29-live-validation-report.md). The
current priority is the approved SM-30 through SM-43 local Bundle improvement
program below, starting with challenger semantics. Broad Spark expansion
waits for its local acceptance gate (SM-43a). SM-15L
provides explicit-period UC output. SM-15I adds automatic new-row scoring
for the first Bundle; SM-18 and streaming remain parked.

Read [HANDOFF.md](HANDOFF.md), [the integration plan](04-databricks-integration-plan.md)
and [the pre-Bundle lifecycle plan](06-prebundle-model-lifecycle-plan.md) before SM-20a.
The [SM-20 Bundle plan](05-sm20-bundle-plan.md) remains the packaging gate.
Historical completion evidence is preserved in
[OPEN_QUEUE_tamamlanmakaydi.md](OPEN_QUEUE_tamamlanmakaydi.md).
SM-26 added a local MLflow package; SM-15L and SM-15I validated UC writers.
The first local-engine Bundle passed generated-project validation and live jobs;
see [SM-20a evidence](21-sm20a-local-bundle-validation-report.md).

Durumlar: READY = başlanabilir; WAIT = önceki görev bekleniyor;
LATER = son aşama; ACTIVE = yürütülüyor; BLOCKED = somut dış engel;
DONE = kanıtla tamamlandı. SM-00 commit: `105a6fe4`.
DEFERRED = user postponed this work; PARKED = do not implement until resumed.
SUPERSEDED = replaced by a later user-directed scope; not a completed feature.

## Next session: integration/template simplification

The user resumed the six simplification tasks on 2026-09-27. **SM-34C1/C2 are
committed as `e16b27e8`; C3 is committed as `f4d654cf`. C4/C5 are committed
as `d8e4949f`. C6 is committed as `d9596763`; live custom-node fix as `5c9798b9`. SM-35 is committed as b67594b7; SM-36 and SM-36f are included in the local delivery commit; SM-36g/h/i are implemented and verified in the nested-policy delivery**. [Live evidence](83-sm36-databricks-live-acceptance.md). [C1 evidence](69-sm34c1-shared-training-preparation.md),
[C2 and 503 tests](70-sm34c2-evidence-result-ownership.md), and
[C3 measured read reduction and 327 tests](71-sm34c3-replay-reuse.md).
[C4 setup reduction and CLI parity](72-sm34c4-initializer-simplification.md), and
[C5 starter reduction and saved-code checks](73-sm34c5-smaller-generated-project.md).
[C6 explicit entrypoints and compatibility](74-sm34c6-explicit-notebook-entrypoints.md).
See [scope, file map and acceptance checks](68-integration-template-simplification-plan.md).

Historical simplification baseline: `b67594b7`; graph implementation: `bd49f49e`; readable-output fix:
`daa1e1c7`. The latter passed one live rendering test (`6136609106178`), but the
two persistent jobs were not updated to that wheel. Queue changes do not authorize
deployment. Preserve existing snapshots, receipts, manual actions and score guards.

## Active priority: local Bundle improvements before Spark

The user approved the comparison findings and clarified that a newly trained
contender remains challenger even when it fails promotion. The user also
explicitly approved separating manual/automatic promotion from score model
selection. These are new requirements; prior DONE rows retain their original
scope and do not imply the improvements below already exist.

Read [the improvement program](37-local-bundle-improvement-program.md) and
[the independent selection/approval design](38-model-selection-and-approval-design.md).
Each task includes implementation locations and concrete acceptance checks.
The user added a mandatory pre-SM-34 sequence for optional dates, clear names,
explicit timezone parsing and Core split/CV reuse. Read the
[training data contract plan](54-training-data-contract-plan.md).
SM-33 was committed as `0c0fd17f`; subsequent tasks are separate work.
**SM-32 through SM-33H3 are DONE for their documented scopes.** H2 and H3 passed
their approved personal serverless checks; H3 run848857785722024 verified saved
custom approval and 240+3/no-op scoring on pandas/Polars. H2/H3 changes are
committed as `d5d2398d`. Matrix63 distinguishes verified recipes from optional
and conditional limits. SM-34 is LOCAL DONE: implementation/review, 647 local
tests and strict CLI checks passed; live clock/queue checks remain unrun.
Quality fixes and SM-34 are committed as `676feddf`. Before SM-35, the user
approved SM-34A: meaningful training/evaluation/lifecycle tasks in the existing
two-job Bundle. See [the task graph plan](66-sm34a-visible-lifecycle-tasks.md).
SM-34A is committed as `3e92d14a`; its user-requested follow-up now uses only
`train` for manual and cron starts. Latest/explicit snapshot selection belongs
to `training_version`, independently of schedules. The follow-up passed local
and CLI generation tests and was included in `bd49f49e`. The subsequent user-authorized
SM-34A live rehearsal found/fixed a branch join, then passed pandas automatic
and Polars manual approval, child scoring, three-row append/no-op and artifact
audit. [Live evidence](rehearsals/sm34a_live/README.md). SM-34B is now DONE:
eight tasks/eight edges, preserved durable phases, shared failure cleanup and
task-state-guarded score handoff. [Evidence](67-sm34b-simplified-lifecycle-graph.md).
The new graph passed [live acceptance](rehearsals/sm34b_live/README.md), including
both engines, manual approval, scoring/no-op and two controlled failures.
SM-35 is committed as b67594b7; SM-36 passed local checks and the [bounded Databricks acceptance](83-sm36-databricks-live-acceptance.md): 34 models, 306 cloud tests, eight score/parity/no-op scenarios and manual lifecycle/incremental scoring. SM-36 and SM-36f are included in this local delivery commit. C1-C6 passed the isolated live check in report75. Latest expanded acceptance: 170/170 model-strategy pairs, 254 setting scenarios, 143 penalty regression tests, 90 pruning tests and 45 CLI generation cases. Logistic Regression legacy-sklearn penalty preservation fixed; that acceptance predates true nested tuning; SM-36f below tracks its implementation. [Report](85-sm36-full-strategy-acceptance.md), [receipt](86-sm36-full-strategy-receipt.json).
New generic row predicates, group-aware splitting and data-quality thresholds
are parked at the user's request.
See [the operation audit and implementation tasks](62-pre-split-cleaning-and-leakage-plan.md).
See [guided setup and live acceptance](60-sm33e-guided-setup-and-live-validation.md):
both engines passed training/CV/MLflow/approval, 240 + 3 predictions and no-op
checks. SM-33D/SM-33E were committed as `516b3f86`; company production acceptance is later.
User-requested operator usability follow-up: readable notebook reports and
automatic full-proof lookup are implemented and verified locally and in the
same personal Bundle; see [the follow-up evidence](52-sm32-operator-output-and-evidence.md).
See [live operator evidence and practice instructions](51-sm32-live-validation-report.md).
The SM-32 delivery includes the history/Bundle/live-fix and operator usability
slices with reports 47 through 52; it follows baseline commit `6a7f9e95`.
Commit `fcfcee31` passed 116 tests and all applicable commit hooks.
The first SM-32 library policy slice passed 78 tests; see
[policy separation progress](43-sm32-policy-separation-progress.md).
The [manual approve library action](45-sm32-manual-approval-progress.md) now
rechecks saved candidate evidence without training. Library reject/rollback
and checked retries are implemented in the [current slice](46-sm32-reject-rollback-progress.md);
145 affected tests passed locally. The subsequent
[challenger history slice](47-sm32-challenger-history-progress.md) is implemented
locally with 155 affected tests passing. Matching Bundle choices, serialized
operator actions and optional score handoff are now implemented locally; see
[Bundle validation](49-sm32-bundle-actions-validation.md). The subsequent
[live rehearsal](51-sm32-live-validation-report.md) passed: 194 local tests,
manual/automatic decisions, history, conditional score, rollback/retry and
170+10-row inference/no-op for both engines. The user has a pending v5 to practice.
[36 misplaced Core test files](44-core-test-relocation.md)
were relocated and verified before this continuation. See
[the implementation evidence](40-sm30-challenger-validation-report.md): 104
tests passed, including real MLflow lifecycles with pandas and Polars.
See [the clean live rehearsal](41-company-tags-and-clean-live-validation.md).
SM-31 passed 110 tests and its deployed thin score notebook passed live; see
[the extraction evidence](42-sm31-refactor-plan-and-evidence.md).
SM-43a remains the later combined acceptance for the full improvement program.
Keep two jobs and reuse Core services. Manual approval must act
on an existing candidate without retraining or reuploading the model.

| Order | Task | Dependencies | Status | Acceptance / deliverable |
| --- | --- | --- | --- | --- |
| SM-30 | Challenger nomination and visible evaluation status | Existing SM-22c/29 | DONE | 104 local tests and live run 607409241163605 passed; company tags, both engines, retained contender, replacement, error and rollback verified |
| SM-31 | Thin template and reusable workflow/publication services | SM-30 | DONE | 44-line notebook, two Core services, 110 tests, wheel installation/import, strict generation validation and live score run 383572337148835 passed |
| SM-32 | Independent score selection and promotion policy | SM-31 | DONE | 194 local tests and personal serverless manual/automatic, approve/reject/rollback/retry, history and score handoff/pin preservation passed; [live evidence](51-sm32-live-validation-report.md) |
| SM-33 | Validated config, migration and runtime parameters | SM-32 | DONE | 219 local checks passed; existing personal Bundle selected v1 then champion v5 without redeploy and rejected invalid v0; [evidence](53-sm33-config-and-runtime-validation.md) |
| SM-33A | Direct readable record/result field names | SM-33 | DONE | No aliases/adapter; regenerate experimental projects/models. Verified pandas/Polars, Spark, Delta and CLI; [evidence](55-sm33a-field-naming-validation.md) |
| SM-33B | Explicit date parsing, source timezone and UTC-safe reads | SM-33A | DONE | Strict parsing before filters; saved-rule approval replay; non-UTC real Delta transport. [Validation](56-sm33b-training-date-validation.md) |
| SM-33C | Date-free training and optional per-row result availability | SM-33B | DONE | Explicit random/temporal split; Core splitter reuse; optional dates, delayed labels, pinned snapshot/holdout and approval replay |
| SM-33D | Core CV, explicit training sampling and independent data-window selection | SM-33C | DONE | 274 native + 41 CLI + 14 local Delta checks; Core CV/FE isolation, deterministic sample/replay, explicit calendar zone; [evidence](59-sm33d-cv-sampling-and-window-validation.md). No cloud deployment |
| SM-33E | Training setup examples, operator guide and live acceptance | SM-33A through SM-33D | DONE | Guided setup/preview; 48 actual CLI tests; live pandas/Polars train/CV/MLflow/approval, 240 + 3 rows and no-op; [evidence](60-sm33e-guided-setup-and-live-validation.md) |
| SM-33F | Clear initializer questions and task-specific choices | SM-33E | DONE | Plain columns, named prediction output, Python Core/custom FE with saved source, task-specific menus and enabled generic cron; 56 CLI + 227 local tests; [evidence](61-bundle-initializer-usability.md) |
| SM-33G | Answer-driven setup without data inspection | SM-33F | DONE | Declare timestamp/local clock/date/text; only relevant parsing questions; CV/schedule/compute branches verified; 37 local + 56 CLI tests, strict dev validation; [evidence](61-bundle-initializer-usability.md#sm-33g-follow-up-answer-driven-questions) |
| SM-33H1 | Core-backed pre-split training cleanup | SM-33G | DONE | Same-file DropMissingRows/ManualBounds recipe, Core leakage admission, both engines, saved recipe and minimum approval replay; local Delta and CLI validation passed; [evidence](62-pre-split-cleaning-and-leakage-plan.md) |
| SM-33H2 | Cleanup artifact evidence and lifecycle replay | SM-33H1 | DONE | Reviewed evidence checks; local suites and one approved serverless run passed both engines, saved-code replay, CV/MLflow, lifecycle and exact 240+3/no-op output identity; [live evidence](62-pre-split-cleaning-and-leakage-plan.md#sm-33h2-live-acceptance-evidence-2026-09-26); committed in d5d2398d |
| SM-33H3 | All existing nodes in the correct pre-split/preprocessing phase | SM-33H2 | DONE | Local code/review/gates and approved live run848857785722024 passed. Registry-complete node/mode matrix; reuse Core nodes, ordered fixed cleanup and existing dedup, fold-local learned FE, train-only resampling, saved normalization/inference replay, pandas/Polars tests and Python examples; [scope](62-pre-split-cleaning-and-leakage-plan.md#sm-33h3--all-existing-nodes-in-the-correct-phase) |
| SM-34 | Independent score/train schedules and training windows | SM-33H3 | LOCAL DONE | Independent cron/timezone/pause, holdout/result lag, early MLflow pin; 647 local tests, 63 CLI generation tests, strict dev validation and independent review passed. [Evidence and live limits](65-sm34-schedules-and-window-plan.md); no new live clock/queue run. |
| SM-34A | Meaningful lifecycle task graph | SM-34 | DONE | Shared phases and durable MLflow references; live join corrected to NONE_FAILED. Pandas automatic and Polars manual approval both reached child score; each wrote 240 + 3 rows, no-op retained Delta v2, exact initial rows/MLflow evidence verified. [Live evidence](rehearsals/sm34a_live/README.md). Two jobs, no new control tables; clocks paused, clock firing/new-graph rollback/reject not tested. |
| SM-34B | Simplify the visible lifecycle graph | SM-34A live acceptance | DONE | Eight tasks/eight edges; durable evidence and guarded handoff preserved. Final 735 local tests/16 optional skips plus 63 CLI checks, lint/type/docs/Bundle gates and independent review passed. Both engines, approval, 240+3/no-op, two intentional failures and audit `683054016979330` passed live. [Evidence](67-sm34b-simplified-lifecycle-graph.md), [live receipt](rehearsals/sm34b_live/README.md). |
| SM-34C1 | Share training preparation and registration operations | SM-34B | DONE | Shared SDK/task preparation and registration; durable boundaries and compatibility preserved. Local tests, Ruff/type checks and independent review passed; committed as e16b27e8, no cloud run. [Evidence](69-sm34c1-shared-training-preparation.md) |
| SM-34C2 | Clarify evidence and result ownership | SM-34C1 | DONE | Shared saved-spec conversion, verified evidence owner and workflow result builder; task-to-notebook dependency removed with compatibility imports preserved. 503 tests after Sourcery readability follow-up, lint/type checks and independent review passed; committed as e16b27e8, no cloud run. [Evidence](70-sm34c2-evidence-result-ownership.md) |
| SM-34C3 | Avoid redundant replay within a task | SM-34C2 | DONE | Both engines/policies: compare/decide source reads 3 to 2, client artifact downloads 27 to 25; registration and mutation guards retained. 327 tests, lint/type checks and review passed; committed as f4d654cf. [Evidence](71-sm34c3-replay-reuse.md) |
| SM-34C4 | Reduce initial setup complexity | After SM-34C3 in delivery order | DONE | Default prompts 33 to 26; 13 advanced settings retain config-file overrides. 127 local tests, 68 CLI cases verified; 12 before/after configs and previews identical. Committed as d8e4949f, no cloud run. [Evidence](72-sm34c4-initializer-simplification.md) |
| SM-34C5 | Keep generated projects small and docs consistent | SM-34C4 | DONE | README 660 to 154 lines, recipe 113 to 24; central custom example, graph2 docs, fresh-process saved-code tests. 241 combined pre-commit tests and hooks passed; committed as d8e4949f. [Evidence](73-sm34c5-smaller-generated-project.md) |
| SM-34C6 | Retire redundant notebook lifecycle routing carefully | SM-34C5; ownership from C2 | DONE | Fixed score entrypoint, no training temp directory, retained sequential API adapter; 204 combined tests plus 10 final output tests (205 distinct), lint/type and review passed. Committed as d9596763; one corrected live scenario passed after custom registration fix 5c9798b9. [Live evidence](75-sm34c-live-acceptance.md). [Evidence](74-sm34c6-explicit-notebook-entrypoints.md) |
| SM-35 | Multi-metric quality gates and clear thresholds | SM-34C6 | DONE | Optional absolute gates enforced across SDK/tasks/first champion/approval; canonical historical digests and per-gate evidence. Sourcery follow-up covers alias commits, workflow checks/preview/dispatch and pre-split validation. 578 local tests and 3,512 identical offline outcomes; committed b67594b7 with DCO/hooks. Later SM-36 acceptance verified live passing/failed gates and corrected UC tag names (uncommitted fix). [Local evidence](77-sm35-quality-gates-evidence.md), [live evidence](83-sm36-databricks-live-acceptance.md) |
| SM-36 | Core tuning, model search and optional explainability | SM-35 | DONE | Five strategies, automatic/custom spaces, four ensemble families, shared diagnostic CV and bounded SHAP. Expanded Databricks acceptance: 170/170 model-strategy pairs / 3,756 trials; 254 setting scenarios; 143 penalty regression and 90 pruning tests; 45 CLI generation cases. Legacy sklearn penalty loss fixed. Earlier eight lifecycle/score/replay/no-op and manual operator checks remain in report83. Included in this local delivery commit; true nested tuning is tracked separately in SM-36f. [Expanded evidence](85-sm36-full-strategy-acceptance.md), [earlier lifecycle evidence](83-sm36-databricks-live-acceptance.md) |
| SM-36a | Project-owned feature engineering and output rules | SM-33D, SM-33H3 | DONE | Saved project packages/custom steps: 9cc81304 (report102); temporal history: 84a7dcb4 (report105). Keyed scoring exclusions/coverage, atomic continuation, bounded assets/exact dependency pins and named output rules passed local and live acceptance ([report106](106-sm36a-project-delivery.md)); final delivery includes 156 passing cloud tests with zero skips (run 530905329987142). Optional H3Index/sentence-model runtime checks, custom pre-split value normalization and new predicate/data-quality gates remain separate. |
| SM-36b | Multiple training branches from one pinned source | SM-36, SM-36a | DONE | Per-target pipelines/labels/tuning/metrics, linked MLflow runs, reproducible splits and bounded execution; keep multiple models rather than selecting one winner |
| SM-36c | Composed multi-model scoring and model-set lifecycle | SM-36b | DONE | Immutable component/rule package, per-rule dependencies, one controlled set champion/rollback, keyed outputs and atomic Delta/history publication. Final-wheel cloud suite: 114 passed; actual Bundle train/package/approve/score and all-excluded/temporal continuation passed. Uncommitted. [Evidence109](109-sm36c-model-set-delivery.md) |
| SM-36d | Databricks segmentation training and scoring | SM-36 | PARKED | Reuse Core K-Means, Mini-Batch K-Means, Gaussian Mixture and Birch; explicit clustering setup, optional reference column, preprocessing, cluster metrics, artifact/MLflow replay and new-row scoring. Define cluster-specific lifecycle policy without assuming supervised CV/tuning support. [Scope and acceptance](81-sm36d-segmentation-task.md) |
| SM-36e | Connect optional SHAP setup and readable results | SM-36 | DONE | Opt-in prompts, explicit training dependencies, bounded global/sample PNG charts in notebook and MLflow, child/winner report provenance. 193 local + five lifecycle tests, 12 CLI cases, three strict validations, 41 cloud contracts and real training/report acceptance passed. Final-wheel and reading-guide replays passed; final synthetic-example removal checked locally. [Evidence121](121-sm36e-shap-delivery.md) |
| SM-36f | True nested tuning across Core, Canvas and Databricks | SM-36 | DONE | Independent inner searches, fold-local preprocessing, separate final search and stored outer evidence. 499 Python regressions, 12 output tests, 113 final frontend tests, 3 CLI generation tests; gates passed. Additional real backend Basic/Advanced x 6 families x ordinary/nested: 24/24. Cloud: 66/66 nested cases across all five strategies and both engines, 12/12 fixed cases, 24/24 reference tests, 1,872 replay predictions. All 255 packaged module hashes verified. Included in this local delivery commit; temporal/group/threshold follow-ups are SM-36g/h/i. [Evidence](88-sm36f-nested-tuning-evidence.md), [receipt](89-sm36f-nested-tuning-receipt.json) |
| SM-36g | Nested temporal cross-validation | SM-36f | DONE | Temporal inner/outer/final search, strict clock and holdout boundaries, row gap/expanding/rolling windows; all layers connected. Final local 291 tests; frontend134; cloud16/16 +384 replay rows; independent persisted audit passed. Included in the nested-policy delivery commit. [Evidence and limits](101-sm36ghi-nested-policy-acceptance.md) |
| SM-36h | Nested group cross-validation | SM-36f | DONE | Group/stratified-group policies preserve metadata and isolate inner/outer/final holdouts; split identifiers excluded from features. Final local 291 tests; frontend134; cloud16/16 +384 replay rows; independent persisted audit passed. Included in the nested-policy delivery commit. [Evidence and limits](101-sm36ghi-nested-policy-acceptance.md) |
| SM-36i | Nested decision-threshold tuning | SM-36f | DONE | Binary training-only inner OOF thresholds, threshold-aware outer scoring, separate final threshold and persisted provenance/artifact parity. Final local 291 tests; frontend134; cloud16/16 +384 replay rows; independent persisted audit passed. Included in the nested-policy delivery commit. [Evidence and limits](101-sm36ghi-nested-policy-acceptance.md) |
| SM-54 | Single-job candidate competition and one champion | SM-35, SM-36f | DONE | Single target only; shared snapshot/folds, candidate tuning/recipes/runs, deterministic CV winner, winner-only registration and existing lifecycle. Guided model-owned Python settings verified on both engines: 419 local + 511 final-wheel cloud tests and real single/competition/multi-model lifecycle/scoring acceptance. Signed delivery includes reports115-119. [Final evidence119](119-model-layout-live-delivery.md), [Delivery116](116-sm54-model-competition-delivery.md), [Plan115](115-sm54-model-competition-plan.md). |
| SM-37 | Production identities and enforced writer ownership | SM-32, SM-33 | PARTIAL | Modular deployment files, optional environment development pairs, same/separate run identities and opt-in job ACLs delivered; every target has an editable permissions list. Current paired layout passed 119 CLI cases locally. Earlier live personal deployment passed train, automatic child score (120 predictions) and unchanged no-op; its seven-target layout validated in one workspace. Current paired layout subsequently deployed for six isolated test_development scoring projects ([Live136](136-sm41-live-cdf-recovery.md)). Separate-identity grants/alias-denial and concurrent lifecycle queue evidence remain open. UC/experiment grants remain external prerequisites, not provisioned resources. [Delivery131](131-sm37-local-deployment-delivery.md), [Live132](132-sm37-live-deployment-verification.md); original scope [report93](93-dbml-reference-recomparison.md). |
| SM-38 | Operational limits, retry/recovery and run summaries | SM-34, SM-37 | PARTIAL | Operational variables and summaries delivered. Live 30-second task/job timeouts, bounded retry, post-commit score recovery, unchanged 120-row/Delta-v1 readback and user-confirmed email passed. Visible generated score HTML fixed and verified in exported run output. Lifecycle retry guards preserved; test jobs PAUSED/idle, normal score restored. Optional webhook delivery remains open; model-set HTML summaries passed live in [Live136](136-sm41-live-cdf-recovery.md). [Delivery133](133-sm38-operational-controls.md), [Live134](134-sm38-live-operations-verification.md); original scope [report93](93-dbml-reference-recomparison.md). |
| SM-39 | Reproducible packaging and per-target compute | SM-33 | DONE | Checked DAB wheel build with content-specific cache identity, centralized direct runtime pins, automatic selected Optuna/model dependencies, custom feature requirements and per-target compute/tags/budget settings delivered. Two clean installs, two real train/registry/score flows and 360 independent prediction comparisons passed; existing models survived changed-wheel and restored-wheel redeployments. 263 distinct affected cases passed. Pins are not a full transitive lock; company policy execution remains SM-43b. [Delivery157](157-sm39-runtime-packaging-delivery.md), original scope [report93](93-dbml-reference-recomparison.md). |
| SM-40 | Generated-project tests and generic CI/CD | SM-37, SM-38, SM-39 | PARTIAL | Standalone project tests/build/validate and optional company adapter; explicit deployment approvals; no default chargeable PR runs; report93 additions: `ci` target, generated recipe/preflight tests and lint config, credential-free render of every init example checked against the Bundle schema in repository CI; per-project working directory for several generated projects in one repository ([report93](93-dbml-reference-recomparison.md)); static smoke delivered locally; isolated CI runner removed/deferred by user, remaining CI/CD scope open ([report129](129-reference-followups-delivery.md)) |
| SM-41 | CDF recovery, full refresh and generation retention | SM-31, SM-33, SM-38 | PARTIAL | Optional default-off CDF recovery verified live: six pandas/Polars single/competition/model-set jobs passed actual retention expiry, separate recovery branch, 123-row replacement, one-row incremental continuation and no-op. Exact model/source/target pins, table IDs, nonempty grants, prior Delta history and HTML reports verified. 20 real-Delta state cases plus four full-rebuild empty/view cases passed (CDF boundary injected in those harnesses); 364 local tests and analysis gates passed. Test retention restored, temporary grants removed, 12 jobs PAUSED/idle. Explicit source-replacement recovery and generation retention remain open. [Delivery135](135-sm41-cdf-recovery-plan.md), [Live136](136-sm41-live-cdf-recovery.md). |
| SM-42 | Complete modular setup, scenario examples and operator guide | SM-30 through SM-41 | WAIT | Progressive setup sections with optional custom/multi-model scenarios; registry-backed parameters, editable config, execution preview and both-engine lifecycle/recovery examples; report93 additions: optional demo source setup so a first deploy runs end to end; generated README links pinned to the docs version matching the wheel ([report93](93-dbml-reference-recomparison.md)) |
| SM-43a | Combined personal-workspace acceptance | SM-42 | WAIT | Representative real-data FE/models, both engines, manual/auto, retained challenger, new rows, no-op, failure/rollback/queue; exact run/resource evidence |
| SM-43b | Company-target production readiness gate | SM-43a; confirmed company environment | WAIT | Approved identities/UC/policy compute/dependency/CI/data checks in the actual company target; no inference from personal serverless tests |

The tasks are sequenced by this table when dependencies allow. SM-43b needs
actual company settings and access; it must not prevent unrelated local work.
No new live resources or company deployments are authorized merely by a queue
status. Existing explicit live authorizations retain their original scope.

## SM-54 - Single-job candidate competition (2026-09-28)

Implemented and verified locally and on Databricks; included in the signed delivery
documented in report119. One lifecycle job trains
several candidates for the same supervised task and target, then selects one
winner. This is separate from SM-36b/36c, which retain and compose multiple
models rather than select one champion.

Delivery: [report116](116-sm54-model-competition-delivery.md). The first delivery
executes candidates sequentially and requires all requested candidates to finish.
The approved 2026-09-30 Bundle follow-up uses separate candidate training tasks
and leaf SHAP report tasks; selection still requires every candidate. The
original sequential Core API remains available. See [report122](122-training-and-shap-nodes.md).
Model-set nesting remains outside the user-approved single-target scope.

Acceptance:

- Setup accepts multiple candidates. Classification permits standalone
  classifiers plus voting/stacking classifiers; regression permits standalone
  regressors plus voting/stacking regressors. Reject mixed tasks or targets.
- Each candidate retains its own model parameters, tuning strategy/settings
  and search space. Reuse Core automatic spaces and ensemble base-model tuning;
  preserve explicit overrides and fixed parameters. Keep single-model setup valid.
- Pin one source snapshot and row membership for the competition. Use identical
  selection folds and one comparison metric/direction across candidates.
  Learn preprocessing only inside the appropriate training folds.
- Choose the winner from training-side validation/CV evidence. Reserve the
  common holdout for final quality gates and comparison with the existing
  champion; do not tune or select candidates repeatedly on that holdout.
- Store a parent run with linked candidate runs, fitted artifacts, parameters,
  metrics and failure evidence for every candidate. Report the leaderboard,
  selection metric, winner and promotion decision separately.
- Register only the winner under the common registered-model name. Its first
  registered candidate is v1, the next is v2, regardless of algorithm family.
  Losing candidates remain inspectable as runs/artifacts without new versions
  in that registered model. A registered winner is not automatically champion.
- Reuse manual/automatic promotion, first-champion quality requirements,
  minimum improvement and previous-champion rollback. If no candidate qualifies,
  preserve the existing champion (or leave it unset on the first run).
- Resolve candidate ties deterministically using a saved rule; a tie against
  the current champion must not cause promotion. Exactly one decision task
  owns alias changes after all candidate results have been collected.
- Bound candidate concurrency, total search budgets and retries. A failed
  requested candidate fails the competition by default; do not silently select
  a winner from incomplete results. Retries must not duplicate registration or
  alias changes. Never claim parallel execution from a sequential loop.
- Verify classification + ensemble and regression + ensemble competitions,
  mixed tuning strategies, shared folds, tie/failure handling, first/subsequent
  champion selection, manual approval and rollback. Include readable job output
  and focused live Databricks evidence in addition to local tests.

## dbml reference re-comparison - 2026-09-28

Static re-read of `/Users/BH7043/repositories/dbml-mlops-template` against the
current template and integrations. Most remaining gaps already belong to
SM-37 through SM-42 and SM-19/21/23; the review adds acceptance items to
those tasks and four new tasks. Full mapping, acceptance and corrections:
[report93](93-dbml-reference-recomparison.md). Planning only; no code change.

Acceptance additions (details in report93):

- **SM-37:** optional UC `registered_models`/`schemas` grants, experiment
  permissions, operator `CAN_MANAGE_RUN`, optional per-target host/catalog prompts,
  shared non-home syst/prod `root_path`.
- **SM-38:** email/webhook notifications, `health.rules` duration limits, task timeouts.
- **SM-39:** DAB `artifacts:` wheel build, one Skyulf/MLflow version variable,
  automatic Optuna dependencies, serverless `budget_policy_id` and job tags.
- **SM-40:** `ci` target, generated recipe/preflight tests and lint config,
  credential-free render of every init example checked against the Bundle schema in repository CI,
  per-project working directory.
- **SM-42:** optional demo source setup so a first deploy runs end to end; docs
  links pinned to the wheel version.
- **SM-23a:** reuse Core `DriftCalculator`, delayed-label performance join,
  optional `quality_monitors` and dashboard.

| Task | Status | Dependency / scope |
| --- | --- | --- |
| SM-45 | WAIT | SM-37, SM-40; suffix-scoped dev/CI cleanup job, dry-run default, production targets refused; separate from SM-41 retention |
| SM-46 | WAIT | SM-34, SM-38; optional `scoring_mode: on_table_update` via Jobs table-update trigger, reusing CDF/queue/no-op |
| SM-47 | WAIT | SM-38; MLflow dataset input for UC lineage, model-version card, optional experiment resource; MLflow 3 deployment jobs investigated without a second alias writer |
| SM-23c | FUNCTIONAL DONE | Drift and independent performance policy delivered. Local gates and isolated Databricks measurement/request/replay acceptance passed. Remaining separate-identity/company production gates belong to SM-37/43b. [Delivery158](158-sm23c-performance-policy.md). Never approves. |
| SM-48 | PARTIAL | Local split complete; live redeploy ID/history check pending. None; split `resources/workflow.jobs.yml` into `train.job.yml`/`score.job.yml` and optionally clearer job display names; keep job keys `train`/`score` (renaming recreates jobs and loses IDs/history) |
| SM-49 | WAIT | SM-39, SM-48; upgrade path for generated projects: regenerate from saved answers into a temporary directory, review the diff, run config migration and deployed-contract checks |
| SM-50 | WAIT | SM-41; optional explicit period backfill operator action reusing `publish_replace_period`, separate from incremental and full rebuild |
| SM-51 | WAIT | SM-37, SM-41; production model-version and MLflow run retention preview; never deletes aliased or receipt-referenced versions |

## Integrations code-quality review - 2026-09-28

Gates are clean for `skyulf-core/skyulf/integrations/`: Ruff/format, ty at the
locked 0.0.75, Lizard CCN 10, 0.4% duplication, no missing docstrings.
Fifteen local subprocess-test failures were caused by a missing editable
install, not code. Details: ([report94](94-integrations-code-quality-review.md)).

| Task | Status | Dependency / scope |
| --- | --- | --- |
| SM-52 | DONE | Four shared modules and directly named domain helpers replace borrowed-private access without helper aliases; 2,149 integration tests passed, 256 skipped; active monkeypatch controls, optional imports, static gates and installed wheel verified. [Delivery129](129-reference-followups-delivery.md) |
| SM-53 | WAIT | SM-36g, SM-36h, SM-36i, SM-52; split `local_retraining.py` (1525 lines) along snapshot read, split, fit and evidence boundaries; `promotion.py` stays as designed (report64) |

Rules for every task touching integrations: 88 functions sit at CCN 9–10, so
budget helper extraction inside the feature task instead of raising the gate;
fix >100-character strings/docstrings in edited lines (no gate reports them).

## Custom feature engineering and multi-model follow-up

| Task | Status | Dependency / scope |
| --- | --- | --- |
| SM-44 | LATER | After SM-43a and SM-36c: migrate the supplied mlmodeltesting example (four models plus color rules), verify original/reconstructed training boundaries, batch imputation and encoding parity, source filters, keys and final publication. User deferred this until the generic Bundle is built. |

## Approved extensions after the local improvement gate

The user explicitly requested serving, A/B, ai_query and online feature lookup.
These existing IDs are required backlog work, implemented after SM-43a in the
listed order, while each feature remains opt-in for generated projects.
See [the delivery contract](39-serving-and-feature-lookup-delivery-plan.md).

| Task | Status | Dependency / scope |
| --- | --- | --- |
| SM-19a | PARTIAL | Minimum pinned HTTP bridge, readiness and real classifier HTTP/local parity delivered for SM-23b. Full supported-artifact, authentication/error-input, warm/cold and bounded-load acceptance remains separate. [Delivery177](177-sm23b-online-monitoring.md). |
| SM-19b | LATER | SM-19a; SQL ai_query invocation, named inputs, privileges and failure behavior |
| SM-19d | LATER | SM-19a; A/B/canary routing, endpoint update/rollback; batch rollout separately explicit |
| SM-21a | LATER | SM-43a; UC feature lookup, keys and point-in-time correctness |
| SM-21b | LATER | SM-21a/19a; optional online publication, freshness and serving lookup |
| SM-23a | BATCH DONE | Quality/drift/delayed outcomes, Spark measurement, shared Delta storage and native AI/BI delivered. All three producer layouts use separate monitoring jobs; native refresh and execution/cost/CPU/RAM views are delivered ([Delivery163](163-sm23h-dashboard-refresh-compute.md)). Overview independently counts current performance evidence ([Delivery164](164-sm23i-monitoring-overview-closure.md)). Optional native Data Profiling (`data_quality`, replacing deprecated `quality_monitors`) and custom slices remain separate enhancements. Company/identity acceptance stays in SM-37/43b. |
| SM-23b | FUNCTIONAL DONE | Actual native telemetry VIEW/backing Delta, pinned endpoint/model versions, Spark drift and delayed-label performance, shared inventory/dashboard and retraining dedup delivered. Native parser: 33 passed; HTTP: 10 predictions matched; two late-label verdicts healthy/degraded, report replay, native refresh and 9 synthetic dashboard queries passed. Seven legacy monitoring readers passed native compatibility and received a wheel-only upgrade; all 12 train/score jobs stayed unchanged. [Delivery177](177-sm23b-online-monitoring.md). Full SM-19a and SM-37/43b acceptance stays separate. |
| SM-23c | FUNCTIONAL DONE | Opt-in drift/performance policies and real request/replay acceptance delivered. Inherited SM-37/43b production gates remain separate. [Delivery158](158-sm23c-performance-policy.md). Never approves ([report93](93-dbml-reference-recomparison.md)). |
| SM-19c | PARKED | Continuous streaming remains outside the current user-approved implementation sequence |

SM-17/24c/20b remain later Spark enhancements after SM-43a. SM-18 stays parked.

### SM-23i — Overview performance and batch closure (DONE, 2026-10-05)

Presentation follow-up: Current monitoring enrollments moves to Drift and data
quality without the displayed Monitoring identity column; Overview retains
counters and removes its duplicate performance evidence table. Dataset and
monitoring semantics are unchanged. [Record167](167-sm23j-overview-layout-cleanup.md).

Four separate performance counters use current enrollment/version evidence;
drift/data-quality counters retain their independent meaning. Old configurations,
backfills and stale/unavailable policy windows cannot appear healthy. Shared model
selectors work in published Chrome. All 28 datasets passed 56 live SQL checks;
37 affected tests passed. Same dashboard ID and permission mode were preserved.
SM-23 batch scope closed here; online SM-23b subsequently completed on 2026-10-06
([Delivery177](177-sm23b-online-monitoring.md)). Production SM-37/43b gates remain
separate. SM-57 subsequently delivered its initial admitted scope through the SM-56 safety inventory ([Delivery170](170-sm57-spark-pyfunc-delivery.md)).
[Evidence164](164-sm23i-monitoring-overview-closure.md).

### SM-23h — Native refresh and cluster utilization (DELIVERED, 2026-10-05)

Native post-monitoring dashboard refresh is deployed to all three existing
producer layouts. Competition run `825802718768390` passed all five tasks,
including `refresh_monitoring_dashboard`, without requesting training.
CPU/RAM cards and trends use scoped classic-node telemetry; current serverless
workloads correctly show unavailable metrics. The redundant Selected model table
is removed; Performance checks separates the measured result from training action.
All 27 datasets passed 54 live SQL checks; 55 distinct focused tests passed.
[Evidence163](163-sm23h-dashboard-refresh-compute.md).

### SM-23g — Performance presentation and template cleanup (DELIVERED, 2026-10-05)

Native performance page now separates latest batch counts, measured metrics,
latest policy decision, historical loss and classification evidence. All 26
datasets passed 52 default/NULL SQL checks; the same dashboard ID was published.
Six unreferenced template notebooks were removed. All three existing layouts
redeployed with nine unchanged job IDs; competition monitoring run
`844667177339309` passed the current three-task graph without requesting training.
Documentation distinguishes Spark calculation, Delta storage, native AI/BI,
separate dashboard refresh, and independently enabled drift/performance triggers.
[Evidence162](162-sm23g-monitoring-presentation-cleanup.md).

### SM-23f — Dashboard interaction and monitoring report (DELIVERED, 2026-10-05)

User-reported empty selectors were reproduced in authenticated Chrome. Independent
option datasets and NULL-safe defaults fix their mutual filtering and blank results.
All 24 datasets passed 48 default/NULL query checks before same-ID publication;
published drift charts, model selection, real RMSE degradation, confusion counts
and job execution rows were verified in Chrome. Attributed billing has no rows.
Runtime reports now separate drift, measured performance and performance loss;
the generated graph is `monitor_model -> monitoring_report -> evaluate_retraining`.
All three layouts redeployed in place; a final model-set monitoring run passed
all three tasks. Historical real-label comparisons show regression degradation
without changing policies or requesting training. All 142 distinct focused cases,
independent review, full CI static gates and applicable pre-commit hooks passed.
[Evidence161](161-sm23f-monitoring-dashboard-runtime-repair.md).

### SM-23e — Native job handoff and dashboard acceptance (DELIVERED, 2026-10-05)

Approved follow-up: generated jobs use Spark directly, with visible native
train -> score -> monitoring job handoff. Remove the redundant inline score
report/retraining graph and engine choice. Repair dashboard model/context
selection, preserve missing-latest-result semantics, expose confusion counts,
and add explicitly scoped native execution/list-price billing views.

[Implementation and evidence160](160-sm23e-monitoring-jobs-dashboard.md) records
297 focused local tests, 25 real Spark cases, all nine original generated
single/competition/model-set train/score/monitor runs and three successful
monitoring-only late-label replays. Coverage rose from 0.5 to 1.0 without changing
prediction snapshots; classification confusion counts rose from 20 to 40.
The existing dashboard is published with 21 validated datasets and 48 successful
selected-context queries. Independent review and static gates passed. Isolated
acceptance schedules remain paused; browser interaction and actual billing data
remain unverified/unavailable respectively, rather than claimed as proven.
This follow-up does not relax the inherited production identity/approval gates.

### SM-23d — Distributed monitoring (DELIVERED, 2026-10-05)

Training/scoring capacity and monitoring capacity remain separate budgets.
The new Spark path removes the local observation cap without changing training
budgets. Existing configurations retain their local behavior; new generated
projects default to a separate serverless monitoring job.

- Evaluate quality, drift and delayed-label metrics on Spark without collecting
  the complete observation into driver memory. Keep local pandas/Polars support.
- Define exact versus approximate metrics explicitly before implementation;
  document sampling, approximation error and coverage if any approximation is used.
- Preserve snapshot/version identity, record-key joins, label availability,
  duplicate detection, model-set semantics and retraining quality gates.
- Give monitoring its own resource and execution budgets, visible failure state
  and retry behavior; large training allowances must not determine these budgets.
- Acceptance: compare distributed results with existing bounded reference cases,
  verify real Databricks/Delta observations beyond current local materialization
  limits, and record memory/runtime evidence. Do not silently truncate input or
  raise the local safety caps to call the task complete.

Status: **DELIVERED — implementation, focused local and isolated cloud acceptance**.
[Implementation and evidence159](159-sm23d-spark-monitoring-plan.md): 1,100,000-row
regression/classification measurements, prepared references, unchanged/new-data
guards, actual independent three-task job, real training request/replay dedup and
native AI/BI SQL/publication. All 12 generated variants passed strict validation.
The scale deployment is single-model; model-set/competition contract coverage
does not claim a new live deployment for every variant. Inherited SM-37/43b
production gates remain separate. This is not another closure in the old defect audit.

Design implemented 2026-10-05:
the user prefers computing monitoring directly on Spark from the outset.
The primary path is **Spark calculation -> Delta evidence -> Databricks
AI/BI dashboard + existing Skyulf policy/retraining guards**. Native InferenceLog
is not a required parallel calculator. A native profiling-generated dashboard
remains a separate optional integration; the AI/BI dashboard can read Skyulf's
metric tables directly.

- Run monitoring as a separate job with independent compute/schedule/retry
  budgets. Scoring publishes durable receipts; late-label changes also cause
  affected windows to be revisited without requiring new predictions.
- Preserve original prediction/source snapshots, physical table identity, stable
  keys, model/component version, label availability and duplicate rejection in
  distributed joins. Only bounded aggregate evidence reaches the driver.
- Prepare version-bound baseline evidence without repeating full local holdout
  replay each observation. Move fresh-training-data eligibility checks to Spark
  as well; otherwise retraining decisions would retain the local bottleneck.
- Establish exact regression and confusion-matrix classification parity first;
  specify drift algorithms and any approximation before enabling their decisions.
  No silent fallback to bounded local data or silent dropping of unsupported metrics.
- Retain single-model, activated competition-winner and per-component model-set
  semantics. Shared request deduplication and evaluation/approval gates remain.
- Validate beyond one million rows on Databricks, including late labels,
  replay/failure recovery, baseline/version changes, metric parity and measured
  memory/runtime. Training/scoring engine capacity is a separate workstream.

The current usable policy example and field explanations are in the
[monitoring guide](../../skyulf-core/examples/databricks_monitoring/README.md#enable-each-independent-model-repository)
and the generated producer README. The guide documents exact/bounded statistical
methods, unsupported types/recipes, explicit reference preparation for legacy
models, a paused-by-default independent schedule and the late-label revisit horizon.

### SM-23c performance policy acceptance - clarified 2026-10-04

The user explicitly requested this policy. Implementation, local gates and isolated
live measurement/request/replay acceptance passed on 2026-10-05. The performance
scope is complete; evidence is in [Delivery158](158-sm23c-performance-policy.md).
`on_drift` and performance policy remain independent signals.

- Add an independent, default-disabled performance policy with off/report/retrain
  behavior; retain independent drift controls. Reuse existing Core metric
  calculations rather than implementing a second metric engine.
- Configure the metric, its improvement direction, an explicit version-bound
  baseline, absolute or relative degradation tolerance, observation window,
  minimum labeled sample count/coverage and consecutive failing windows.
  Example only, not defaults: F1 falls by at least 0.05 absolute for three windows.
- Compare matching metric definitions, class conventions and eligible populations.
  Distinguish a training holdout reference from a production reference window;
  changing the concrete model version requires a matching baseline and resets
  the consecutive-window state.
- Join original predictions to actual outcomes by stable keys. Respect outcome
  availability and maturity; absent, stale, insufficient or nonfinite evidence
  means unavailable, never an inferred degradation or improvement.
- Count distinct completed observation windows. Replays must not advance the
  failure streak; gaps and invalid windows must not silently satisfy a consecutive
  failure requirement. Report baseline/current values, direction, degradation,
  coverage, evaluated windows and the exact action/skip reason.
- Show performance degradation explicitly in the existing monitoring report and
  dashboard, separately from drift. Display model/version, metric, reference and
  current values, absolute/relative change, tolerance, labeled coverage, window
  timestamps, consecutive failure count and resulting action. Distinguish
  disabled, insufficient/stale evidence, healthy, degraded and retraining
  requested/skipped states; missing labels must never appear green/healthy.
  Provide a time-series view of metric versus baseline/tolerance and drill-down
  to the observation and training request. Verify dashboard values against the
  stored evidence for both drift-only and performance-only failures.
- Retraining reuses the existing request gate, cooldown, active/queued-run checks
  and fresh labeled training-data guard. Concurrent drift/performance signals
  must not create duplicate requests. Monitoring labels are not implicitly merged
  into the upstream-owned training table. Existing evaluation/promotion rules remain.
- Prove degradation without drift, drift without degradation, higher/lower-is-better
  metrics, threshold boundaries, delayed/missing labels, repeated windows, model
  changes and combined triggers. Recalculate metrics independently and verify a
  real Databricks request, unchanged-data skip and replay without duplicate runs.

### SM-39 wheel setup simplification - proposed 2026-10-04

The user requested an easier alternative to manually maintaining `artifact.json`.
Proposed normal path: place one `skyulf_core-*.whl` in a project-owned input
`wheels/` directory, read its distribution/version from wheel metadata, and keep
the content-tagged prepared output in `dist/skyulf/`. Missing or multiple input
wheels must fail clearly rather than silently selecting the newest file. Retain
source-checkout builds as an explicit advanced path. No mandatory duplicate
version field in the ordinary workflow. This is a UX proposal; current runtime
still reads `deployment/artifact.json`, and the SM-39 delivered evidence above
describes that current implementation.

## Earlier implementation and live evidence

| Sıra | Görev | Bağımlılık | Durum | Kısa bitiş ölçütü |
| --- | --- | --- | --- | --- |
| SM-00 | Baseline ve Spark test ortamı | — | DONE | Güncel master: 231 baseline; 2 Spark smoke; [kanıt](BASELINE.md) |
| SM-01 | Execution/capability kuralları | SM-00 | DONE | Immutable config; registry operation desteği; aşağıda kanıt |
| SM-02 | Spark engine ve conversion sınırı | SM-01 | DONE | Gerçek Spark adapter; conversion korumaları; aşağıda kanıt |
| SM-03 | Dispatcher, schema, row keys | SM-02 | DONE | Keyed Spark FE giriş/preflight; aşağıda kanıt |
| SM-04 | Versioned portable state | SM-03 | DONE | Tagged state, limit/version/schema kontrolü; aşağıda kanıt |
| SM-05 | Native SimpleImputer | SM-04 | DONE | Mean/constant train-only fit, Spark apply; aşağıda kanıt |
| SM-06 | Native StandardScaler | SM-05 | DONE | Population variance, dört flag ve numeric parity; aşağıda kanıt |
| SM-07 | Uçtan uca FE kapısı | SM-06 | DONE | Üç engine çapraz fit/apply; kayıt/yükleme; aşağıda kanıt |
| SM-08 | Ortak inference bundle | SM-07 | DONE | Raw/features ayrımı; model metadata round-trip; aşağıda kanıt |
| SM-09 | Native FE + worker model | SM-08 | DONE | Spark tahmini local ile key bazında aynı; aşağıda kanıt |
| SM-10 | Worker Python FE + model | SM-09 | DONE | Batch-safe portable FE; window gibi yollar açık ret |
| SM-11 | Classification ve ölçek kapısı | SM-10 | DONE | Class/proba/threshold parity; transport, packaging and scale evidence |
| SM-12 | Opsiyonel MLflow tracking | SM-11 | DONE | Off bağımsız; gerçek run lifecycle ve izolasyon |
| SM-13 | MLflow model packaging | SM-12 | DONE | Temiz ortamda pyfunc yükleme ve parity |
| SM-14 | Registry/Unity Catalog adapter | SM-13 | DONE | Explicit publish; alias/version pinning; local evidence |
| SM-15 | Monthly batch + Delta sink | SM-14 | DONE | 43 local Delta/contract/admission tests; evidence below |
| SM-16 | Gerçek Databricks kapısı | SM-15 | DONE | Selected serverless workflow passed; live evidence below |
| SM-15L | Explicit-period local predictions -> UC Delta | SM-24a | DONE | Score requested period; preserve rows outside it; Spark only for bounded I/O |
| SM-15I | Automatic incremental local scoring | SM-15L | DONE | No date/source-version input; 80+80 synthetic and 200+100 real taxi rows; no-op replay; [core evidence](13-sm15i-live-validation-report.md), [real-data proof](15-sm15i-real-nyctaxi-live-report.md) |
| SM-17 | Complete existing Spark node/model coverage | SM-20a | LATER | Resume only after the first local Bundle; preserve all inventoried gaps |
| SM-18 | Backend/Canvas and legacy artifact bridge | SM-17 | PARKED | Capability UI/API, DAG contracts and existing deployment compatibility |
| SM-26 | Local pandas/Polars pipeline MLflow packaging | SM-16, existing local persistence | DONE | Fitted FE/model/engine, schema and thresholds preserved; MLflow 3.16.1 isolated-process parity; Spark/HTTP scopes rejected |
| SM-25 | Local-first config / SDK preflight | SM-26 | DONE | Immutable local config; pinned artifact load, bounded frame and actionable preflight; no node/model allowlist |
| SM-24a | Local training and bounded batch prediction | SM-25 | DONE | Two UC source tables, five live cross-job models, 62-ID FE audit and bounded monthly reads; [evidence](08-sm24a-live-validation-report.md) |
| SM-22a | Pinned candidate/champion validation | Current evaluation/registry | DONE | Local comparison/report and isolated UC metrics gate passed; [local evidence](16-sm22a-local-validation-report.md), [live evidence](17-sm22a-live-metrics-report.md) |
| SM-22b | Explicit promotion and rollback | SM-22a | DONE | Version-checked alias changes, prior/new-version receipt, conflicts and permission tests; [live evidence](18-sm22b-live-validation-report.md) |
| SM-22c | Challenger and previous-champion aliases | SM-22b | DONE | Validated challenger staging; guarded three-alias promotion/rollback; legacy receipts and partial-write tests; [local and UC evidence](19-sm22c-lifecycle-alias-validation-report.md) |
| SM-28a | Label-aware retraining service | SM-22a, SM-22b, SM-22c, SM-24a | DONE | Pinned training snapshot, temporal holdout, candidate registration/comparison; no automatic promotion; [evidence](20-sm28a-label-aware-retraining-report.md) |
| SM-20a | First local-engine Bundle/template | SM-24a, SM-15I, SM-22b, SM-22c, SM-28a | DONE | Generated/deployed Polars Bundle; train/compare/stage/promote and 2+2 incremental score/no-op live; [evidence](21-sm20a-local-bundle-validation-report.md) |
| SM-20P | Company-shaped Bundle draft | SM-20a | SUPERSEDED | Replaced by the generic four-target SM-20R direction; no company workspace deployment |
| SM-20R | Reset personal tests; generic minimal Bundle | SM-20a | DONE | Ten test jobs and three schemas removed; four target shapes validated, personal dev deployed with three jobs; one source, one prediction, one internal control, one UC model; Polars 600+50-row scoring and no-op replay; [live evidence](24-sm20r-clean-generic-bundle-validation-report.md) |
| SM-20S | Two-job, table-minimal local Bundle | SM-20R | DONE | First score creates only prediction output; no admission tables; 650+1 rows, replay no-op, final two jobs/two tables; [live evidence](26-sm20s-two-job-live-validation-report.md) |
| SM-28b | Optional monthly retraining schedule | SM-20S, SM-28a | DONE | Two-job Bundle with optional paused train schedule, pinned monthly label window/version and explicit activation; [plan](27-sm28b-monthly-retraining-plan.md), [validation](28-sm28b-monthly-retraining-validation-report.md) |
| SM-24b | Optional Databricks Jobs API operations | SM-20a | LATER | Add dynamic submit/status/cancel only if Bundle jobs are insufficient |
| SM-24d | Hard transport budget for wide UC rows | SM-24a | LATER | Add a proven paged/size-limited source adapter when exact transfer-byte enforcement is required; current Spark iterator bounds accepted decoded rows and frame memory only |
| SM-27 | Full-history local rescore and selectable Bundle mode | SM-20a, SM-15L | DONE | Live append retained 160 v1 rows and added ten v2 rows; full view exposes 170 v2 rows, retains v1 and its SELECT grant; [plan](29-sm27-model-change-scoring-plan.md), [live evidence](35-sm27-sm29-live-validation-report.md) |
| SM-29 | Gated automatic champion and score handoff | SM-22a/b/c, SM-28a/b, SM-27 | DONE | Live first champion, v2 promotion, tied v3 rejection, two-job handoff, score-failure recovery, queue/no-op and restricted alias-write denial passed; [design](31-sm29-auto-champion-design.md), [live evidence](35-sm27-sm29-live-validation-report.md) |
| SM-19 | Optional live HTTP / SQL ai_query / endpoint operations | SM-20a, compatible pyfunc package | PARTIAL | Minimum pinned HTTP bridge delivered with SM-23b; full SM-19a acceptance, SQL invocation and endpoint rollout remain open. Streaming remains parked. |
| SM-21 | Optional Databricks feature tables / online lookup | SM-20a; SM-19a for online serving | LATER | Point-in-time lookups and optional online freshness; declare any Spark dependency |
| SM-23 | Monitoring and inference observability | SM-20a, relevant batch/serving adapter | FUNCTIONAL DONE | Batch and online Spark measurement, independent drift/performance, delayed labels, shared native dashboard and refresh delivered. Online HTTP/log/version/replay acceptance passed on 2026-10-06. [Delivery177](177-sm23b-online-monitoring.md). Full serving acceptance remains SM-19a; production gates remain SM-37/43b. |
| SM-23c | Drift/performance-triggered retraining | SM-23a, SM-37 | FUNCTIONAL DONE | Independent default-off performance policy delivered: pinned metric/baseline/tolerance, mature labels, coverage, consecutive windows and shared retraining guards. Local gates, dashboard SQL/publish/readback, real training request, unchanged-data skip and replay/combined-trigger dedup passed live. Inherited SM-37/43b production gates remain separate. [Delivery158](158-sm23c-performance-policy.md). Gates unchanged, never approves; [report93](93-dbml-reference-recomparison.md). |
| SM-24c | Spark batch workflow adapter | SM-20a, existing Spark sink | LATER | Expose tested Spark runner after first local Bundle |
| SM-20b | Spark Bundle enhancement | SM-24c, selected SM-17 slices | LATER | Add tested Spark engine choice while preserving local variant |
| SM-45 | Suffix-scoped dev/CI resource cleanup | SM-37, SM-40 | WAIT | Dry-run default, confirmation, production targets refused; deleted set equals preview; other suffixes untouched; [report93](93-dbml-reference-recomparison.md) |
| SM-46 | Optional table-update scoring trigger | SM-34, SM-38 | WAIT | `scoring_mode: on_table_update`; bursts queue without duplicate rows, unchanged source no-op, paused in dev; [report93](93-dbml-reference-recomparison.md) |
| SM-47 | MLflow/UC lineage and model-version card | SM-38 | WAIT | Pinned source logged as MLflow dataset input, lineage visible live, card matches saved evidence; no second alias writer; [report93](93-dbml-reference-recomparison.md) |
| SM-48 | Split Bundle job resources and clearer display names | — | PARTIAL | Two resource files, same job keys/IDs on redeploy, cross-file `${resources.jobs.score.id}` resolves; template/generation tests, guide and README updated; strict validation passes; graph-3 live task/operator/scoring and persisted-data acceptance passed; pre-existing graph-2 ID/history migration still pending; [report93](93-dbml-reference-recomparison.md) |
| SM-49 | Generated-project upgrade path | SM-39, SM-48 | WAIT | Regenerate from saved init answers into a temporary directory, reviewed diff, `migrate_workflow_config` and deployed-contract checks; same job IDs after upgrade; guide section; ([report93](93-dbml-reference-recomparison.md)) |
| SM-50 | Explicit period backfill action | SM-41 | WAIT | Optional operator action on the score job reusing `publish_replace_period`; rows outside the period preserved, pinned model version, receipt, no-op replay; ([report93](93-dbml-reference-recomparison.md)) |
| SM-51 | Production model/run retention | SM-37, SM-41 | WAIT | Preview then scoped delete of old registry versions/runs; champion, previous_champion, challenger and receipt-referenced versions protected; rollback still works; ([report93](93-dbml-reference-recomparison.md)) |
| SM-52 | Integrations internal-helper boundary | — | DONE | Shared primitives and directly named domain helpers without aliases; 2,149 passed, 256 skipped; Ruff, full CI Ty, Lizard and installed wheel passed. [Delivery129](129-reference-followups-delivery.md); original [report94](94-integrations-code-quality-review.md) |
| SM-53 | Split `local_retraining.py` | SM-36g/h/i, SM-52 | WAIT | Cohesive modules for snapshot read, split, fit and evidence; public imports and saved evidence unchanged; complexity not increased; ([report94](94-integrations-code-quality-review.md)) |
| SM-55 | Optional prediction columns on an existing source table | SM-41 | LATER | Low priority, after the current Bundle work: separate prediction table remains the default; optional keyed updates of prediction/provenance columns on the source. Validate unique keys, column ownership, stale-row checks, CDF feedback prevention, permissions, idempotency and rollback; consider a joined view for unified reading. Not implemented. |
| SM-56 | Spark inference for pandas/Polars-trained pipelines with row-local FE | SM-29 | INVENTORY DONE; IMPLEMENTATION LATER | Actual apply-body inventory completed for built-in FE families. Portable JSON predict_spark still supports only SimpleImputer mean/constant and StandardScaler; extending it needs codecs and capability declarations. Trusted-pickle SM-57 can transport more fitted objects but requires independent batch-safety proof. Confirmed power-transform, replacement, binning and casting hazards; fitted group_agg is a mapping candidate, not a current-batch aggregate. Effective inference skips and worker Polars dependencies are documented. No new nodes certified. [Inventory165](165-sm57-partition-safety-inventory.md). |
| SM-57 | Project setup and Bundle pyfunc through mlflow.pyfunc.spark_udf | SM-56 inventory | DONE - INITIAL ADMITTED SCOPE | Independent local/Spark setup, fitted partition-safety gate, exact worker package, named UDF and distributed Delta lifecycle delivered. Single/competition/model-set each published 4,096 rows with exact local parity; no-op preserved table/receipt hashes. All initial and replay monitoring/native refresh chains passed on serverless. Initial nodes: mean/constant SimpleImputer and StandardScaler; Linear/LogisticRegression with reviewed tuning wrappers, independent model sets. Cluster virtualenv remains unverified; SM-58 now records the successful serverless scale matrix. [Delivery170](170-sm57-spark-pyfunc-delivery.md), [Plan166](166-sm57-spark-udf-plan.md). |
| SM-58 | Scale and memory validation of Spark inference | SM-56, SM-39 | PARTIAL — SERVERLESS PASSED; CLASSIC BLOCKED | [Validation171](171-sm58-scale-memory-validation.md): 18 serverless 1M/5M configurations, 108M verified predictions, worker load/RSS probes and operator guidance. Classic job creation rejected: workspace supports only serverless. Full executor RSS and policy-cluster virtualenv acceptance remain open. |
| SM-59 | Native Spark expressions for hot FE nodes (optional) | SM-58 | LATER | Only for nodes SM-58 shows as bottlenecks; not every node. |

SM-57 acceptance criteria (user-agreed setup flow, 2026-10-04):

- Ask about expected **inference input per run**, using rows, input size and
  available local capacity for guidance; do not hard-code a two-million-row
  cutoff. Ask once during project setup and save the choice.
- Large inference input selects pandas for training/preprocessing and generates
  actual `mlflow.pyfunc.spark_udf` scoring over Spark partitions. Local-suitable
  input keeps local scoring and the pandas/Polars engine choice. Store training
  engine and inference mode independently; do not silently switch routes.
- Keep training memory limits separate: pandas training still has to fit its
  training capacity. Spark UDF distributes inference only. This flow applies to
  new projects; do not silently convert existing fitted Polars artifacts.
- Validate supported preprocessing as row independent, row preserving and safe
  across worker batches. Lag/rolling, inference-time whole-frame aggregates and
  row-changing steps need explicit support or a clear rejection before execution.
  Wrapping a bundle in pyfunc alone does not establish partition safety.
- Verify signature columns, worker dependencies and `env_manager`, per-worker
  model loading and Arrow batch size. Test local/UDF prediction parity and parity
  with `predict_spark` where supported, then validate the generated route on
  Databricks. Large-data throughput and memory measurements belong to SM-58.

SM-48 resource-only implementation evidence (2026-09-28, before the graph-3 follow-up): split files reconstruct the
original job definitions exactly, including names, keys and cross-job references.
63 template tests, 6 tuning-template tests and 25 real CLI generation tests pass
(serverless/policy-cluster, manual/scheduled, both engines/tasks and policies).
All four schedule combinations pass `bundle validate --strict --target dev`
with a freshly built Core wheel. Ruff, formatting and full CI Ty scope pass.
No deployment or job execution was performed; the live redeploy check for
unchanged job IDs/history remains open. Documentation build was not run.

SM-48 readable-graph follow-up (2026-09-28, uncommitted):
`initialize_run -> load_data -> prepare_dataset -> train_and_tune ->
select_best_model -> register_model -> evaluate_model -> model_decision ->
training_report`, with conditional training and optional scoring handoff.
Manual actions share model_decision and skip data/training. Graph 3 stores
bounded source/split Parquet artifacts with digests; learned transforms remain
fold-local. Graph 2 runtime support stays intact. Selection explicitly has one
candidate until SM-54. See [implementation plan/evidence](98-readable-training-graph-plan.md).
Final local evidence: 318 integration tests, all 69 CLI generation cases across
reruns, four strict bundle validations, Ruff/format, full Ty and Lizard CCN 10
passed. Authorized graph-3 live acceptance passed: classification, regression, voting
ensemble/nested CV, automatic/manual promotion, rollback, rejection, guarded
failures and child scoring. Both prediction tables have 123 unique rows;
no-op preserves commits; MLflow receipt chains and Parquet hashes passed audit
`316654296326563`. See [live report](99-readable-graph-live-acceptance.md). The old-job
migration/history check remains open. The user chose to keep two jobs and defer
the long false-edge visual issue; no third model-actions job will be added.

SM-28b added no second score writer; shared-target admission remains a gate
before any later scoring workflow writes the same prediction table.
SM-20R replaces SM-20P after the user's reset and generic-template direction.
Company profile/policy execution remains unverified. SM-22a/b/c and SM-28a
are prerequisites for the first Bundle. SM-28b only
wires their optional monthly schedule after the Bundle exists. SM-27 remains
post-Bundle and independent. Endpoint, feature lookup and monitoring remain
optional; broad Spark expansion follows the local Bundle.

## SM-52 closure record - 2026-09-30

The original checkpoint below is superseded by the direct-name follow-up in
[delivery129](129-reference-followups-delivery.md). Historical private-name
compatibility is not claimed after the requested removal of helper aliases.

Inspected baseline: `bffb5bc4`; implementation is uncommitted. Four shared helper
modules and declared domain entrypoints preserve existing workflow behavior,
public exports and saved identities. User Python-style edits were retained.
The complete 102-file integration suite passed 2,110 cases with 252 explicit
environment/opt-in skips and zero failures/errors. Missing-MLflow subprocess
checks, static gates and installed-wheel verification passed. Python 3.12.10,
MLflow 3.16.1, Ty 0.0.75 and Core wheel 0.9.1 were used. Exact commands,
negative-scenario coverage, baseline fixture repairs and skip breakdown are in
[delivery127](127-sm52-internal-api-delivery.md). No new cloud run or deployment;
SM-37 is READY. SM-36d and configurable data-quality gates remain PARKED.

## Reference follow-ups and direct-name closure - 2026-09-30

**Superseded scope:** The user subsequently deferred acceptance, monitoring and
cost tools. Their scripts, dedicated library code/tests and acceptance-only CI
settings were removed. Static smoke and SM-52 direct helper names remain.
The following results describe the earlier checkpoint, not the reduced scope.

Inspected baseline: `bffb5bc4`; changes remain uncommitted. Shared helpers now
have one directly defined name; callers and tests use it. Active MLflow sentinel
tests verify that patches are reached, and structural checks reject new helper
aliases. Static smoke, run-isolated opt-in acceptance, optional job-attributed
cost reporting and bounded drift/freshness reports are implemented locally.

All 106 integration test files ran exactly once: 2,149 passed, 256 skipped,
zero failures/errors. Separately, 94 real CLI generation tests and seven strict
Bundle validations passed. Ruff check/format, full CI Ty, Lizard CCN 10 and an
isolated installed-wheel check passed. Review findings on CI schedule overrides
and smoke training-date validation were fixed and regression-covered.
See [delivery129](129-reference-followups-delivery.md) for commands and limits.
No live acceptance deploy/train/score or billing-system query was executed;
SM-40 and SM-23a remain PARTIAL, and SM-37 remains the next READY task.

## Supplied dbml reference review - 2026-09-30

[Review126](126-sm52-reference-review.md) inspects the user's supplied local
`dbml-mlops-template` copy against the current implementation and this queue.
Production operations, CI/CD, serving load tests, feature lookup, resource
isolation and monitoring already have owners; no duplicate tasks were added.

The user subsequently deferred acceptance, cost and monitoring implementation.
Keep these requirements for later tasks; only static smoke remains delivered.
Historical implementation and the removal are recorded in
[delivery129](129-reference-followups-delivery.md):

- **SM-40:** distinguish credential-free/no-write smoke execution from opt-in
  live acceptance; sampling alone is not a no-write guarantee. Isolate concurrent
  CI runs by run identity, including runs by the same service principal.
- **SM-23a with SM-38/39:** optional project/target/run-scoped infrastructure
  cost reporting; system-table permission handling and list-price estimate
  labeling. Cost tags and budget settings alone are not a report.
- **SM-23a:** real timestamp/freshness semantics; failed or missing observations
  cannot look healthy; explicit monitor types and optional slices; retain
  Core's KS-statistic contract when translating reference thresholds.

Current delivered slice: generated static smoke. Isolated CI acceptance, cost
reporting and drift/freshness monitoring are deferred at the user's request;
their added tools and implementation have been removed. SM-40 remains PARTIAL
because smoke is retained; SM-23a returns to LATER. Configurable data-quality
rejection gates remain PARKED by the existing user decision.

## SM-27/SM-29 closure record - 2026-09-24

Implementation inspected: `68f64935`; approved test plan: `556b63f3`.
The selected Polars/serverless generated Bundle passed first-champion
initialization, improved v2 promotion, tied v3 rejection, both scoring
policies, failed-score recovery, queued no-op requests and a restricted
principal's real alias-write denial. Final rows: append 160 v1 + ten v2;
full view 170 v2; retained v1 generation unchanged. Exactly two persistent
test jobs and no admission table. The English
[live report](35-sm27-sm29-live-validation-report.md) records every run URL,
expected negative runs, exact commands, wheel hash, runtime configuration
and scope limits. Strict Bundle validation, wheel build, local pandas/Polars
precheck, live-evidence assertions and strict docs build passed. The prior
94-test implementation gate is preserved in the SM-29 local report.
No library fix was required by this rehearsal. Source updates/deletes,
company targets, policy compute and broad Spark execution remain outside
this evidence. Production must still enforce one serialized alias writer.

The subsequent [reference comparison and readiness review](36-bundle-reference-comparison-and-readiness.md)
separates this completed functional scope from remaining production adoption
work: manual approval UX, runtime parameters, independent score scheduling,
workflow simplification, identities/operations and generated-project CI.
The user subsequently approved that work; the new task definitions and the
corrected challenger semantics are recorded in SM-30 through SM-43 above.

## Local-first path to the first Bundle

These tasks were identified in the reference review; the latest user direction
moves the model lifecycle services before the first local Bundle. Spark and
other optional integrations follow the Bundle.
Full task scope, proposed code areas, reference sources and acceptance checks are
in [04-databricks-integration-plan.md](04-databricks-integration-plan.md).

1. **SM-26:** fitted pandas/Polars pipeline -> MLflow package -> same local predictions.
2. **SM-25:** config, pinned local artifact loading, limits and preflight.
3. **SM-24a:** local training and bounded local batch prediction in Databricks.
   Its [live validation report](08-sm24a-live-validation-report.md) records two
   source tables, five cross-job models and every preprocessing registration ID.
4. **SM-15L:** explicit-period local predictions reach guarded Delta. The live
   test used scripted period and source-version values; it is not automatic.
5. **SM-15I:** add automatic first-snapshot and later new-row scoring with a
   committed source-version receipt and append-safe output. See the
   [incremental plan](12-sm15i-incremental-scoring-plan.md).
6. **SM-22a/b/c:** compare concrete candidate/champion versions on one labeled
   holdout, stage a validated `@challenger`, then explicitly promote with
   `@previous_champion` and guarded rollback.
7. **SM-28a:** train a candidate from a pinned, label-aware snapshot and
   temporal holdout; compare it without automatic promotion.
8. **SM-20a:** generate, validate, deploy and run the first local-engine Bundle
   using the [two-run rehearsal](05-sm20-bundle-plan.md) without manual dates
   or source versions. Package the already-tested lifecycle services. **DONE:**
   [generated-project and live evidence](21-sm20a-local-bundle-validation-report.md).
9. **After SM-20a:** SM-28b wires optional monthly retraining; SM-27 adds
   explicit `full_rebuild` alongside default `incremental_append`. SM-15L's
   `period_update` remains an explicit backfill.
10. **Later:** optional SM-24b/24d/19/21/23 adapters; Spark SM-24c/SM-17, then SM-20b Bundle enhancement.

SM-26 is the explicit packaging task added after the local-first MLflow discussion.
It reuses the fitted local pipeline rather than requiring every FE node to gain a
portable Spark codec first. Each advertised local pipeline/model variant needs
save/load/prediction evidence. This does not make its artifact Spark-compatible.
Backend joblib dictionary bridging stays in parked SM-18. Local prediction
and UC Delta publication remain separate tasks; SM-15I is now on the first
Bundle's critical path, with an explicit small-data memory limit. Spark can
perform UC table I/O here without becoming the FE or model execution engine.

Pandas/Polars training -> Spark inference is available only for the currently
compatible FE/model bundle contract. Both Spark modes remain capability-gated.
Delaying SM-17 does not enable arbitrary local pipelines or all registered models.
SM-25 extracts the supported-workflow usability subset from later SM-17j; wider
native model/engine coverage waits until the first local Bundle passes.

## Later SM-17 scope and completion contract

The user requested coverage of the existing nodes and models, not closure by
listing the rest as unsupported. Missing combinations stay open unless the user
explicitly defers them. This expansion starts after SM-20a; none of its missing
features is marked DONE. No implicit collection or algorithm swap.

| Task | Status | Deliverable |
| --- | --- | --- |
| SM-17-00 | LATER | Expand the [100-ID inventory](NODE_SUPPORT.md) into per-option fit/apply/state/model/runtime coverage and a registry coverage guard |
| SM-17a | LATER | Column/cast/cleaning/date/math/interaction/polynomial and basic scaler native paths |
| SM-17b | LATER | Remaining imputation, categorical encoders, binning/ranges and robust/quantile state |
| SM-17c | LATER | Feature selection, transforms and outliers; fit/apply distinction and row alignment |
| SM-17d | LATER | Group/window/history and target/WOE OOF semantics; deterministic splits and leakage gates |
| SM-17e | LATER | Text/vectorization/embeddings and geo; sparse/vector contracts and worker dependencies |
| SM-17f | LATER | Split/resampling/inspection; training-only row changes, stable membership and bounded previews |
| SM-17g | LATER | Existing model-family bundle/inference coverage: sklearn variants, XGBoost/LightGBM, ensembles/calibration and clustering |
| SM-17h | LATER | Explicit native Spark model training/transform, artifact kind and MLflow persistence; all existing model IDs retain decisions/tasks |
| SM-17i | LATER | Evaluation/CV/tuning/thresholds and explainability/SHAP contracts; parallel trials versus distributed fit; bounded execution |
| SM-17j | LATER | Easier config/SDK entry point, preflight and independent platform/FE/model/sink/tracking choices; no Canvas or DAB work |
| SM-17k | LATER | Family regression gates and selected Databricks runtime validation; no missing requested coverage silently marked DONE |

Family tasks depend on SM-17-00 and their relevant preceding contracts. Model
work need not wait for every independent FE family, but no task should claim
support before its artifact, inference and runtime requirements pass.

Details, reference-repo comparison and parked endpoint tasks:
[Spark/Databricks gap review](reports/2026-09-22-spark-databricks-gap-review.md).

## Detay planlar

- Completed SM-16 evidence and scope: [PLATFORM_VALIDATION.md](PLATFORM_VALIDATION.md).
- SM-15 and SM-15L provide Spark and explicit-period local UC writers.
  SM-15I is required for automatic new-row output in SM-20a. The first
  template precedes Spark expansion.
- Latest user direction supersedes the prior Spark-first and template-last order:
  deliver a local pandas/Polars Bundle, then expand Spark and add a tested Spark
  option. SM-18 and continuous streaming remain parked.

- SM-00–SM-07: [Core/Spark](01-core-spark-plan.md)
- SM-08–SM-11: [Inference](02-inference-plan.md)
- SM-12–SM-20: [MLflow/batch/son teslimatlar](03-mlflow-batch-delivery-plan.md)
- Current integration lane: [Local-first integration](04-databricks-integration-plan.md)
  and [SM-20 Bundle rehearsal](05-sm20-bundle-plan.md).
- Ortak kurallar: [Mimari](ARCHITECTURE.md), [Doğrulama](VALIDATION.md)

SM-17 requires the full tracked scope above. Some algorithms may need an explicit
alternative backend rather than an equivalent native Spark implementation; that
decision must not hide missing support. SM-19 streaming remains optional and parked.

## SM-20a closure record - 2026-09-24

Inspected branch `090` at the SM-28a baseline and added the custom template,
generated-workflow tests, Databricks Bundle guide and isolated rehearsal example.
`databricks bundle init skyulf-core/templates/databricks --config-file ... --output-dir ...`,
`databricks bundle validate --strict -t dev --profile skyulf`,
`databricks bundle deploy -t dev --profile skyulf` and Bundle `run` commands for
train/compare/stage/promote/score all succeeded in the test workspace. The
score job passed 2 initial + 2 new rows, then returned a no-op with target
version unchanged. `python -m pytest tests/integration/test_sm20a_bundle_template.py`
passed four tests; Ruff lint/format and `mkdocs build --strict` passed. The
first local wheel upload and a test-only alias verifier failed, were diagnosed,
corrected and rerun; [the report](21-sm20a-local-bundle-validation-report.md)
retains those failure details and live run links. This Bundle uses Polars for
the live test, and remote forced-write failure was not repeated. SM-28b is next;
SM-27, broad Spark work and endpoints remain later.

## Bir görevi kapatma kaydı

Her DONE satırına aynı dosyada aşağıdaki bilgileri içeren tarihli kayıt ekle:

```text
Task ID / tarih / incelenen commit / değişiklik commit'i veya uncommitted
Değişen dosyalar ve sağlanan davranış
Çalıştırılan exact komutlar / passed-failed-skipped / runtime sürümleri
Olumsuz senaryo kanıtı (unsupported, leakage, retry vb.)
Bilinen sınırlamalar / sonraki görev
```

Komutlar çalıştırılmadan kutular işaretlenmez. Spark lane'inde tamamı skip olmuş
suite DONE kanıtı değildir. Platform erişimi gerektiğinde BLOCKED nedeni somut
olarak yazılır; yerel test sonucu gerçek UC/Databricks sonucu yerine geçmez.

## Devam oturumu için kısa talimat

Önce README, mimari ve bu kuyruğu oku. Test öncesi ortamı CI ile eşitle: `uv pip install -r requirements-ci.txt` (editable `skyulf-core`, ty 0.0.75); aksi halde subprocess testleri ve ty yanlış hata verir ([report94](94-integrations-code-quality-review.md)). Güncel git durumunu ve ilgili kaynakları
doğrula; kullanıcı değişikliklerini koru. İlk READY görevi ACTIVE yap, kendi
test döngüsüyle tamamla ve kanıtı yaz. Sonraki bağımlılığı aç. İlk local Bundle
kapısından önce Spark veya endpoint işine başlama. SM-00 yalnız test/runtime
hazırlığıydı; güncel destek ve sınırlar tamamlanan görevlerin kayıtlarında belirtilir.
