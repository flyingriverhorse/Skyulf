# SM-23e — Spark job integration and dashboard acceptance

User-authorized follow-up to SM-23d, 2026-10-05. Base: 177316b7.

## Requirements and design

Generated Databricks jobs always use Spark monitoring without an engine choice.
Training activates/enrolls the selected model or pinned model-set components and
may call scoring; scoring hands the exact successful scoring receipt to a visible
native Run Job task. The separate monitoring job owns measurement, drift reporting
and guarded retraining. Preserve independent scheduled observation for late labels.
Keep the existing bounded local library API compatible; its budgets are not Spark
observation limits. Do not silently promote candidates, loosen provenance, or
fabricate performance/cost data. Existing user reference dashboards are read-only.

## Task 1 — Producer integration

- Reproduce template graph and engine-selection gaps with focused tests.
- Replace configurable template engine with literal Spark; remove the engine
  variable and legacy inline report/retraining graph from scoring.
- Validate producer receipt in a preparation notebook and publish serialized
  monitoring_request_json plus a boolean ready value. Use a condition and native
  run_job_task to call the dedicated monitoring resource, passing that exact
  request. Disabled monitoring must skip child dispatch. Recovery and model-set
  receipt binding must remain intact; reject local configs in this new path.
- Retain the legacy asynchronous/public API for existing callers, including its
  invocation idempotency contract. New native Run Job path needs no exposed
  monitoring_invocation_id. Preserve training -> scoring graph semantics.
- Document local max_rows default versus hard cap and clarify Spark ignores raw
  observation caps. Correct touched misleading task descriptions and README flow.
- Run affected template/task tests and strict generated Bundle validation.

## Task 2 — Dashboard data and presentation

- Reproduce missing model visibility from published selectors and live inventory.
- Compare reference inference and compute/cost dashboard datasets/pages read-only.
- Fix selectors, report semantics and text based on evidence; support model-set
  component identity and distinguish no labels from healthy performance.
- Expose bounded Spark confusion counts if absent; add useful native visualization
  with actual labels and explicit unavailable states for unsupported tasks.
- Add suitable execution/compute views only from verified available evidence;
  billing availability/access and lag must remain explicit.
- Execute every final dataset SQL via CLI before update + publish of the existing
  Skyulf dashboard ID. Check widget fields/filters and published readback; identify
  any browser-only verification not available rather than claiming visual proof.

## Task 3 — Three live layouts

- Deploy isolated generated single, competition and multi_target bundles using
  profile skyulf. Use small reproducible fixtures, real activation gates and current
  wheel/source. Reuse central acceptance monitoring namespace.
- Run actual training/scoring/native monitoring child jobs. Verify exact models,
  model-set components, receipt binding and dashboard results. Provide job links.
- Record final source revision/digest, commands, results and remaining limitations.

## Global verification constraints

Preserve unrelated AGENTS.md/.claude/.tmp-review-model. No push. Focused explicit
test files only; no full pytest run. Full CI Ruff, Ty and CCN gates plus applicable
pre-commit on task changes. Cloud validation required for changed Databricks flow.
Do not reuse test evidence after relevant source changes. Profile is already
authorized; engine choice is already decided. Do not alter reference dashboards.

## Progress

- [x] Task 1 implementation and review
- [x] Task 2 implementation and review
- [ ] Task 3 live acceptance
- [ ] Final static gates and integrated review

Independent read-only investigations: layout_live, dashboard_audit.

### Investigation evidence and decisions

- Dashboard source compiled, but all five chart datasets were empty by default.
  Even selected regressor v1 matched two monitor IDs (main/guard), so the SQL's
  single-monitor guard correctly refused to blend them while UI lacked a selector.
  Remedy: explicit monitoring context with visible deterministic default, inventory
  based choices and preserved model/version/monitor boundaries.
- Existing Spark classification aggregates already calculated confusion counts
  but discarded cells. Save the bounded known-class grid with actual/predicted
  labels and indices, distinguishing absent labels from measured zero counts.
- Reference dashboards are read-only. Native job telemetry and list-price billing
  are available in system tables; endpoint token metrics do not describe batch
  models. Use explicit workspace/job scope, disclose billing delay and unpriced
  usage. Do not copy reference demo table names or endpoint zeroes.
- Old local MonitorConfig remains API-compatible; generated bundles remove engine
  choice and inline legacy nodes. Native Run Job owns child dispatch/failure state.

### Verification log (ongoing)

- Root confusion regression test reproduced discarded cells (RED); source fixed,
  final combined confusion/store group: 9 passed. Explicit collection: 9 nodes.
- Inventory view test reproduced missing pinned version fallback (RED); view now
  exposes model-set identity and pre-observation pinned version. Actual owned
  current_health view refreshed successfully after ownership verification.
- Full Ruff, full CI Ty scope and full backend/Core CCN<=10 passed on runtime
  state packaged below. Subsequent test/dashboard changes require final recheck.
- Current source wheel SHA256:
  `71876aa7914371e84e7e2b889a63d4764013054a116904d3a6fff855cd53b0ca`.
  Focused real Spark metrics/drift run: 271410534042281; task 492074882188274
  returned pytest exit code 0, all 25 cases passed (output read back).
- Three actual generated Bundle jobs deployed; live train runs started:
  single 552686286815455, competition 941536479335710,
  model-set 195752207318772. Completion and child evidence pending.
- Local offloader health check unavailable: configured Python launcher absent;
  explicit venv Python then hit sandbox denial writing its external server log.
  Continued task directly; no local-model output was applied.
- Producer verification: 61 runtime, 14 monitoring generation matrix, 145 shared
  generation, 23 branch-template and 34 direct generation-helper consumer cases
  passed. Shared helper incorrectly assumed only two jobs since SM-23d; updated
  affected consumers for the dedicated monitoring job and existing named candidate
  tasks. Production behavior was not changed to satisfy those stale assertions.
- Dashboard: 21 dataset SQLs passed on the live warehouse, including native
  execution/list-price queries and updated context selection. Nine focused source
  contract tests passed; exact selected-context and final publication checks pending.
- Full CI Ruff format scope: 1334 files already formatted. No frontend changes,
  whole pytest suites or MkDocs run; CI retains full coverage/docs validation.
- Final independent review: no runtime handoff/report blockers. SM23E-IR-1 found
  an incorrect old-wheel explanation for new failed reports lacking a matrix.
  Fixed with explicit failed/no_data priority and neutral remaining absence text;
  reproduced with focused test and four live SQL cases. Scoped re-review approved
  dashboard source `c01433a25b4c42179b2e1db5c0c69223f79e842804da627abdb8656fe10be41c`.
- Every deployment-copy SQL (21) passed with the actual execution defaults before
  publication. Published scope workspace 7474646244882000/job 742248207939853.
  At validation time telemetry/billing rows were absent; this is not zero cost.
- Applicable pre-commit hooks passed on all task files: whitespace, EOF, JSON,
  Ruff, formatting, CCN and full CI Ty. YAML/schema/frontend hooks had no applicable
  source changes. Full Ruff/Ty rechecked after test-helper edits and passed.
- Existing dashboard 01f1c090238e1b6da5d633032ad9960b updated and published at
  2026-10-05T08:15:50.538Z: four pages, 21 datasets. Saved payload equality verified
  after only normalizing the server's multiline text segmentation. Published
  metadata matches the new revision/name/warehouse; its API does not return the
  serialized dashboard. Browser visual interaction remains unverified.
- Live pre-observation inventory proof passed for single model monitor
  `204a5ddca1b70c1cbfed361ac3cab604cf9514b3ef704c7f25b631155f7302d1`:
  pinned version 1 remains in current_health and selector choices while health is
  never_observed. The first reference prepared in 349 seconds; native scoring child
  120594366867189 started automatically after real initial activation.
- All four new v1 model enrollments and their selector choices verified live,
  including model-set parent sm23e_model_set_set v1 on both component rows. Nine
  generated project jobs appear in execution suggestions. All three actual score
  and prepare_monitoring tasks passed; independent monitoring children are running.
- Execution nonempty-path proof: owned older monitoring job 918715320090438 has
  two timeline runs (ERROR/SUCCEEDED), train job 1069046353781634 one SUCCEEDED.
  Billing for both is absent. Actual cost SQL separately passed read-only typed
  inline fixtures: signed DBU correction 10-2=8 at USD2 gives16; wrong workspace,
  job, currency and unit cannot enter the estimate; unpriced usage remains null.
  Synthetic arithmetic proof is not actual billing or an invoice claim.
- Single full native chain SUCCESS: train 552686286815455 -> score
  120594366867189 -> monitoring 731262534722948. Competition full native chain
  SUCCESS: train 941536479335710 -> score 739554295284075 -> monitoring
  451915098832566. Model-set monitoring 878182024119180 started after the natural
  workspace queue drained; no cancellation or promotion-gate bypass.
- Single actual saved report has 40 predictions, 20 eligible labels and coverage
  0.5, healthy feature evidence, regression confusion status not_applicable.
  All 12 single-context dashboard datasets passed with this exact observation.

### Operator walkthrough

1. Open a layout's train run. `model_decision` records the actual promotion gate;
   `register_monitor` enrolls only an activated version and prepares its reference.
2. Open `run_batch_scoring` to reach the native score child. After successful
   `score`, `prepare_monitoring` validates and serializes the exact receipt.
3. `monitor_model` is now a native Run Job task. Open its child run to see actual
   `monitor_model -> drift_report -> retrain_on_drift` execution and saved evidence.
4. Open the shared dashboard. Overview shows enrolled models/components. In Drift
   or Performance, the context summary states the model, version and monitoring
   identity selected; use the monitoring context selector when multiple producers
   monitor the same registered model. Confusion status distinguishes missing labels,
   regression, failed observations and older reports without stored class counts.
5. Execution and compute uses a concrete workspace/job selection. Its list-price
   estimate covers attributed usage only; late or missing billing stays explicit.
   Refreshing the dashboard reads saved results; it does not launch measurement.

The legacy asynchronous API's `monitoring_invocation_id` is a producer run ID:
repairs of one run deduplicate, whereas a later no-op run can revisit new labels.
The generated native Run Job path no longer exposes or requires this setting.
