# End-User Observability & Debuggability Audit

**Date:** 2026-08-11  
**Scope:** What a data scientist or analyst can discover inside Skyulf after
an unexpected execution or result. This deliberately excludes operator-facing
Sentry, server logging, metrics, and tracing covered in
[backend-blockers.md](2026-08-11-backend-blockers.md).

## Executive judgment

Skyulf has a substantially better foundation than a black-box “job failed”
product: job details retain an error string and live logs, preview execution
returns per-node results/warnings, the canvas can mark failed preview nodes,
and Experiments provides unusually strong model-evaluation visualizations.

The important boundary is **preview versus submitted training job**. The
rich per-node result/warning payload exists only for the synchronous preview
path. A real training/tuning job persists a job-level error, logs, final
metrics, and graph snapshot, but not a durable per-node execution trace or
data-quality profile. Thus the moment users most need diagnosis—after a
long-running production-like run—still often makes them infer the failing
step from an ID embedded in a raw exception.

“Frustration” below estimates the impact on an end user trying to answer
*what happened, where, and what should I do next?*

## Findings

### 1. Submitted-job failures show raw error text and logs, not a structured diagnosis

**Frustration: High**

**What the user gets**

- The Jobs detail view renders `job.error` verbatim under **Error Log** in a
  `<pre>` (`frontend/ml-canvas/src/components/panels/jobs/JobDetailsView.tsx:653-663`),
  and exposes a **Live Logs** tab (`:531-555`) with copy-all support. It also
  offers retry for failed/cancelled training and tuning runs (`:217-241`,
  `:505-523`).
- The backend does preserve a useful raw failure string for a normal pipeline
  failure: it searches node results and writes
  `Error in node {node_id}: {node_res.error}` (`backend/ml_pipeline/_services/pipeline_execution_service.py:137-149`).
  An unhandled exception instead persists only `str(exc)` (`:152-165`), while
  its traceback stays server-side (`:154-155`).
- The HTTP job-detail contract carries only `error` and optional `logs`
  (`frontend/ml-canvas/src/core/api/jobs.ts:6-18`). It has no typed error
  code/category, failed-node label, failing input/column, remediation, or
  traceback/details discriminator.

This is better than the earlier canvas toast, “Pipeline execution failed —
Check console for details,” already recorded in
[smooth-experience-fixes.md](2026-08-11-smooth-experience-fixes.md:45-49).
It is nevertheless a new, deeper product gap: the durable UI’s primary
diagnosis is an implementation exception. Some errors will be informative;
others expose Python/library terminology or only say a conversion failed.
The user has no consistent “cause / affected node or column / suggested
repair” experience.

**Recommendation:** persist a typed `failure_diagnosis` alongside the raw
error: failed node ID **and label**, step type, column/parameter when known,
classification (configuration/data/schema/resource/internal), plain-language
explanation, and concrete next action. Keep raw error and traceback only in
a collapsible Technical details section; make retry retain the diagnosis.

### 2. Node attribution is good for preview but incomplete for a real job

**Frustration: High**

The backend engine stops at the first failed node and stores a per-node result
(`backend/ml_pipeline/_execution/engine/__init__.py:113-150`). During preview,
the frontend receives `node_results` keyed by node ID
(`frontend/ml-canvas/src/core/api/client.ts:126-147`), gives the failed card a
red border/status chip (`components/canvas/CustomNodeWrapper.tsx:235-245,
269-285`), and the Results panel displays the error (`components/layout/
ResultsPanel.tsx:244-257`). That is a good feedback loop.

For submitted jobs, however, `JobStrategy.handle_success()` persists only
final metrics, timing roll-ups, and aggregate dropped columns
(`backend/ml_pipeline/_execution/strategies.py:109-141`); the failure path
persists only status and a string (`:143-147`). The Job Detail’s **Inspect
node** link is wired to `job.node_id` (`JobDetailsView.tsx:625-641`), which is
the training/tuning terminal node recorded when the job was created—not
necessarily the preprocessing node named inside the error string. The user
can see `Error in node <UUID>`, but must map that opaque ID to a canvas card
or search logs manually.

**Recommendation:** persist the complete ordered per-node run ledger for
submitted jobs (status, started/ended/duration, warnings, metrics, error).
Render it as a clickable pipeline stepper in Job Details; select and focus
the failed graph node by stable ID, and link directly to its node inspector.

### 3. Data-quality feedback exists for selected preview nodes but has no universal post-run gate

**Frustration: High**

There are meaningful positive examples:

- Drop Rows shows exact removed, remaining, and total counts
  (`frontend/ml-canvas/src/modules/nodes/processing/DropRowsNode.tsx:32-53`).
- Feature Selection shows dropped columns, scores/p-values/importances, and
  its own execution error (`FeatureSelectionNode.tsx:143-205`).
- Engine warnings are captured with node IDs (`backend/ml_pipeline/_execution/
  engine/__init__.py:202-207`) and preview warnings are placed in the
  notification bell and persisted to pipeline logs
  (`frontend/ml-canvas/src/core/hooks/useExecutionWarnings.ts:32-73`).
- Merge warnings are explicitly surfaced in Results (`ResultsPanel.tsx:
  135-170`).

But this is an uneven, node-specific UI convention—not a data-quality
contract. There is no cross-node output profile or quality gate that says,
for example, “this transform is 100% null,” “row count fell 99%,” “a category
was unseen,” “target has one class,” or “this output has no usable numeric
features,” before a later model node fails. More importantly,
`useExecutionWarnings` only acts on the preview `executionResult`; a Celery
training job does not return that result to the browser. The submitted-job
strategy aggregates dropped columns but does not persist per-node warnings or
output health (`backend/ml_pipeline/_execution/strategies.py:53-64,
109-141`).

This directly supports the “black-box automation” concern in the
differentiation strategy, but is distinct from its broader proposal for
leakage guardrails: it is the missing **after-each-step, user-visible**
quality feedback loop.

**Recommendation:** standardize a `NodeDataQualityReport` for every run:
input/output rows, columns, missingness, duplicates, invalid/infinite values,
schema changes, row/column loss, and warnings. Persist it per node for jobs,
show deltas on the canvas/stepper, and block or prominently warn on
configurable catastrophic thresholds (all-null output, zero rows, no
features, severe row loss).

### 4. Model-performance transparency is a strength, but it lacks an automated diagnosis

**Frustration: Medium**

This is **not** a “bare metric only” product:

- Job Details renders all returned metrics and a top-five feature-importance
  view (`JobDetailsView.tsx:681-699`).
- Experiments compares feature importances with explicit “not reported”
  treatment and raw-value table (`components/pages/ExperimentsPage/components/
  FeatureImportanceView.tsx:120-193`).
- It includes SHAP summary, beeswarm, dependence, waterfall, force, and
  interaction views (`ShapExplainabilityView.tsx:27-35,99-178`).
- Classification has confusion matrices, ROC/PR/calibration-related views and
  threshold exploration (`ClassificationChartsForSplit.tsx:1-8,58-195`);
  regression includes actual-vs-predicted, residual, histogram, Q-Q,
  scale-location, and percentile diagnostics
  (`RegressionChartsForSplit.tsx:1-8,103-193`).

The remaining gap is interpretation. None of these views assesses whether a
metric is poor for the task/baseline, detects train/test generalization gaps,
flags likely leakage/imbalance/too-small validation sets from the completed
run, or turns a diagnostic pattern into next steps. Users get powerful
charts, but must already know how to read them.

**Recommendation:** add a run-quality summary above evaluation: scoring
metric and baseline comparison, train-vs-validation/test gap, class support,
error concentration, and ranked hypotheses (“likely overfit,” “minority class
recall is low,” “residual variance rises with prediction”), each linked to the
relevant chart and corrective node/template.

### 5. Active jobs are status-visible but effectively have no execution progress or ETA

**Frustration: High**

The detail screen explicitly shows **“Not reported”** for any non-terminal
job (`JobDetailsView.tsx:593-600`), and duration is unavailable until both
timestamps exist (`:585-591`). It does not show the current pipeline node,
step count, progress bar, historical duration estimate, or ETA.

The backend model actually has `progress` and `current_step`
(`backend/database/models.py:266-267,291-292`), and the realtime event
schema carries both (`backend/realtime/events.py:23-34`). But the public
`JobInfo` client type omits them (`frontend/ml-canvas/src/core/api/jobs.ts:
6-38`) and the pipeline service only initializes progress to zero, publishes
it, and sets it to 100 at success (`pipeline_execution_service.py:131-149,
185-189`). No execution code updates `current_step`; log callbacks merely
republish existing values (`:89-119`). Thus the apparently available
instrumentation is neither complete nor rendered.

This expands, rather than repeats, the prior smooth-experience finding that
training jumps from “queued” to Jobs: the missing contract is **per-step
progress and a credible time estimate**, not just a better spinner.

**Recommendation:** instrument node start/finish and known subprogress (CV
folds/tuning trials), persist and return `current_step`, `completed_steps`,
`total_steps`, and progress; render a stepper plus elapsed time and
historical/uncertainty-marked ETA. Never invent an ETA when no estimate is
available.

### 6. Run comparison is strong for configuration and metrics, absent for data-profile changes

**Frustration: Medium**

Users can select two runs and get a side-by-side graph/config diff:
`PipelineDiffView` states this intent and fetches saved job graphs
(`frontend/ml-canvas/src/components/pages/experiments/PipelineDiffView.tsx:
1-12,100-138`), lists modified node fields (`:337-364`), and Experiments
also supplies metric/branch comparisons (`components/pages/ExperimentsPage.tsx:
653-733`). Pipeline Audit Log shows chronological saves and
node-level additions/removals/modifications
(`frontend/ml-canvas/src/pages/AuditLogPage.tsx:340-347,180-213`).

However, neither compares the input data profile nor output quality between
two runs: row counts, schema, missingness, category distributions, target
balance, transformations’ data loss, and train/test split changes are not
first-class run comparison data. The Pipeline Audit Log is a save history
keyed by dataset (`core/api/pipelineVersions.ts:169-239`), not a reproducible
run ledger. When identical configs behave differently because data changed,
the user still has to infer why.

**Recommendation:** snapshot compact data/profile fingerprints at source,
each key node, and train/test boundaries; add a “Why different?” comparison
tab aligning configuration diff, profile diff, quality warnings, execution
timings, and metric deltas for two selected runs.

### 7. Self-service history exists, but it is fragmented and not reliably scoped to the user’s submitted job

**Frustration: Medium**

There are real user-facing facilities:

- Jobs offers status/history/detail/logs (`frontend/ml-canvas/src/pages/
  Jobs.tsx:321-353`; `JobDetailsView.tsx:531-555`).
- The navigation exposes **Errors** and **Audit Log**
  (`frontend/ml-canvas/src/components/Layout.tsx:175-181`); Errors has filters, job links, diagnostic IDs, and can
  show a full traceback (`pages/ErrorLogPage.tsx:121-160,163-180`).
- The monitoring API supports searching pipeline logs by level, node type/ID,
  pipeline ID, and text (`frontend/ml-canvas/src/core/api/monitoring.ts:
  167-195,427-451`).

This is a positive answer to “is there any self-service log/history?” Yet it
is fragmented: job-local logs live in Job Details; preview warnings/failures
are copied into a global Pipeline Logs/Error page
(`frontend/ml-canvas/src/components/layout/MainLayout.tsx:50`); persisted job failures are
not automatically represented as that per-node ledger. The Error page’s
“resolve” and “delete all” controls (`core/api/monitoring.ts:400-412`) also
read like an operational incident console rather than a personal run journal.
Finally, current backend access is globally unscoped—an enterprise defect
already covered in backend blockers—so it cannot become trustworthy
per-user history until identity/workspace scoping is added.

**Recommendation:** make Job Details the canonical run timeline, including
all events, warnings, node logs, artifacts, retries, and links to the exact
executed graph. Keep a separate admin incident view. Once tenancy exists,
scope every timeline/query to the current user/workspace and retain immutable
per-run records.

## Prioritized top five fixes

1. **Persist and render a submitted-job per-node execution ledger.** Clickable
   graph/stepper, labels not UUIDs, durations, warnings, metrics, and failure
   details solve the largest attribution and black-box gap.
2. **Replace raw job-error-first UX with structured actionable diagnosis.**
   Cause, node/column/config, likely remedy, retry, and collapsible technical
   details.
3. **Add universal per-node data-quality reports and guardrails.** Surface
   catastrophic output changes immediately and retain them for later jobs and
   comparisons.
4. **Implement real progress/current-step/ETA semantics end-to-end.** The
   database/event shape exists but is not updated or consumed.
5. **Add a “Why did these runs differ?” view.** Combine existing graph/metric
   comparison with persisted data-profile, quality-warning, timing, and
   artifact differences.

## What to preserve

Do not replace the existing strengths with a generic “job failed” banner:
raw job logs, retry, preview-node failure cards, notification history, merge
advisories, configuration diff, feature importance/SHAP, and detailed
classification/regression diagnostics are valuable technical escape hatches.
The missing layer is a durable, plain-language explanation that connects
those artifacts to the exact run and next action.
