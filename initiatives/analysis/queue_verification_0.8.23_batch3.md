# 0.8.23 third queue batch — verification

The preceding twelve-finding batch and the empty-undo follow-up were committed
as `8a745e1c` with DCO sign-off and passing hooks. This batch repairs nine
findings: **OC-50/111/223/225/226/253/269/302/318**. The live queue moves from
**34 open / 4 parked** to **25 open / 4 parked**. Two closures belong to
OC-271–318, bringing that Qwen group's total to **41/48 closed**.
OC-71/72/73/185 remain parked.

## Changes and examples

| Findings | Previous behavior | Verified behavior |
|---|---|---|
| OC-223 | A completed disposition could clear the note typed for a different alert. Late requests could also overwrite newer details or pending/error state. | Requests belong to their alert lifetime. New drafts survive old actions, and a successful disposition invalidates older detail reads. Same-alert success/error races are covered. |
| OC-225 | Opening a job removed the heading named by the drawer; the icon-only Back action had no accessible name. | History and detail views both have valid dialog names. Back/Close actions are named, and real keyboard navigation keeps focus within the drawer. |
| OC-226 | Delayed hyperparameter definitions could restore an earlier model/reference or write through an obsolete callback after unmount. | Definition requests follow the active model and node lifetime, merge into current configuration, and preserve user-entered parameters. |
| OC-253 | For labels `{1,2}`, identical predictions scored `10/11` in tuning and `0.8` in evaluation and threshold tuning. | Named binary F1/precision/recall/PR-AUC use the sorted-last class, matching probability column 1. All five search strategies score the example F1 as `0.8`; an explicitly supplied class-1 scorer still gives `10/11`. |
| OC-269 | Selecting a pruner created its object but left `OptunaSearchCV.enable_pruning=False`. | Supported SGD trials record finite per-epoch scores and can actually become PRUNED. Unsupported wrappers/options and models without an epoch budget retain ordinary fitting with an explanatory log. |
| OC-50 | Boolean targets lacked balance advice; binary integer targets could switch between classification and regression solely because the dataset grew. | Supplied Boolean and supported binary integer targets retain classification and balance advice at 20/40/100 rows. Explicit Regression remains authoritative. Numeric class labels do not receive numeric-transform or feature-encoding advice. |
| OC-111 | Integer encoding advice required categorical statistics that numeric profiles never had. | Repeated integer features with more than 50 values receive conditional advice only if the numbers represent category codes. Floats, unique IDs, low-cardinality features, targets and unsupported types do not receive this suggestion. |
| OC-302 | `2500 outliers / 5%` from a 50000-row sample lacked its denominator and looked like a full-dataset result. | Optional `analyzed_rows` and `total_rows` survive JSON and HTTP responses. UI and console identify sampled results; old reports explicitly retain an unknown denominator. A filtered 200000-row input reports its analyzed 50000 rows. |
| OC-318 | Clearing a range input produced NaN, passed validation and became a null bound in JSON. | Empty drafts remain visibly empty and invalid through save/reload; Issues opens the relevant control. Core raises a named ValueError for malformed/nonfinite bounds. Valid JSON and programmatic numeric ranges retain their scaled values. |

## Verification

| Check | Result |
|---|---|
| Final full Core suite, sklearn 1.9.1 | **9185 passed, 80 skipped**, 3 snapshots; 300.02 s |
| Final Core statement/branch coverage | **97.44%**, above the 90% CI floor |
| Final full frontend suite including pruning settings | **2597 passed** across 198 files; 71.75 s |
| Final full backend suite including pruning metadata | **3785 passed**, 7 snapshots; 190.22 s |
| Backend EDA, saved reports, threshold service/routes and calibrated-tuning controls | **110 passed**, 17 warnings |
| Modeling compatibility | **376 passed** on sklearn 1.9; final **37 regression cases passed** on 1.8 |
| Pruning capability/runtime compatibility | **34 passed** on each of sklearn 1.8 and 1.9, including actual MLP fallback trials |
| Backend fold replay/model alias compatibility | **1216 passed** and 7 snapshots on each of sklearn 1.8 and 1.9 |
| Scaling compatibility | **149 passed** on sklearn 1.9, including public fit/apply on both dataframe engines |
| EDA final guarded implementation | **83 passed**, including target inference, recommendation safety, population metadata and console output |
| Canvas settings/history/scaling browser checks | **9 passed**, including a real checkbox edit with one undo/redo step and zero panel-visibility edits |
| Jobs, segmentation and drift browser checks | **9 scenarios passed**; one initial page-navigation timeout passed on an unchanged isolated retry |
| Outlier/EDA desktop and mobile browser checks | **6 passed** |
| Pruning controls browser checks | **9 passed**; 23.4 s, covering task variants, Apply/run payloads, history and late replies |
| Real Canvas requests replayed through production FastAPI | **5 passed**, all HTTP 200 with expected support; unsupported distribution-object negative control returns 422 |
| Python gates | Ruff, normal Ty scope and full production Core/backend CCN ≤10 passed |
| Frontend gates | ESLint, CCN ≤10, TypeScript/Vite build and size check passed |

The final run covers **125/128 changed executable Core lines**. Two uncovered
lines conservatively inspect non-list search-space distributions; the declared
public tuning configuration uses lists. The third is the capability fallback for
calculators with no tunable model class. All changed scaling and profiling
executable lines are covered. No artificial unsupported configuration was added
solely to increase coverage.

The complete frontend output is rebuilt in `static/ml_canvas`. The main chunk
is **326.3 KiB gzip**, up from 325.6 KiB before the requested pruning controls
and separate model-support information tooltip.
Its explicit budget moves from **326 to 327 KiB** for the capability request and
UI state; other chunk limits are unchanged. Package metadata remains 0.8.22,
and these release notes remain under v0.8.23 as previously requested.

## Reproduction and review

Before repair, focused tests recorded 15 Canvas failures with 80 passing
controls, 19 modeling failures with 15 controls, 20 EDA failures with 11
controls, four EDA UI failures, 54 Core scaling failures with six controls,
and 12 scaling UI failures with 21 controls. The real Jobs keyboard test
failed at both tested desktop widths before its accessible-name repair.
The new pruning controls first recorded five failing component cases and a
failing Random Forest browser case against the prior UI. A follow-up reopen
test caught a stale supported result being shown during a fresh request; it
passes after clearing request state on context change and close.

Review added concrete regressions for same-alert GET/disposition races,
numeric class-label transformation advice, unsupported Int128 recommendations,
and programmatic scaler ranges. The scaler review reproduced nine compatibility
failures, then restored iterable normalization and finite Decimal support.
The first complete Core run also caught integer bounds being converted to
floats in fitted artifacts. The final validator preserves original numeric
types; all three existing snapshots pass without regenerating their baselines.
GaussianNB keeps one ordinary fit per fold; enabling repeated full-fold
`partial_fit` epochs would change its accumulated statistics and cost without
providing meaningful training epochs.

## Durable tests and practical limits

- [Scaling range contracts](../../skyulf-core/tests/integration/test_scaling_range_validation.py)
  and [browser editing](../../frontend/ml-canvas/e2e/scaling-range-validation.spec.ts)
  preserve `0,2,4 → -1,0,1` for MinMax `[-1,1]` and `-0.5,0,0.5` for Robust
  quantiles `[0,100]`. Invalid drafts remain invalid after JSON serialization.
- [Positive-class contracts](../../skyulf-core/tests/integration/test_tuning_positive_class.py)
  cover Grid, Random, both Halving strategies and Optuna. Named-score rankings
  for `{1,2}` may intentionally change on future tuning runs; existing fitted
  models and explicit scorer choices are preserved.
- [Actual Optuna pruning](../../skyulf-core/tests/integration/test_optuna_incremental_pruning.py)
  covers public tuning, intermediate values, real pruning decisions, ordinary-fit
  fallback, and training-fold-only scaling. Pruning needs an outer `partial_fit`
  and a fixed positive integer `max_iter`. Pipelines/fold wrappers are never
  bypassed. Incompatible `early_stopping`, balanced class weights or searched
  epoch budgets fall back to fit. `pruner='none'` and explicit `pruning=False`
  preserve ordinary fitting.
- [Pruning metadata](../../tests/integration/test_pruning_support_api.py) shares
  Core estimator preparation and backend fold-step selection. Canvas asks about
  the selected model, search candidates and actual converted graph, rather than
  maintaining another model allowlist. Unsupported configurations show disabled
  None with a reason; only Apply changes the saved pruner, preserving sampler and
  timeout. Loading/errors retain the saved choice. Closing/reopening and changing
  a model or connection invalidate earlier requests without creating undo edits.
  Merged input paths are conservatively unconfirmed; no fitting, artifact reads
  or database access is used to infer support. Data validity remains the training
  run's responsibility. Browser tests use mocked responses; the additional wire
  probe sends captured requests to the production FastAPI router.
  These cover Random Forest (disabled), custom SGD with only alpha candidates
  (enabled), searched max_iter (disabled), imported early_stopping=True (disabled)
  and Split/Scaling/SGD (disabled). Default SGD search spaces include max_iter,
  so the enabled example is an explicit custom search, not the default panel.
  Candidate values on the actual wire are arrays. The first browser fixture
  incorrectly mocked object distributions; that fixture was corrected after
  checking the real editor and default provider, without adding new input syntax.
- [Target recommendations](../../skyulf-core/tests/integration/test_profiling_target_recommendation_inference.py)
  preserve explicit Regression and ordinary feature typing. Fractional numeric
  measurements and unsupported native types do not acquire a new categorical
  interpretation. Missing-target imputation policy is outside this repair.
- [Outlier population](../../skyulf-core/tests/integration/test_profiling_outlier_population.py)
  runs actual analysis on 200000 rows. [Backend serialization](../../tests/integration/test_eda_outlier_population.py)
  runs the real analyzer and both saved-report HTTP endpoints with a mocked DB
  session; it does not claim a live PostgreSQL round trip. Browser API responses
  are mocked. Regenerate saved reports to obtain previously absent denominators;
  sample counts are never extrapolated into claimed full-data outlier counts.
  The deterministic 200000-row fixture measured **2497 / 50000 = 4.994%**;
  score ties mean a nominal 5% contamination need not produce exactly 2500 rows.
- Canvas async tests resolve real component/hook requests in reversed order,
  after switching alerts/models/nodes and after unmount. Browser tests exercise
  actual controls and keyboard focus. The additional history test confirms
  meaningful setting changes remain undoable after the preceding ghost-undo fix.

Root Ruff excludes only the user's unrelated untracked `tmp_polars_e2e`
investigations for this verification; no application/test lint exemptions were
added. Existing expected-error tests may print jsdom network/error-boundary
messages while passing. The user's Qwen report and temporary investigation
folders were left outside the preceding commit.

## Model-support tooltip follow-up

The nine-finding batch and dynamic pruning controls were committed as
`e221f3de` with DCO sign-off and all hooks passing. The requested separate
information tooltip names the eligible supervised Canvas model found by a live
registry/capability check: SGD Classifier (text / linear), offered under Text
Classification. Its default max_iter search disables pruning; the supported
case requires a compatible custom search and pipeline. Core MLP regression
tests do not imply an MLP option exists in Canvas, and MiniBatchKMeans belongs
to Segmentation's fixed-mode workflow.

The shared help component retains its existing default icon and accessible
name. The additional icon has the distinct name "Models with pruning support".
The follow-up passes 77 focused modeling tests, ESLint, CCN <=10,
TypeScript/Vite build and the existing 327 KiB bundle budget.
The existing modeling accessibility and pruning browser suites pass 11 cases.
A separate real-browser probe confirms focus/hover show the new information,
Escape dismisses the tooltip before its modal, and graph/history stay unchanged
with zero undo entries. The screenshot/probe artifacts remain ignored.
