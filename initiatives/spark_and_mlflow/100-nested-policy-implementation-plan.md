# Nested temporal, group and threshold CV implementation plan

Specs: 90-sm36g, 91-sm36h and 92-sm36i in this directory.
User requested finishing all three; keep the existing two Databricks jobs.
Baseline graph commit: 42bc0d37. Branch 091 is the existing task workspace.

## Shared contract

- Preserve `cv_type="nested_cv"`; add `cv_nested_type="auto"` with choices
  auto, k_fold, stratified_k_fold, time_series_split, group_k_fold,
  stratified_group_k_fold. Auto preserves classification/regression defaults.
- Ordinary group CV uses cv_type group_k_fold / stratified_group_k_fold.
- `cv_group_column`: optional split-only group column, excluded from features.
- `cv_time_column`: existing split-only time column.
- `cv_gap=0`, `cv_test_size=None`, `cv_max_train_size=None`: row-count time
  settings; unbounded training expands, bounded training rolls. Both nested
  levels and the separate final search use the same explicit policy.
- Stable time ordering; reject missing times and tied timestamps crossing a
  fold boundary. Groups must be nonnull and isolated at both nested levels.
- `tune_threshold=True` on nested binary classification selects thresholds
  from inner out-of-fold predictions of the selected recipe, never outer or
  final holdout labels. Probability/ranking scores retain probabilities.
- Save split policy/boundary/count evidence, per-fold thresholds and an
  independent final threshold. Preserve existing raw-label mappings.
- Basic fixed models must evaluate the requested policy honestly; no ignored
  settings or silent fallback. Final holdout must respect groups/time.

## Work and ownership

- [x] Core policy and nested orchestration: schemas, metadata extraction,
  precomputed CV partitions shared by every strategy, inner metadata slicing,
  final refit, fixed-model evaluation and independent reference tests.
- [x] Nested threshold helper and pipeline persistence: probability/OOF
  selection, threshold-aware outer scoring, final artifact parity tests.
- [x] Backend/Canvas: full payload mapping, controls and results, aligned
  split metadata before fitting, focused backend and frontend tests/build.
- [x] Databricks: shared LocalCVSpec/search plumbing, metadata retention and
  isolated final holdout, Bundle schema/config/preview/docs and tests.
- [x] Cross-layer integration review, Ruff/full Ty/CCN10, relevant Python
  suites, frontend lint/complexity/tests/build/size; bounded live Databricks
  pandas/Polars regression/classification/ensemble and persisted evidence.

## Verification constraints

Failing tests first for new contracts; meaningful reference and membership
assertions. Use five bounded strategy cases, not a claim of full Cartesian
coverage. Keep all failure semantics, optional dependency gates and existing
ordinary CV defaults. No threshold/gate suppression; production CCN <=10.
No push or third job. Record partial work accurately until all acceptance is
verified. Temporary fixtures and cloud payloads stay under rehearsals.

## Progress

- Baseline graph commit and hooks complete; nested implementation starting.
- Design ruling: use row-count gap/window sizes with strict cross-boundary
  timestamp checks, and explicit group split metadata. This matches current
  sklearn conventions while avoiding an ambiguous time-duration gap.

- Implementation now includes temporal/group/threshold policies across all layers.
- Review reproduced and fixed missing SDK test-holdout validation, old sklearn
  classifier recognition, threshold-aware evaluation metrics and stored OOF provenance.
- Exact inner/outer/final preprocessing membership verified for pandas and Polars.
- Three generated bundles passed real CLI generation and strict validation.
- Wheel SHA256 715a1f559753f41bf4a4aca6d9e3f6fb3d5020f4ccc69f36a0bbbe3bd5e8e202;
  all 259 packaged Python source files matched the working tree.
- Local 16-case acceptance including persisted MLflow artifact replay passed.
- First cloud run 417057667951020: first seven pandas cases passed; Python SIGSEGV
  during temporal stacking classifier. Diagnostic run 595790100383979 pending.
  Completion remains pending cloud diagnosis and combined regressions.

- Final acceptance complete: cloud run329510165594261 passed both engines (16/16),
  independent16-receipt/384-prediction audit passed. Final291 Python and134 frontend
  checks passed, CI analysis scopes and three strict Bundle validations passed.
  First native crash retained as unresolved transient evidence; repeat did not
  skip cases or change production code. See report101. Nested changes included in the delivery commit.
