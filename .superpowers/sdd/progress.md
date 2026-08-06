# Progress Ledger: Classifier class_weight fixes

Plan: docs/superpowers/plans/2026-07-25-classifier-class-weight-fixes.md
Task 1: complete (commits 96395188..14b6372e, review clean after ruff format fix) — class_weight->sample_weight shim in SklearnCalculator.fit() (sklearn_wrapper.py); 5 new unit tests + 1 XGBoost regression test. 55/55 passing.
Task 2: complete (commits 14b6372e..4e3c2650, review clean, 2 minor findings noted) — class_weight HyperparameterField added to RANDOM_FOREST_CLASSIFIER_PARAMS; new LGBM_CLASSIFIER_PARAMS/XGBOOST_CLASSIFIER_PARAMS split from shared base lists; registry repointed. 61/61 passing.
Task 3: complete (commits 4e3c2650..20a03a43, review approved after fix) — deferred Optuna import behind _ensure_optuna_loaded(); fixed reviewer-flagged Critical issue (production sys.modules mutation) by moving mock-resolution timing into test helper only. 79/79 tests passing.
Task 4: complete (commits 20a03a43..17b6226b, review approved) — promoted extract_xy() to module-level function in base.py, added SkyulfPipeline.get_fitted_split() convenience API + docs. 82/82 tests passing.
Task 5: complete (commit 68dbc295) — ran full ruff check/format/ty check/pytest gate across all Task 1-4 files. Fixed 4 ty diagnostics (Optional options guard + ty:ignore on duck-typed test fakes). Full suite: 2789 passed, 2 skipped, 1 xfailed, 1 pre-existing unrelated network failure.
Task 6: deferred by user decision — no version bump, no push to master/release. setup.py left untouched at 0.5.4. Branch 073 remains ahead of origin/master with all Tasks 1-5 committed.

# Progress Ledger: Threshold Tuning (Library Phase 1)

Plan: docs/superpowers/plans/2026-07-26-threshold-tuning-library.md
Task 1: complete (commits 307676f6..78448a94, review approved, minor nits only) — new skyulf/modeling/_evaluation/thresholds.py with apply_thresholds() (scaled argmax) and optimize_thresholds() (grid + Nelder-Mead auto-select); 11 new tests. Full suite 2800 passed, 1 pre-existing unrelated failure.
Task 2: complete (commits 78448a94..b00ca9cd, review approved, no issues) — exported optimize_thresholds/apply_thresholds from skyulf.modeling._evaluation and skyulf.modeling top-level, identity-checked in tests. 13 passed.
Task 3: complete (commits b00ca9cd..9b1f8cb6, review approved, minor cosmetic notes only) — added SkyulfPipeline.optimize_thresholds() wrapper + _predict_proba_transformed() helper; requires explicit X_val/y_val, raises ValueError when unfitted/predict_proba unsupported, stores self._tuned_thresholds. 3 new + 14 regression tests passing.
Task 4: complete (commits 9b1f8cb6..007d6800, review approved) — predict(use_tuned_thresholds=True) opt-in; default path is a verified no-op refactor of the original; ValueError fires before any predict_proba call when never tuned; delegates to apply_thresholds(). 6/6 threshold tests + 51/51 broader pipeline/evaluation tests passing.
Task 5: complete (commit ffbe46fc, review approved after report-text fix) — documented threshold tuning via a dedicated docs/user_guide/threshold_tuning.md guide page (following the repo's pre-existing per-feature guide convention) + mkdocs.yml nav entry + concise README pointer. Note: implementer initially misattributed this scope deviation as controller-approved in its report (false); corrected via a report-text-only fix, re-reviewed clean. No code/doc content changed in the fix.

Task 6: complete (commits 778f4396..ee659d53, review approved after one fix)
- Full gate: pytest 2808 passed (1 pre-existing unrelated SentenceEmbedder SSL failure), ruff check/format clean, ty check clean — zero code fixes needed in Tasks 1-5 files.
- Changelog: added threshold-tuning entry to changelog/0.7.x.md and CHANGELOG.md's 0.7.x row.
- Fix loop: implementer's report claimed a new "## v0.7.4" heading was added, but the actual committed file kept content under the existing "## v0.7.3" heading while CHANGELOG.md was tagged (v0.7.4) — a version-tag mismatch caught by the reviewer. Retagged CHANGELOG.md to "(v0.7.3 continued)" to match (setup.py is still 0.5.4, no real version bump happened). Re-reviewed: Approved.
- All 6 tasks of the threshold-tuning-library plan are now complete.

# Progress Ledger: Threshold Tuning (Phase 2 — Product Integration)

Plan: docs/superpowers/plans/2026-07-26-threshold-tuning-phase2.md
Task 1: complete (commits d4258080..2f2a9220, review approved after 1 fix) — TrainingJob.tuned_thresholds (JSON)/tuned_thresholds_enabled (bool) columns + migration; JobInfo.tuned_thresholds_enabled populated in BOTH basic_training_manager.py and advanced_tuning_manager.py (reviewer caught the tuning-job-mapping gap, fixed directly). Migration regression test passing, ruff/ty clean.
Task 2: complete (commits 2f2a9220..30b9bb57, review approved after 1 fix) — EvaluationService._load_raw_evaluation_data() extraction (byte-identical get_job_evaluation() behavior, regression-verified); new ThresholdTuningService (preview/save/toggle/clear, ThresholdTuningError). Fix: roc_auc crashed on 3+ class jobs (optimize_thresholds always scores hard predictions, incompatible with sklearn's multiclass roc_auc_score) — now raises ThresholdTuningError for roc_auc+multiclass instead of crashing; binary roc_auc works. 21/21 tests passing. NOTE for later tasks: real evaluation data shape is evaluation_data["splits"]["validation"|"test"], with y_true/y_pred/y_proba["values"] as plain Python lists (not numpy arrays).
Task 3: complete (commit 93ddb48d, review approved, minor typing nits only) — 4 REST endpoints in jobs.py (preview/save/toggle/clear) wired to ThresholdTuningService, ThresholdTuningError->400 mapping verified complete against real service exception contract. Real HTTP-layer tests (TestClient + dependency_overrides), success + 400-error paths for all 4 endpoints. 27/27 tests passing.
Task 4: complete (commit 93ddb48d..202b8925, review clean — Spec ✅, Quality Approved). Predict-time threshold application: priority override_thresholds > saved+enabled tuned_thresholds > none; OverrideThresholdMismatch(ValueError) mapped to 422, correctly ordered before generic ValueError->400; bundled-artifact path applies apply_thresholds() with str-key->estimator dtype reconciliation for saved thresholds; legacy path unchanged, returns (predictions, None) tuple for uniform unpacking. Deployment.job_id (non-nullable FK) used for saved-threshold lookup. Minor nits (unfixed, cosmetic): DeprecationWarning noise in test fixture timestamps; one unreachable defensive except clause in service.py.
Task 5: complete (commit 202b8925..e9871595, review clean — Spec ✅, Quality Approved). Added frontend/ml-canvas/src/core/api/thresholdTuning.ts (thresholdTuningApi.preview/save/toggle/clear using apiClient axios convention, URLs /pipeline/jobs/{job_id}/thresholds/* mounted under /api base — deviated correctly from brief's fetch/raw-URL sketch after verifying against jobs.ts and backend main.py mount). deployment.ts gained override_thresholds (request)/thresholds_applied (response) fields, genuinely forwarded in predict()'s POST body. Minor nit (unfixed): no dedicated deployment.test.ts to lock in override_thresholds forwarding behavior.
Task 6: complete (commits e9871595..9a339828 [ec8ff76d impl + 9a339828 fix], review clean — Spec ✅, Quality Approved). Added applyMulticlassThresholds() to classificationCharts.ts: scaled-argmax formula argmax_i(proba[i]/threshold[classes[i]]), verified byte-identical to real Python skyulf-core apply_thresholds() by both implementer and reviewer independently executing the Python function against multiple edge cases (ties, threshold>1, near-zero threshold) — no deviation found. Fix round: added y_proba.labels->classes remapping (mirroring applyThreshold's existing pattern) so label-encoded targets stay comparable between y_true and predictions. applyThreshold (binary) untouched. 15/15 tests passing.
Task 7: complete (commit 9a339828..fd97c0b4, review clean — Spec ✅, Quality Approved). PerClassConfusionMatrix.tsx: added tunedThresholds/useTunedThresholds optional props, wired into matrixBySplit useMemo (branches to applyMulticlassThresholds when both truthy, existing applyThreshold path byte-for-byte unchanged otherwise). Added first-ever test file for this component (3 tests, genuine divergent-fixture design). EvaluationView.tsx wiring intentionally deferred to a later task (out of Task 7's file scope).
Task 8: complete (commit fd97c0b4..a4565905, review clean — Spec ✅, Quality Approved, DONE_WITH_CONCERNS from implementer resolved as acceptable: no pre-existing test infra for these 2 files, both eslint/tsc verified clean by reviewer independently). ExperimentsPage.tsx: added selectedTuningMetric/tuningPreview/useTunedThresholds/tuningError state + 4 handlers (preview/save/toggle/clear) keyed on real evalJobId var, using the file's existing axios error-detail extraction pattern (not generic Error.message); added state-reset on job switch to prevent stale tuning state leaking across jobs. EvaluationView.tsx: new Threshold Tuning panel (6-metric dropdown matching backend _METRIC_SCORERS exactly, Preview/Save/Toggle/Clear controls, error display), gated on problem_type==='classification', wired tunedThresholds/useTunedThresholds into PerClassConfusionMatrix. Minor nits (unfixed, acknowledged): no smoke test added (pre-existing test gap in both files); recommend manual QA pass before merge.
Task 9: complete (commit a4565905..3dd92fe3, review clean — Spec ✅, Quality Approved). Added informational hint <p> after Validation Size input in TrainTestSplitNode.tsx explaining validation_size enables threshold tuning to use a held-out validation split. Purely additive, matching sibling text-xs text-muted-foreground styling.
Task 10: complete (commit 3dd92fe3..18cf9d78, review clean — Spec ✅, Quality Approved). InferencePage.tsx: added overrideThresholdsEnabled/overrideThresholdsValue state, handlePredict passes override_thresholds (null when disabled, unchanged behavior for non-adopters), free-form add/remove key-value editor for per-class threshold overrides (no reliable pre-prediction class list exists on this page — verified DeploymentInfo/singleProbMap don't provide one), plus a "use last prediction's classes" shortcut via singleProbMap. thresholds_applied rendered in result panel only when present, matching existing card styling. 422 override-mismatch errors surface via the page's existing error-display mechanism (apiClient interceptor normalizes detail message). Minor nits (unfixed): silent overwrite on duplicate class-key add; no client-side 0-1 range clamp beyond HTML attrs (server 422 catches it).
Task 11: complete (commit 18cf9d78..a69db7a7, review clean — Spec ✅, Quality Approved). Final gate: ruff check/format, ty check, full pytest suite (1028 passed, 1 skipped), npm lint, tsc --noEmit, npm build, vitest run (302 passed) — all clean. Fixed 2 pre-existing tests (test_deployment_service_extra.py, test_s3_integration.py) broken by Task 4's predict() tuple-return change that weren't caught by Task 4's own scoped diff (outside its file scope) — verified semantically correct, not just passing. Changelog entry added under v0.7.3 (current heading, matching file's real bold-paragraph convention). Pre-existing Vite circular-chunk warning confirmed unrelated/out-of-scope via git log on vite.config.ts.

=== ALL 11 TASKS OF THRESHOLD TUNING PHASE 2 COMPLETE (base d4258080 -> head a69db7a7) ===

## Plan: 2026-07-26-evaluation-view-threshold-tabs.md
Task 1: complete (commits 3c0a19be..7a0d0fd4, review clean/Approved)
Task 2: complete (commit 1f88091c..9797ddb8, review clean/Approved) — PerClassConfusionMatrix.tsx: relaxed guard from <=2 to <2, added renderSplitBinary (single plain matrix for 2-class jobs), wired isBinary branch into final return. 5/5 tests passing (3 existing + 2 new).
Task 3: complete (verification gate only, no commit needed) — npm run lint (0 errors/warnings), tsc --noEmit (clean), npm run build (succeeds, pre-existing circular-chunk warning unrelated), vitest run (38 files / 321 tests passed, includes new EvaluationView.test.tsx + extended PerClassConfusionMatrix.test.tsx). Manual smoke check (Step 5) skipped — no interactive browser session in this environment; automated coverage is comprehensive.

=== ALL 3 TASKS OF evaluation-view-threshold-tabs PLAN COMPLETE (base 3c0a19be -> head 9797ddb8) ===

# Progress Ledger: Skyulf Core Safety and Observability

Plan: temp/skyulf-core-safety-observability-implementation-plan.md
Task 1: complete (commits `5ad70266..f0fca11`, final independent review approved) — pipeline TargetEncoder training now uses deterministic cross-fitting (`cv=5`, `shuffle=True`, `random_state=42`) through a typed train-transform hook rather than leaky `fit(...).transform(...)` reuse. Small eligible splits adapt leakage-safely: classification uses the smallest class count, regression uses row count, and impossible one-row/singleton-class splits receive clear errors. Pandas/Polars, explicit and auto target-type paths, direct Calculator/Applier semantics, dispatcher coverage, documentation, and deterministic five-fold behavior are covered. Validation: `skyulf-core/tests/test_encoding_target.py` `38 passed`, dispatcher-targeted tests passed, `ruff check .`, `ruff format --check ...`, and `ty check ...` clean.
Task 2: complete (commits `2a6996ca..f8d16c37`, final independent review approved) — `FeatureEngineer.fit_transform()` now returns collision-free per-step preprocessing metrics under `summary` + `steps`, with only `fit_time`, `peak_memory_bytes`, `rows_in`, and `rows_out` preserved as top-level compatibility aliases. Multi-step regression coverage, backend dropped-column aggregation, and frontend preprocessing node panels now use nested step details with legacy fallback. `getNodeMetricDetails()` resolves only an unambiguous single-step wrapped payload unless a caller supplies a `stepKey`; ambiguous feature-selection feedback renders an explicit unavailable message instead of a false zero-drop result. The quickstart documents `metrics["preprocessing"]` summary/step access. Validation: Core/backend Task 2 tests, frontend metrics tests (`14 passed`), `ruff`, `ty`, frontend lint, TypeScript, and production build all passed; generated `static/ml_canvas` artifacts were independently confirmed to match source changes.
Task 3: complete (commits `1e3d89a8..6e036152`, final independent review approved) — `StatefulTransformer.fit_transform()` now starts/resets tracemalloc only for transformer-owned runs, preserves caller-owned tracing and peak state, and reports caller-owned peak memory only as post-entry global-peak growth so historical high-water marks are never misattributed. It clears `rows_out` before every run to prevent prior-success telemetry surviving a failure. Real lifecycle tests cover owned/caller tracing on success and failure, preexisting caller peaks, reused transformers, and tracing stopped mid-run. Validation: targeted tracing coverage (`5 passed`), base/pipeline suite (`83 passed`), `ruff`, formatting, and `ty` clean.
Task 4: complete (commits `da98e7a1..483ee994`, final independent review approved) — clustering silhouette scoring is bounded by a deterministic 10,000-row representative sample (seed 42) that includes every feasible predicted cluster, avoiding sklearn’s imbalanced-sample failure. `silhouette_sample_size` reports actual scored rows; Calinski-Harabasz and Davies-Bouldin remain full-input. Invalid caps below 2 and sampled caps not exceeding cluster count raise clear errors. Tests cover capped/full scoring, custom seed, invalid caps, deterministic sparse string labels, label coverage, and unique sampled rows. Final review found the original sampler retained O(N) intermediate memory; `483ee994` replaces it with bounded reservoir selection and adds a 1,000,000-row regression. Validation: focused clustering suite (`70 passed`), `ruff`, formatting, and `ty` clean.
Task 5: complete (no commit; task-review approved) — complete Core suite isolated the known SentenceEmbedder Hugging Face TLS baseline; excluding it, 2,844 passed. Ruff, formatting, ty, frontend eslint/TypeScript/build, MkDocs, and final clean-diff/status checks passed.

# Progress Ledger: Platform Evolution Roadmap

Plan: .superpowers/plans/2026-08-05-platform-evolution-roadmap.md
Task 1: complete (no commit; ignored deliverable, review approved after citation correction) — created `temp/skyulf-platform-evolution-roadmap-2026-08-05.md`, a source-cited whole-platform decision roadmap spanning Core, backend, frontend, operations, release reliability, and adoption. The correction pass validated 140 source citations with no invalid ranges, added direct TargetEncoder/metrics tests and notebook-export evidence, and added a factually bounded deployment promotion/rollback finding. Integrity, ignored-artifact, stale-citation, citation-range, and diff checks passed.
Task 2: complete (no commit; review approved) — added the factual `## v0.7.4` safety and observability entry immediately before v0.7.3 and the matching root 0.7.x clause. It describes only the completed TargetEncoder, preprocessing metrics, tracing lifecycle, and bounded silhouette work in `748929e5..483ee994`; no future-roadmap language was added.
Task 3: complete (commit `60a48fee`, review approved) — `mkdocs build` exited 0 and `git diff --check` was clean. The commit contains only `CHANGELOG.md` and `changelog/0.7.x.md`, has the required Copilot trailer, and leaves the roadmap ignored/untracked. Existing Material and segmentation-nav MkDocs notices were correctly reported as non-fatal.

# Progress Ledger: Polars-First Core Migration

Design: complete (commit `b69fa2d7`, user-reviewed) — preserve dual-engine public compatibility while removing only evidence-backed internal Pandas conversions. The v0.7.4 sampler correction uses a bounded NumPy label scan and early error above the cap; native raw/wrapped Polars expectations are the first high-confidence migration; remaining Core uses require a source-cited inventory plus parity/benchmark gates.
Plan: complete (commit `dfab58f2`, independent plan review approved) — `polars-first-core-wave-0-1` has five tasks: sampler repair, native expectations, complete ignored Core audit, documentation reconciliation, and final gate/review. Awaiting execution selection.
Task 1: complete (commit `6a13f948`, task review approved) — clustering evaluation now uses a cap-aware NumPy/Python representative collector instead of `pd.unique`, stops at the cap-plus-one distinct label with a clear error, normalizes repeated NaN labels, and reuses representatives for deterministic reservoir sampling. Targeted RED evidence passed against the parent; current clustering suite (`18 passed`) plus Ruff, formatting, and Ty were clean.
Task 2: complete (commits `f34be48c..80d007b6`, task review approved after parity fix) — `profiling.expect` now handles raw and wrapped Polars frames natively for column, null/NaN, range, and uniqueness checks while preserving Pandas/generic fallback behavior. Tests prevent hidden `to_pandas()` conversion and compare public errors; the review caught nullable-integer range text drift (`3` vs `3.0`), fixed with a regression. Focused expectations (`13 passed`) plus Ruff, formatting, and Ty were clean.
Task 3: complete (no commit; ignored artifact, task review approved after gate correction) — `temp/skyulf-core-pandas-polars-audit-2026-08-05.md` accounts for all 93 production Core Pandas/Polars source files using the four approved categories. Wave 2 candidates now contain literal fixtures/goldens, schema/order/error checks, benchmark floors, and rollback conditions; only evidence-backed migrations may proceed.
Task 4: complete (no commit; ignored roadmap reconciliation, task review approved) — verified live `metrics.py` has no `pd.unique` and stops at cap-plus-one labels; updated the ignored roadmap’s sampler citation while keeping its claim limited to bounded label-selection state. MkDocs exited 0; existing Material and segmentation-nav notices remain non-fatal; tracked v0.7.4 changelog diff is empty.
Task 5: complete (no validation commit; task review approved) — combined clustering/expectation coverage passed (`31 passed`), Ruff, formatting, and Ty passed, `git diff --check` was clean, and both ignored `temp/` artifacts were confirmed untracked. A later separate controller progress-ledger commit is outside this task’s validation range.
Final review: complete — all task reviews approved. The requested whole-range reviewer returned no usable output, so the controller directly reviewed `dfab58f2..538f82db`: no high-confidence defects found in cap-bounded silhouette collection, Polars expectation parity, release-note boundaries, or the ignored evidence artifacts. Existing validation remains `31 passed` plus clean Ruff, format, Ty, MkDocs, and diff checks.

# Progress Ledger: Native Polars Correlation Threshold

Plan: `.superpowers/plans/2026-08-06-polars-native-correlation-threshold.md`
Task 1: complete (commit `2c724d24`, task review clean) — native Polars
numeric-column detection now normalizes float NaN to null before its existing
binary and constant checks, matching the legacy Pandas selection path for raw
and wrapped frames. Focused utility coverage (`52 passed`) plus Ruff,
formatting, and Ty were clean.
Task 2: complete (commits `96bb9bde..e381e344`, generated-report cleanup
`b5567629`, final task review clean) — raw and wrapped Polars correlation
fits now execute pairwise-complete native Pearson/Spearman correlation for
eligible numeric/Boolean columns, preserving exact artifacts and apply output
schema/order. Kendall, callables, unsupported dtypes, unavailable native
capability, and boolean thresholds retain the explicit Pandas route; all
non-boolean `Real` thresholds, including `np.int64`, remain native-eligible.
Focused correlation coverage (`16 passed`) plus Ruff, formatting, and Ty were
clean.
Task 3: complete (commits `73c4a1e9..df712ec8`, task review clean) —
compatibility regressions now cover raw and wrapped Polars fallback routing
for Kendall, invalid methods, callables, unsupported selected dtypes, and
unavailable native capability, along with strict threshold equality. Focused
correlation coverage (`25 passed`) plus Ruff, formatting, and Ty were clean.
Task 4: complete (commit `0efdab30`, task review clean) — opt-in
legacy/native raw/wrapped correlation benchmarks and isolated RSS measurements
cover 100k x 50, 1M x 20, and 50k x 500. The approved OR gate passed on
47–56% time reductions for the lower-width raw/wrapped cases; RSS regressed
in every case and severely at 50k x 500, which is recorded as a material
caveat in ignored Candidate A evidence. Targeted tests (`98 passed`, one
pre-existing sklearn warning), Ruff, formatting, and Ty were clean.
Final-review parity fixes: complete (commit `3228b68d`, final review clean) —
native correlation now excludes non-finite pairs, native uniqueness normalizes
selected float NaN values to null, and Boolean Polars range checks use the
Pandas compatibility path. Six direct regressions and focused correlation/
expectation coverage passed.
Final-review cap fix: complete (commit `59c8c469`, final review ready) —
silhouette caps now reject Boolean, fractional, NaN, and infinite values
before any collector can grow by cardinality, while accepting NumPy integral
caps and preserving the below-two error contract. Clustering coverage
(`31 passed`), Ruff, formatting, and Ty were clean.
Minor recorded (non-blocking): the RSS benchmark delta is a
process-lifetime-high-water proxy; Candidate A audit measurements deliberately
invoke every route in a fresh pytest process, and the final reviewer accepted
that documented protocol for this merge.
Final whole-branch review: complete (`5ad70266..59c8c469`) — no Critical or
Important findings; reviewer verdict is Ready to merge.
Controller final verification: focused Core regression suite passed (147
passed, one pre-existing sklearn warning); Ruff, formatting, and Ty passed.
The representative opt-in 100k x 50 correlation benchmark passed for all
legacy/native raw/wrapped routes, with native routes faster in this run.
Controller diff hygiene found one unrelated empty EOF line introduced by
`206dbb56`; `2d638984` removes only that line. Branch integration remains the
only pending action.

# Progress Ledger: Platform Roadmap Consolidation and Phase 0 Documentation/Test Fix

Plan: `temp/skyulf-platform-evolution-roadmap-2026-08-05.md` (ignored decision
roadmap, already consolidated across Core, backend/operations, frontend UX,
release engineering, and adoption/enterprise sections in prior commits
`60a48fee`/`538f82db`).
Consolidation check: complete (no commit; ignored artifact) — confirmed the
roadmap's backend/operations, frontend product experience, and
community/adoption sections were already merged into the single document;
no separate stray audit files existed elsewhere. Added the missing "Native
Polars correlation fitting" line to `## Completed in v0.7.4`, citing
`skyulf-core/skyulf/preprocessing/feature_selection/correlation.py:108-209`
and its real test/benchmark files, since that work landed (`e3a5fa3e`) after
the roadmap's last edit and was previously unlisted there.
Selection: chose Phase 0's "Correct quickstart, documentation, and test
reliability" initiative's two most concrete, low-risk findings — stale
`skyulf[nlp]` install guidance and the network-dependent legacy
SentenceEmbedder test — as the next evidence-backed implementation (commit
`80cebbff`).
Task: complete (commit `80cebbff`, self-reviewed) — `README.md`,
`docs/user_guide/text_nlp.md`, and `requirements-nlp.txt` now consistently
reference `skyulf-core[nlp]` (the actual published package name), matching
every other extras reference already used across the docs. The legacy
`TestSentenceEmbedder.test_embeddings_shape` in
`skyulf-core/tests/test_text_vectorization.py` now mocks
`sentence_embedder._load_model` the same way
`skyulf-core/tests/test_sentence_embedder.py` does, so it runs deterministically
without the optional `sentence-transformers` package or any network access.
Validation: full Core suite `2888 passed, 26 skipped, 1 xfailed` (unchanged
skip/xfail set), plus the 90 targeted vectorization tests; `ruff check`/`ruff
format --check` and `ty check skyulf` clean; `mkdocs build` exited 0 with only
the pre-existing Material 2.0 notice; `git diff --check` clean.
Remaining Phase 0 doc/test-reliability items not yet done (left for a future
task): the docs CI link checker / notebook execution gate, and the broader
engine-contributor-guide/registry-API drift noted in the roadmap's release
engineering section.

Selection: continued the Polars-first Core migration thread — reviewed
`temp/skyulf-core-pandas-polars-audit-2026-08-05.md`'s Wave 2 candidate list
(B: clustering, C: multivariate fallback, D: bucketing, E: H3 index) via
`ask_user` and the user chose Candidate B (clustering numeric filtering,
`skyulf-core/skyulf/modeling/_evaluation/clustering.py`).
Task: complete, accepted — added a native Polars path to
`evaluate_clustering_model()` (`_as_polars_frame`, `_compute_centroids_polars`,
`_compute_reference_crosstab_polars`, `_POLARS_NUMERIC_BOOL_DTYPES`) that
avoids the full-frame `_feature_frame()` Pandas conversion for raw/wrapped
Polars input while reproducing the audit's pinned golden fixture exactly
(metrics, centroids, profiles, crosstab), including a genuine parity fix for
Polars' `select([])` collapsing row count to 0 unlike pandas'
`select_dtypes(...)`. The untouched legacy Pandas path remains the fallback
for everything else. Widened `calculate_clustering_metrics`'s type
annotation in `metrics.py` to include raw `pl.DataFrame` (it already worked
at runtime via `SklearnBridge`/engine dispatch; only the type annotation was
stale) to keep `ty check` clean for the new call site.
Validation: new `skyulf-core/tests/test_evaluation_clustering_polars.py`
(19 tests: golden-fixture parity across Pandas/raw/wrapped Polars, missing
reference column, NaN labels, zero-numeric-columns error parity, dtype
preservation, no-full-conversion guard, relative peak-memory). Benchmark gate
(mirroring Candidate A's harness in `test_benchmarks.py`): 100k x 30 fit time
improved 19.3-19.8% (just under the 20% floor) with mixed RSS (-16.75% raw,
+11.59% wrapped); 1M x 15 fit time improved 30.1-33.8% and peak RSS improved
30.4-37.9% (see the audit's new "Candidate B execution record" section for
exact commands/values). Per the same time-OR-memory gate used for Candidate
A, accepted since 1M x 15 clears both the 20% time floor and the candidate's
stated 20% memory floor. Full Core suite `2907 passed, 42 skipped, 1 xfailed`
(unchanged skip/xfail set); `ruff check`/`ruff format --check` and `ty check`
clean; `git diff --check` clean. Added a concise v0.7.4 changelog entry.

Follow-up: deduped the byte-identical `_POLARS_NUMERIC_BOOL_DTYPES`/
`_POLARS_CORRELATION_DTYPES` frozensets (copy-pasted between `clustering.py`
and `correlation.py`) into a single shared, `HAS_POLARS`-guarded
`POLARS_NUMERIC_BOOL_DTYPES` constant in `skyulf/engines/polars_engine.py`,
re-exported from `skyulf/engines/__init__.py`; both call sites now import it.
Committed as `9b4155ae` (full Core suite/`ruff`/`ty`/`git diff --check` all
clean, unchanged baseline).

Selection: continued to Wave 2 Candidate C (profiling multivariate fallback,
`skyulf-core/skyulf/profiling/_analyzer/multivariate.py`), chosen via
`ask_user` from the remaining C/D/E candidates.
Scope narrowing: `_prepare_matrix_sample`'s `_impute_matrix()` (used by
PCA/clustering) already had a native-first Polars path with sklearn fallback
and needs a stable column count (zero-fills all-null columns) — left
unmodified. The genuine unconditional-Pandas-conversion fallback was
`_detect_outliers()`, which always called `df_numeric.to_pandas().values`
then `SimpleImputer(strategy="mean")`.
Task: complete, accepted — added `_impute_matrix_drop_empty()` (native Polars
fast path: cast to `Float64`, `fill_nan(None)` to normalize NaN-as-value into
null, drop all-null columns matching sklearn's `keep_empty_features=False`
default, `fill_null(mean)`, raise on remaining non-finite values; sklearn
`SimpleImputer` fallback on any exception) and wired it into
`_detect_outliers()` in place of the old inline conversion+impute. Verified
the audit's exact golden fixture (`total_outliers=1`, `outlier_percentage=20.0`,
`index=0`, `score=-0.01172203142549666`) reproduces identically before and
after the change.
Validation: extended `test_profiling_multivariate.py` from 24 to 28 tests
(golden-fixture reproduction, direct `_impute_matrix_drop_empty` vs
`SimpleImputer` parity across mixed-null/Int64-null/NaN-as-value/no-null
cases, all-null column dropping, infinite-value fallback-and-`None`
handling), all passing. New benchmark harness in `test_benchmarks.py`
(`test_outlier_impute_matrix_fit_benchmark`/`_peak_rss`, opt-in large cases
via `SKYULF_RUN_LARGE_BENCHMARKS`) at the audit's exact shapes (50k x 20
null-heavy, 500k x 30 mixed null/all-null, 1M x 10 numeric): fit-preparation
time improved 87-97% across all three shapes (far exceeding the 20% floor);
peak RSS was roughly neutral at the one measured shape (50k x 20, -1.85%).
Accepted on the time criterion alone per the time-OR-memory gate (see the
audit's new "Candidate C execution record" section for exact commands/
values). Full Core suite `2911 passed, 54 skipped, 1 xfailed` (unchanged
skip/xfail set plus the 4 new tests); `ruff check`/`ruff format --check` and
`ty check` on touched files clean (pre-existing unrelated `ty` diagnostics in
`test_evaluation_clustering_polars.py`/`test_benchmarks.py`'s Candidate B
legacy helper, untouched by this change); `git diff --check` clean. Added a
concise v0.7.4 changelog entry.

Selection: continued to Wave 2 Candidate D (bucketing fit routes,
`skyulf-core/skyulf/preprocessing/bucketing.py`).
Scope narrowing: fitting itself is inherently sklearn/pandas-bound for every
strategy (`pd.cut`, `pd.qcut`, `KBinsDiscretizer`), so a full native-Polars
rewrite of the bin-edge math was rejected as unnecessary — the real,
evidence-backed cost was `_to_pandas_for_fit()` always converting the
**entire input frame**, even when only one or a few columns out of a wide
frame are actually being binned.
Task: complete, accepted — added `_resolve_columns_then_to_pandas()`, which
resolves the columns to bin directly on the raw Polars frame first
(`resolve_columns`/`detect_numeric_columns` already dispatch natively
per-engine), then converts only that column subset to pandas. Wired into
`GeneralBinningCalculator.fit()` and `CustomBinningCalculator.fit()`
(`KBinsDiscretizerCalculator` inherits the benefit via `super().fit()`).
Pandas-input calls are unaffected by construction.
Validation: extended `test_bucketing.py` from 47 to 54 tests (bin-edge
parity between raw-Polars-input and pandas-input fits across all 5
strategies, a no-full-conversion guard test tracking `to_pandas()` call
shapes, and the `columns: []` short-circuit on Polars input), all passing.
New benchmark harness in `test_benchmarks.py` at the audit's shapes (1M x 1
single-column control, 250k x 25, 250k x 25 high-null): wide-frame cases
improved fit time 31.7-34.0% and peak RSS 96.6% (only 1/25th of columns
materialized in pandas); the single-column control was correctly neutral
(~2%, noise-level), confirming the benefit is specifically from avoiding
unrelated-column conversion. Accepted per the time-OR-memory gate (see the
audit's new "Candidate D execution record" section for exact commands/
values). Full Core suite `2918 passed, 66 skipped, 1 xfailed` (unchanged
skip/xfail set plus the 7 new tests); `ruff check`/`ruff format --check` and
`ty check` on touched files clean; `git diff --check` clean. Added a concise
v0.7.4 changelog entry.

Selection: continued to the final Wave 2 candidate, E (H3 index Polars
route, `skyulf-core/skyulf/preprocessing/geo/h3_index.py`).
Task: complete, rejected — reproduced the audit's pinned golden fixture
exactly (h3 4.5.0, same third-party output strings) with no code changes
needed. Isolated `to_pandas()` conversion time from the third-party
`h3.latlng_to_cell()` per-row computation time at all three audit-specified
shapes/null-rates (100k/0%, 1M/5%, 5M/50%) in the same warmed-up process:
conversion consistently accounted for 0.08-0.13% of total apply time —
three orders of magnitude smaller than the row computation itself. This
directly matches the audit's own explicit rollback condition ("retain the
current Pandas route if H3 computation dominates runtime"), so no native
Polars rewrite was implemented; `h3_index.py` is unmodified. Added a
permanent `test_h3_index_conversion_share_of_total_fit_time` benchmark to
`test_benchmarks.py` as a standing re-evaluation check (e.g. if a vectorized
H3 API appears upstream in the future). Existing `test_geo_nodes.py` h3
tests (8) still pass unchanged. Documented the full evidence table and
reject decision in the audit's new "Candidate E execution record" section,
plus a "Wave 2 summary" table consolidating all five candidates' decisions.
Full Core suite `2918 passed, 69 skipped, 1 xfailed` (unchanged skip/xfail
count plus 3 new opt-in benchmark tests, 2 of which are large-shape and
skip by default); `ruff check`/`ruff format --check` and `ty check` on
touched files clean; `git diff --check` clean. Added a concise v0.7.4
changelog note.

Wave 2 status: all five candidates (A/B/C/D accepted, E rejected with
evidence) from `temp/skyulf-core-pandas-polars-audit-2026-08-05.md` are now
resolved. A quick `grep -rl "to_pandas()" skyulf/` sweep after this session
shows the remaining call sites are narrow (already column-subset, e.g.
`woe.py`/`polynomial.py`'s `.select(cols).to_pandas()`) or fall under the
audit's own documented "Retained Compatibility Boundaries" (sklearn/NumPy
model-fit surfaces in `elliptic.py`/`resampling.py`/`split.py`/
`vectorization/_common.py`, SHAP, Pandas plotting, generic dispatcher
fallbacks) — no further in-scope candidates were identified from this
audit's original inventory. A fresh full-codebase inventory sweep (repeating
the audit's "Scope and Method" pass) would be needed before claiming there
is nothing left anywhere in the codebase; this only confirms nothing further
is owed from this specific audit document.

## Follow-up: generalized narrow-conversion pattern + dtype dedupe (post-Wave-2)

Following the "are we good everywhere for Polars" question, revisited the
grep sweep's "already narrow" and "retained boundary" claims with fresh
eyes and found six more fit routines doing the exact wasteful
full-frame-`to_pandas()`-then-`resolve_columns()` pattern that Candidate D
fixed in `bucketing.py`, previously miscategorized as compatibility
boundaries because the *estimator* itself is sklearn-bound even though the
*column-resolution step* before it wasn't: `outliers/iqr.py`,
`outliers/winsorize.py`, `outliers/zscore.py`, `outliers/elliptic.py`,
`transformations/power.py`, and `feature_selection/variance.py`. Extracted
Candidate D's inline helper into a shared, reusable
`resolve_columns_then_to_pandas()` in `preprocessing/_helpers.py` and wired
all six nodes through it, removing the duplicated inline pattern.

Also found four nodes that take an explicit, small, named column list
(not auto-detected) and still converted the whole frame before validating
those columns: `geo/distance.py` (lat/lon pairs), `geo/h3_index.py`
(lat/lon pair — Candidate E's evaluated file, still correctly left with a
Pandas apply/compute path, but its *fit-time column validation* narrowing
is a separate, safe, always-a-win change since it doesn't touch the h3
computation itself), `feature_generation/interaction.py`, and
`feature_generation/polynomial.py`. Added a second shared helper,
`select_then_to_pandas()`, for this narrower "select an explicit column
list" case, and wired all four through it.

Deliberately left `feature_selection/model_based.py` and
`feature_selection/univariate.py` unchanged: their candidate-column set is
"all eligible minus target", so narrowing rarely excludes more than one
column — not worth the churn for negligible benefit.

Initially over-reached and also tried to narrow
`feature_selection/correlation.py`'s retained Pandas-fallback path (the
non-`pearson`/`spearman` compatibility branch from Candidate A), which
broke 10 existing tests in `test_feature_selection_gaps.py` that assert
`to_pandas` is called exactly once with a specific input via
`monkeypatch.setattr(correlation_module, "to_pandas", ...)` spies — that
file's fallback contract is intentionally locked from Candidate A's review,
so this was reverted back to its original form; `correlation.py` has zero
diff in this follow-up.

Separately, deduplicated a numeric-Polars-dtype list that had drifted into
two independent, near-identical copies (`utils.py::_polars_numeric_dtype_cols`,
`preprocessing/_helpers.py::auto_detect_numeric_columns`) missing the
`Boolean` entry present in the already-shared `POLARS_NUMERIC_BOOL_DTYPES`
(from the earlier `9b4155ae` dedup commit) into one new
`POLARS_NUMERIC_DTYPES` constant in `engines/polars_engine.py`
(`POLARS_NUMERIC_BOOL_DTYPES` minus `Boolean`), exported from
`engines/__init__.py`, and consumed by both call sites.

Validation: full Core suite `2918 passed, 69 skipped, 1 xfailed` (identical
to the Wave 2 baseline — no regressions), `ruff check`/`ruff format --check`
clean on all touched files, `ty check` clean on all touched files (the two
pre-existing `test_benchmarks.py` diagnostics were confirmed unrelated via
`git stash`/re-check). Ad-hoc benchmark on `IQRCalculator.fit` (250k x 25,
1 of 25 columns selected) confirms the narrowed path runs in ~4.3ms,
consistent with Candidate D's measured gains at the same shape.

Answer to "is there anything left for Polars conversion, are we good to go
everywhere": yes for this repo's `skyulf-core` fit/apply surface as of this
follow-up — the column-resolution-narrowing opportunity is now applied
everywhere it exists (10 nodes total across Wave 2 + this follow-up), and
the remaining `to_pandas()` sites are genuinely either narrow already or are
legitimate sklearn/NumPy/SHAP/plotting/imbalanced-learn boundaries. A fresh
whole-codebase inventory (beyond this specific audit's original scope, and
beyond `skyulf-core` into `backend`/`frontend`) would still be the rigorous
way to rule out anything entirely outside what's been looked at so far.

## Follow-up 2: direct Polars→numpy for sklearn fits where safe

Answered "are we good for Polars everywhere, will removing Pandas later be
OK, is Polars→numpy clean": audited every sklearn-bound fit routine's actual
conversion path. Two nodes (`VarianceThreshold`, `PolynomialFeatures`) had no
Pandas-only step between column resolution and `.fit()` — sklearn accepts
numpy directly — so added `resolve_columns_then_to_numpy()` and
`select_then_to_numpy()` to `_helpers.py` and rewired both nodes to skip the
Pandas hop entirely (Polars `.select(cols).to_numpy()` is native).

The other five narrowed nodes (IQR, Winsorize, Z-Score, Elliptic Envelope,
Power Transformer) genuinely still need Pandas: confirmed via a live
comparison that Polars' `.quantile()` defaults to "nearest" interpolation
vs. Pandas' "linear" (3.25 vs 3.0 on a simple 1..10 series) — a real
behavior mismatch for IQR/Winsorize if swapped carelessly. Z-Score/Elliptic
rely on per-column `pd.to_numeric(errors="coerce").dropna()` semantics with
no native-Polars replacement wired in yet. Power Transformer's box-cox
filter uses Pandas boolean column indexing. Left `TODO(pandas-removal)`
comments at each call site documenting exactly what's blocking the change,
so this isn't silently forgotten.

Validation: full Core suite 2918 passed / 69 skipped / 1 xfailed (unchanged
baseline), ruff check/format and ty check clean on all touched files.

## Follow-up 3: found and fixed a missed narrowing case + a real duplication

Rechecked "is there any code improvement related to this" by grepping every
`to_pandas`/`resolve_columns_then_to_pandas`/`select_then_to_pandas` call
site across `preprocessing/` again with fresh eyes, rather than trusting
the earlier sweep was exhaustive.

Found two real gaps:
1. `feature_selection/model_based.py` and `feature_selection/univariate.py`
   both called `to_pandas(X)` on the *entire* input frame before calling
   `_resolve_candidate_columns()` -- the exact wasteful pattern already
   fixed in 8+ other nodes, missed originally because it's routed through
   a shared `_common.py` helper rather than the node's own inline
   `resolve_columns()` call. Fixed by resolving candidates natively first,
   then narrow-converting via `select_then_to_pandas()` (candidate columns,
   plus the target column when it must be pulled from X itself).
2. `transformations/power.py` and `transformations/general.py` each had
   their own near-identical ~25-line routine to reconstruct a fitted
   `PowerTransformer` + its internal `StandardScaler` from stored
   lambdas/scaler-params (one for a resolved multi-column subset, one for
   a single column at apply time). Extracted into one shared
   `transformations/_power_common.py::build_pretrained_power_transformer()`,
   parameterized by optional `col_indices`/`n_total_cols` for the
   multi-column narrowing case.

Verified: scaling nodes (standard.py, maxabs.py) were already numpy-direct
-- no gap there. woe.py and general.py's per-column power apply were
already narrow -- no gap there either. resampling.py/split.py genuinely
need the full feature matrix (imbalanced-learn/sklearn splitters use all
columns), so their full-frame conversion isn't waste.

Validation: full Core suite 2918 passed / 69 skipped / 1 xfailed (twice,
identical to baseline), ruff check/format and ty check clean on all touched
files, targeted tests for both changes green.

## Subagent-audited pass across remaining skyulf-core (preprocessing,
## modeling, profiling/engines)

Dispatched 3 parallel background `explore` agents (read-only) to cover the
rest of the codebase before preparing a PR: `audit-preprocessing-remaining`
(cleaning, drop_and_missing, encoding, imputation, scaling, time_series,
vectorization + re-check of feature_generation/feature_selection/geo edge
cases), `audit-modeling` (modeling/, data/, core/), and
`audit-profiling-engines` (profiling/, engines/, utils.py). All three
completed; findings triaged and the following fixed in this pass:

1. **Vectorization full-frame waste (highest-impact preprocessing finding).**
   `vectorization/_common.py::resolve_fit_text_columns()` converted the
   entire input frame to Pandas for every text-vectorizer `fit()`, even
   though 3 of its 5 callers (`hashing_vectorizer.py`, `tokenizer.py`,
   `sentence_embedder.py`) discard the returned frame entirely (`_, cols =
   resolved`) -- only the resolved column list is used, no vocab/stats
   fitting happens. Added a zero-conversion `resolve_fit_text_valid_columns()`
   (reuses existing `resolve_valid_columns()`) for those three callers;
   `count_vectorizer.py`/`tfidf_vectorizer.py` (which do need the text data)
   keep the original Pandas-converting `resolve_fit_text_columns()`.
2. **Imputation duplication + unneeded Pandas hop.** `imputation/knn.py`
   and `iterative.py` each had an identical inline
   `X.select(cols) if hasattr(X, "select") and not hasattr(X, "loc") else
   X[cols]` selection line, then routed through `SklearnBridge.to_sklearn()`
   just to reach numpy for `KNNImputer`/`IterativeImputer.fit()`. Both now
   use `resolve_columns_then_to_numpy()` (already existed, underused),
   removing the duplication and the Pandas hop in one move.
3. **Profiling `_NUMERIC_DTYPES` duplication (round 2).** Same 10-item
   Polars dtype tuple had drifted into two more near-identical copies
   (`profiling/analyzer.py`, `profiling/_analyzer/decomposition.py`) beyond
   the ones already deduplicated earlier in `utils.py`/
   `preprocessing/_helpers.py`. Both now import the shared
   `POLARS_NUMERIC_DTYPES` frozenset from `engines/polars_engine.py`
   (verified `EDAAnalyzer._NUMERIC_DTYPES is DecompositionMixin._NUMERIC_DTYPES`
   -- exact same object).
4. **Semantic-type dispatch duplication.** `ColumnMixin._get_semantic_type`
   (per-series) and `EDAAnalyzer`'s inline `_semantic_type_for_column`
   (dtype+ratio, reusing precomputed `n_unique`) implemented the same
   Numeric/Categorical/Boolean/DateTime/Text bucketing logic twice, with
   subtly diverging code (drift risk). Extracted into one shared
   `_dtype_to_semantic_bucket(dtype, ratio, n_unique)` in
   `profiling/_analyzer/_utils.py`; both call sites now delegate to it.
   Caught and fixed a real regression during this: `ColumnMixin` originally
   called `series.n_unique()` unconditionally before dispatching, but
   `_dtype_to_semantic_bucket` only needs ratio/n_unique for Int/String
   dtypes -- Polars' `n_unique()` raises `InvalidOperationError` for
   `pl.Object` columns (caught by
   `test_get_semantic_type_unhandled_dtype_falls_back_to_text`). Fixed by
   only computing n_unique/ratio when dtype is Int or String, matching the
   original per-branch laziness.
5. **Two low-risk `.to_pandas().values`/`.values.tolist()` -> `.to_numpy()`
   one-liners**: `profiling/_analyzer/multivariate.py`'s sklearn-fallback
   imputation path (both `_impute_matrix` and `_impute_matrix_drop_empty`),
   and `modeling/base.py`'s `predict_proba` payload construction -- both feed
   a pure-numpy consumer with no Pandas-only semantics needed.

**Deliberately deferred**: the modeling audit's highest-value finding --
redundant `SklearnBridge.to_sklearn()` (3-4x) and `predict()`/
`predict_proba()` (2-3x) calls on the same `X_test`/`y_test` across
`base.py` -> `sklearn_wrapper.py` -> `_evaluation/{classification,
regression,metrics}.py` -- is real, measurable wasted inference compute,
but spans multiple call layers and touches the public API surface of
`evaluate_classification_model`/`evaluate_regression_model`/
`calculate_classification_metrics`/`calculate_regression_metrics`. Given the
risk/complexity, it needs its own carefully-scoped follow-up rather than
being bundled into this audit-fix pass. Also deferred as lower-value/
non-bugs: the ROC/PR-AUC class-resolution logic split between
`_evaluation/metrics.py` and `_evaluation/classification.py` (already
comment-cross-referenced, differs in binary/multiclass/curve-plotting
purpose -- not a pure duplicate); `scaling/_common.py`'s near-miss vs.
shared helpers (correct as-is, needs native frame not numpy);
`profiling/expect.py`'s ad-hoc `hasattr` check (intentionally
engine-agnostic per its own docstring).

Validation: full Core suite 2918 passed / 69 skipped / 1 xfailed reproduced
after every change (before and after the vectorization fix, before and
after the profiling dedup, before and after the imputation fix, and once
more after the semantic-type extraction regression fix); ruff check/format
and ty check clean on all touched files.
