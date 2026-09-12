# CCN refactor - 0.8.22

Baseline: `e1bc7b440ba1c8d6ddd9c26ff9712c2dd343cced` (2026-09-13).

The user requested all 59 production functions above CCN 10 be simplified in
0.8.22. The scope is the complete `skyulf-core/skyulf/` and `backend/` trees;
tests are not part of the complexity scan. Existing report-above-8 and
fail-above-10 CI settings remain unchanged, without exemptions or suppressions.

| Production tree | Previous >10 | Current >10 | Current >8 report | Maximum |
|---|---:|---:|---:|---:|
| `skyulf-core/skyulf/` | 26 | 0 | 55 | 10 |
| `backend/` | 33 | 0 | 40 | 10 |

## Function measurements

| File | Function | Before | After |
|---|---|---:|---:|
| `backend/ml_pipeline/_execution/_cycle_validation.py` | `validate_no_cycles` | 20 | 9 |
| `backend/ml_pipeline/_execution/_leakage_validation.py` | `validate_no_preprocessing_before_split` | 39 | 10 |
| `backend/ml_pipeline/_execution/diagram.py` | `build_pipeline_diagram` | 15 | 4 |
| `backend/ml_pipeline/_execution/engine/__init__.py` | `_build_node_metadata` | 11 | 8 |
| `backend/ml_pipeline/_execution/engine/_artifacts.py` | `_normalize_train_frame` | 22 | 9 |
| `backend/ml_pipeline/_execution/engine/_feature_eng.py` | `_safe_split_merge_anchor` | 12 | 9 |
| `backend/ml_pipeline/_execution/engine/_feature_eng.py` | `_upstream_fe_chain` | 15 | 10 |
| `backend/ml_pipeline/_execution/engine/_feature_eng.py` | `_branch_chain_up_to_loader` | 11 | 7 |
| `backend/ml_pipeline/_execution/engine/_feature_eng.py` | `_run_feature_engineering` | 14 | 8 |
| `backend/ml_pipeline/_execution/engine/_inspection.py` | `_table_parts` | 11 | 5 |
| `backend/ml_pipeline/_execution/engine/_inspection.py` | `_cell` | 13 | 8 |
| `backend/ml_pipeline/_execution/engine/_inspection.py` | `_snapshot_table` | 11 | 10 |
| `backend/ml_pipeline/_execution/engine/_merge.py` | `_column_modifiers` | 11 | 10 |
| `backend/ml_pipeline/_execution/engine/_merge.py` | `_merge_frames_columnwise` | 11 | 3 |
| `backend/ml_pipeline/_execution/engine/_merge.py` | `_merge_frames_rowwise` | 13 | 9 |
| `backend/ml_pipeline/_execution/engine/_merge.py` | `_merge_frames` | 12 | 7 |
| `backend/ml_pipeline/_execution/engine/_merge.py` | `_sibling_fan_in_overlap_columns` | 13 | 9 |
| `backend/ml_pipeline/_execution/engine/_node_runners.py` | `_run_data_loader` | 12 | 8 |
| `backend/ml_pipeline/_execution/engine/_node_runners.py` | `_assert_numeric_training_frame` | 15 | 8 |
| `backend/ml_pipeline/_execution/engine/_node_runners.py` | `_resolve_train_feature_columns` | 14 | 8 |
| `backend/ml_pipeline/_execution/engine/_node_runners.py` | `_run_training_tuned` | 13 | 10 |
| `backend/ml_pipeline/_internal/_routers/pipelines_io.py` | `get_pipeline_audit_log._matches` | 15 | 4 |
| `backend/ml_pipeline/_internal/_routers/pipelines_io.py` | `get_pipeline_audit_log` | 11 | 9 |
| `backend/ml_pipeline/_internal/_routers/preview.py` | `_inspection_path` | 18 | 6 |
| `backend/ml_pipeline/_internal/_routers/preview.py` | `_run_preview_sub_pipelines` | 21 | 10 |
| `backend/ml_pipeline/_internal/_routers/preview.py` | `preview_pipeline` | 20 | 9 |
| `backend/ml_pipeline/_internal/_routers/run_pipeline.py` | `resubmit_job_from_graph` | 11 | 9 |
| `backend/ml_pipeline/_services/evaluation_service.py` | `_decode_reference_column` | 18 | 8 |
| `backend/monitoring/router.py` | `calculate_drift` | 15 | 8 |
| `backend/monitoring/router.py` | `list_error_events` | 17 | 9 |
| `backend/monitoring/router.py` | `list_pipeline_logs` | 16 | 10 |
| `backend/monitoring/router.py` | `_parse_inspector_node` | 15 | 8 |
| `backend/monitoring/router.py` | `_build_node_inspector_response` | 14 | 9 |
| `skyulf-core/skyulf/leakage.py` | `leakage_exemption_reason` | 38 | 5 |
| `skyulf-core/skyulf/leakage.py` | `validate_leakage_safety` | 20 | 9 |
| `skyulf-core/skyulf/modeling/_evaluation/clustering.py` | `evaluate_clustering_model` | 13 | 4 |
| `skyulf-core/skyulf/modeling/_evaluation/thresholds.py` | `apply_thresholds` | 14 | 10 |
| `skyulf-core/skyulf/modeling/_explainability/shap_explanation.py` | `_compute_interaction_summary` | 12 | 7 |
| `skyulf-core/skyulf/modeling/_explainability/shap_explanation.py` | `compute_shap_explanation` | 19 | 9 |
| `skyulf-core/skyulf/modeling/_tuning/engine.py` | `_align_time_series_validation` | 12 | 9 |
| `skyulf-core/skyulf/modeling/_tuning/engine.py` | `fit` | 16 | 9 |
| `skyulf-core/skyulf/modeling/_tuning/engine.py` | `tune` | 45 | 10 |
| `skyulf-core/skyulf/modeling/_tuning/grid_random.py` | `fit_and_score_candidate_fold` | 12 | 8 |
| `skyulf-core/skyulf/modeling/_tuning/params.py` | `normalize_logistic_search_config` | 12 | 9 |
| `skyulf-core/skyulf/modeling/_tuning/refit.py` | `refit_best_model` | 11 | 8 |
| `skyulf-core/skyulf/modeling/_tuning/strategies/runner.py` | `execute_search` | 12 | 8 |
| `skyulf-core/skyulf/modeling/clustering.py` | `_select_numeric_features` | 12 | 8 |
| `skyulf-core/skyulf/modeling/cross_validation.py` | `_detect_datetime_columns` | 12 | 9 |
| `skyulf-core/skyulf/pipeline/_pipeline.py` | `_fit_tuning_pipeline` | 16 | 8 |
| `skyulf-core/skyulf/pipeline/diagram.py` | `build_mermaid_diagram` | 12 | 6 |
| `skyulf-core/skyulf/pipeline/seal.py` | `_feed_canonical` | 39 | 10 |
| `skyulf-core/skyulf/preprocessing/casting.py` | `_build_polars_cast_exprs` | 19 | 10 |
| `skyulf-core/skyulf/preprocessing/casting.py` | `fit` | 11 | 7 |
| `skyulf-core/skyulf/preprocessing/fold_adapter.py` | `_run_branches` | 13 | 7 |
| `skyulf-core/skyulf/preprocessing/pipeline.py` | `fit_transform` | 13 | 9 |
| `skyulf-core/skyulf/preprocessing/vectorization/_common.py` | `apply_text_dual_engine` | 14 | 9 |
| `skyulf-core/skyulf/preprocessing/vectorization/_common.py` | `resolve_fit_text_valid_columns` | 11 | 3 |
| `skyulf-core/skyulf/profiling/_analyzer/_utils.py` | `_dtype_to_semantic_bucket` | 15 | 8 |
| `skyulf-core/skyulf/profiling/visualizer.py` | `_plot_pca` | 13 | 9 |

All extracted helpers are included in the same complete-tree scan.
New helper docstrings were checked through the Python AST, including private modules.

## Behavior and compatibility

The refactor separates existing stages of tuning, evaluation, preprocessing,
graph admission, execution, preview and monitoring into named helpers. It
preserves public signatures, fitted state, target/index alignment, warning
order, API payloads, SQL filters and optional dependency boundaries.

Independent review found a recursion-depth regression in the initial serializer
extraction. The final encoder yields child values and keeps recursion in the
original dispatch frame. Frozen digest regressions pin scalar/container and
numeric/object/structured array encodings; further regressions pin 220 nested
dicts, 220 mixed list/dict levels and 350 nested object arrays. No recursion
limit is changed. Existing artifacts retain their fingerprints, so this release
requires no model retraining or frontend workflow change.

HEAD-versus-current differential checks verified 13,029 leakage exemption cases,
500 Core leakage verdicts, 500 cycle graphs with exact error text, 300 backend
leakage graphs with warning order, and 600 preview path hashes/labels. Serializer
checks covered 519 payloads; final independent review additionally confirmed eight
deep container shapes and five cyclic shapes. No review finding remains open.

## Verification

Commands used `.venv/Scripts/python.exe`; tests disabled the pytest cache provider,
used separate temporary directories, and ran with Hugging Face offline flags.
Core and backend suites ran separately to avoid duplicate module names.

| Check | Result |
|---|---|
| Full final Core, sklearn 1.9.1 / imbalanced-learn 0.14.2, Polars default | **8,846 passed, 80 skipped**, 3 snapshots |
| Full Core, sklearn 1.8.0, pandas default, before final serializer depth adjustment | **8,843 passed, 80 skipped**, 3 snapshots; branch coverage **97.09%**, above the 90% gate |
| Final serializer suite on sklearn 1.8.0 | **42 passed**, including all seven new compatibility vectors |
| Backend, pandas default | **3,646 passed**, 7 snapshots; one local smoke test deselected as explained below |
| Backend, Polars default | **3,646 passed**, 7 snapshots; one local smoke test failed during external-directory setup as explained below |
| Actual complexity `run` steps from both committed CI workflows | Reports exit 0; both gates exit 0 with **0 findings** |
| Full Ruff check, format check, and configured global ty check | Passed |
| Pre-commit over changed files and this report | Passed; frontend source lint skipped because no frontend source changed |
| Frontend `check-version`, TypeScript/Vite build, all 11 bundle size budgets | Passed; generated assets unchanged |
| App/Core/frontend/npm lock/uv lock/local editable Core version | **0.8.22** |

The existing `tests/integration/test_full_inference_pipeline.py` is marked to skip
on CI unless a hard-coded local `C:\Users\Murat\Desktop\skyulf-mlflow` workspace
exists. Here it attempted to delete that external workspace's
`temp_test_artifacts_full` directory and failed with `PermissionError` before
pipeline execution. It was excluded from the pandas run; neither the test nor
that external directory was modified. This is a local verification limitation,
not a passing assertion or a production regression.

Full test logs and workflow output remain in ignored `tmp_repro_artifacts/`:
`ccn_full_core_sk19_polars.log`, `ccn_full_core_pandas.log`,
`ccn_core_pipeline_seal_final.log`, `ccn_full_backend_pandas.log`,
`ccn_full_backend_polars.log`, `ccn_0822_workflows.log`,
`ccn_0822_hooks.log`, and `ccn_frontend_build.log`.

The refactor left the audit queue at **58 open / 4 parked** and did not close
unrelated bug findings. The subsequent requested
[pipeline exercise](ccn_pipeline_verification_0.8.22.md) found pre-existing
serving defect OC-320, bringing the live queue to **59 open / 4 parked**.
