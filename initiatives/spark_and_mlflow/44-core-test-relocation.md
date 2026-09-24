# Core test relocation

Date: 2026-09-24. Scope: collected pytest files that test Core without backend dependencies.

36 files moved into Core. The three Databricks Bundle tests belong in
`skyulf-core/tests/integrations`, alongside the integration services. The
preprocessing chain belongs in `tests/integration`, and isolated node/model
checks belong in `tests/unit`. Existing same-named Core files were preserved;
conflicting names gained a descriptive `_nodes` suffix.

Backend/Core integration tests remain in root `tests`. The EDA producer
contract also stays there because it compares Core output with a frontend
fixture. Manual diagnostic and benchmark scripts were not reclassified as
pytest tests by this relocation.

## Verification and supporting corrections

- Before relocation: 224 cases collected and passed.
- After relocation and path updates: the same 224 cases passed.
- All Core tests collected: 10,900 cases, no duplicate-module or import failures.
- Moving tests into Core brought 60 previously out-of-scope ty diagnostics into
  scope. Explicit frame/tuple/None assertions and typed test parameters resolve
  them; required checks were not suppressed globally.
- One precise waiver retains the native-Polars CV runtime regression because
  its public annotation omits that accepted input. No input conversion was added.
- The former manual registry check changed from print-only diagnostics and
  global stdout/sys.path mutations to real metadata assertions.
- Moved tests plus the workflow service suite: 281 passed. Full repository ty
  and scoped Ruff passed. Independent review found no removed test functions,
  reduced assertion counts or broken Bundle paths.
- Historical reports retain their original evidence paths; use this mapping
  to locate the current test files.

## File mapping

| Previous path | Current path |
| --- | --- |
| `tests/integration/test_bundle_challenger_lifecycle.py` | `skyulf-core/tests/integrations/test_databricks_bundle_lifecycle.py` |
| `tests/integration/test_pipeline_integration.py` | `skyulf-core/tests/integration/test_preprocessing_chain.py` |
| `tests/integration/test_sm20a_bundle_template.py` | `skyulf-core/tests/integrations/test_databricks_bundle_template.py` |
| `tests/integration/test_sm31_notebook_boundary.py` | `skyulf-core/tests/integrations/test_databricks_bundle_notebook.py` |
| `tests/unit/test_advanced_cleaning.py` | `skyulf-core/tests/unit/test_advanced_cleaning.py` |
| `tests/unit/test_binning_comprehensive.py` | `skyulf-core/tests/unit/test_binning_comprehensive.py` |
| `tests/unit/test_casting.py` | `skyulf-core/tests/unit/test_casting_nodes.py` |
| `tests/unit/test_cleaning.py` | `skyulf-core/tests/unit/test_cleaning.py` |
| `tests/unit/test_cleaning_nodes.py` | `skyulf-core/tests/unit/test_cleaning_nodes.py` |
| `tests/unit/test_cross_validation_all_methods.py` | `skyulf-core/tests/unit/test_cross_validation_all_methods.py` |
| `tests/unit/test_cv_basic_vs_advanced.py` | `skyulf-core/tests/unit/test_cv_basic_vs_advanced.py` |
| `tests/unit/test_cv_polars_final.py` | `skyulf-core/tests/unit/test_cv_polars_final.py` |
| `tests/unit/test_drift.py` | `skyulf-core/tests/unit/test_drift.py` |
| `tests/unit/test_dynamic_registry_manual.py` | `skyulf-core/tests/unit/test_registry_metadata.py` |
| `tests/unit/test_encoding.py` | `skyulf-core/tests/unit/test_encoding.py` |
| `tests/unit/test_encoding_scenarios.py` | `skyulf-core/tests/unit/test_encoding_scenarios.py` |
| `tests/unit/test_encoding_target_guard.py` | `skyulf-core/tests/unit/test_encoding_target_guard.py` |
| `tests/unit/test_feature_generation.py` | `skyulf-core/tests/unit/test_feature_generation.py` |
| `tests/unit/test_feature_generation_speed.py` | `skyulf-core/tests/unit/test_feature_generation_speed.py` |
| `tests/unit/test_feature_selection.py` | `skyulf-core/tests/unit/test_feature_selection.py` |
| `tests/unit/test_feature_selection_comprehensive.py` | `skyulf-core/tests/unit/test_feature_selection_comprehensive.py` |
| `tests/unit/test_feature_selection_polars.py` | `skyulf-core/tests/unit/test_feature_selection_polars.py` |
| `tests/unit/test_filtering.py` | `skyulf-core/tests/unit/test_filtering.py` |
| `tests/unit/test_imputation.py` | `skyulf-core/tests/unit/test_imputation.py` |
| `tests/unit/test_imputation_backend.py` | `skyulf-core/tests/unit/test_imputation_backend.py` |
| `tests/unit/test_inspection.py` | `skyulf-core/tests/unit/test_inspection_nodes.py` |
| `tests/unit/test_label_encoder_target.py` | `skyulf-core/tests/unit/test_label_encoder_target.py` |
| `tests/unit/test_outliers.py` | `skyulf-core/tests/unit/test_outliers.py` |
| `tests/unit/test_preprocessing_comprehensive_suite.py` | `skyulf-core/tests/unit/test_preprocessing_comprehensive_suite.py` |
| `tests/unit/test_preprocessing_feature_nodes.py` | `skyulf-core/tests/unit/test_preprocessing_feature_nodes.py` |
| `tests/unit/test_profiling.py` | `skyulf-core/tests/unit/test_profiling.py` |
| `tests/unit/test_resampling.py` | `skyulf-core/tests/unit/test_resampling_nodes.py` |
| `tests/unit/test_scaling.py` | `skyulf-core/tests/unit/test_scaling_nodes.py` |
| `tests/unit/test_split.py` | `skyulf-core/tests/unit/test_split_nodes.py` |
| `tests/unit/test_transformations.py` | `skyulf-core/tests/unit/test_transformations.py` |
| `tests/unit/test_transformations_polars.py` | `skyulf-core/tests/unit/test_transformations_polars.py` |
