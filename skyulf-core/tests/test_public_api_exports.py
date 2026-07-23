"""Tests for the supported public import surfaces."""


def test_top_level_eda_exports_match_profiling_exports() -> None:
    """Top-level Skyulf exposes the profiling API's primary entry points."""
    import skyulf
    from skyulf import profiling

    assert skyulf.EDAAnalyzer is profiling.EDAAnalyzer
    assert skyulf.EDAVisualizer is profiling.EDAVisualizer
    assert skyulf.DatasetProfile is profiling.DatasetProfile
    assert skyulf.DriftCalculator is profiling.DriftCalculator
    assert skyulf.expect_columns_exist is profiling.expect_columns_exist
    assert skyulf.expect_no_nulls is profiling.expect_no_nulls
    assert skyulf.expect_unique is profiling.expect_unique
    assert skyulf.expect_value_range is profiling.expect_value_range


def test_modeling_evaluation_and_explainability_exports_match_internal_api() -> None:
    """Modeling exposes supported evaluation and SHAP helper functions."""
    from skyulf import modeling
    from skyulf.modeling._evaluation import (
        calculate_classification_metrics,
        calculate_clustering_metrics,
        calculate_regression_metrics,
    )
    from skyulf.modeling._explainability import compute_shap_explanation

    assert modeling.calculate_classification_metrics is calculate_classification_metrics
    assert modeling.calculate_regression_metrics is calculate_regression_metrics
    assert modeling.calculate_clustering_metrics is calculate_clustering_metrics
    assert modeling.compute_shap_explanation is compute_shap_explanation
