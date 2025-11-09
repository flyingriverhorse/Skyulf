import numpy as np
import pandas as pd
import pytest

from core.feature_engineering.nodes.feature_eng.feature_selection import apply_feature_selection


def _build_node(config):
    return {"id": "feature-selection", "data": {"config": config}}


def _build_classification_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feat_high": [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
            "feat_mid": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1],
            "feat_noise": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def _build_regression_frame() -> pd.DataFrame:
    x = np.linspace(0.0, 9.0, num=10)
    noise = np.array([0.1, 0.4, 0.2, 0.5, 0.7, 0.3, 0.6, 0.8, 0.2, 0.9])
    return pd.DataFrame(
        {
            "feat_linear": x,
            "feat_quadratic": x ** 2,
            "feat_noise": noise,
            "target": 2.0 * x + noise,
        }
    )


def test_feature_selection_select_k_best_keeps_top_features():
    frame = pd.DataFrame(
        {
            "signal_high": [0, 1, 0, 1, 0, 1],
            "signal_medium": [0, 1, 1, 0, 0, 1],
            "signal_noise": [5, 3, 4, 2, 1, 0],
            "target": [0, 0, 1, 1, 0, 1],
        }
    )
    node = _build_node(
        {
            "columns": ["signal_high", "signal_medium", "signal_noise"],
            "target_column": "target",
            "method": "select_k_best",
            "k": 2,
            "drop_unselected": True,
            "auto_detect": False,
        }
    )

    result, summary, signal = apply_feature_selection(frame, node)

    assert summary.startswith("Feature selection: kept 2 of 3")
    assert "SelectKBest" in summary
    assert set(signal.selected_columns) == {"signal_high", "signal_medium"}
    assert signal.dropped_columns == ["signal_noise"]
    assert list(result.columns) == ["signal_high", "signal_medium", "target"]


def test_feature_selection_variance_threshold_without_target():
    frame = pd.DataFrame(
        {
            "flat": [1, 1, 1, 1],
            "slight": [1, 2, 1, 2],
            "spread": [1, 2, 3, 4],
        }
    )
    node = _build_node(
        {
            "columns": ["flat", "slight", "spread"],
            "method": "variance_threshold",
            "threshold": 0.3,
            "drop_unselected": True,
        }
    )

    result, summary, signal = apply_feature_selection(frame, node)

    assert summary.startswith("Feature selection: kept 1 of 3")
    assert "VarianceThreshold" in summary
    assert signal.selected_columns == ["spread"]
    assert set(signal.dropped_columns) == {"flat", "slight"}
    assert list(result.columns) == ["spread"]
    assert signal.target_column is None


@pytest.mark.parametrize("score_func", ["f_classif", "mutual_info_classif", "chi2"])
def test_feature_selection_select_percentile_classification_scores(score_func):
    frame = _build_classification_frame()
    node = _build_node(
        {
            "columns": ["feat_high", "feat_mid", "feat_noise"],
            "target_column": "target",
            "method": "select_percentile",
            "percentile": 50,
            "score_func": score_func,
            "drop_unselected": False,
            "auto_detect": False,
        }
    )

    result, summary, signal = apply_feature_selection(frame, node)

    assert summary.startswith("Feature selection: kept")
    assert "SelectPercentile" in summary
    assert signal.problem_type == "classification"
    assert signal.score_func == score_func
    assert set(signal.selected_columns).issubset({"feat_high", "feat_mid", "feat_noise"})
    assert "target" in result.columns


@pytest.mark.parametrize("score_func", ["f_regression", "mutual_info_regression", "r_regression"])
def test_feature_selection_generic_univariate_regression_scores(score_func):
    frame = _build_regression_frame()
    node = _build_node(
        {
            "columns": ["feat_linear", "feat_quadratic", "feat_noise"],
            "target_column": "target",
            "method": "generic_univariate_select",
            "mode": "k_best",
            "k": 2,
            "score_func": score_func,
            "drop_unselected": False,
            "problem_type": "regression",
            "auto_detect": False,
        }
    )

    result, summary, signal = apply_feature_selection(frame, node)

    assert summary.startswith("Feature selection: kept")
    assert "GenericUnivariateSelect" in summary
    assert signal.problem_type == "regression"
    assert signal.score_func == score_func
    assert set(signal.selected_columns).issubset({"feat_linear", "feat_quadratic", "feat_noise"})
    assert "target" in result.columns


@pytest.mark.parametrize(
    "method,expected_label",
    [
        ("select_fpr", "SelectFpr"),
        ("select_fdr", "SelectFdr"),
        ("select_fwe", "SelectFwe"),
    ],
)
def test_feature_selection_alpha_based_methods(method, expected_label):
    frame = _build_classification_frame()
    node = _build_node(
        {
            "columns": ["feat_high", "feat_mid", "feat_noise"],
            "target_column": "target",
            "method": method,
            "score_func": "f_classif",
            "alpha": 0.5,
            "drop_unselected": False,
            "auto_detect": False,
        }
    )

    _, summary, signal = apply_feature_selection(frame, node)

    assert summary.startswith("Feature selection: kept")
    assert expected_label in summary
    assert signal.score_func == "f_classif"
    assert signal.problem_type == "classification"
    assert set(signal.selected_columns).issubset({"feat_high", "feat_mid", "feat_noise"})


def test_feature_selection_select_from_model_random_forest():
    frame = _build_classification_frame()
    node = _build_node(
        {
            "columns": ["feat_high", "feat_mid", "feat_noise"],
            "target_column": "target",
            "method": "select_from_model",
            "estimator": "random_forest",
            "drop_unselected": True,
            "auto_detect": False,
        }
    )

    result, summary, signal = apply_feature_selection(frame, node)

    assert summary.startswith("Feature selection: kept")
    assert "SelectFromModel" in summary
    assert signal.problem_type == "classification"
    assert set(result.columns).issubset({"feat_high", "feat_mid", "feat_noise", "target"})
    assert signal.selected_columns


def test_feature_selection_rfe_random_forest():
    frame = _build_classification_frame()
    node = _build_node(
        {
            "columns": ["feat_high", "feat_mid", "feat_noise"],
            "target_column": "target",
            "method": "rfe",
            "estimator": "random_forest",
            "k": 2,
            "drop_unselected": False,
            "auto_detect": False,
        }
    )

    _, summary, signal = apply_feature_selection(frame, node)

    assert summary.startswith("Feature selection: kept")
    assert "RFE" in summary
    assert signal.selected_columns
    assert signal.problem_type == "classification"
