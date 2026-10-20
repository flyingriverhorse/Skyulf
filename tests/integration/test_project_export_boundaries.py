"""Invalid saved contracts and estimator outputs must fail before misleading predictions ship."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from backend.ml_pipeline.project_export.service import build_project
from skyulf.preprocessing.pipeline import FeatureEngineer
from tests.integration.test_project_export import export_job, fitted_bundle
from tests.integration.test_project_export_runtime import _fit_project, _write_runtime_project


@pytest.mark.parametrize(
    "schema, message",
    [
        pytest.param(["x"], "name and dtype", id="non-column-entry"),
        pytest.param([{"name": "x", "dtype": None}], "name and dtype", id="missing-dtype"),
        pytest.param([{"name": "", "dtype": "float64"}], "nonempty", id="empty-name"),
        pytest.param(
            [{"name": "x", "dtype": "float64"}, {"name": "x", "dtype": "float64"}],
            "unique",
            id="duplicate-name",
        ),
        pytest.param(
            [{"name": "target", "dtype": "float64"}],
            "prediction target",
            id="target-leakage",
        ),
    ],
)
def test_export_rejects_malformed_raw_schema(schema, message):
    """A corrupt input contract must not become a runnable-looking project or require a target."""
    bundle = fitted_bundle()
    bundle["input_schema"] = schema

    with pytest.raises(ValueError, match=message):
        build_project(bundle, export_job(), "both")


@pytest.mark.parametrize("columns", [[], ["x", "x"]], ids=["empty", "duplicate"])
def test_export_rejects_invalid_model_feature_order(columns):
    """Feature identity must be unambiguous before serializing the fitted model."""
    bundle = fitted_bundle()
    bundle["feature_columns"] = columns

    with pytest.raises(ValueError, match="ordered model feature columns"):
        build_project(bundle, export_job(), "both")


@pytest.mark.parametrize(
    "matrix, message",
    [
        pytest.param([[0.5, 0.5]], "shape", id="wrong-row-count"),
        pytest.param([[1.0], [1.0]], "shape", id="wrong-class-count"),
        pytest.param([[np.nan, np.nan], [0.5, 0.5]], "finite", id="nonfinite"),
        pytest.param([[-0.1, 0.5], [0.5, 0.5]], "between zero and one", id="negative"),
        pytest.param([[1.1, 0.5], [0.5, 0.5]], "between zero and one", id="above-one"),
        pytest.param([[0.3, 0.4], [0.5, 0.5]], "sum to one", id="unnormalized"),
    ],
)
def test_runtime_rejects_invalid_estimator_probabilities(tmp_path, monkeypatch, matrix, message):
    """Malformed estimator output must not be paired with incorrect rows or class labels."""
    predictor = _fit_project(tmp_path)
    monkeypatch.setattr(predictor.state["model"], "predict_proba", lambda frame: np.array(matrix))

    with pytest.raises(ValueError, match=message):
        predictor.predict(pd.DataFrame({"x": [0.0, 5.0]}))


def test_runtime_rejects_multi_target_probabilities(tmp_path):
    """A model with two classification targets must not collapse probabilities into one mapping."""
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    model = RandomForestClassifier(n_estimators=2, random_state=0).fit(
        frame, [[0, 1], [0, 0], [1, 1], [1, 0]]
    )
    predictor = _write_runtime_project(tmp_path, model, FeatureEngineer([]))

    with pytest.raises(ValueError, match="single classification target"):
        predictor.predict(frame)


def test_runtime_rejects_prediction_row_count_changes(tmp_path, monkeypatch):
    """Every returned prediction must correspond to exactly one eligible input row."""
    predictor = _fit_project(tmp_path, classifier=False)
    monkeypatch.setattr(predictor.state["model"], "predict", lambda frame: np.array([1.0]))

    with pytest.raises(ValueError, match="prediction changed row count"):
        predictor.predict(pd.DataFrame({"x": [0.0, 5.0]}))


def test_runtime_rejects_missing_model_features(tmp_path):
    """A schema/preprocessor mismatch must fail before the model can consume a wrong column."""
    predictor = _fit_project(tmp_path)
    predictor.state["feature_columns"] = ["missing_feature"]

    with pytest.raises(ValueError, match="Missing model features.*missing_feature"):
        predictor.predict(pd.DataFrame({"x": [0.0, 5.0]}))


def test_runtime_rejects_unlabelled_preprocessing_output(tmp_path, monkeypatch):
    """Array output must not silently bypass the saved feature-name and ordering contract."""
    predictor = _fit_project(tmp_path)
    monkeypatch.setattr(
        predictor.state["feature_engineer"],
        "transform",
        lambda frame, **kwargs: np.array([[0.0], [5.0]]),
    )

    with pytest.raises(ValueError, match="pandas DataFrame"):
        predictor.predict(pd.DataFrame({"x": [0.0, 5.0]}))
