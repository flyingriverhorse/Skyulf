"""Unit tests for `ArtifactsMixin._extract_shap_summary` (backend data-shape resolution).

The actual SHAP computation is implemented and tested in
`skyulf.modeling._explainability` (see `skyulf-core/tests/test_explainability.py`).
These tests only cover the backend-specific glue: resolving pipeline data
shapes (`SplitDataset`, `(X, y)` tuple, plain DataFrame) into a feature-only
DataFrame and unwrapping `(model, tuning_result)` tuples before delegating.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from backend.ml_pipeline._execution.engine._artifacts import ArtifactsMixin
from skyulf.data.dataset import SplitDataset


class _Host(ArtifactsMixin):
    """Minimal host exposing the attributes `ArtifactsMixin` methods expect."""

    dataset_name = "test-dataset"


@pytest.fixture
def host() -> _Host:
    """Return a fresh `_Host` instance."""
    return _Host()


@pytest.fixture
def classification_frame() -> pd.DataFrame:
    """Small deterministic binary-classification dataset with a target column."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.random(60),
            "b": rng.random(60),
            "c": rng.random(60),
        }
    )
    df["target"] = (df["a"] + df["b"] > 1).astype(int)
    return df


def test_extract_shap_summary_with_dataframe(host: _Host, classification_frame: pd.DataFrame):
    """A plain DataFrame is resolved to its feature columns (target dropped)."""
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(
        classification_frame[["a", "b", "c"]], classification_frame["target"]
    )

    with patch(
        "backend.ml_pipeline._execution.engine._artifacts.compute_shap_summary"
    ) as mock_compute:
        mock_compute.return_value = {"a": 1.0, "b": 0.5, "c": 0.1}
        result = host._extract_shap_summary(model, classification_frame, "target")

    assert result == {"a": 1.0, "b": 0.5, "c": 0.1}
    passed_model, passed_X = mock_compute.call_args[0]
    assert passed_model is model
    assert list(passed_X.columns) == ["a", "b", "c"]
    assert "target" not in passed_X.columns


def test_extract_shap_summary_with_split_dataset(host: _Host, classification_frame: pd.DataFrame):
    """A `SplitDataset` resolves via its `.train` DataFrame."""
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(
        classification_frame[["a", "b", "c"]], classification_frame["target"]
    )
    split = SplitDataset(train=classification_frame, test=classification_frame)

    with patch(
        "backend.ml_pipeline._execution.engine._artifacts.compute_shap_summary"
    ) as mock_compute:
        mock_compute.return_value = {"a": 1.0, "b": 0.5, "c": 0.1}
        result = host._extract_shap_summary(model, split, "target")

    assert result == {"a": 1.0, "b": 0.5, "c": 0.1}
    passed_X = mock_compute.call_args[0][1]
    assert list(passed_X.columns) == ["a", "b", "c"]


def test_extract_shap_summary_with_xy_tuple(host: _Host, classification_frame: pd.DataFrame):
    """An `(X, y)` tuple resolves its feature-only `X` frame."""
    X = classification_frame[["a", "b", "c"]]
    y = classification_frame["target"]
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

    with patch(
        "backend.ml_pipeline._execution.engine._artifacts.compute_shap_summary"
    ) as mock_compute:
        mock_compute.return_value = {"a": 1.0, "b": 0.5, "c": 0.1}
        result = host._extract_shap_summary(model, (X, y), "target")

    assert result == {"a": 1.0, "b": 0.5, "c": 0.1}
    passed_X = mock_compute.call_args[0][1]
    assert list(passed_X.columns) == ["a", "b", "c"]


def test_extract_shap_summary_unwraps_tuning_tuple(host: _Host, classification_frame: pd.DataFrame):
    """A `(model, tuning_result)` tuple from advanced tuning is unwrapped before delegating."""
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(
        classification_frame[["a", "b", "c"]], classification_frame["target"]
    )

    with patch(
        "backend.ml_pipeline._execution.engine._artifacts.compute_shap_summary"
    ) as mock_compute:
        mock_compute.return_value = {"a": 1.0}
        host._extract_shap_summary((model, {"best_params": {}}), classification_frame, "target")

    passed_model = mock_compute.call_args[0][0]
    assert passed_model is model


def test_extract_shap_summary_returns_none_for_unresolvable_data(host: _Host):
    """Data that can't be resolved to a feature frame short-circuits to `None`."""
    result = host._extract_shap_summary(RandomForestClassifier(), object(), "target")

    assert result is None


def test_extract_shap_summary_end_to_end_with_real_model(
    host: _Host, classification_frame: pd.DataFrame
):
    """Smoke test the full path (no mocking) with a real fitted model, if `shap` is installed."""
    pytest.importorskip("shap")
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(
        classification_frame[["a", "b", "c"]], classification_frame["target"]
    )

    result = host._extract_shap_summary(model, classification_frame, "target")

    assert result is not None
    assert set(result.keys()) == {"a", "b", "c"}
