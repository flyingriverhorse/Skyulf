"""Cross-validated selectors save only fixed feature membership for inference."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.feature_selection import RFECV, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LogisticRegression

from skyulf import SkyulfPipeline
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.registry import NodeRegistry


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("problem", ["classification", "regression"])
@pytest.mark.parametrize(
    "method,direction",
    [
        ("rfecv", "forward"),
        ("sequential_feature_selection", "forward"),
        ("sequential_feature_selection", "backward"),
    ],
)
def test_cv_selection_matches_sklearn_and_replays_fixed_names(engine, problem, method, direction):
    """Persisted selections must match real sklearn CV and apply without rerunning a selector."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(36, 4)), columns=list("abcd"))
    y = 4 * frame.a + 0.1 * frame.b
    y = (y > 0).astype(int) if problem == "classification" else y
    estimator = (
        LogisticRegression(max_iter=1000) if problem == "classification" else LinearRegression()
    )
    config = {
        "method": method,
        "columns": list("abcd"),
        "problem_type": problem,
        "cv": 3,
        "n_features_to_select": 2,
        "direction": direction,
    }
    selector = (
        RFECV(estimator, cv=3)
        if method == "rfecv"
        else SequentialFeatureSelector(estimator, cv=3, n_features_to_select=2, direction=direction)
    )
    selector.fit(frame.to_numpy(), y.to_numpy())
    native = pl.from_pandas(frame) if engine == "polars" else frame
    target = pl.Series("label", y) if engine == "polars" else y.rename("label")
    state = NodeRegistry.get_calculator("feature_selection")().fit((native, target), config)
    assert state["selected_columns"] == list(frame.columns[selector.get_support()])
    restored = pickle.loads(pickle.dumps(state))
    for request in (frame, pl.from_pandas(frame)):
        result = NodeRegistry.get_applier("feature_selection")().apply(request, restored)
        assert list(result.columns) == state["selected_columns"]
        assert (
            len(NodeRegistry.get_applier("feature_selection")().apply(request.head(0), restored))
            == 0
        )
    assert not any(callable(v) or hasattr(v, "fit") for v in state.values())


@pytest.mark.parametrize("method", ["rfecv", "sequential_feature_selection"])
@pytest.mark.parametrize("cv", [1, True, 2.5])
def test_cv_selection_rejects_invalid_fold_counts(method, cv):
    """A malformed CV recipe must raise before fitting a model or returning passthrough state."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0], "target": [1, 0, 1]})
    with pytest.raises(ValueError, match="cv"):
        NodeRegistry.get_calculator("feature_selection")().fit(
            frame, {"method": method, "target_column": "target", "cv": cv}
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["rfecv", "sequential_feature_selection"])
def test_cv_selection_real_pipeline_persistence(engine, method, tmp_path, monkeypatch):
    """Feature membership survives disk export and inference on reordered and empty requests."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    frame = pd.DataFrame(np.random.default_rng(17).normal(size=(30, 4)), columns=list("abcd"))
    train = frame.assign(target=frame.a * 3 + frame.b / 2)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "select",
                    "transformer": "feature_selection",
                    "params": {
                        "method": method,
                        "problem_type": "regression",
                        "cv": 3,
                        "n_features_to_select": 2,
                    },
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(pl.from_pandas(train) if engine == "polars" else train, target_column="target")
    save_pipeline(pipeline, tmp_path / "selected")
    restored = load_pipeline(tmp_path / "selected")
    expected = predict_pipeline(frame, restored).prediction.to_numpy()
    np.testing.assert_allclose(
        predict_pipeline(pl.from_pandas(frame), restored).prediction, expected
    )
    np.testing.assert_allclose(
        predict_pipeline(frame.iloc[::-1], restored).prediction, expected[::-1]
    )
    assert len(predict_pipeline(frame.head(0), restored)) == 0
