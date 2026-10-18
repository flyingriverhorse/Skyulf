"""Selection and binning must reject failed fits and keep targets out of discovery."""

import pickle
from copy import deepcopy

import pandas as pd
import polars as pl
import pytest

from skyulf import SkyulfPipeline
from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.preprocessing.bucketing import GeneralBinningApplier, GeneralBinningCalculator
from skyulf.preprocessing.feature_generation.polynomial import (
    PolynomialFeaturesApplier,
    PolynomialFeaturesCalculator,
)
from skyulf.preprocessing.feature_selection.facade import FeatureSelectionCalculator
from skyulf.preprocessing.feature_selection.model_based import (
    ModelBasedSelectionApplier,
    ModelBasedSelectionCalculator,
)


def _frame(engine):
    """Keep enough distinct numeric values to exercise automatic feature discovery."""
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "z": [4.0, 1.0, 3.0, 2.0, 7.0],
            "target": [2.5, 4.5, 6.5, 8.5, 10.5],
        }
    )
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"target_column": "missing"}, "requires target"),
        ({"method": "telepathy"}, "select_from_model.*rfe"),
        ({"estimator": "mystery"}, "auto.*linear_regression.*random_forest"),
        ({"problem_type": "forecast"}, "classification.*regression"),
    ],
)
def test_model_selection_rejects_unresolved_configuration(engine, config, message):
    """A supervised selector must never save a successful-looking passthrough after misconfiguration."""
    config = {"columns": ["x", "z"], "target_column": "target", **config}
    before = deepcopy(config)
    with pytest.raises(ValueError, match=message):
        ModelBasedSelectionCalculator().fit(_frame(engine), config)
    assert config == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_selection_facade_rejects_unknown_method(engine):
    """A misspelled facade method must name supported choices instead of disabling selection."""
    with pytest.raises(ValueError, match="telepathy.*variance.*select_k_best.*rfe"):
        FeatureSelectionCalculator().fit(_frame(engine), {"method": "telepathy"})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_model_selection_preserves_real_estimator_fit_failure(engine):
    """A fitted selector with an impossible feature count must fail rather than keep every feature."""
    with pytest.raises(ValueError, match="max_features"):
        ModelBasedSelectionCalculator().fit(
            _frame(engine), {"target_column": "target", "max_features": -1}
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("override", [False, True])
def test_binning_rejects_unknown_global_and_column_strategies(engine, override):
    """A typo must fail for the affected column even when earlier columns fitted successfully."""
    config = {"columns": ["x", "z"], "strategy": "telepathy"}
    if override:
        config.update(strategy="equal_width", column_strategies={"z": {"strategy": "telepathy"}})
    before = deepcopy(config)
    with pytest.raises(
        ValueError, match="telepathy.*equal_width.*equal_frequency.*kmeans.*custom.*kbins"
    ):
        GeneralBinningCalculator().fit(_frame(engine), config)
    assert config == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("mixed", [False, True])
def test_binning_rejects_unknown_strategy_before_skipping_all_missing_values(engine, mixed):
    """Missing observations must not hide a typo or permit a partially fitted artifact."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "missing": [float("nan")] * 3})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    config = {"columns": ["missing"], "strategy": "telepathy"}
    if mixed:
        config = {
            "columns": ["x", "missing"],
            "column_strategies": {"missing": {"strategy": "telepathy"}},
        }
    with pytest.raises(ValueError, match="telepathy.*equal_width.*equal_frequency"):
        GeneralBinningCalculator().fit(frame, config)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("strategy", ["equal_width", "kmeans", "kbins"])
def test_binning_rejects_failed_column_fit(engine, strategy):
    """Invalid bin counts must not leave a partially fitted artifact that silently skips a feature."""
    config = {
        "columns": ["x", "z"],
        "strategy": "equal_width",
        "column_strategies": {
            "z": {"strategy": strategy, "equal_width_bins": -1, "n_bins": -1, "kbins_n_bins": -1}
        },
    }
    with pytest.raises(ValueError):
        GeneralBinningCalculator().fit(_frame(engine), config)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("selection", [{}, {"columns": []}])
def test_polynomial_auto_detection_excludes_target_and_replays_without_it(engine, selection):
    """Training target powers must never enter saved features or disappear when scoring lacks labels."""
    frame = _frame(engine)
    config = {"auto_detect": True, "target_column": "target", **selection}
    state = PolynomialFeaturesCalculator().fit(frame, config)
    assert state["columns"] == ["x", "z"]
    saved = pickle.dumps(state)
    applier = PolynomialFeaturesApplier()
    request = frame.drop(columns="target") if engine == "pandas" else frame.drop("target")
    output = applier.apply(request, pickle.loads(saved))
    training = applier.apply(frame, state)
    expected = training.drop(columns="target") if engine == "pandas" else training.drop("target")
    if engine == "pandas":
        pd.testing.assert_frame_equal(output, expected)
    else:
        assert output.equals(expected)
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["selection", "binning", "polynomial"])
def test_valid_saved_selection_and_binning_state_replays_chunks_and_empty(engine, kind):
    """New fit validation must leave existing fitted decisions, output schemas and state unchanged."""
    frame = _frame(engine)
    calculators = {
        "selection": (
            ModelBasedSelectionCalculator(),
            ModelBasedSelectionApplier(),
            {"target_column": "target", "max_features": 1},
        ),
        "binning": (
            GeneralBinningCalculator(),
            GeneralBinningApplier(),
            {"columns": ["x"], "n_bins": 3},
        ),
        "polynomial": (
            PolynomialFeaturesCalculator(),
            PolynomialFeaturesApplier(),
            {"columns": ["x", "z"]},
        ),
    }
    calculator, applier, config = calculators[kind]
    state = calculator.fit(frame, config)
    saved = pickle.dumps(state)
    expected = applier.apply(frame, state)
    for size in (len(frame), 1, 0):
        actual = applier.apply(frame.head(size), pickle.loads(saved))
        if engine == "pandas":
            pd.testing.assert_frame_equal(actual, expected.head(size))
        else:
            assert actual.equals(expected.head(size))
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["ModelBasedSelection", "GeneralBinning", "PolynomialFeatures"])
def test_fitted_pipeline_selection_and_binning_roundtrip(engine, node, tmp_path, monkeypatch):
    """Persisted model predictions must retain the same selected, binned or generated features."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    frame = _frame(engine)
    params = {
        "ModelBasedSelection": {"columns": ["x", "z"], "max_features": 1},
        "GeneralBinning": {"columns": ["x"], "n_bins": 3},
        "PolynomialFeatures": {"auto_detect": True},
    }[node]
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": "features", "transformer": node, "params": params}],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    request = frame.drop(columns="target") if engine == "pandas" else frame.drop("target")
    expected = pipeline.predict(request)
    save_pipeline(pipeline, tmp_path / "pipeline")
    loaded = load_pipeline(tmp_path / "pipeline")
    actual = predict_pipeline(request, loaded)
    pd.testing.assert_series_equal(actual.prediction, pd.Series(expected, name="prediction"))
