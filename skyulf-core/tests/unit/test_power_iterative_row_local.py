"""Invalid power values and entirely missing rows must not change their neighbours."""

import logging
import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")

from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.imputation.iterative import (
    IterativeImputerApplier,
    IterativeImputerCalculator,
)
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.preprocessing.transformations.power import (
    PowerTransformerApplier,
    PowerTransformerCalculator,
)


def _native(frame, engine):
    """Keep positional comparison separate from duplicate pandas row labels."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _values(frame):
    """Compare nulls as numeric NaNs without discarding native frame assertions."""
    return frame.to_numpy()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [False, True])
def test_power_invalid_values_are_local_and_counted(engine, method, standardize, caplog):
    """A bad selected value excludes its row while valid neighbours keep their learned transform."""
    training = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0], "z": [2.0, 5.0, 7.0, 11.0]})
    state = PowerTransformerCalculator().fit(
        _native(training, engine),
        {"columns": ["x", "z"], "method": method, "standardize": standardize},
    )
    state = pickle.loads(pickle.dumps(state))
    saved = artifact_digest(state)
    bad = -1.0 if method == "box-cox" else np.inf
    sample = pd.DataFrame(
        {"x": [2.0, bad, np.nan, 2.0], "z": [bad, 5.0, np.nan, 5.0]}, index=[4, 4, 1, 4]
    )
    native = _native(sample, engine)
    before = deepcopy(native)
    applier = PowerTransformerApplier()
    with caplog.at_level(logging.WARNING):
        whole = applier.apply(native, state)
    assert "excluded 2 invalid row" in caplog.text
    valid = applier.apply(_native(pd.DataFrame({"x": [2.0], "z": [5.0]}), engine), state)
    np.testing.assert_array_equal(_values(whole), np.vstack([[np.nan, np.nan], _values(valid)]))
    for positions in ([0], [1], [2], [3], [3, 2, 0, 1], []):
        batch = native.iloc[positions] if engine == "pandas" else native[positions]
        output = applier.apply(batch, state)
        kept = [index for index in positions if index in (2, 3)]
        expected = np.array(
            [[np.nan, np.nan] if index == 2 else _values(valid)[0] for index in kept]
        ).reshape(-1, 2)
        np.testing.assert_array_equal(_values(output), expected)
    if engine == "pandas":
        pd.testing.assert_frame_equal(native, before)
        assert whole.index.tolist() == [1, 4]
        assert whole.dtypes.tolist() == [np.dtype("float64")] * 2
    else:
        assert native.equals(before) and whole.dtypes == [pl.Float64, pl.Float64]
    capability = get_inference_capability("PowerTransformer", {}, state, engine=engine)
    assert capability is not None and (capability.context, capability.row_effect) == (
        "row",
        "filter",
    )
    assert artifact_digest(state) == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("estimator", ["BayesianRidge", "DecisionTree", "ExtraTrees", "KNeighbors"])
@pytest.mark.parametrize("rounds", [0, 10])
def test_iterative_all_missing_rows_match_mixed_requests(engine, estimator, rounds, monkeypatch):
    """Saved iterative rounds must run for all-null requests without fitting or state mutation."""
    training = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0, np.nan, 4.0, 5.0], "b": [0.0, 2.0, np.nan, 6.0, 8.0, 10.0]}
    )
    state = IterativeImputerCalculator().fit(
        _native(training, engine),
        {"columns": ["a", "b"], "estimator": estimator, "max_iter": rounds},
    )
    state = pickle.loads(pickle.dumps(state))
    saved = artifact_digest(state)
    imputer = state["imputer_object"]

    def forbidden(*args, **kwargs):
        """Saved inference must not learn statistics or refit any estimator."""
        raise AssertionError("unexpected fit")

    monkeypatch.setattr(type(imputer), "fit", forbidden)
    sample = pd.DataFrame(
        {"a": [np.nan, 2.0, np.nan, 7.0], "b": [np.nan, np.nan, 3.0, 14.0]}, index=[7, 2, 2, 0]
    )
    native = _native(sample, engine)
    applier = IterativeImputerApplier()
    whole = applier.apply(native, state)
    reference = imputer.transform(sample.to_numpy())
    np.testing.assert_allclose(_values(whole), reference, rtol=1e-9, atol=1e-12)
    for positions in ([0], [0, 0], [1, 2], [3, 2, 1, 0], []):
        batch = native.iloc[positions] if engine == "pandas" else native[positions]
        output = applier.apply(batch, state)
        np.testing.assert_allclose(
            _values(output), _values(whole)[positions], rtol=1e-9, atol=1e-12
        )
    capability = get_inference_capability("IterativeImputer", {}, state, engine=engine)
    assert capability is not None and capability.context == "row"
    assert artifact_digest(state) == saved


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
@pytest.mark.parametrize("standardize", [False, True])
def test_power_extreme_and_missing_values_keep_valid_negative_or_positive_rows(
    engine, method, standardize, caplog
):
    """Overflow and nonfinite inputs must not abort valid values or turn missing data into zero."""
    state = {
        "type": "power_transformer",
        "columns": ["x"],
        "method": method,
        "standardize": standardize,
        "lambdas": [100.0],
        "scaler_params": {"mean": [1.0], "scale": [2.0]} if standardize else {},
    }
    good = 0.5 if method == "box-cox" else -2.0
    sample = pd.DataFrame({"x": [good, 1e308, np.nan, np.inf, -np.inf, 0.0]})
    applier = PowerTransformerApplier()
    with caplog.at_level(logging.WARNING):
        output = applier.apply(_native(sample, engine), state)["x"].to_numpy()
    assert (
        "excluded 4 invalid row" in caplog.text
        if method == "box-cox"
        else "excluded 3 invalid row" in caplog.text
    )
    assert len(output) == (2 if method == "box-cox" else 3)
    singleton = applier.apply(_native(sample.iloc[:1], engine), state)["x"].to_numpy()
    np.testing.assert_array_equal(output[:1], singleton)
    assert np.isnan(output[1])
    assert np.isfinite(output[[0, -1]]).all() if method == "yeo-johnson" else np.isfinite(output[0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["float32", "Float32", "Int64"])
def test_iterative_active_rounds_preserve_dtype_and_null_partition_contract(engine, dtype):
    """The sentinel must not widen float32 or change nullable conversion relative to mixed rows."""
    training = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0, np.nan, 4.0, 5.0], "b": [0.0, 2.0, np.nan, 6.0, 8.0, 10.0]}
    )
    state = IterativeImputerCalculator().fit(training, {"columns": ["a", "b"]})
    sample = pd.DataFrame({"a": [None, 2, 5], "b": [None, None, 10]}).astype(dtype)
    native = _native(sample, engine)
    applier = IterativeImputerApplier()
    whole = applier.apply(native, state)
    empty = native.head(0)
    single = applier.apply(native.head(1), state)
    assert (
        single.dtypes == whole.dtypes if engine == "polars" else single.dtypes.equals(whole.dtypes)
    )
    np.testing.assert_allclose(_values(single), _values(whole)[:1], rtol=1e-9, atol=1e-12)
    assert applier.apply(empty, state).shape == (0, 2)


def test_iterative_callable_neighbor_weights_have_no_row_contract():
    """An exact KNN class cannot make a batch-dependent weight callback safe."""
    training = pd.DataFrame(
        {"a": [0.0, 1.0, 2.0, np.nan, 4.0, 5.0], "b": [0.0, 2.0, np.nan, 6.0, 8.0, 10.0]}
    )
    state = IterativeImputerCalculator().fit(
        training, {"columns": ["a", "b"], "estimator": "KNeighbors"}
    )
    state["imputer_object"].imputation_sequence_[0].estimator.weights = lambda distances: distances
    assert get_inference_capability("IterativeImputer", {}, dict(state), engine="pandas") is None
