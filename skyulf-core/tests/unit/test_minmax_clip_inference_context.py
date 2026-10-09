"""Existing affine and fixed-bound appliers expose their reviewed local Polars context."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from skyulf.core.capabilities import UnsupportedExecutionError, require_capability
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint64"])
@pytest.mark.parametrize(
    "node,config",
    [
        ("MinMaxScaler", {"columns": ["x"], "feature_range": [-2, 7]}),
        ("ClipValues", {"bounds": {"x": {"lower": 1, "upper": 7}}}),
        ("ClipValues", {"bounds": {"x": {"lower": 0.5, "upper": 7.5}}}),
    ],
)
def test_local_context_reuses_fitted_bounds_and_affine_state(
    engine, dtype, node, config, monkeypatch
):
    """Local declarations must describe unchanged saved arithmetic across request boundaries."""
    training = pd.DataFrame({"x": pd.Series([0, 1, 2, 3, 10], dtype=dtype), "keep": range(5)})
    floating = dtype.startswith("float")
    values = [None, np.nan, np.inf, -0.0, 6.0, 100.0] if floating else [0, 6, 100]
    sample = pd.DataFrame({"x": pd.Series(values, dtype=dtype), "keep": range(len(values))})
    if engine == "polars":
        training, sample = pl.from_pandas(training), pl.from_pandas(sample)
    calculator = NodeRegistry.get_calculator(node)
    state = calculator().fit(training, config)
    saved = pickle.dumps(state)
    before = pickle.dumps(sample)

    def forbidden(*args, **kwargs):
        """Replaying saved parameters must never learn statistics from scoring rows."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(calculator, "fit", forbidden)
    capability = get_inference_capability(node, config, state, engine=engine)
    assert capability is not None
    assert (capability.context, capability.row_effect, capability.execution_kind) == (
        "row",
        "preserve",
        "local" if engine == "polars" else "python_batch",
    )
    applier = NodeRegistry.get_applier(node)()
    full = applier.apply(sample, state)
    if isinstance(sample, pl.DataFrame):
        pieces = [
            applier.apply(sample.slice(i, 1).lazy(), state).collect() for i in range(len(sample))
        ]
        assert_frame_equal(pl.concat(pieces), full, check_exact=True)
        assert_frame_equal(applier.apply(sample.lazy(), state).collect(), full, check_exact=True)
        assert_frame_equal(applier.apply(sample.reverse(), state), full.reverse(), check_exact=True)
        assert_frame_equal(applier.apply(sample.head(0).lazy(), state).collect(), full.head(0))
    else:
        pieces = [applier.apply(sample.iloc[i : i + 1], state) for i in range(len(sample))]
        pd.testing.assert_frame_equal(pd.concat(pieces), full, check_exact=True)
        pd.testing.assert_frame_equal(
            applier.apply(sample.iloc[::-1], state), full.iloc[::-1], check_exact=True
        )
        pd.testing.assert_frame_equal(applier.apply(sample.head(0), state), full.head(0))
    assert pickle.dumps(sample) == before
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize(
    "node,config,bad_config",
    [
        ("MinMaxScaler", {"columns": ["x"]}, {"columns": ["x"], "feature_range": [-1, 1]}),
        ("ClipValues", {"bounds": {"x": {"lower": 0}}}, {"bounds": {"x": {"lower": 1}}}),
    ],
)
def test_local_metadata_keeps_validation_and_worker_boundaries(
    node, config, bad_config, monkeypatch
):
    """Adding local metadata must retain pure inspection, abstention and execution-kind admission."""
    calculator = NodeRegistry.get_calculator(node)
    applier = NodeRegistry.get_applier(node)
    state = calculator().fit(pl.DataFrame({"x": [0.0, 1.0, 2.0]}), config)
    saved = pickle.dumps(state)

    def forbidden(*args, **kwargs):
        """Metadata inspection must not invoke a transform or fit."""
        raise AssertionError("Unexpected execution")

    monkeypatch.setattr(calculator, "fit", forbidden)
    monkeypatch.setattr(applier, "apply", forbidden)
    assert get_inference_capability(node, config, state, engine="polars") is not None
    assert get_inference_capability(node, bad_config, state, engine="polars") is None
    assert (
        get_inference_capability(node, config, {**state, "type": "invalid"}, engine="polars")
        is None
    )
    overridden = applier()
    overridden.apply = forbidden
    assert (
        get_inference_capability(node, config, state, engine="polars", applier=overridden) is None
    )
    for kind in ("python_batch", "native"):
        with pytest.raises(UnsupportedExecutionError):
            require_capability(node, "apply", "polars", config=config, execution_kind=kind)
    assert pickle.dumps(state) == saved
