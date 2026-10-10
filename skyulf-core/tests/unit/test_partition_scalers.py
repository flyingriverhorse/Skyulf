"""Fitted affine scaling must remain immutable across independent pandas workers."""

import numpy as np
import pandas as pd
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError, require_capability
from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.pipeline import SkyulfPipeline


@pytest.fixture
def minmax_artifact(tmp_path, request):
    """Persist actual learned scalar coefficients, including a constant training feature."""
    extra = getattr(request, "param", {})
    data = pd.DataFrame(
        {
            "x": [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
            "z": [5.0] * 6,
            "target": [1.0, 5.0, 9.0, 13.0, 17.0, 21.0],
        }
    ).astype({"x": "Float64", "z": "Float64"})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x", "z"]}},
                {
                    "name": "range",
                    "transformer": "MinMaxScaler",
                    "params": {"columns": ["x", "z"], **extra},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=data.iloc[:4], test=data.iloc[4:]), target_column="target")
    save_pipeline(pipeline, tmp_path / "minmax")
    return load_pipeline(tmp_path / "minmax")


@pytest.mark.parametrize(
    "minmax_artifact",
    [{}, {"feature_range": [-2.0, 3.0]}, {"feature_range": (-1.0, 1.0)}],
    indirect=True,
)
def test_minmax_saved_state_has_partition_reorder_null_and_empty_parity(minmax_artifact):
    """Null fills and out-of-range values must reuse the training extrema and affine map."""
    evidence = require_partition_safe_pipeline(minmax_artifact)
    query = pd.DataFrame(
        {
            "x": pd.array([None, -6.0, 12.0, None], dtype="Float64"),
            "z": pd.array([None, 5.0, 5.0, None], dtype="Float64"),
        },
        index=[9, 2, 7, 1],
    )
    whole = predict_pipeline(query, minmax_artifact)
    split = pd.concat(
        [predict_pipeline(query.iloc[part], minmax_artifact) for part in ([0], [1, 2], [3])]
    )
    shuffled = predict_pipeline(query.iloc[[3, 1, 0, 2]], minmax_artifact)
    pd.testing.assert_frame_equal(whole, split)
    pd.testing.assert_frame_equal(whole.sort_index(), shuffled.sort_index())
    engineer = minmax_artifact.pipeline.feature_engineer
    empty = engineer.transform(query.iloc[:0], preserve_rows=True)
    features = engineer.transform(query, preserve_rows=True)
    state = engineer.fitted_steps[1]["artifact"]
    low, high = state["feature_range"]
    np.testing.assert_allclose(
        features["x"],
        [
            low + (high - low) / 2,
            low - (high - low),
            low + 2 * (high - low),
            low + (high - low) / 2,
        ],
    )
    np.testing.assert_allclose(features["z"], [low] * 4)
    assert empty.shape == (0, 2)
    assert empty.dtypes.to_dict() == features.dtypes.to_dict()
    assert require_partition_safe_pipeline(minmax_artifact) == evidence


@pytest.mark.parametrize(
    "change",
    [
        "callback",
        "subclass",
        "unknown_field",
        "native_scaler",
        "columns",
        "width",
        "nonfinite",
        "boolean",
        "range",
        "range_recipe",
        "zero_scale",
        "inverted_extrema",
        "registry",
    ],
)
def test_minmax_admission_rejects_custom_or_malformed_affine_state(
    minmax_artifact, monkeypatch, change
):
    """Reviewed affine arithmetic cannot inherit callbacks, ignored options or malformed state."""
    record = minmax_artifact.pipeline.feature_engineer.fitted_steps[1]
    state = record["artifact"]
    if change == "callback":
        record["applier"]._apply_pandas = lambda *args: None
    elif change == "subclass":
        record["applier"].__class__ = type("CustomMinMax", (type(record["applier"]),), {})
    elif change == "unknown_field":
        state["clip"] = True
    elif change == "native_scaler":
        state["scale"] = object()
    elif change == "columns":
        state["columns"] = ["x", "x"]
    elif change == "width":
        state["min"] = [0.0]
    elif change == "nonfinite":
        state["scale"][0] = np.inf
    elif change == "boolean":
        state["scale"][0] = True
    elif change == "range":
        state["feature_range"] = (1.0, 0.0)
    elif change == "range_recipe":
        record["params"]["feature_range"] = [-1.0, 1.0]
    elif change == "zero_scale":
        state["scale"][0] = 0.0
    elif change == "inverted_extrema":
        state["data_max"][0] = state["data_min"][0] - 1
    else:
        from skyulf.registry import NodeRegistry

        monkeypatch.setitem(NodeRegistry._calculators, "MinMaxScaler", object)
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(minmax_artifact)


def test_minmax_learned_affine_change_invalidates_evidence(minmax_artifact):
    """Workers must compare learned coefficients rather than trusting just a node label."""
    before = require_partition_safe_pipeline(minmax_artifact)
    minmax_artifact.pipeline.feature_engineer.fitted_steps[1]["artifact"]["min"][0] += 0.25
    assert require_partition_safe_pipeline(minmax_artifact).state_sha256 != before.state_sha256


def test_minmax_worker_admission_does_not_enable_native_spark(minmax_artifact):
    """Pandas worker support must not grant native Spark fitting or portable-state codecs."""
    from skyulf.core.portable_state import validate_state

    require_partition_safe_pipeline(minmax_artifact)
    with pytest.raises(UnsupportedExecutionError):
        require_capability("MinMaxScaler", "apply", "spark", config={})
    with pytest.raises(ValueError, match="Unsupported portable"):
        validate_state(
            "MinMaxScaler", minmax_artifact.pipeline.feature_engineer.fitted_steps[1]["artifact"]
        )
