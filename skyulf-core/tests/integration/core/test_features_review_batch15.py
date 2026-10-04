"""Feature nodes preserve group identity and report unsupported input without deprecations."""

import logging
import os
import pickle
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import PolynomialFeatures as SkPolynomialFeatures

import skyulf
from skyulf.preprocessing.feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
    PolynomialFeaturesApplier,
    PolynomialFeaturesCalculator,
)
from skyulf.preprocessing.transformations.power import (
    PowerTransformerApplier,
    PowerTransformerCalculator,
)


def test_imports_use_selected_source_tree():
    """Local snapshot verification must import its own production sources."""
    expected = os.environ.get("SKYULF_EXPECTED_SOURCE_ROOT")
    if expected is not None:
        assert Path(skyulf.__file__).resolve().parent == Path(expected).resolve() / "skyulf"
    assert Path(skyulf.__file__).is_file()


def _native(frame, engine):
    """Preserve engine-specific scalar types at the public API boundary."""
    return pl.from_pandas(frame) if engine == "polars" else frame.copy(deep=True)


def _pandas(frame):
    """Compare native results without changing the execution engine."""
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


def _group_artifact(frame):
    """Learn real training aggregates and replay their serialized artifact."""
    artifact = FeatureGenerationCalculator().fit(
        frame,
        {
            "operations": [
                {
                    "operation_type": "group_agg",
                    "method": "mean",
                    "input_columns": ["g"],
                    "secondary_columns": ["v"],
                    "output_column": "group_mean",
                }
            ]
        },
    )
    return pickle.loads(pickle.dumps(artifact))


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "training,query,expected",
    [
        ([1, 2], ["1", "2", "absent"], [np.nan, np.nan, np.nan]),
        (["1", "2"], [1, 2, 3], [np.nan, np.nan, np.nan]),
        ([True, False], [1, 0, 2], [np.nan, np.nan, np.nan]),
        ([1, 0], [True, False, True], [np.nan, np.nan, np.nan]),
        ([1.25, 2.0], [1, 2, 3], [np.nan, 20.0, np.nan]),
        ([1, 2], [1.0, 2.0, 3.0], [10.0, 20.0, np.nan]),
        (
            [9007199254740993, 9007199254740994],
            [9007199254740994, 9007199254740993, 9007199254740995],
            [20.0, 10.0, np.nan],
        ),
        (["a", None], ["a", None, "b"], [10.0, 20.0, np.nan]),
        ([1.0, np.nan], ["1", None, "absent"], [np.nan, 20.0, np.nan]),
    ],
    ids=[
        "integer-string",
        "string-integer",
        "boolean-integer",
        "integer-boolean",
        "fractional-integer",
        "exact-numeric-promotion",
        "large-integers",
        "string-null",
        "numeric-string-null",
    ],
)
def test_group_aggregation_preserves_key_identity(
    fit_engine, apply_engine, training, query, expected
):
    """Different key types must not alias while exact numeric and null groups still replay."""
    fitted = _group_artifact(_native(pd.DataFrame({"g": training, "v": [10.0, 20.0]}), fit_engine))
    inference = pd.DataFrame({"g": query, "row_id": [9007199254740993, 2, 3]}, index=[9, 3, 7])
    original = inference.copy(deep=True)
    result = _pandas(FeatureGenerationApplier().apply(_native(inference, apply_engine), fitted))
    np.testing.assert_allclose(result["group_mean"], expected, equal_nan=True)
    pd.testing.assert_frame_equal(
        result[inference.columns].reset_index(drop=True), original.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(inference, original)
    if apply_engine == "pandas":
        assert result.index.tolist() == [9, 3, 7]
    assert result["row_id"].tolist() == inference["row_id"].tolist()


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
def test_group_aggregation_preserves_categorical_keys(fit_engine, apply_engine):
    """Categorical physical encodings must not replace their original group values."""
    training = pd.DataFrame({"g": pd.Categorical(["b", "a", "b"]), "v": [10, 20, 30]})
    fitted = _group_artifact(_native(training, fit_engine))
    query = pd.DataFrame({"g": pd.Categorical(["a", "b", "new"], categories=["new", "a", "b"])})
    result = _pandas(FeatureGenerationApplier().apply(_native(query, apply_engine), fitted))
    np.testing.assert_allclose(result["group_mean"], [20.0, 20.0, np.nan], equal_nan=True)
    assert result["g"].tolist() == ["a", "b", "new"]


@pytest.fixture(scope="module")
def h3_nodes():
    """Skip only H3 cases before importing the optional node implementation."""
    h3 = pytest.importorskip("h3", reason="H3 is an optional geo dependency")
    from skyulf.preprocessing.geo.h3_index import (  # noqa: PLC0415 - optional dependency checked first
        H3IndexApplier,
        H3IndexCalculator,
    )

    return h3, H3IndexCalculator, H3IndexApplier


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
def test_h3_rejects_out_of_range_coordinates(h3_nodes, fit_engine, apply_engine):
    """Invalid latitude or longitude must stay missing instead of wrapping to a real cell."""
    h3, calculator, applier = h3_nodes
    points = [
        (90.0, 180.0),
        (-90.0, -180.0),
        (0.0, 0.0),
        (90.00001, 0.0),
        (-90.00001, 0.0),
        (0.0, 180.00001),
        (0.0, -180.00001),
        (float("inf"), 0.0),
        (0.0, -float("inf")),
        (float("nan"), 0.0),
        (0.0, float("nan")),
        (None, None),
    ]
    frame = pd.DataFrame(points, columns=["lat", "lon"])
    frame["row_id"] = pd.Series(range(len(frame)), dtype="Int64")
    frame.loc[0, "row_id"] = 9007199254740993
    native = _native(frame, apply_engine)
    fitted = calculator().fit(_native(frame, fit_engine), {"lat_col": "lat", "lon_col": "lon"})
    result = _pandas(applier().apply(native, pickle.loads(pickle.dumps(fitted))))
    expected = [h3.latlng_to_cell(lat, lon, 9) for lat, lon in points[:3]] + [None] * 9
    assert result["h3_index"].tolist() == expected
    assert result["row_id"].tolist() == frame["row_id"].tolist()
    assert "h3_index" not in native.columns


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("all_invalid", [False, True])
def test_box_cox_warns_for_excluded_columns(engine, all_invalid, caplog):
    """Fit must identify excluded nonpositive columns while retaining the documented policy."""
    data = {"has_zero": [0.0, 2.0, 4.0], "has_negative": [-1.0, 2.0, 4.0]}
    if not all_invalid:
        data["positive"] = [1.0, 2.0, 4.0]
    frame = _native(pd.DataFrame(data), engine)
    with caplog.at_level(logging.WARNING, logger="skyulf.preprocessing.transformations.power"):
        artifact = PowerTransformerCalculator().fit(
            frame, {"method": "box-cox", "columns": list(data)}
        )
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "skyulf.preprocessing.transformations.power"
    ]
    assert len(messages) == 1
    assert "Box-Cox" in messages[0]
    assert "has_zero" in messages[0] and "has_negative" in messages[0]
    assert "positive" not in messages[0].replace("non-positive", "")
    assert artifact.get("columns", []) == ([] if all_invalid else ["positive"])
    result = _pandas(PowerTransformerApplier().apply(frame, artifact))
    pd.testing.assert_frame_equal(
        result[["has_zero", "has_negative"]], pd.DataFrame(data)[["has_zero", "has_negative"]]
    )
    assert result.columns.tolist() == list(data)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["box-cox", "yeo-johnson"])
def test_supported_power_columns_do_not_warn(engine, method, caplog):
    """Valid Box-Cox and signed Yeo-Johnson inputs must retain warning-free fitting."""
    values = [1.0, 2.0, 4.0] if method == "box-cox" else [-1.0, 0.0, 4.0]
    with caplog.at_level(logging.WARNING, logger="skyulf.preprocessing.transformations.power"):
        artifact = PowerTransformerCalculator().fit(
            _native(pd.DataFrame({"x": values}), engine), {"method": method, "columns": ["x"]}
        )
    assert artifact["columns"] == ["x"]
    assert not [
        record
        for record in caplog.records
        if record.name == "skyulf.preprocessing.transformations.power"
    ]


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
@pytest.mark.parametrize("include_bias", [False, True])
def test_polynomial_append_emits_no_deprecation(fit_engine, apply_engine, include_bias):
    """Appending equal-height generated features must remain compatible with supported Polars."""
    frame = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "row_id": [9007199254740993, 2, 3]},
        index=[9, 3, 7],
    )
    native = _native(frame, apply_engine)
    config = {"columns": ["a", "b"], "degree": 2, "include_bias": include_bias}
    fitted = PolynomialFeaturesCalculator().fit(_native(frame, fit_engine), config)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = _pandas(
            PolynomialFeaturesApplier().apply(native, pickle.loads(pickle.dumps(fitted)))
        )
    oracle = SkPolynomialFeatures(degree=2, include_bias=include_bias)
    expected = oracle.fit_transform(frame[["a", "b"]])
    keep = np.sum(oracle.powers_, axis=1) != 1
    names = [
        "poly_" + name.replace(" ", "_").replace("^", "_pow_")
        for name in oracle.get_feature_names_out(["a", "b"])[keep]
    ]
    np.testing.assert_allclose(result[names], expected[:, keep])
    pd.testing.assert_frame_equal(
        result[frame.columns].reset_index(drop=True), frame.reset_index(drop=True)
    )
    assert list(native.columns) == list(frame.columns)
    assert result.columns.tolist() == list(frame.columns) + names


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "keys",
    [
        [date(2026, 1, 1), date(2026, 1, 2)],
        [pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-01-02", tz="UTC")],
        [b"first", b"second"],
    ],
    ids=["date", "timezone", "binary"],
)
def test_group_aggregation_preserves_other_scalar_keys(fit_engine, apply_engine, keys):
    """Identity checks must retain existing temporal and binary group lookup support."""
    fitted = _group_artifact(_native(pd.DataFrame({"g": keys, "v": [10.0, 20.0]}), fit_engine))
    query = _native(pd.DataFrame({"g": keys[::-1]}), apply_engine)
    result = _pandas(FeatureGenerationApplier().apply(query, fitted))
    np.testing.assert_allclose(result["group_mean"], [20.0, 10.0])
    assert list(query.columns) == ["g"]


@pytest.mark.parametrize(
    "query,expected", [(["1", "other"], [20.0, np.nan]), ([1, 2], [10.0, np.nan])]
)
def test_mixed_training_keys_do_not_collapse_on_polars_replay(query, expected):
    """A pandas artifact with separate numeric and string keys must replay without collisions."""
    fitted = _group_artifact(pd.DataFrame({"g": [1, "1"], "v": [10.0, 20.0]}))
    result = FeatureGenerationApplier().apply(pl.DataFrame({"g": query}), fitted)
    np.testing.assert_allclose(result["group_mean"].to_numpy(), expected, equal_nan=True)
    assert result["g"].to_list() == query
