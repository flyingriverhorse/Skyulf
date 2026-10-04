"""New dummy fits exclude float NaN without changing saved-artifact replay."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from skyulf.engines import PolarsEngine
from skyulf.preprocessing.encoding.dummy import DummyEncoderApplier, DummyEncoderCalculator


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
@pytest.mark.parametrize("wrapped", [False, True], ids=["native", "wrapped"])
@pytest.mark.parametrize("drop_first", [False, True])
@pytest.mark.parametrize(
    "values, expected_categories",
    [([1.0, np.nan, None, 2.0], ["1", "2"]), ([np.nan, None, np.nan], [])],
    ids=["mixed-missing", "all-missing"],
)
def test_dummy_fit_excludes_native_float_nan(
    dtype, wrapped, drop_first, values, expected_categories
):
    """Native float missing values must not add a category absent from pandas fits."""
    native = pl.DataFrame({"feature": pl.Series(values, dtype=dtype)})
    original = native.clone()
    X = PolarsEngine.wrap(native) if wrapped else native
    config = {"columns": ["feature"], "drop_first": drop_first}
    calculator, applier = DummyEncoderCalculator(), DummyEncoderApplier()
    expected_artifact = calculator.fit(native.to_pandas(), config)

    artifact = calculator.fit(X, config)
    restored = pickle.loads(pickle.dumps(artifact))
    output = applier.apply(X, restored)

    assert artifact["categories"]["feature"] == expected_categories
    assert artifact == expected_artifact
    converted = pl.from_pandas(native.to_pandas(), nan_to_null=True)
    assert calculator.fit(converted, config) == artifact
    expected = applier.apply(native.to_pandas(), expected_artifact)
    pd.testing.assert_frame_equal(output.to_pandas(), expected, check_column_type=False)
    heldout = pl.DataFrame({"feature": pl.Series([np.nan, None, 7.0], dtype=dtype)})
    actual = applier.apply(PolarsEngine.wrap(heldout) if wrapped else heldout, artifact)
    assert actual.shape == (3, expected.shape[1])
    assert np.count_nonzero(actual.to_numpy()) == 0
    assert_frame_equal(native, original)


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64])
@pytest.mark.parametrize("version", [None, 1], ids=["unversioned", "version-one"])
def test_dummy_saved_nan_category_keeps_legacy_replay(dtype, version):
    """Existing models must retain the indicator their stored NaN category trained on."""
    artifact = {
        "type": "dummy_encoder",
        "columns": ["feature"],
        "categories": {"feature": ["1", "NaN"]},
        "drop_first": False,
    }
    if version is not None:
        artifact["category_key_version"] = version
    restored = pickle.loads(pickle.dumps(artifact))
    X = pl.DataFrame({"feature": pl.Series([1.0, np.nan, None], dtype=dtype)})

    output = DummyEncoderApplier().apply(X, restored)

    assert output.to_dict(as_series=False) == {"feature_1": [1, 0, 0], "feature_NaN": [0, 1, 0]}


@pytest.mark.parametrize("wrapped", [False, True], ids=["native", "wrapped"])
def test_dummy_literal_nan_strings_are_still_categories(wrapped):
    """Missing-value normalization must leave literal string categories untouched."""
    native = pl.DataFrame({"feature": ["NaN", "nan", None]})
    X = PolarsEngine.wrap(native) if wrapped else native
    artifact = DummyEncoderCalculator().fit(X, {"columns": ["feature"]})

    output = DummyEncoderApplier().apply(X, artifact)

    assert artifact["categories"]["feature"] == ["NaN", "nan"]
    assert output.to_dict(as_series=False) == {"feature_NaN": [1, 0, 0], "feature_nan": [0, 1, 0]}
