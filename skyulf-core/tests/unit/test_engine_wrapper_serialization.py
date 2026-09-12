"""Regression coverage for dataframe wrapper restoration before its state exists."""

import copy
import pickle

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as assert_polars_frame_equal

from skyulf.core.serialization import JoblibModelSerializer
from skyulf.engines.registry import EngineRegistry


@pytest.fixture(params=["pandas", "polars"])
def native_frame(request):
    """Include nullable and typed values plus a non-default pandas index."""
    frame = pd.DataFrame(
        {
            "count": pd.array([1, None, 3], dtype="Int64"),
            "label": pd.array(["first", None, "last"], dtype="string"),
            "enabled": pd.array([True, None, False], dtype="boolean"),
            "category": pd.Categorical(["low", None, "high"]),
            "amount": [1.5, float("nan"), -2.5],
            "observed": pd.to_datetime(["2026-01-01", "NaT", "2026-01-03"], utc=True),
        },
        index=pd.Index([7, 2, 9], name="row_id"),
    )
    if request.param == "polars":
        return pl.from_pandas(frame, nan_to_null=False)
    return frame


def assert_native_frame_equal(actual, expected):
    """Check native schema, row order, nulls, values, and pandas index fidelity."""
    assert type(actual) is type(expected)
    if isinstance(expected, pd.DataFrame):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    else:
        assert_polars_frame_equal(actual, expected, check_exact=True)


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
@pytest.mark.parametrize("protocol", [4, 5])
def test_pickle_restores_wrapper_and_native_delegation(native_frame, nested, protocol):
    """Pickle must restore usable wrappers, including inside model state containers."""
    wrapped = EngineRegistry.wrap(native_frame)
    payload = {"frames": [(wrapped,)]} if nested else wrapped

    loaded = pickle.loads(pickle.dumps(payload, protocol=protocol))
    restored = loaded["frames"][0][0] if nested else loaded

    assert type(restored) is type(wrapped)
    assert restored is not wrapped
    assert_native_frame_equal(restored.to_native(), native_frame)
    assert_native_frame_equal(restored.head(2), native_frame.head(2))
    assert restored.columns == list(native_frame.columns)


@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_joblib_serializer_restores_wrapper(native_frame, nested, tmp_path):
    """Core's serializer must load dataframe wrappers it successfully persisted."""
    wrapped = EngineRegistry.wrap(native_frame)
    payload = {"frames": [(wrapped,)]} if nested else wrapped
    serializer = JoblibModelSerializer()
    path = tmp_path / "wrapper.joblib"

    serializer.dump(payload, path)
    assert path.is_file()
    loaded = serializer.load(path)
    restored = loaded["frames"][0][0] if nested else loaded

    assert type(restored) is type(wrapped)
    assert_native_frame_equal(restored.to_native(), native_frame)
    assert_native_frame_equal(restored.head(2), native_frame.head(2))
    with pytest.raises(AttributeError):
        _ = restored.missing_wrapper_attribute


@pytest.mark.parametrize("name", ["_df", "__setstate__", "head"])
def test_uninitialized_wrapper_attribute_lookup_raises_attribute_error(native_frame, name):
    """State probes during restoration must terminate normally before _df exists."""
    wrapper_type = type(EngineRegistry.wrap(native_frame))
    uninitialized = object.__new__(wrapper_type)

    with pytest.raises(AttributeError):
        getattr(uninitialized, name)
    assert not hasattr(uninitialized, name)


def test_shallow_copy_restores_wrapper(native_frame):
    """The copy protocol must reconstruct a wrapper while sharing its native frame."""
    wrapped = EngineRegistry.wrap(native_frame)

    cloned = copy.copy(wrapped)

    assert type(cloned) is type(wrapped)
    assert cloned is not wrapped
    assert cloned.to_native() is native_frame
    assert cloned.columns == list(native_frame.columns)
