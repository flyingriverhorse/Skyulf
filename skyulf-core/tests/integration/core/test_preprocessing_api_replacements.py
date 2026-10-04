"""Public replacement and ordinal APIs accept portable scalar/list inputs."""

import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.cleaning.value_replacement import (
    ValueReplacementApplier,
    ValueReplacementCalculator,
)
from skyulf.preprocessing.encoding.ordinal import (
    OrdinalEncoderApplier,
    OrdinalEncoderCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"to_replace": [1, 2], "value": 0}, [0, 0, 3]),
        ({"to_replace": [1, 2], "value": [10, 20]}, [10, 20, 3]),
        ({"to_replace": ["1", "2"], "value": [10, 20]}, [10, 20, 3]),
        ({"to_replace": ["invalid", "2"], "value": [10, 20]}, [1, 20, 3]),
        ({"to_replace": "1", "value": 10}, [10, 2, 3]),
        ({"to_replace": {"1": 10}, "value": 999}, [10, 2, 3]),
        ({"to_replace": [], "value": 0}, [1, 2, 3]),
        ({"to_replace": "invalid", "value": 0}, [1, 2, 3]),
        ({"mapping": {"3": 30}, "to_replace": [1, 2], "value": 0}, [1, 2, 30]),
    ],
)
def test_replacement_list_and_scalar_configs_match_mapping(engine, config, expected):
    """JSON-shaped replacement configs must give the same values on both engines."""
    create = pd.DataFrame if engine == "pandas" else pl.DataFrame
    frame = create({"value": [1, 2, 3], "untouched": [4, 5, 6]})
    artifact = ValueReplacementCalculator().fit(frame, {"columns": ["value"], **config})
    restored = pickle.loads(pickle.dumps(artifact))
    result = ValueReplacementApplier().apply(frame, restored)
    assert result["value"].to_list() == expected
    assert result["untouched"].to_list() == [4, 5, 6]
    assert frame["value"].to_list() == [1, 2, 3]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_replacement_rejects_unequal_list_lengths(engine):
    """Unequal replacement lists must fail instead of silently truncating a rule."""
    create = pd.DataFrame if engine == "pandas" else pl.DataFrame
    frame = create({"value": [1, 2, 3]})
    artifact = ValueReplacementCalculator().fit(
        frame, {"columns": ["value"], "to_replace": [1, 2], "value": [10]}
    )
    with pytest.raises(ValueError, match="length"):
        ValueReplacementApplier().apply(frame, artifact)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("source", "keys", "expected"),
    [
        ([True, False, True], ["true"], [False, False, False]),
        ([1.5, 2.5, 3.5], ["1.5", "2.5"], [0.0, 0.0, 3.5]),
    ],
)
def test_replacement_lists_coerce_boolean_and_float_keys(engine, source, keys, expected):
    """The list API must reuse the established mapping-key conversion contract."""
    create = pd.DataFrame if engine == "pandas" else pl.DataFrame
    frame = create({"value": source})
    replacement = False if isinstance(source[0], bool) else 0.0
    artifact = ValueReplacementCalculator().fit(
        frame, {"columns": ["value"], "to_replace": keys, "value": replacement}
    )
    result = ValueReplacementApplier().apply(frame, artifact)
    assert result["value"].to_list() == expected


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("container", [list, tuple, np.asarray])
@pytest.mark.parametrize("labels", [["a", "b", "a"], []])
def test_ordinal_target_accepts_sequence_without_using_index_method(engine, container, labels):
    """List.index is a method, not a row index, including for empty holdouts."""
    create = pd.DataFrame if engine == "pandas" else pl.DataFrame
    train = create({"value": [1, 2, 3]})
    target = pd.Series(["a", "b", "a"]) if engine == "pandas" else pl.Series(["a", "b", "a"])
    artifact = OrdinalEncoderCalculator().fit((train, target), {"columns": []})
    restored = pickle.loads(pickle.dumps(artifact))
    score = create({"value": list(range(len(labels)))})
    result, encoded = OrdinalEncoderApplier().apply((score, container(labels)), restored)
    assert result.shape == (len(labels), 1)
    assert encoded.to_list() == ([0.0, 1.0, 0.0] if labels else [])


def test_ordinal_target_preserves_real_pandas_series_index():
    """Accepting sequence targets must not reset an explicitly indexed Series."""
    train = pd.DataFrame({"value": [1, 2, 3]}, index=[7, 2, 9])
    target = pd.Series(["a", "b", "a"], index=train.index, name="label")
    artifact = OrdinalEncoderCalculator().fit((train, target), {"columns": []})
    _, encoded = OrdinalEncoderApplier().apply((train, target), artifact)
    assert encoded.index.equals(target.index)
    assert encoded.name == target.name
    assert encoded.to_list() == [0.0, 1.0, 0.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("replay", ["json", "pickle"])
@pytest.mark.parametrize("keys", [[True, 1], [1, True], [False, 0], [0, False]])
@pytest.mark.parametrize("source", [[0, 1, 2], [0.0, 1.0, 2.0], [False, True, False]])
def test_replacement_lists_keep_boolean_and_numeric_key_identity(engine, replay, keys, source):
    """Distinct boolean and numeric rules must survive normalization and artifact replay."""
    replacements = [True, False] if isinstance(source[0], bool) else [10, 20]
    reference: pd.Series[Any] = pd.Series(source)
    expected = reference.replace(keys, replacements)
    create = pd.DataFrame if engine == "pandas" else pl.DataFrame
    frame = create({"value": source})
    artifact = ValueReplacementCalculator().fit(
        frame, {"columns": ["value"], "to_replace": keys, "value": replacements}
    )
    restored = (
        json.loads(json.dumps(artifact))
        if replay == "json"
        else pickle.loads(pickle.dumps(artifact))
    )
    result = ValueReplacementApplier().apply(frame, restored)
    assert result["value"].to_list() == expected.to_list()
    assert result["value"].dtype == frame["value"].dtype


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("values", [[10], [10, 20], np.array([10, 20]), {"value": 9}])
def test_replacement_scalar_key_rejects_list_values(engine, values):
    """Invalid scalar rules must not distribute replacement values over matching rows."""
    create = pd.DataFrame if engine == "pandas" else pl.DataFrame
    frame = create({"value": [1, 1, 2]})
    artifact = ValueReplacementCalculator().fit(
        frame, {"columns": ["value"], "to_replace": 1, "value": values}
    )
    with pytest.raises(TypeError, match="scalar"):
        ValueReplacementApplier().apply(frame, artifact)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("keys", [{1, 2}, frozenset({1, 2})])
def test_replacement_set_keys_keep_scalar_replacement_support(engine, keys):
    """Previously accepted set keys must remain usable after list normalization."""
    create = pd.DataFrame if engine == "pandas" else pl.DataFrame
    frame = create({"value": [1, 2, 3]})
    artifact = ValueReplacementCalculator().fit(
        frame, {"columns": ["value"], "to_replace": keys, "value": 9}
    )
    restored = pickle.loads(pickle.dumps(artifact))
    result = ValueReplacementApplier().apply(frame, restored)
    assert result["value"].to_list() == [9, 9, 3]


def test_replacement_pandas_object_lists_keep_native_duplicate_semantics():
    """A mixed object column retains pandas's treatment of overlapping replacement keys."""
    frame = pd.DataFrame({"value": pd.Series([True, 1, False, 0, "1"], dtype=object)})
    keys, values = [True, 1, "1"], [10, 20, 30]
    with pd.option_context("future.no_silent_downcasting", True):
        expected = frame["value"].replace(keys, values).infer_objects()
    artifact = ValueReplacementCalculator().fit(
        frame, {"columns": ["value"], "to_replace": keys, "value": values}
    )
    result = ValueReplacementApplier().apply(frame, artifact)
    pd.testing.assert_series_equal(result["value"], expected)


@pytest.mark.parametrize(
    ("source", "dtype", "key"),
    [([0, 1, 2], "Int64", 0), ([0, 1, 2], "Float64", 0), ([False, True], "boolean", True)],
)
def test_replacement_scalar_null_retains_pandas_nullable_dtype(source, dtype, key):
    """Scalar null replacement must preserve nullable types during artifact replay."""
    frame = pd.DataFrame({"value": pd.Series(source, dtype=dtype)})
    expected = frame["value"].replace(key, None)
    artifact = ValueReplacementCalculator().fit(
        frame, {"columns": ["value"], "to_replace": key, "value": None}
    )
    result = ValueReplacementApplier().apply(frame, json.loads(json.dumps(artifact)))
    pd.testing.assert_series_equal(result["value"], expected)
