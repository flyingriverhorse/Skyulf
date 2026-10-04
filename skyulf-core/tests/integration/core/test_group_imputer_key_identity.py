"""Group lookup must preserve learned scalar identity across engines and artifacts."""

import json
import pickle
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines import SkyulfPolarsWrapper
from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.preprocessing import FeatureEngineer
from skyulf.preprocessing.cleaning.value_replacement import (
    ValueReplacementApplier,
    ValueReplacementCalculator,
)
from skyulf.preprocessing.imputation.group import GroupImputerApplier, GroupImputerCalculator

_FRAME_KINDS = ["pandas", "polars", "pandas_wrapper", "polars_wrapper"]
_CONFIG = {"columns": ["x"], "group_by": "group", "strategy": "mean"}


def _score_frame(groups: list[Any], dtype: Any, frame_kind: str) -> Any:
    """Keep inference key precision and nulls while varying native frames and wrappers."""
    pandas_dtypes = {
        pl.Int64: "Int64",
        pl.UInt64: "UInt64",
        pl.UInt8: "UInt8",
        pl.Float64: "Float64",
        pl.Float32: "Float32",
        pl.Boolean: "boolean",
        pl.String: "object",
        pl.Null: "object",
    }
    if frame_kind.startswith("pandas"):
        key_dtype = pandas_dtypes.get(dtype, "category")
        frame = pd.DataFrame(
            {"group": pd.Series(groups, dtype=key_dtype), "x": [np.nan] * len(groups)}
        )
        return SkyulfPandasWrapper(frame) if frame_kind == "pandas_wrapper" else frame
    frame = pl.DataFrame(
        {
            "group": pl.Series(groups, dtype=dtype),
            "x": pl.Series([None] * len(groups), dtype=pl.Float64),
        }
    )
    return SkyulfPolarsWrapper(frame) if frame_kind == "polars_wrapper" else frame


def _artifact(groups: list[Any], values: list[float], replay: str = "json") -> dict[str, Any]:
    """Fit typed group pairs without pandas inferring a lossy common numeric dtype."""
    train = pd.DataFrame({"group": pd.Series(groups, dtype=object), "x": values})
    artifact = GroupImputerCalculator().fit(train, _CONFIG)
    if replay == "pickle":
        return pickle.loads(pickle.dumps(artifact))
    return json.loads(json.dumps(artifact))


@pytest.mark.parametrize("frame_kind", _FRAME_KINDS)
@pytest.mark.parametrize("replay", ["json", "pickle"])
@pytest.mark.parametrize(
    ("trained", "scored", "dtype", "expected"),
    [
        ([1, 2], ["1", "2", "3", None], pl.String, [50.0, 50.0, 50.0, 50.0]),
        (["1", "2"], [1, 2, 3, None], pl.Int64, [50.0, 50.0, 50.0, 50.0]),
        ([1, 2], [1, 2, 3, None], pl.Int64, [10.0, 90.0, 50.0, 50.0]),
        (["1", "2"], ["1", "2", "3", None], pl.String, [10.0, 90.0, 50.0, 50.0]),
        ([1, 2], [1.0, 2.0, 3.0, None], pl.Float64, [10.0, 90.0, 50.0, 50.0]),
        ([1.0, 2.0], [1, 2, 3, None], pl.Int64, [10.0, 90.0, 50.0, 50.0]),
        ([False, True], [0, 1, 2, None], pl.Int64, [10.0, 90.0, 50.0, 50.0]),
        ([0, 1], [False, True, None], pl.Boolean, [10.0, 90.0, 50.0]),
        ([2, 0], [True, False, None], pl.Boolean, [50.0, 90.0, 50.0]),
        ([False, True], ["false", "true", None], pl.String, [50.0, 50.0, 50.0]),
    ],
)
def test_group_lookup_preserves_numeric_equality_and_text_identity(
    frame_kind: str,
    replay: str,
    trained: list[Any],
    scored: list[Any],
    dtype: Any,
    expected: list[float],
) -> None:
    """Only equal learned keys may supply fills after dtype changes or artifact replay."""
    artifact = _artifact(trained, [10.0, 90.0], replay)
    original = deepcopy(artifact)
    frame = _score_frame(scored, dtype, frame_kind)
    result = GroupImputerApplier().apply(frame, artifact)
    assert result["x"].to_list() == expected
    assert artifact == original


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("trained", "scored", "dtype", "expected"),
    [
        ([1.5, 2.5], [1, 2, None], pl.Int64, [50.0, 50.0, 50.0]),
        ([1.0, 1.5], [1, 2, None], pl.Int64, [10.0, 50.0, 50.0]),
        (
            [2**53 + 1, 2**53 + 2],
            [float(2**53), float(2**53 + 2), None],
            pl.Float64,
            [50.0, 90.0, 50.0],
        ),
        ([1.1, 2.0], [1.1, 2.0, None], pl.Float32, [50.0, 90.0, 50.0]),
        (
            [2**24 + 1, 2**24 + 2],
            [float(2**24), float(2**24 + 2), None],
            pl.Float32,
            [50.0, 90.0, 50.0],
        ),
        ([2**64 - 1, 2], [2**64 - 1, 2, None], pl.UInt64, [10.0, 90.0, 50.0]),
        ([-1, 255], [0, 255, None], pl.UInt8, [50.0, 90.0, 50.0]),
    ],
)
def test_narrowed_group_keys_cannot_match_a_different_numeric_value(
    frame_kind: str,
    trained: list[Any],
    scored: list[Any],
    dtype: Any,
    expected: list[float],
) -> None:
    """Truncation, range overflow and floating-point rounding must not select another group."""
    artifact = _artifact(trained, [10.0, 90.0])
    result = GroupImputerApplier().apply(_score_frame(scored, dtype, frame_kind), artifact)
    assert result["x"].to_list() == expected


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("scored", "dtype", "expected"),
    [
        ([1, 2, 3, None], pl.Int64, [10.0, 20.0, 82.5, 82.5]),
        (["1", "2", "3", None], pl.String, [100.0, 200.0, 82.5, 82.5]),
    ],
)
def test_mixed_learned_group_keys_keep_their_separate_statistics(
    frame_kind: str, scored: list[Any], dtype: Any, expected: list[float]
) -> None:
    """An artifact containing integer and text groups must remain usable without collisions."""
    artifact = _artifact([1, "1", 2, "2"], [10.0, 100.0, 20.0, 200.0])
    result = GroupImputerApplier().apply(_score_frame(scored, dtype, frame_kind), artifact)
    assert result["x"].to_list() == expected


@pytest.mark.parametrize("frame_kind", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", [pl.Categorical, pl.Enum(["1", "2", "other"])])
@pytest.mark.parametrize("trained", [[1, 2], ["1", "2"]])
def test_categorical_group_keys_preserve_text_identity(
    frame_kind: str, dtype: Any, trained: list[Any]
) -> None:
    """Categorical and Enum scoring keys must match their text values rather than numeric codes."""
    artifact = _artifact(trained, [10.0, 90.0], "pickle")
    frame = _score_frame(["1", "2", "other", None], dtype, frame_kind)
    expected = [50.0, 50.0, 50.0, 50.0] if trained == [1, 2] else [10.0, 90.0, 50.0, 50.0]
    result = GroupImputerApplier().apply(frame, artifact)
    assert result["x"].to_list() == expected


@pytest.mark.parametrize("frame_kind", _FRAME_KINDS)
@pytest.mark.parametrize("dtype", [pl.Null, pl.Float64])
def test_missing_group_keys_keep_global_fallback_and_observed_values(
    frame_kind: str, dtype: Any
) -> None:
    """Null and NaN groups must fall back without changing existing values or target alignment."""
    artifact = _artifact([1, 2, None], [10.0, 90.0, 200.0])
    groups = [None, None, None] if dtype == pl.Null else [None, float("nan"), 1.0]
    frame = _score_frame(groups, dtype, frame_kind)
    if hasattr(frame, "to_native"):
        frame = frame.to_native()
    if isinstance(frame, pd.DataFrame):
        frame.loc[1, "x"] = 123.0
        target: Any = pd.Series([3, 2, 1], name="target")
    else:
        frame = frame.with_columns(pl.Series("x", [None, 123.0, None], dtype=pl.Float64))
        target = pl.Series("target", [3, 2, 1])
    if frame_kind == "pandas_wrapper":
        frame = SkyulfPandasWrapper(frame)
    elif frame_kind == "polars_wrapper":
        frame = SkyulfPolarsWrapper(frame)
    result, result_target = GroupImputerApplier().apply((frame, target), artifact)
    assert result_target.to_list() == [3, 2, 1]
    assert result["x"].to_list() == [100.0, 123.0, 100.0 if dtype == pl.Null else 10.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_public_group_imputer_does_not_reinterpret_scoring_group_strings(engine: str) -> None:
    """Fitted preprocessing replay must share the calculator's typed group identity contract."""
    train: Any = pd.DataFrame({"group": [1, 2], "x": [10.0, 90.0]})
    score = _score_frame(["1", "2", "3"], pl.String, engine)
    if engine == "polars":
        train = pl.from_pandas(train)
    worker = FeatureEngineer(
        [{"name": "group_fill", "transformer": "GroupImputer", "params": _CONFIG}]
    )
    worker.fit_transform(train)
    worker = pickle.loads(pickle.dumps(worker))
    result = worker.transform(score)
    assert result["x"].to_list() == [50.0, 50.0, 50.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_value_replacement_still_accepts_json_numeric_string_keys(engine: str) -> None:
    """Learned group identity must not alter intentional JSON config-key coercion elsewhere."""
    frame: Any = pd.DataFrame({"x": [1, 2]}) if engine == "pandas" else pl.DataFrame({"x": [1, 2]})
    artifact = ValueReplacementCalculator().fit(frame, {"columns": ["x"], "mapping": {"1": 7}})
    result = ValueReplacementApplier().apply(frame, artifact)
    assert result["x"].to_list() == [7, 2]
