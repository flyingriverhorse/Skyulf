"""Label encoding must apply to every target container accepted during fit."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.encoding.label import (
    LabelEncoderApplier,
    LabelEncoderCalculator,
)
from skyulf.preprocessing.pipeline import FeatureEngineer


def _frame(engine: str, values: list[Any]) -> Any:
    """Build the feature frame on the requested engine."""
    data = {"feature": values}
    return pl.DataFrame(data) if engine == "polars" else pd.DataFrame(data)


def _target(engine: str, container: str, values: list[Any]) -> Any:
    """Keep neutral targets neutral and give native Series a name and row metadata."""
    if container == "list":
        return values.copy()
    if container == "numpy":
        return np.asarray(values, dtype=object)
    if engine == "polars":
        return pl.Series("outcome", values)
    return pd.Series(values, name="outcome", index=range(20, 20 + len(values)), dtype=object)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("container", ["list", "numpy", "series"])
def test_label_target_fit_apply_accepts_supported_containers(engine: str, container: str) -> None:
    """Targets accepted at fit must encode without relying on input-specific Series methods."""
    X = _frame(engine, [10, 20, 30])
    y = _target(engine, container, ["yes", "no", "yes"])
    params = LabelEncoderCalculator().fit((X, y), {})

    X_out, y_out = LabelEncoderApplier().apply((X, y), dict(params))

    assert params["classes_count"] == {"__target__": 2}
    assert list(y_out) == [1, 0, 1]
    assert list(y) == ["yes", "no", "yes"]
    assert list(X["feature"]) == list(X_out["feature"]) == [10, 20, 30]
    if engine == "polars":
        assert isinstance(y_out, pl.Series)
        assert y_out.dtype == pl.Int64
        assert y_out.name == ("outcome" if container == "series" else "target")
    else:
        assert isinstance(y_out, pd.Series)
        assert y_out.dtype == np.int64
        assert y_out.name == ("outcome" if container == "series" else None)
        assert list(y_out.index) == ([20, 21, 22] if container == "series" else [0, 1, 2])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("container", ["list", "numpy", "series"])
def test_label_feature_only_encoding_preserves_target(engine: str, container: str) -> None:
    """Feature-only encoding must leave all supported target containers and values intact."""
    X = _frame(engine, ["b", "a", "b"])
    y = _target(engine, container, ["yes", "no", "yes"])
    params = LabelEncoderCalculator().fit((X, y), {"columns": ["feature"]})

    X_out, y_out = LabelEncoderApplier().apply((X, y), dict(params))

    assert "__target__" not in params["encoders"]
    assert type(y_out) is type(y)
    assert list(y_out) == list(y) == ["yes", "no", "yes"]
    assert list(X["feature"]) == ["b", "a", "b"]
    assert list(X_out["feature"]) == [1, 0, 1]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("container", ["list", "numpy", "series"])
def test_label_target_missing_and_unknown_codes_remain_stable(engine: str, container: str) -> None:
    """Learned missing labels retain their code and unseen labels use missing_code."""
    X = _frame(engine, [10, 20, 30])
    y_fit = _target(engine, container, ["yes", None, "no"])
    params = LabelEncoderCalculator().fit((X, y_fit), {"missing_code": -99})
    y_apply = _target(engine, container, [None, "new", "yes"])

    _, y_out = LabelEncoderApplier().apply((X, y_apply), dict(params))

    assert list(y_fit) == ["yes", None, "no"]
    assert list(y_apply) == [None, "new", "yes"]
    assert list(y_out) == [0, -99, 2]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("container", ["list", "numpy", "series"])
def test_label_targets_preserve_existing_missing_string_meanings(
    engine: str, container: str
) -> None:
    """Container normalization must not change the fitted None and literal-string classes."""
    X = _frame(engine, [10, 20, 30])
    y = _target(engine, container, [None, "None", "nan"])
    params = LabelEncoderCalculator().fit((X, y), {})

    _, y_out = LabelEncoderApplier().apply((X, y), dict(params))

    expected = [1, 0, 1] if engine == "polars" and container == "series" else [0, 0, 1]
    assert list(y) == [None, "None", "nan"]
    assert list(y_out) == expected


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("container", ["list", "numpy"])
def test_feature_engineer_encodes_neutral_target_containers(engine: str, container: str) -> None:
    """The public pipeline must encode neutral targets at fit and later transform."""
    X = _frame(engine, [10, 20, 30])
    y = _target(engine, container, ["yes", "no", "yes"])
    engineer = FeatureEngineer([{"name": "labels", "transformer": "LabelEncoder"}])

    (_, y_train), _ = engineer.fit_transform((X, y))
    _, y_test = engineer.transform((X, _target(engine, container, ["no", "new", "yes"])))

    assert list(y_train) == [1, 0, 1]
    assert list(y) == ["yes", "no", "yes"]
    assert list(y_test) == [0, -1, 1]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_label_targets_still_reject_mixed_native_engines(engine: str) -> None:
    """Supporting neutral containers must retain the dispatcher's native-engine guard."""
    X = _frame(engine, [10, 20, 30])
    native_y = _target(engine, "series", ["yes", "no", "yes"])
    other_engine = "pandas" if engine == "polars" else "polars"
    mixed_y = _target(other_engine, "series", ["yes", "no", "yes"])
    params = LabelEncoderCalculator().fit((X, native_y), {})

    with pytest.raises(TypeError, match="Mixed engines"):
        LabelEncoderCalculator().fit((X, mixed_y), {})
    with pytest.raises(TypeError, match="Mixed engines"):
        LabelEncoderApplier().apply((X, mixed_y), dict(params))
