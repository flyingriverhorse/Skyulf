"""Raw temporal features require an explicit representation before sklearn conversion."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.engines.sklearn_bridge import SklearnBridge
from skyulf.modeling.sklearn_wrapper import SklearnApplier, SklearnCalculator
from skyulf.preprocessing.time_series.date_features import (
    DateFeaturesApplier,
    DateFeaturesCalculator,
)


def _frame(engine, kind, companion):
    """Preserve temporal units while varying the numeric companion independently."""
    dates = pd.date_range("2024-01-01", periods=12)
    values = dates.date if kind == "date" else dates.as_unit(kind)
    frame = pd.DataFrame({"when": values})
    if companion:
        frame["value"] = np.arange(12, dtype=companion)
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("kind", ["date", "ns", "us", "ms"])
@pytest.mark.parametrize("companion", [None, "int64", "float64"])
@pytest.mark.parametrize("model_class", [LinearRegression, DecisionTreeRegressor])
@pytest.mark.parametrize("phase", ["fit", "predict"])
def test_raw_temporal_features_are_explicitly_rejected(engine, kind, companion, model_class, phase):
    """Fit and predict must not silently reinterpret dates differently across engines."""
    frame = _frame(engine, kind, companion)
    calculator = SklearnCalculator(model_class, {}, "regression")
    y = pd.Series(np.arange(12, dtype=float))
    with pytest.raises(ValueError, match="temporal.*when.*DateFeatures"):
        if phase == "fit":
            calculator.fit(frame, y, {})
        else:
            model = model_class().fit(np.ones((12, len(frame.columns))), y)
            SklearnApplier().predict(frame, model)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("drop_original", [False, True])
@pytest.mark.parametrize("model_class", [LinearRegression, DecisionTreeRegressor])
def test_date_features_require_removing_raw_column(engine, drop_original, model_class):
    """Explicit calendar features remain supported without silently dropping the raw date."""
    frame = _frame(engine, "us", "float64")
    params = DateFeaturesCalculator().fit(
        frame, {"columns": ["when"], "features": ["day"], "drop_original": drop_original}
    )
    converted = DateFeaturesApplier().apply(frame, params)
    calculator = SklearnCalculator(model_class, {}, "regression")
    if not drop_original:
        with pytest.raises(ValueError, match="temporal.*when.*DateFeatures"):
            calculator.fit(converted, pd.Series(np.arange(12)), {})
    else:
        model = calculator.fit(converted, pd.Series(np.arange(12)), {})
        np.testing.assert_allclose(
            SklearnApplier().predict(converted, model), np.arange(12), atol=1e-10
        )
    assert ("when" in converted.columns) is not drop_original


def test_temporal_target_is_not_mistaken_for_feature():
    """The feature restriction must not change the separate target conversion contract."""
    y = pd.Series(pd.date_range("2024-01-01", periods=2))
    values, target = SklearnBridge.to_sklearn(
        (pd.DataFrame({"x": [1, 2]}), y), validate_features=True
    )
    np.testing.assert_array_equal(values, [[1], [2]])
    np.testing.assert_array_equal(target, y.to_numpy())


@pytest.mark.parametrize(
    "container", ["pandas_wrapper", "polars_wrapper", "polars_objects", "numpy", "objects", "list"]
)
def test_temporal_input_containers_cannot_bypass_model_guard(container):
    """Wrapper and array conversion must not erase temporal types before validation."""
    frame = _frame("pandas", "ns", None)
    if container == "pandas_wrapper":
        data = SkyulfPandasWrapper(frame)
    elif container == "polars_wrapper":
        data = SkyulfPolarsWrapper(pl.from_pandas(frame))
    elif container == "polars_objects":
        data = pl.DataFrame(
            {"when": pl.Series("when", list(frame["when"].to_numpy()), dtype=pl.Object)}
        )
    elif container == "numpy":
        data = frame.to_numpy()
    elif container == "objects":
        data = np.array([[value.date()] for value in frame["when"]], dtype=object)
    else:
        data = [[value.date()] for value in frame["when"]]
    with pytest.raises(ValueError, match="Raw temporal features"):
        SklearnBridge.to_sklearn(data, validate_features=True)


@pytest.mark.parametrize("unit", ["ns", "us", "ms"])
@pytest.mark.parametrize("centered", [False, True])
def test_explicit_numeric_time_conversion_is_preserved(unit, centered):
    """An explicitly chosen epoch unit/centering remains numeric and is never rewritten."""
    values = (
        pd.date_range("2024-01-01", periods=10)
        .as_unit(unit)
        .to_numpy()
        .astype("int64")
        .astype(float)
    )
    if centered:
        values = (values - values.mean()) / values.std()
    values = values.reshape(-1, 1)
    result, _ = SklearnBridge.to_sklearn(values, validate_features=True)
    assert result is values
