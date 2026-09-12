"""Decimal columns must participate in numeric preprocessing on both engines."""

from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from skyulf.core.schema import SkyulfSchema
from skyulf.engines.registry import EngineRegistry
from skyulf.preprocessing._helpers import auto_detect_numeric_columns
from skyulf.preprocessing.bucketing import GeneralBinningApplier, GeneralBinningCalculator
from skyulf.preprocessing.feature_generation.polynomial import (
    PolynomialFeaturesApplier,
    PolynomialFeaturesCalculator,
)
from skyulf.preprocessing.feature_selection.correlation import CorrelationThresholdCalculator
from skyulf.preprocessing.feature_selection.model_based import ModelBasedSelectionCalculator
from skyulf.preprocessing.feature_selection.univariate import UnivariateSelectionCalculator
from skyulf.preprocessing.feature_selection.variance import VarianceThresholdCalculator
from skyulf.preprocessing.imputation.iterative import (
    IterativeImputerApplier,
    IterativeImputerCalculator,
)
from skyulf.preprocessing.imputation.knn import KNNImputerApplier, KNNImputerCalculator
from skyulf.preprocessing.imputation.simple import SimpleImputerApplier, SimpleImputerCalculator
from skyulf.preprocessing.inspection import DatasetProfileCalculator
from skyulf.preprocessing.outliers.iqr import IQRApplier, IQRCalculator
from skyulf.preprocessing.outliers.winsorize import WinsorizeApplier, WinsorizeCalculator
from skyulf.preprocessing.outliers.zscore import ZScoreApplier, ZScoreCalculator
from skyulf.preprocessing.scaling.maxabs import MaxAbsScalerApplier, MaxAbsScalerCalculator
from skyulf.preprocessing.scaling.minmax import MinMaxScalerApplier, MinMaxScalerCalculator
from skyulf.preprocessing.scaling.robust import RobustScalerApplier, RobustScalerCalculator
from skyulf.preprocessing.scaling.standard import StandardScalerApplier, StandardScalerCalculator
from skyulf.preprocessing.transformations.power import (
    PowerTransformerApplier,
    PowerTransformerCalculator,
)
from skyulf.utils import detect_numeric_columns, resolve_columns


@pytest.fixture(params=["pandas", "polars", "pandas_wrapped", "polars_wrapped"])
def make_frame(request: pytest.FixtureRequest) -> Any:
    """Exercise native and wrapped inputs without changing decimal values."""

    def build(data: dict[str, Any]) -> Any:
        """Build the requested engine's frame from exact values."""
        frame = pd.DataFrame(data) if request.param.startswith("pandas") else pl.DataFrame(data)
        return EngineRegistry.wrap(frame) if request.param.endswith("wrapped") else frame

    return build


def _pandas(frame: Any) -> pd.DataFrame:
    """Unwrap output only for assertions shared between engines."""
    if hasattr(frame, "to_native"):
        frame = frame.to_native()
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


def test_decimal_detection_preserves_selection_rules(make_frame: Any) -> None:
    """Decimals follow numeric exclusions without opting text or targets into transforms."""
    frame = make_frame(
        {
            "price": [Decimal("1.25"), None, Decimal("3.75")],
            "binary": [Decimal("0"), None, Decimal("1")],
            "constant": [Decimal("2"), None, Decimal("2")],
            "qty": [2, 3, 4],
            "text": ["1.25", None, "3.75"],
            "flag": [True, False, True],
        }
    )
    original = _pandas(frame).copy(deep=True)

    assert detect_numeric_columns(frame) == ["price", "qty"]
    assert detect_numeric_columns(frame, False, False) == ["price", "binary", "constant", "qty"]
    assert auto_detect_numeric_columns(frame) == ["price", "binary", "constant", "qty"]
    assert resolve_columns(frame, {"target_column": "price"}, detect_numeric_columns) == ["qty"]
    assert StandardScalerCalculator().fit(frame, {"columns": []}) == {}
    pd.testing.assert_frame_equal(_pandas(frame), original)


def test_decimal_detection_rejects_mixed_objects() -> None:
    """A decimal next to a string must not silently turn an object column into numbers."""
    frame = pd.DataFrame(
        {
            "mixed": [Decimal("1.25"), "2.50", None],
            "empty": [None, None, None],
            "price": [Decimal("1.25"), pd.NA, Decimal("2.50")],
        }
    )
    assert detect_numeric_columns(frame) == ["price"]
    assert auto_detect_numeric_columns(frame) == ["price"]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "calculator,applier,reference",
    [
        (StandardScalerCalculator, StandardScalerApplier, StandardScaler),
        (RobustScalerCalculator, RobustScalerApplier, RobustScaler),
        (MinMaxScalerCalculator, MinMaxScalerApplier, MinMaxScaler),
        (MaxAbsScalerCalculator, MaxAbsScalerApplier, MaxAbsScaler),
    ],
)
def test_decimal_scalers_fit_and_replay(
    make_frame: Any,
    explicit: bool,
    calculator: Any,
    applier: Any,
    reference: Any,
) -> None:
    """Fitted scaling must handle decimal inputs, nulls and unseen values without mutation."""
    frame = make_frame(
        {
            "price": [Decimal("1.25"), Decimal("2.50"), None, Decimal("5.00")],
            "target": [Decimal("10.00"), Decimal("20.00"), Decimal("30.00"), Decimal("40.00")],
            "label": ["a", "b", "c", "d"],
        }
    )
    original = _pandas(frame).copy(deep=True)
    config = {"columns": ["price"]} if explicit else {"target_column": "target"}
    fitted = reference().fit(np.array([[1.25], [2.5], [np.nan], [5.0]]))
    params = calculator().fit(frame, config)
    out = _pandas(applier().apply(frame, params))

    assert params["columns"] == ["price"]
    np.testing.assert_allclose(out[["price"]], fitted.transform([[1.25], [2.5], [np.nan], [5.0]]))
    assert pd.api.types.is_float_dtype(out["price"])
    pd.testing.assert_frame_equal(out[["target", "label"]], original[["target", "label"]])
    pd.testing.assert_frame_equal(_pandas(frame), original)

    heldout = make_frame({"price": [Decimal("7.50"), None, Decimal("0.00")]})
    replay = _pandas(applier().apply(heldout, params))
    np.testing.assert_allclose(replay[["price"]], fitted.transform([[7.5], [np.nan], [0.0]]))


@pytest.mark.parametrize(
    "calculator,applier,config",
    [
        (IQRCalculator, IQRApplier, {}),
        (ZScoreCalculator, ZScoreApplier, {"threshold": 2.0}),
    ],
)
def test_decimal_outliers_keep_nulls_and_target_alignment(
    make_frame: Any,
    calculator: Any,
    applier: Any,
    config: dict[str, Any],
) -> None:
    """Filtering decimal outliers must preserve exact retained values and matching targets."""
    values = [Decimal(str(n)) for n in [1, 2, 3, 4, 5, 6, 100]] + [None]
    frame = make_frame({"price": values})
    target = np.arange(len(values))
    params = calculator().fit(frame, config)
    out, out_target = applier().apply((frame, target), params)

    assert _pandas(out)["price"].tolist() == values[:6] + [None]
    np.testing.assert_array_equal(out_target, [0, 1, 2, 3, 4, 5, 7])


def test_decimal_winsorization_applies_fitted_bounds(make_frame: Any) -> None:
    """Fitting decimal quantiles must actually clip values during apply on either engine."""
    frame = make_frame({"price": [Decimal("0"), Decimal("10"), None, Decimal("20")]})
    params = WinsorizeCalculator().fit(frame, {"lower_percentile": 25, "upper_percentile": 75})
    out = _pandas(WinsorizeApplier().apply(frame, params))

    np.testing.assert_allclose(out["price"].to_numpy(dtype=float), [5, 10, np.nan, 15])
    assert pd.api.types.is_float_dtype(out["price"])


@pytest.mark.parametrize(
    "calculator,applier,config",
    [
        (SimpleImputerCalculator, SimpleImputerApplier, {"strategy": "mean"}),
        (SimpleImputerCalculator, SimpleImputerApplier, {"strategy": "median"}),
        (KNNImputerCalculator, KNNImputerApplier, {}),
        (IterativeImputerCalculator, IterativeImputerApplier, {}),
    ],
)
def test_decimal_imputation_remains_numeric_for_next_node(
    make_frame: Any,
    calculator: Any,
    applier: Any,
    config: dict[str, Any],
) -> None:
    """Filled decimal columns must remain discoverable by the next numeric pipeline node."""
    frame = make_frame({"price": [Decimal("1.25"), None, Decimal("3.75")]})
    params = calculator().fit(frame, config)
    out = applier().apply(frame, params)

    assert params["columns"] == ["price"]
    np.testing.assert_allclose(_pandas(out)["price"].to_numpy(dtype=float), [1.25, 2.5, 3.75])
    assert detect_numeric_columns(out) == ["price"]


def test_decimal_binning_generates_expected_bins(make_frame: Any) -> None:
    """Decimal fit must not be silently skipped by the bin-edge calculator."""
    frame = make_frame({"price": [Decimal("1"), Decimal("2"), None, Decimal("3")]})
    params = GeneralBinningCalculator().fit(frame, {"n_bins": 2})
    out = _pandas(GeneralBinningApplier().apply(frame, params))

    assert params["bin_edges"]["price"] == [1.0, 2.0, 3.0]
    np.testing.assert_allclose(out["price_binned"].to_numpy(dtype=float), [0, 0, np.nan, 1])


def test_decimal_feature_selection_and_generation(make_frame: Any) -> None:
    """Decimal features must reach variance, correlation and polynomial calculations."""
    frame = make_frame(
        {
            "price": [Decimal("1"), Decimal("2"), Decimal("3")],
            "duplicate": [Decimal("2"), Decimal("4"), Decimal("6")],
        }
    )
    variance = VarianceThresholdCalculator().fit(frame, {"threshold": 0.1})
    correlation = CorrelationThresholdCalculator().fit(frame, {})
    polynomial = PolynomialFeaturesCalculator().fit(frame, {"auto_detect": True})
    out = _pandas(PolynomialFeaturesApplier().apply(frame, polynomial))

    assert variance["selected_columns"] == ["price", "duplicate"]
    assert variance["variances"]["price"] == pytest.approx(2 / 3)
    assert correlation["columns_to_drop"] == ["duplicate"]
    assert polynomial["columns"] == ["price", "duplicate"]
    np.testing.assert_allclose(out["poly_price_pow_2"], [1, 4, 9])


def test_decimal_dataset_profile_reports_numeric_statistics(make_frame: Any) -> None:
    """Decimal profiling must expose numeric statistics instead of categorical counts."""
    frame = make_frame({"price": [Decimal("1.25"), None, Decimal("3.75")]})
    artifact = DatasetProfileCalculator().fit(frame, {})

    assert artifact["profile"]["numeric_stats"]["price"]["mean"] == pytest.approx(2.5)


@pytest.mark.parametrize("method", ["yeo-johnson", "box-cox"])
def test_decimal_power_transform_matches_float_reference(make_frame: Any, method: str) -> None:
    """Automatic power fitting and replay must use the same numeric values as float inputs."""
    values = [1.25, 2.5, np.nan, 5.0, 10.0]
    frame = make_frame({"price": [Decimal(str(v)) if np.isfinite(v) else None for v in values]})
    reference = PowerTransformer(method=method).fit(np.array(values).reshape(-1, 1))
    params = PowerTransformerCalculator().fit(frame, {"method": method})
    out = _pandas(PowerTransformerApplier().apply(frame, params))

    assert params["columns"] == ["price"]
    np.testing.assert_allclose(out[["price"]], reference.transform(np.array(values).reshape(-1, 1)))


@pytest.mark.parametrize(
    "calculator", [UnivariateSelectionCalculator, ModelBasedSelectionCalculator]
)
def test_decimal_supervised_selection_excludes_target(make_frame: Any, calculator: Any) -> None:
    """Numeric candidate discovery must include decimals while keeping the target out."""
    frame = make_frame(
        {
            "price": [Decimal(str(v)) for v in [1, 3, 2, 5, 7, 8]],
            "target": [2.0, 6.0, 4.0, 10.0, 14.0, 16.0],
        }
    )
    params = calculator().fit(
        frame, {"target_column": "target", "problem_type": "regression", "k": 1}
    )

    assert params["candidate_columns"] == ["price"]
    assert params["selected_columns"] == ["price"]


@pytest.mark.parametrize(
    "calculator,applier",
    [
        (StandardScalerCalculator, StandardScalerApplier),
        (RobustScalerCalculator, RobustScalerApplier),
        (MinMaxScalerCalculator, MinMaxScalerApplier),
        (MaxAbsScalerCalculator, MaxAbsScalerApplier),
        (KNNImputerCalculator, KNNImputerApplier),
        (IterativeImputerCalculator, IterativeImputerApplier),
        (PowerTransformerCalculator, PowerTransformerApplier),
    ],
)
@pytest.mark.parametrize("arrow", [False, True])
def test_decimal_pandas_na_matches_float_missing(
    calculator: Any, applier: Any, arrow: bool
) -> None:
    """Pandas' missing sentinel in a decimal object column must survive numeric boundaries."""
    decimal_frame = pd.DataFrame({"price": [Decimal("1.25"), pd.NA, Decimal("3.75")]})
    if arrow:
        pa = pytest.importorskip("pyarrow")
        decimal_frame["price"] = decimal_frame["price"].astype(pd.ArrowDtype(pa.decimal128(10, 2)))
    numeric_frame = pd.DataFrame({"price": [1.25, np.nan, 3.75]})
    params = calculator().fit(decimal_frame, {})
    reference = calculator().fit(numeric_frame, {})
    out = applier().apply(decimal_frame, params)
    expected = applier().apply(numeric_frame, reference)

    np.testing.assert_allclose(out[["price"]].to_numpy(dtype=float), expected[["price"]])


@pytest.mark.parametrize("precision,scale", [(10, 2), (20, 4), (38, 10)])
def test_parameterized_decimal_schema_and_empty_selection(precision: int, scale: int) -> None:
    """Decimal precision/scale must not prevent detection or advertise unchanged scaler dtypes."""
    dtype = pl.Decimal(precision, scale)
    frame: Any = pl.DataFrame({"price": pl.Series([Decimal("1.25"), Decimal("3.75")], dtype=dtype)})
    empty = frame.with_columns(pl.lit(None, dtype=dtype).alias("price"))
    schema = SkyulfSchema.from_dataframe(frame)

    assert auto_detect_numeric_columns(frame) == ["price"]
    assert detect_numeric_columns(frame) == ["price"]
    assert detect_numeric_columns(empty, False, False) == []
    assert StandardScalerCalculator().infer_output_schema(schema, {}).dtypes["price"] == "float64"
