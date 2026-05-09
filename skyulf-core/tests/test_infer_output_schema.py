"""Tests for the C7 lineage primitive (`SkyulfSchema` + `infer_output_schema`)."""

from __future__ import annotations

import pandas as pd
import pytest

from skyulf.preprocessing import SkyulfSchema
from skyulf.preprocessing.base import BaseCalculator
from skyulf.preprocessing.scaling import (
    MaxAbsScalerCalculator,
    MinMaxScalerCalculator,
    RobustScalerCalculator,
    StandardScalerCalculator,
)
from skyulf.preprocessing.drop_and_missing import (
    DeduplicateCalculator,
    DropMissingColumnsCalculator,
    DropMissingRowsCalculator,
    MissingIndicatorCalculator,
)
from skyulf.preprocessing.imputation import (
    IterativeImputerCalculator,
    KNNImputerCalculator,
    SimpleImputerCalculator,
)
from skyulf.preprocessing.outliers import (
    EllipticEnvelopeCalculator,
    IQRCalculator,
    ManualBoundsCalculator,
    WinsorizeCalculator,
    ZScoreCalculator,
)
from skyulf.preprocessing.transformations import (
    GeneralTransformationCalculator,
    PowerTransformerCalculator,
    SimpleTransformationCalculator,
)
from skyulf.preprocessing.cleaning import (
    AliasReplacementCalculator,
    InvalidValueReplacementCalculator,
    TextCleaningCalculator,
    ValueReplacementCalculator,
)
from skyulf.preprocessing.casting import CastingCalculator
from skyulf.preprocessing.inspection import (
    DataSnapshotCalculator,
    DatasetProfileCalculator,
)
from skyulf.preprocessing.resampling import (
    OversamplingCalculator,
    UndersamplingCalculator,
)
from skyulf.preprocessing.encoding.label import LabelEncoderCalculator
from skyulf.preprocessing.encoding.ordinal import OrdinalEncoderCalculator
from skyulf.preprocessing.encoding.one_hot import OneHotEncoderCalculator
from skyulf.preprocessing.encoding.hash import HashEncoderCalculator
from skyulf.preprocessing.encoding.target import TargetEncoderCalculator
from skyulf.preprocessing.split import (
    FeatureTargetSplitCalculator,
    SplitCalculator,
)
from skyulf.preprocessing.feature_selection import VarianceThresholdCalculator


# ---------- SkyulfSchema dataclass ----------


def test_schema_from_columns() -> None:
    s = SkyulfSchema.from_columns(["a", "b"], {"a": "int64"})
    assert s.column_list() == ["a", "b"]
    assert s.dtypes == {"a": "int64"}
    assert "a" in s
    assert len(s) == 2


def test_schema_from_dataframe_pandas() -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [1.0, 2.0]})
    s = SkyulfSchema.from_dataframe(df)
    assert s.column_list() == ["x", "y"]
    assert s.dtypes["x"] == "int64"
    assert s.dtypes["y"] == "float64"


def test_schema_drop_returns_new_instance() -> None:
    s = SkyulfSchema.from_columns(["a", "b", "c"], {"a": "int64", "b": "int64", "c": "int64"})
    s2 = s.drop(["b"])
    assert s.column_list() == ["a", "b", "c"]  # unchanged
    assert s2.column_list() == ["a", "c"]
    assert "b" not in s2.dtypes


def test_schema_add_and_rename() -> None:
    s = SkyulfSchema.from_columns(["a"]).add("b", "float64").rename({"a": "alpha"})
    assert s.column_list() == ["alpha", "b"]
    assert s.dtypes == {"b": "float64"}


def test_schema_with_dtype() -> None:
    s = SkyulfSchema.from_columns(["a"], {"a": "int64"}).with_dtype("a", "float64")
    assert s.dtypes["a"] == "float64"
    # No-op for unknown column.
    assert s.with_dtype("missing", "int64") is not None


# ---------- BaseCalculator default ----------


def test_base_calculator_default_returns_none() -> None:
    class _Stub(BaseCalculator):
        def fit(self, df, config):  # type: ignore[override]
            return {}

    s = SkyulfSchema.from_columns(["a"])
    assert _Stub().infer_output_schema(s, {}) is None


# ---------- Scalers (passthrough) ----------


@pytest.mark.parametrize(
    "cls",
    [
        StandardScalerCalculator,
        MinMaxScalerCalculator,
        RobustScalerCalculator,
        MaxAbsScalerCalculator,
    ],
)
def test_scalers_passthrough_schema(cls) -> None:
    s = SkyulfSchema.from_columns(["a", "b"], {"a": "float64", "b": "float64"})
    assert cls().infer_output_schema(s, {"columns": ["a"]}) == s


# ---------- DropMissingColumns ----------


def test_drop_missing_explicit_columns() -> None:
    s = SkyulfSchema.from_columns(["a", "b", "c"])
    out = DropMissingColumnsCalculator().infer_output_schema(s, {"columns": ["b"]})
    assert out is not None
    assert out.column_list() == ["a", "c"]


def test_drop_missing_threshold_returns_none() -> None:
    # Threshold path is data-dependent → cannot infer pre-fit.
    s = SkyulfSchema.from_columns(["a", "b"])
    out = DropMissingColumnsCalculator().infer_output_schema(s, {"missing_threshold": 50.0})
    assert out is None


def test_drop_missing_no_config_passes_through() -> None:
    s = SkyulfSchema.from_columns(["a"])
    assert DropMissingColumnsCalculator().infer_output_schema(s, {}) == s


# ---------- Calculator without override falls back to None ----------


def test_unimplemented_calculator_returns_none() -> None:
    s = SkyulfSchema.from_columns(["a"])
    # OneHotEncoder is data-dependent (cardinality unknown pre-fit) → default returns None.
    assert OneHotEncoderCalculator().infer_output_schema(s, {}) is None
    # Feature selection is also data-dependent.
    assert VarianceThresholdCalculator().infer_output_schema(s, {}) is None


# ---------- Phase A: passthrough Calculators ----------


PASSTHROUGH_CALCULATORS = [
    SimpleImputerCalculator,
    KNNImputerCalculator,
    IterativeImputerCalculator,
    IQRCalculator,
    ZScoreCalculator,
    WinsorizeCalculator,
    ManualBoundsCalculator,
    EllipticEnvelopeCalculator,
    PowerTransformerCalculator,
    SimpleTransformationCalculator,
    GeneralTransformationCalculator,
    TextCleaningCalculator,
    InvalidValueReplacementCalculator,
    ValueReplacementCalculator,
    AliasReplacementCalculator,
    DeduplicateCalculator,
    DropMissingRowsCalculator,
    OversamplingCalculator,
    UndersamplingCalculator,
    DatasetProfileCalculator,
    DataSnapshotCalculator,
    LabelEncoderCalculator,
    OrdinalEncoderCalculator,
    HashEncoderCalculator,
    TargetEncoderCalculator,
    SplitCalculator,
]


@pytest.mark.parametrize("calc_cls", PASSTHROUGH_CALCULATORS)
def test_phase_a_passthrough(calc_cls: type) -> None:
    s = SkyulfSchema.from_columns(["a", "b", "c"], {"a": "float64"})
    assert calc_cls().infer_output_schema(s, {}) == s


# ---------- Phase A: config-driven Calculators ----------


def test_casting_rewrites_dtypes() -> None:
    s = SkyulfSchema.from_columns(["price", "qty"], {"price": "int64", "qty": "int64"})
    out = CastingCalculator().infer_output_schema(s, {"column_types": {"price": "float"}})
    assert out is not None
    assert out.dtypes["price"] == "float64"
    assert out.dtypes["qty"] == "int64"
    assert out.column_list() == ["price", "qty"]


def test_casting_columns_plus_target_type() -> None:
    s = SkyulfSchema.from_columns(["a", "b"], {"a": "int64", "b": "int64"})
    out = CastingCalculator().infer_output_schema(
        s, {"columns": ["a", "b"], "target_type": "string"}
    )
    assert out is not None
    assert out.dtypes["a"] == "string"
    assert out.dtypes["b"] == "string"


def test_missing_indicator_explicit_columns_adds_indicator_cols() -> None:
    s = SkyulfSchema.from_columns(["a", "b"])
    out = MissingIndicatorCalculator().infer_output_schema(s, {"columns": ["a"]})
    assert out is not None
    assert out.column_list() == ["a", "b", "a_missing"]
    assert out.dtypes["a_missing"] == "bool"


def test_missing_indicator_no_columns_returns_none() -> None:
    s = SkyulfSchema.from_columns(["a", "b"])
    # Without explicit columns the indicator set depends on data.
    assert MissingIndicatorCalculator().infer_output_schema(s, {}) is None


# ---------- Phase A follow-up: split.py ----------


def test_feature_target_split_drops_target() -> None:
    s = SkyulfSchema.from_columns(["a", "b", "label"])
    out = FeatureTargetSplitCalculator().infer_output_schema(s, {"target_column": "label"})
    assert out is not None
    assert out.column_list() == ["a", "b"]


def test_feature_target_split_target_alias() -> None:
    s = SkyulfSchema.from_columns(["a", "y"])
    out = FeatureTargetSplitCalculator().infer_output_schema(s, {"target": "y"})
    assert out is not None
    assert out.column_list() == ["a"]


def test_feature_target_split_unknown_target_passes_through() -> None:
    # Validator (Phase D) flags the typo separately; here we just don't crash.
    s = SkyulfSchema.from_columns(["a", "b"])
    out = FeatureTargetSplitCalculator().infer_output_schema(s, {"target_column": "missing"})
    assert out == s


# ---------- Data-dependent Calculators must return None ----------
#
# These nodes can only know their output schema after seeing data
# (cardinality, fitted thresholds, etc.). The contract: their
# ``infer_output_schema`` must return ``None`` so the schema graph
# treats the downstream chain as opaque rather than guessing wrong.

from skyulf.preprocessing.bucketing import (  # noqa: E402
    CustomBinningCalculator,
    GeneralBinningCalculator,
    KBinsDiscretizerCalculator,
)
from skyulf.preprocessing.encoding.dummy import DummyEncoderCalculator  # noqa: E402
from skyulf.preprocessing.feature_generation import (  # noqa: E402
    FeatureGenerationCalculator,
    PolynomialFeaturesCalculator,
)
from skyulf.preprocessing.feature_selection import (  # noqa: E402
    CorrelationThresholdCalculator,
    ModelBasedSelectionCalculator,
    UnivariateSelectionCalculator,
)


DATA_DEPENDENT_CALCULATORS = [
    # Encoders whose output column count depends on cardinality.
    OneHotEncoderCalculator,
    DummyEncoderCalculator,
    HashEncoderCalculator,  # passthrough actually — kept here pending review
    TargetEncoderCalculator,  # passthrough actually — kept here pending review
    # Bucketing — output column set depends on fitted bin edges.
    GeneralBinningCalculator,
    CustomBinningCalculator,
    KBinsDiscretizerCalculator,
    # Feature generation — new column names emerge from fit.
    PolynomialFeaturesCalculator,
    FeatureGenerationCalculator,
    # Feature selection — selected_columns is data-dependent.
    VarianceThresholdCalculator,
    CorrelationThresholdCalculator,
    UnivariateSelectionCalculator,
    ModelBasedSelectionCalculator,
]


@pytest.mark.parametrize("calc_cls", DATA_DEPENDENT_CALCULATORS)
def test_data_dependent_returns_none_or_passthrough(calc_cls: type) -> None:
    """Each data-dependent Calculator must return either ``None`` (opaque)
    or the input schema unchanged (passthrough). Anything else would mean
    the schema graph is making up columns that may not exist at runtime."""
    s = SkyulfSchema.from_columns(["a", "b", "c"])
    out = calc_cls().infer_output_schema(s, {})
    assert out is None or out == s, (
        f"{calc_cls.__name__}.infer_output_schema returned {out!r}; "
        f"expected None (opaque) or {s!r} (passthrough)"
    )


# ---------- Cross-cutting: every Calculator with an override returns
#            either SkyulfSchema or None (never raises, never returns
#            something else) ----------


@pytest.mark.parametrize(
    "calc_cls",
    PASSTHROUGH_CALCULATORS
    + DATA_DEPENDENT_CALCULATORS
    + [
        CastingCalculator,
        MissingIndicatorCalculator,
        DropMissingColumnsCalculator,
        FeatureTargetSplitCalculator,
    ],
)
def test_infer_output_schema_contract(calc_cls: type) -> None:
    """``infer_output_schema(input_schema, config)`` must always return
    ``Optional[SkyulfSchema]`` for any reasonable config. It must never
    raise and never return a non-schema, non-None value."""
    s = SkyulfSchema.from_columns(["a", "b", "c"], {"a": "float64"})
    try:
        out = calc_cls().infer_output_schema(s, {})
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"{calc_cls.__name__}.infer_output_schema raised {type(e).__name__}: {e}")
    assert out is None or isinstance(out, SkyulfSchema), (
        f"{calc_cls.__name__} returned {type(out).__name__}, " f"expected SkyulfSchema or None"
    )
