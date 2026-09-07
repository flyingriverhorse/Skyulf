"""Tests for the C7 lineage primitive (`SkyulfSchema` + `infer_output_schema`)."""

import pandas as pd
import pytest
from sklearn.preprocessing import TargetEncoder

from skyulf.preprocessing import SkyulfSchema
from skyulf.preprocessing.base import BaseCalculator
from skyulf.preprocessing.casting import CastingCalculator
from skyulf.preprocessing.cleaning import (
    AliasReplacementCalculator,
    InvalidValueReplacementCalculator,
    TextCleaningCalculator,
    ValueReplacementCalculator,
)
from skyulf.preprocessing.drop_and_missing import (
    DeduplicateCalculator,
    DropMissingColumnsCalculator,
    DropMissingRowsCalculator,
    MissingIndicatorCalculator,
)
from skyulf.preprocessing.encoding.hash import HashEncoderCalculator
from skyulf.preprocessing.encoding.label import LabelEncoderCalculator
from skyulf.preprocessing.encoding.one_hot import OneHotEncoderCalculator
from skyulf.preprocessing.encoding.ordinal import OrdinalEncoderCalculator
from skyulf.preprocessing.encoding.target import TargetEncoderCalculator
from skyulf.preprocessing.feature_selection import VarianceThresholdCalculator
from skyulf.preprocessing.imputation import (
    IterativeImputerCalculator,
    KNNImputerCalculator,
    SimpleImputerCalculator,
)
from skyulf.preprocessing.inspection import (
    DatasetProfileCalculator,
    DataSnapshotCalculator,
)
from skyulf.preprocessing.outliers import (
    EllipticEnvelopeCalculator,
    IQRCalculator,
    ManualBoundsCalculator,
    WinsorizeCalculator,
    ZScoreCalculator,
)
from skyulf.preprocessing.resampling import (
    OversamplingCalculator,
    UndersamplingCalculator,
)
from skyulf.preprocessing.scaling import (
    MaxAbsScalerCalculator,
    MinMaxScalerCalculator,
    RobustScalerCalculator,
    StandardScalerCalculator,
)
from skyulf.preprocessing.split import (
    FeatureTargetSplitCalculator,
    SplitCalculator,
)
from skyulf.preprocessing.time_series import LagFeaturesCalculator
from skyulf.preprocessing.transformations import (
    GeneralTransformationCalculator,
    PowerTransformerCalculator,
    SimpleTransformationCalculator,
)
from skyulf.registry import NodeRegistry

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


# ---------- OC-03: dtype-evolution nodes ----------


@pytest.mark.parametrize(
    "node_id, calc_cls, df, cfg",
    [
        (
            "StandardScaler",
            StandardScalerCalculator,
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            {"columns": ["x"]},
        ),
        (
            "MinMaxScaler",
            MinMaxScalerCalculator,
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            {"columns": ["x"]},
        ),
        (
            "RobustScaler",
            RobustScalerCalculator,
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            {"columns": ["x"]},
        ),
        (
            "MaxAbsScaler",
            MaxAbsScalerCalculator,
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            {"columns": ["x"]},
        ),
        (
            "SimpleImputer",
            SimpleImputerCalculator,
            pd.DataFrame({"x": [1.0, None, 3.0], "y": [1.0, 2.0, 3.0]}),
            {"strategy": "mean", "columns": ["x"]},
        ),
        (
            "KNNImputer",
            KNNImputerCalculator,
            pd.DataFrame(
                {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
            ),
            {"columns": ["x"]},
        ),
        (
            "IterativeImputer",
            IterativeImputerCalculator,
            pd.DataFrame(
                {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
            ),
            {"columns": ["x"]},
        ),
        (
            "SimpleTransformation",
            SimpleTransformationCalculator,
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            {"transformations": [{"column": "x", "method": "log"}]},
        ),
        (
            "GeneralTransformation",
            GeneralTransformationCalculator,
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            {"transformations": [{"column": "x", "method": "log"}]},
        ),
        (
            "PowerTransformer",
            PowerTransformerCalculator,
            pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}),
            {"method": "yeo-johnson", "columns": ["x"]},
        ),
        (
            "Winsorize",
            WinsorizeCalculator,
            pd.DataFrame({"x": list(range(1, 7)), "y": [0, 0, 0, 0, 0, 0]}),
            {"columns": ["x"], "lower_percentile": 10, "upper_percentile": 90},
        ),
        (
            "LagFeatures",
            LagFeaturesCalculator,
            pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}),
            {"columns": ["x"], "lags": [1]},
        ),
    ],
)
def test_oc03_infer_output_schema_matches_runtime_dtype_changes(
    node_id: str,
    calc_cls: type[BaseCalculator],
    df: pd.DataFrame,
    cfg: dict[str, object],
) -> None:
    """OC-03: infer_output_schema should match real output dtypes for int->float transforms."""
    calc = calc_cls()
    schema_in = SkyulfSchema.from_dataframe(df)
    inferred = calc.infer_output_schema(schema_in, cfg)
    assert inferred is not None
    artifact = calc.fit(df, cfg)
    if not artifact:
        pytest.skip("No-op config in schema-inference parity probe")

    applier = NodeRegistry.get_applier(node_id)
    out = applier().apply(df, artifact)
    assert inferred == SkyulfSchema.from_dataframe(out)


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
    IQRCalculator,
    ZScoreCalculator,
    ManualBoundsCalculator,
    EllipticEnvelopeCalculator,
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
    SplitCalculator,
]


@pytest.mark.parametrize("calc_cls", PASSTHROUGH_CALCULATORS)
def test_phase_a_passthrough(calc_cls: type) -> None:
    s = SkyulfSchema.from_columns(["a", "b", "c"], {"a": "float64"})
    assert calc_cls().infer_output_schema(s, {}) == s


def test_target_encoder_binary_continuous_is_passthrough() -> None:
    """Explicit binary/continuous ``target_type`` encodes in place, so its schema is predictable.

    Breaks if the encoder ever starts fanning out columns for these two target
    types, or if the in-place prediction is lost for the regression case —
    ``"continuous"`` is sklearn's spelling, not ``"regression"``.
    """
    s = SkyulfSchema.from_columns(["a", "b", "c"], {"a": "float64"})
    for target_type in ("binary", "continuous"):
        out = TargetEncoderCalculator().infer_output_schema(s, {"target_type": target_type})
        assert out == s


def test_target_encoder_regression_spelling_is_rejected_and_opaque() -> None:
    """``"regression"`` is not a sklearn ``target_type``, so the schema must stay opaque for it.

    Pins OC-22: this file used to assert passthrough for ``"regression"``, which
    promised an output shape no working pipeline could ever produce — the value
    is forwarded to ``TargetEncoder(target_type=...)`` verbatim and raises at
    fit. Breaks if sklearn starts accepting the alias, or if the prediction
    starts answering for a config that cannot fit.
    """
    X = pd.DataFrame({"cat": ["a", "b", "a", "b"]})
    y = pd.Series([0, 1, 0, 1])
    # sklearn raises InvalidParameterError, which subclasses ValueError; the class
    # itself lives in the private sklearn.utils._param_validation, so match the base.
    with pytest.raises(ValueError, match="target_type"):
        TargetEncoder(target_type="regression").fit(X, y)

    s = SkyulfSchema.from_columns(["cat"], {"cat": "object"})
    assert TargetEncoderCalculator().infer_output_schema(s, {"target_type": "regression"}) is None


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
    assert out.dtypes["a_missing"] == "int64"


def test_missing_indicator_no_columns_returns_none() -> None:
    s = SkyulfSchema.from_columns(["a", "b"])
    # Without explicit columns the indicator set depends on data.
    assert MissingIndicatorCalculator().infer_output_schema(s, {}) is None


# ---------- Phase A follow-up: split.py ----------


def test_feature_target_split_keeps_target_in_schema() -> None:
    # FT split outputs (X, y); the target lives in the y slot but is still
    # part of the dataset, so downstream column pickers must still see it.
    s = SkyulfSchema.from_columns(["a", "b", "label"])
    out = FeatureTargetSplitCalculator().infer_output_schema(s, {"target_column": "label"})
    assert out is not None
    assert out.column_list() == ["a", "b", "label"]


def test_feature_target_split_target_alias() -> None:
    s = SkyulfSchema.from_columns(["a", "y"])
    out = FeatureTargetSplitCalculator().infer_output_schema(s, {"target": "y"})
    assert out is not None
    assert out.column_list() == ["a", "y"]


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

from skyulf.preprocessing.bucketing import (
    CustomBinningCalculator,
    GeneralBinningCalculator,
    KBinsDiscretizerCalculator,
)
from skyulf.preprocessing.encoding.dummy import DummyEncoderCalculator
from skyulf.preprocessing.feature_generation import (
    FeatureGenerationCalculator,
    PolynomialFeaturesCalculator,
)
from skyulf.preprocessing.feature_selection import (
    CorrelationThresholdCalculator,
    ModelBasedSelectionCalculator,
    UnivariateSelectionCalculator,
)

DATA_DEPENDENT_CALCULATORS = [
    # Encoders whose output column count depends on cardinality.
    OneHotEncoderCalculator,
    DummyEncoderCalculator,
    HashEncoderCalculator,  # passthrough actually — kept here pending review
    TargetEncoderCalculator,  # returns None for the default "auto"/multiclass
    # target_type (data-dependent column fan-out); binary/continuous stays
    # passthrough, covered separately in test_encoding_target.py.
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
    the schema graph is making up columns that may not exist at runtime.
    """
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
    raise and never return a non-schema, non-None value.
    """
    s = SkyulfSchema.from_columns(["a", "b", "c"], {"a": "float64"})
    try:
        out = calc_cls().infer_output_schema(s, {})
    except Exception as e:  # noqa: BLE001 - contract: any raise is converted into an explicit pytest.fail
        pytest.fail(f"{calc_cls.__name__}.infer_output_schema raised {type(e).__name__}: {e}")
    assert out is None or isinstance(out, SkyulfSchema), (
        f"{calc_cls.__name__} returned {type(out).__name__}, expected SkyulfSchema or None"
    )
