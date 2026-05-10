"""C4 — Typed `NodeArtifact` shape tests.

For every preprocessing Calculator wired with a TypedDict artifact in
``skyulf.preprocessing._artifacts``, drive ``fit(df, config)`` with
synthetic data and assert:

1. The result is a ``dict``.
2. The ``"type"`` discriminator matches the documented value (when the
   calculator returns a populated artifact).
3. Every key present in the result is declared on the TypedDict
   (catches typos and stale keys; ``total=False`` allows missing keys
   on purpose for the ``return {}`` early-exit).

This is the runtime counterpart of the static ``ty`` enforcement and
the only thing that catches a calculator silently emitting an unknown
key (which Appliers would then ignore at runtime).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
import pytest

from skyulf.preprocessing import _artifacts as art
from skyulf.preprocessing.base import BaseCalculator
from skyulf.preprocessing.bucketing import (
    CustomBinningCalculator,
    GeneralBinningCalculator,
)
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
from skyulf.preprocessing.encoding.dummy import DummyEncoderCalculator
from skyulf.preprocessing.encoding.hash import HashEncoderCalculator
from skyulf.preprocessing.encoding.label import LabelEncoderCalculator
from skyulf.preprocessing.encoding.one_hot import OneHotEncoderCalculator
from skyulf.preprocessing.encoding.ordinal import OrdinalEncoderCalculator
from skyulf.preprocessing.encoding.target import TargetEncoderCalculator
from skyulf.preprocessing.feature_generation import (
    FeatureGenerationCalculator,
    PolynomialFeaturesCalculator,
)
from skyulf.preprocessing.feature_selection import (
    CorrelationThresholdCalculator,
    ModelBasedSelectionCalculator,
    UnivariateSelectionCalculator,
    VarianceThresholdCalculator,
)
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
from skyulf.preprocessing.transformations import (
    GeneralTransformationCalculator,
    PowerTransformerCalculator,
    SimpleTransformationCalculator,
)


# ---------- Synthetic data factories ----------


def _numeric_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, 50),
            "b": rng.normal(5, 2, 50),
            "c": rng.uniform(0, 10, 50),
        }
    )


def _numeric_with_nans_df() -> pd.DataFrame:
    df = _numeric_df()
    df.loc[0:5, "a"] = np.nan
    return df


def _categorical_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color": ["red", "blue", "red", "green", "blue"] * 10,
            "size": ["S", "M", "L", "M", "S"] * 10,
        }
    )


def _mixed_with_target_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "x1": rng.normal(0, 1, 60),
            "x2": rng.normal(0, 1, 60),
            "cat": (["a", "b", "c"] * 20),
            "target": rng.integers(0, 2, 60),
        }
    )


def _text_df() -> pd.DataFrame:
    return pd.DataFrame({"name": [" Alice ", "BOB ", " carol "] * 5})


def _bool_alias_df() -> pd.DataFrame:
    return pd.DataFrame({"flag": ["yes", "no", "y", "n", "true", "false"] * 5})


# ---------- Test matrix ----------
#
# Each entry: (calculator class, artifact TypedDict, expected "type" string,
#             data factory, fit-config dict).


CASES: list[
    Tuple[Type[BaseCalculator], Type[Any], Optional[str], Callable[[], Any], Dict[str, Any]]
] = [
    # Scalers --------------------------------------------------------------
    (
        StandardScalerCalculator,
        art.StandardScalerArtifact,
        "standard_scaler",
        _numeric_df,
        {"columns": ["a", "b"]},
    ),
    (
        MinMaxScalerCalculator,
        art.MinMaxScalerArtifact,
        "minmax_scaler",
        _numeric_df,
        {"columns": ["a", "b"]},
    ),
    (
        RobustScalerCalculator,
        art.RobustScalerArtifact,
        "robust_scaler",
        _numeric_df,
        {"columns": ["a", "b"]},
    ),
    (
        MaxAbsScalerCalculator,
        art.MaxAbsScalerArtifact,
        "maxabs_scaler",
        _numeric_df,
        {"columns": ["a", "b"]},
    ),
    # Imputers -------------------------------------------------------------
    (
        SimpleImputerCalculator,
        art.SimpleImputerArtifact,
        "simple_imputer",
        _numeric_with_nans_df,
        {"columns": ["a"], "strategy": "mean"},
    ),
    (
        KNNImputerCalculator,
        art.KNNImputerArtifact,
        "knn_imputer",
        _numeric_with_nans_df,
        {"columns": ["a", "b"], "n_neighbors": 3},
    ),
    (
        IterativeImputerCalculator,
        art.IterativeImputerArtifact,
        "iterative_imputer",
        _numeric_with_nans_df,
        {"columns": ["a", "b"]},
    ),
    # Outliers -------------------------------------------------------------
    (IQRCalculator, art.IQRArtifact, "iqr", _numeric_df, {"columns": ["a"]}),
    (ZScoreCalculator, art.ZScoreArtifact, "zscore", _numeric_df, {"columns": ["a"]}),
    (
        WinsorizeCalculator,
        art.WinsorizeArtifact,
        "winsorize",
        _numeric_df,
        {"columns": ["a"], "lower_percentile": 5, "upper_percentile": 95},
    ),
    (
        ManualBoundsCalculator,
        art.ManualBoundsArtifact,
        "manual_bounds",
        _numeric_df,
        {"bounds": {"a": {"lower": -3, "upper": 3}}},
    ),
    (
        EllipticEnvelopeCalculator,
        art.EllipticEnvelopeArtifact,
        "elliptic_envelope",
        _numeric_df,
        {"columns": ["a", "b"], "contamination": 0.1},
    ),
    # Transformations ------------------------------------------------------
    (
        PowerTransformerCalculator,
        art.PowerTransformerArtifact,
        "power_transformer",
        _numeric_df,
        {"columns": ["a", "b"], "method": "yeo-johnson"},
    ),
    (
        SimpleTransformationCalculator,
        art.SimpleTransformationArtifact,
        "simple_transformation",
        _numeric_df,
        {"transformations": [{"column": "a", "method": "log"}]},
    ),
    (
        GeneralTransformationCalculator,
        art.GeneralTransformationArtifact,
        "general_transformation",
        _numeric_df,
        {"transformations": [{"column": "a", "method": "log", "params": {}}]},
    ),
    # Resampling -----------------------------------------------------------
    (
        OversamplingCalculator,
        art.OversamplingArtifact,
        "oversampling",
        _mixed_with_target_df,
        {"method": "smote", "target_column": "target"},
    ),
    (
        UndersamplingCalculator,
        art.UndersamplingArtifact,
        "undersampling",
        _mixed_with_target_df,
        {"method": "random", "target_column": "target"},
    ),
    # Drop / missing -------------------------------------------------------
    (DeduplicateCalculator, art.DeduplicateArtifact, "deduplicate", _numeric_df, {"keep": "first"}),
    (
        DropMissingColumnsCalculator,
        art.DropMissingColumnsArtifact,
        "drop_missing_columns",
        _numeric_with_nans_df,
        {"missing_threshold": 50.0},
    ),
    (
        DropMissingRowsCalculator,
        art.DropMissingRowsArtifact,
        "drop_missing_rows",
        _numeric_with_nans_df,
        {"how": "any"},
    ),
    (
        MissingIndicatorCalculator,
        art.MissingIndicatorArtifact,
        "missing_indicator",
        _numeric_with_nans_df,
        {"columns": ["a"]},
    ),
    # Casting --------------------------------------------------------------
    (
        CastingCalculator,
        art.CastingArtifact,
        "casting",
        _numeric_df,
        {"columns": ["a"], "target_type": "float"},
    ),
    # Bucketing ------------------------------------------------------------
    (
        GeneralBinningCalculator,
        art.GeneralBinningArtifact,
        "general_binning",
        _numeric_df,
        {"columns": ["a"], "strategy": "quantile", "n_bins": 4},
    ),
    (
        CustomBinningCalculator,
        art.GeneralBinningArtifact,
        "general_binning",
        _numeric_df,
        {"columns": ["a"], "bins": [-3.0, 0.0, 3.0]},
    ),
    # Feature generation --------------------------------------------------
    (
        PolynomialFeaturesCalculator,
        art.PolynomialFeaturesArtifact,
        "polynomial_features",
        _numeric_df,
        {"columns": ["a", "b"], "degree": 2},
    ),
    (
        FeatureGenerationCalculator,
        art.FeatureGenerationArtifact,
        "feature_generation",
        _numeric_df,
        {"operations": []},
    ),
    # Feature selection ---------------------------------------------------
    (
        VarianceThresholdCalculator,
        art.VarianceThresholdArtifact,
        "variance_threshold",
        _numeric_df,
        {"threshold": 0.0, "columns": ["a", "b", "c"]},
    ),
    (
        CorrelationThresholdCalculator,
        art.CorrelationThresholdArtifact,
        "correlation_threshold",
        _numeric_df,
        {"threshold": 0.95},
    ),
    (
        UnivariateSelectionCalculator,
        art.UnivariateSelectionArtifact,
        "univariate_selection",
        _mixed_with_target_df,
        {"method": "select_k_best", "k": 1, "target_column": "target", "columns": ["x1", "x2"]},
    ),
    (
        ModelBasedSelectionCalculator,
        art.ModelBasedSelectionArtifact,
        "model_based_selection",
        _mixed_with_target_df,
        {
            "method": "select_from_model",
            "estimator": "random_forest",
            "target_column": "target",
            "columns": ["x1", "x2"],
        },
    ),
    # Cleaning ------------------------------------------------------------
    (
        TextCleaningCalculator,
        art.TextCleaningArtifact,
        "text_cleaning",
        _text_df,
        {"columns": ["name"], "operations": [{"op": "trim", "mode": "both"}]},
    ),
    (
        InvalidValueReplacementCalculator,
        art.InvalidValueReplacementArtifact,
        "invalid_value_replacement",
        _numeric_df,
        {"columns": ["a"], "rule": "clip", "min_value": -3, "max_value": 3},
    ),
    (
        ValueReplacementCalculator,
        art.ValueReplacementArtifact,
        "value_replacement",
        _numeric_df,
        {"columns": ["a"], "to_replace": 0, "value": np.nan},
    ),
    (
        AliasReplacementCalculator,
        art.AliasReplacementArtifact,
        "alias_replacement",
        _bool_alias_df,
        {"columns": ["flag"], "alias_type": "boolean"},
    ),
    # Inspection ----------------------------------------------------------
    (DatasetProfileCalculator, art.DatasetProfileArtifact, "dataset_profile", _numeric_df, {}),
    (DataSnapshotCalculator, art.DataSnapshotArtifact, "data_snapshot", _numeric_df, {"n_rows": 5}),
    # Encoders ------------------------------------------------------------
    (
        OneHotEncoderCalculator,
        art.OneHotArtifact,
        "onehot",
        _categorical_df,
        {"columns": ["color"]},
    ),
    (
        OrdinalEncoderCalculator,
        art.OrdinalArtifact,
        "ordinal",
        _categorical_df,
        {"columns": ["color"]},
    ),
    (
        LabelEncoderCalculator,
        art.LabelEncoderArtifact,
        "label_encoder",
        _categorical_df,
        {"columns": ["color"]},
    ),
    (
        TargetEncoderCalculator,
        art.TargetEncoderArtifact,
        "target_encoder",
        _mixed_with_target_df,
        {"columns": ["cat"], "target_column": "target"},
    ),
    (
        HashEncoderCalculator,
        art.HashEncoderArtifact,
        "hash_encoder",
        _categorical_df,
        {"columns": ["color"], "n_features": 4},
    ),
    (
        DummyEncoderCalculator,
        art.DummyEncoderArtifact,
        "dummy_encoder",
        _categorical_df,
        {"columns": ["color"]},
    ),
]


def _ids(
    case: Tuple[Type[BaseCalculator], Type[Any], Optional[str], Callable[[], Any], Dict[str, Any]]
) -> str:
    return case[0].__name__


@pytest.mark.parametrize(
    "calc_cls,artifact_cls,expected_type,make_data,config", CASES, ids=[_ids(c) for c in CASES]
)
def test_artifact_shape(
    calc_cls: Type[BaseCalculator],
    artifact_cls: Type[Any],
    expected_type: Optional[str],
    make_data: Callable[[], Any],
    config: Dict[str, Any],
) -> None:
    """Drive ``calc.fit(df, config)`` and validate the returned artifact."""
    df = make_data()
    result = calc_cls().fit(df, config)

    # 1) Always a dict.
    assert isinstance(
        result, dict
    ), f"{calc_cls.__name__}.fit returned {type(result).__name__}, not dict"

    # 2) Empty-result early-exits are allowed (e.g. user picked no columns).
    if not result:
        return

    # 3) "type" discriminator must match.
    if expected_type is not None:
        assert result.get("type") == expected_type, (
            f"{calc_cls.__name__}.fit returned type={result.get('type')!r}, "
            f"expected {expected_type!r}"
        )

    # 4) Every key in the result must be declared on the TypedDict.
    declared_keys = set(artifact_cls.__annotations__.keys())
    extra_keys = set(result.keys()) - declared_keys
    assert not extra_keys, (
        f"{calc_cls.__name__}.fit returned undeclared keys "
        f"{sorted(extra_keys)} (declared: {sorted(declared_keys)})"
    )


# ---------- Split calculators (typed Artifact since C8) ----------


def test_split_calculator_returns_typed_artifact() -> None:
    """SplitCalculator builds a typed SplitArtifact from known config keys."""
    cfg = {"test_size": 0.2, "random_state": 42, "shuffle": True}
    result = SplitCalculator().fit(_numeric_df(), cfg)
    assert isinstance(result, dict)
    assert result["type"] == "split"
    # Known keys are forwarded; extra config keys are dropped.
    assert result["test_size"] == 0.2
    assert result["random_state"] == 42
    assert result["shuffle"] is True
    extra = set(result.keys()) - set(art.SplitArtifact.__annotations__.keys())
    assert not extra, f"SplitCalculator produced undeclared keys: {sorted(extra)}"


def test_feature_target_split_calculator_returns_typed_artifact() -> None:
    cfg = {"target_column": "target"}
    result = FeatureTargetSplitCalculator().fit(_mixed_with_target_df(), cfg)
    assert isinstance(result, dict)
    assert result["type"] == "feature_target_split"
    assert result["target_column"] == "target"
    extra = set(result.keys()) - set(art.FeatureTargetSplitArtifact.__annotations__.keys())
    assert not extra, f"FeatureTargetSplitCalculator produced undeclared keys: {sorted(extra)}"


# ---------- Cross-cutting: every TypedDict has a "type" field ----------


def test_every_artifact_typeddict_declares_type_field() -> None:
    """Sanity: every artifact TypedDict in _artifacts.py declares 'type'."""
    import inspect

    for name, obj in inspect.getmembers(art):
        # TypedDicts are classes that have __annotations__ and subclass dict.
        if not (
            inspect.isclass(obj)
            and getattr(obj, "__module__", "") == art.__name__
            and hasattr(obj, "__annotations__")
        ):
            continue
        assert (
            "type" in obj.__annotations__
        ), f"{name} TypedDict is missing the 'type' discriminator field"
