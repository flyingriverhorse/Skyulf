"""Exact built-in step identities eligible for observed partition admission."""

from collections.abc import Iterable
from importlib import import_module
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder

from ..preprocessing._exclusion import is_exclusion_step
from ..preprocessing.inference_context import get_inference_capability
from ..registry import NodeRegistry
from ._fitted_contract import resolve_fitted_step

# Explicit module/class names avoid treating arbitrary registry replacements as
# reviewed code. Aliases point to their actual owner; optional imports stay lazy.
_OWNERS = {
    "AliasReplacement": ("cleaning.alias", "AliasReplacement"),
    "Casting": ("casting", "Casting"),
    "CorrelationThreshold": ("feature_selection.correlation", "CorrelationThreshold"),
    "CustomBinning": ("bucketing", "CustomBinning"),
    "DateFeatures": ("time_series.date_features", "DateFeatures"),
    "DropMissingColumns": ("drop_and_missing.drop_columns", "DropMissingColumns"),
    "DummyEncoder": ("encoding.dummy", "DummyEncoder"),
    "EllipticEnvelope": ("outliers.elliptic", "EllipticEnvelope"),
    "FeatureGeneration": ("feature_generation.generation", "FeatureGeneration"),
    "FeatureGenerationNode": ("feature_generation.generation", "FeatureGeneration"),
    "FeatureMath": ("feature_generation.generation", "FeatureGeneration"),
    "FrequencyEncoder": ("encoding.frequency", "FrequencyEncoder"),
    "GeneralBinning": ("bucketing", "GeneralBinning"),
    "GeneralTransformation": ("transformations.general", "GeneralTransformation"),
    "GeoDistance": ("geo.distance", "GeoDistance"),
    "H3Index": ("geo.h3_index", "H3Index"),
    "HashEncoder": ("encoding.hash", "HashEncoder"),
    "InvalidValueReplacement": ("cleaning.invalid_value", "InvalidValueReplacement"),
    "IQR": ("outliers.iqr", "IQR"),
    "IsolationForest": ("outliers.isolation_forest", "IsolationForest"),
    "LocalOutlierFactor": ("outliers.local_outlier_factor", "LocalOutlierFactor"),
    "IterativeImputer": ("imputation.iterative", "IterativeImputer"),
    "KBinsDiscretizer": ("bucketing", "KBinsDiscretizer"),
    "KNNImputer": ("imputation.knn", "KNNImputer"),
    "LabelEncoder": ("encoding.label", "LabelEncoder"),
    "ManualBounds": ("outliers.manual_bounds", "ManualBounds"),
    "MaxAbsScaler": ("scaling.maxabs", "MaxAbsScaler"),
    "MissingIndicator": ("drop_and_missing.missing_indicator", "MissingIndicator"),
    "ModelBasedSelection": ("feature_selection.model_based", "ModelBasedSelection"),
    "Normalizer": ("scaling.normalizer", "Normalizer"),
    "OrdinalEncoder": ("encoding.ordinal", "OrdinalEncoder"),
    "PolynomialFeatures": ("feature_generation.polynomial", "PolynomialFeatures"),
    "PolynomialFeaturesNode": ("feature_generation.polynomial", "PolynomialFeatures"),
    "PowerTransformer": ("transformations.power", "PowerTransformer"),
    "QuantileTransformer": ("scaling.quantile", "QuantileTransformer"),
    "RareCategoryGrouping": ("encoding.rare", "RareCategoryGrouping"),
    "RobustScaler": ("scaling.robust", "RobustScaler"),
    "SimpleTransformation": ("transformations.simple", "SimpleTransformation"),
    "TargetEncoder": ("encoding.target", "TargetEncoder"),
    "TextCleaning": ("cleaning.text", "TextCleaning"),
    "UnivariateSelection": ("feature_selection.univariate", "UnivariateSelection"),
    "ValueReplacement": ("cleaning.value_replacement", "ValueReplacement"),
    "VarianceThreshold": ("feature_selection.variance", "VarianceThreshold"),
    "WOEEncoder": ("encoding.woe", "WOEEncoder"),
    "Winsorize": ("outliers.winsorize", "Winsorize"),
    "ZScore": ("outliers.zscore", "ZScore"),
    "feature_selection": ("feature_selection.facade", "FeatureSelection"),
    "count_vectorizer": ("vectorization.count_vectorizer", "CountVectorizer"),
    "hashing_vectorizer": ("vectorization.hashing_vectorizer", "HashingVectorizer"),
    "tfidf_vectorizer": ("vectorization.tfidf_vectorizer", "TfidfVectorizer"),
    "tokenizer": ("vectorization.tokenizer", "Tokenizer"),
}
PROBED_BUILTINS = tuple(_OWNERS)
_DTYPE_OWNERS = {CountVectorizer, HashingVectorizer, TfidfVectorizer, OrdinalEncoder}
_NUMERIC_TYPES = frozenset(np.dtype(code).type for code in "?bhilqBHILQefdgFDG")


def _state_fields(value: Any) -> Iterable[tuple[Any, Any]]:
    """Visit nested estimator attributes and containers, including object arrays."""
    if isinstance(value, dict):
        return value.items()
    if isinstance(value, (list, tuple, set, frozenset)):
        return enumerate(value)
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        return enumerate(value.flat)
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return vars(value).items()
    return ()


def _dtype_constructor(owner: Any, name: Any, value: Any) -> bool:
    """Allow exact numeric dtype constructors in their known sklearn configuration slots."""
    return (
        type(owner) in _DTYPE_OWNERS
        and name == "dtype"
        and isinstance(value, type)
        and value in _NUMERIC_TYPES
    )


def _check_nested_state(value: Any, seen: set[int]) -> None:
    """Reject executable overrides that the semantic state digest deliberately omits."""
    if id(value) in seen:
        return
    seen.add(id(value))
    for name, item in _state_fields(value):
        replaced_method = (
            not isinstance(value, dict)
            and isinstance(name, str)
            and callable(getattr(type(value), name, None))
        )
        if replaced_method or (callable(item) and not _dtype_constructor(value, name, item)):
            raise ValueError("Overridden built-in transform methods or callbacks are unsupported.")
        _check_nested_state(item, seen)


def builtin_applier(node: str) -> type | None:
    """Resolve the exact built-in owner without accepting registry substitutions."""
    owner = _OWNERS.get(node)
    if owner is None:
        return None
    module_name, prefix = owner
    module = import_module(f"skyulf.preprocessing.{module_name}")
    calculator = getattr(module, f"{prefix}Calculator")
    applier = getattr(module, f"{prefix}Applier")
    if NodeRegistry.get_calculator(node) is not calculator:
        raise ValueError("Unreviewed built-in calculator registration.")
    if NodeRegistry.get_applier(node) is not applier:
        raise ValueError("Unreviewed built-in applier registration.")
    return applier


def builtin_contract(record: dict, config: dict) -> tuple[Any, dict]:
    """Require fitted row-local behavior, explicitly preserving row exclusion outcomes."""
    node = record["type"]
    applier = record["applier"]
    expected = builtin_applier(node)
    if expected is None:
        raise ValueError(f"Built-in step {node!r} is not eligible for partition probes.")
    if type(applier) is not expected or vars(applier):
        raise ValueError("A stateless exact built-in applier is required.")
    _check_nested_state(record["artifact"], set())
    state, params, _ = resolve_fitted_step(record, config, require_portable=False)
    capability = get_inference_capability(node, params, state, engine="pandas", applier=applier)
    row_effects = {"filter", "preserve"} if is_exclusion_step(record) else {"preserve"}
    if (
        capability is None
        or capability.context != "row"
        or capability.row_effect not in row_effects
    ):
        raise ValueError(
            f"Built-in step {record['name']!r} ({node}) requires validated row-preserving "
            "inference. Filtering rows or using other request rows cannot be admitted by a probe."
        )
    return state, params


def fresh_builtin_record(record: dict) -> dict:
    """Restore a new stateless applier around the caller's detached fitted state."""
    expected = builtin_applier(record["type"])
    if expected is None or type(record["applier"]) is not expected:
        raise ValueError("A known built-in applier is required for fresh replay.")
    return {**record, "applier": expected()}
