"""Strict Python-batch validation for fitted company preprocessing and tree models.

These checks deliberately do not extend the portable JSON codecs or native Spark
execution. Estimator objects remain trusted, checksum-verified Python artifacts.
"""

import json
import math
from importlib import import_module
from typing import Any

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from ..core.portable_state import _normalize, validate_state
from ..preprocessing.encoding.one_hot import OneHotEncoderApplier, OneHotEncoderCalculator
from ..preprocessing.imputation.group import GroupImputerApplier, GroupImputerCalculator
from ..preprocessing.outliers.clip_values import (
    ClipValuesApplier,
    ClipValuesCalculator,
    _clean_bound,
)
from ..registry import NodeRegistry

APPLIERS = {
    "ClipValues": ClipValuesApplier,
    "GroupImputer": GroupImputerApplier,
    "OneHotEncoder": OneHotEncoderApplier,
}
CALCULATORS = {
    "ClipValues": ClipValuesCalculator,
    "GroupImputer": GroupImputerCalculator,
    "OneHotEncoder": OneHotEncoderCalculator,
}


def _fields(value: Any, names: set[str]) -> None:
    """Reject unknown or missing state fields rather than ignoring new behavior."""
    if type(value) is not dict or set(value) != names:
        raise ValueError("Unexpected Python-batch state fields.")


def _columns(value: Any) -> list[str]:
    """Require unambiguous ordered feature names."""
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError("Fitted columns must be a list of strings.")
    if len(set(value)) != len(value):
        raise ValueError("Fitted columns must be unique.")
    return value


def _scalar(value: Any) -> None:
    """Allow only finite learned scalars without callbacks or nested objects."""
    if type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("Learned values must be scalar.")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Learned values must be finite.")


def _bounds(raw: Any) -> dict:
    """Normalize only explicit finite clipping limits."""
    if type(raw) is not dict or any(type(key) is not str for key in raw):
        raise ValueError("Clip bounds must map column names to limits.")
    result = {}
    for column, bound in raw.items():
        if type(bound) is not dict or set(bound) - {"lower", "upper"}:
            raise ValueError("Unknown clipping option.")
        result[column] = _clean_bound(column, bound)
    return result


def _clip_state(raw: dict) -> dict:
    """Bind the fitted clipping artifact to its known fixed-bound schema."""
    state = _normalize(raw)
    _fields(state, {"type", "bounds"})
    if state["type"] != "clip_values":
        raise ValueError("Wrong clipping artifact type.")
    if state["bounds"] != _bounds(state["bounds"]):
        raise ValueError("Fitted clipping bounds must be normalized.")
    return state


def _group_pairs(pairs: Any, numeric: bool) -> None:
    """Require unique scalar group keys and finite per-group replacements."""
    if type(pairs) is not list:
        raise ValueError("Group replacements must be a list.")
    keys = set()
    for pair in pairs:
        if type(pair) is not list or len(pair) != 2:
            raise ValueError("Group replacements must contain key/value pairs.")
        key, value = pair
        _scalar(key)
        _scalar(value)
        if key is None or key in keys:
            raise ValueError("Group keys must be non-null and unique.")
        keys.add(key)
        if numeric and type(value) not in (int, float):
            raise ValueError("Group means must be numeric.")


def _group_state(raw: dict) -> dict:
    """Validate learned group maps and the global fallback without recomputing either."""
    state = _normalize(raw)
    _fields(state, {"type", "group_by", "strategy", "columns", "group_values", "fill_values"})
    columns = _columns(state["columns"])
    if state["type"] != "group_imputer" or state["strategy"] not in ("mean", "most_frequent"):
        raise ValueError("Unsupported group imputer state.")
    if type(state["group_by"]) is not str or state["group_by"] in columns:
        raise ValueError("Invalid group key.")
    _fields(state["group_values"], set(columns))
    _fields(state["fill_values"], set(columns))
    for column in columns:
        fallback = state["fill_values"][column]
        _scalar(fallback)
        if state["strategy"] == "mean" and type(fallback) not in (int, float, type(None)):
            raise ValueError("Global group means must be numeric.")
        _group_pairs(state["group_values"][column], state["strategy"] == "mean")
    return state


def _mode_state(raw: dict) -> dict:
    """Reuse scalar/count validation without expanding the portable strategy vocabulary."""
    if raw.get("strategy") != "most_frequent":
        raise ValueError("Expected a most-frequent imputer.")
    state = validate_state("SimpleImputer", {**raw, "strategy": "mean"})
    for value in state["fill_values"].values():
        _scalar(value)
    return {**state, "strategy": "most_frequent"}


def _check_encoder(encoder: Any, columns: list[str]) -> None:
    """Admit the exact dense fitted sklearn encoder without arbitrary callables."""
    if type(encoder) is not OneHotEncoder:
        raise ValueError("Expected exact OneHotEncoder.")
    expected = {
        "categories",
        "sparse_output",
        "dtype",
        "handle_unknown",
        "drop",
        "min_frequency",
        "max_categories",
        "feature_name_combiner",
        "_infrequent_enabled",
        "n_features_in_",
        "categories_",
        "_drop_idx_after_grouping",
        "drop_idx_",
        "_n_features_outs",
    }
    _fields(vars(encoder), expected)
    _encoder_options(encoder)
    if encoder.drop not in (None, "first") or encoder.n_features_in_ != len(columns):
        raise ValueError("Encoder feature count or dropped-category policy disagrees.")
    _encoder_categories(encoder, columns)


def _encoder_options(encoder: Any) -> None:
    """Restrict serving to deterministic dense encoding without infrequent categories."""
    if encoder.dtype is not np.int8 or encoder.feature_name_combiner != "concat":
        raise ValueError("Unsupported encoder dtype or feature-name callback.")
    if encoder.categories != "auto" or encoder.sparse_output is not False:
        raise ValueError("Encoder requires automatic categories and dense output.")
    if encoder.max_categories is not None or encoder.min_frequency is not None:
        raise ValueError("Infrequent-category grouping requires separate batch admission.")
    if encoder._infrequent_enabled is not False or encoder.handle_unknown not in (
        "ignore",
        "error",
    ):
        raise ValueError("Unsupported encoder inference policy.")


def _encoder_categories(encoder: Any, columns: list[str]) -> None:
    """Validate category widths and dropped indices before calling known name generation."""
    if type(encoder.categories_) is not list or len(encoder.categories_) != len(columns):
        raise ValueError("Encoder categories must align with columns.")
    widths = []
    for categories in encoder.categories_:
        if type(categories) is not np.ndarray or categories.ndim != 1 or not len(categories):
            raise ValueError("Encoder categories must be nonempty vectors.")
        values = _normalize(categories.tolist())
        for value in values:
            _scalar(value)
        if len(set(values)) != len(values):
            raise ValueError("Encoder categories must be unique.")
        widths.append(len(values) - int(encoder.drop == "first"))
    if encoder._n_features_outs != widths:
        raise ValueError("Encoder output widths disagree with fitted categories.")
    _encoder_drop_indices(encoder, len(columns))


def _encoder_drop_indices(encoder: Any, count: int) -> None:
    """Bind both sklearn dropped-index fields to the supported drop policy."""
    for indices in (encoder.drop_idx_, encoder._drop_idx_after_grouping):
        if encoder.drop is None:
            if indices is not None:
                raise ValueError("Unexpected dropped category indices.")
        elif type(indices) is not np.ndarray or indices.tolist() != [0] * count:
            raise ValueError("Dropped indices must select the first fitted category.")


def _onehot_state(raw: dict) -> dict:
    """Keep the estimator in the certificate while validating every execution option."""
    _fields(
        raw,
        {
            "type",
            "columns",
            "encoder_object",
            "feature_names",
            "prefix_separator",
            "drop_original",
            "include_missing",
        },
    )
    scalar = _normalize({key: value for key, value in raw.items() if key != "encoder_object"})
    columns = _columns(scalar["columns"])
    _columns(scalar["feature_names"])
    if scalar["type"] != "onehot" or scalar["include_missing"] is not False:
        raise ValueError("Only observed-category one-hot artifacts are admitted.")
    if type(scalar["drop_original"]) is not bool or type(scalar["prefix_separator"]) is not str:
        raise ValueError("Invalid encoder output options.")
    encoder = raw["encoder_object"]
    _check_encoder(encoder, columns)
    if encoder.get_feature_names_out(columns).tolist() != scalar["feature_names"]:
        raise ValueError("Encoder feature names disagree with fitted categories.")
    return {**scalar, "encoder_object": encoder}


def batch_state(node: str, raw: dict) -> dict:
    """Inspect the narrow Python object vocabulary independently of JSON codecs."""
    validators = {
        "ClipValues": _clip_state,
        "GroupImputer": _group_state,
        "OneHotEncoder": _onehot_state,
        "SimpleImputer": _mode_state,
    }
    return validators[node](raw)


def batch_config(node: str, raw: dict, state: dict) -> dict:
    """Resolve defaults and bind configured behavior to the saved fitted state."""
    params = _normalize(raw)
    params.pop("target_column", None)
    auto = params.pop("_auto_columns", False)
    if node == "ClipValues":
        _fields(params, {"bounds"})
        bounds = _bounds(params["bounds"])
        if bounds != state["bounds"]:
            raise ValueError("Configured bounds disagree with fitted clipping bounds.")
        return {"bounds": bounds}
    columns = params.get("columns")
    if columns is not None and not auto and columns != state["columns"]:
        raise ValueError("Configured columns disagree with fitted columns.")
    params["columns"] = state["columns"]
    return (
        _imputation_config(node, params, state)
        if node != "OneHotEncoder"
        else _onehot_config(params, state)
    )


def _imputation_config(node: str, params: dict, state: dict) -> dict:
    """Bind group and modal fill strategies to their learned artifact."""
    defaults: dict[str, Any] = {"strategy": "mean"}
    if node == "SimpleImputer":
        defaults["fill_value"] = None
    resolved = {**defaults, **params}
    fields = {"columns", "strategy", "group_by" if node == "GroupImputer" else "fill_value"}
    _fields(resolved, fields)
    if resolved["strategy"] == "mode" and node == "GroupImputer":
        resolved["strategy"] = "most_frequent"
    if resolved["strategy"] != state["strategy"]:
        raise ValueError("Configured strategy disagrees with fitted strategy.")
    if node == "GroupImputer" and resolved["group_by"] != state["group_by"]:
        raise ValueError("Configured group key disagrees with fitted key.")
    if node == "SimpleImputer" and resolved["fill_value"] is not None:
        raise ValueError("Mode imputation does not use a configured constant.")
    return resolved


def _onehot_config(params: dict, state: dict) -> dict:
    """Check recipe options against the encoder that will actually transform rows."""
    resolved = {
        "drop_first": False,
        "max_categories": 20,
        "handle_unknown": "ignore",
        "prefix_separator": "_",
        "drop_original": True,
        "include_missing": False,
        **params,
    }
    _fields(
        resolved,
        {
            "columns",
            "drop_first",
            "max_categories",
            "handle_unknown",
            "prefix_separator",
            "drop_original",
            "include_missing",
        },
    )
    encoder = state["encoder_object"]
    expected = {
        "drop_first": encoder.drop == "first",
        "max_categories": encoder.max_categories,
        "handle_unknown": encoder.handle_unknown,
        **{key: state[key] for key in ("prefix_separator", "drop_original", "include_missing")},
    }
    for key, value in expected.items():
        if type(resolved[key]) is not type(value) or resolved[key] != value:
            raise ValueError(f"Configured encoder {key} disagrees with fitted state.")
    return resolved


def xgboost_applier(model: Any) -> type | None:
    """Lazily admit exact CPU gbtree regressors and their registered built-in applier."""
    if type(model).__module__ != "xgboost.sklearn" or type(model).__name__ != "XGBRegressor":
        return None
    xgb = import_module("xgboost")
    regression = import_module("skyulf.modeling.regression")
    if type(model) is not xgb.XGBRegressor:
        raise ValueError("Expected exact XGBRegressor.")
    if NodeRegistry.get_calculator("xgboost_regressor") is not regression.XGBRegressorCalculator:
        raise ValueError("Unreviewed XGBoost calculator registration.")
    applier = regression.XGBRegressorApplier
    if NodeRegistry.get_applier("xgboost_regressor") is not applier:
        raise ValueError("Unreviewed XGBoost applier registration.")
    _check_xgboost_state(model, xgb.Booster)
    return applier


def _check_xgboost_state(model: Any, booster_type: type) -> None:
    """Reject custom objectives/callbacks, DART and substituted native booster wrappers."""
    if any(callable(value) for value in vars(model).values()):
        raise ValueError("Overridden XGBoost inference methods are unsupported.")
    if model.callbacks is not None or model.booster not in (None, "gbtree"):
        raise ValueError("XGBoost callbacks and non-gbtree boosters are unsupported.")
    if model.objective not in ("reg:squarederror", "reg:logistic"):
        raise ValueError("Only squared-error and logistic regression objectives are admitted.")
    _normalize(getattr(model, "kwargs", {}))
    _check_booster(model, booster_type)


def _check_booster(model: Any, booster_type: type) -> None:
    """Bind the trusted estimator wrapper to its actual native tree/objective payload."""
    booster = model.get_booster()
    if type(booster) is not booster_type or any(
        callable(value) for value in vars(booster).values()
    ):
        raise ValueError("Overridden XGBoost booster methods are unsupported.")
    learner = json.loads(booster.save_config())["learner"]
    if learner["gradient_booster"]["name"] != "gbtree":
        raise ValueError("Only fitted gbtree boosters are admitted.")
    if learner["objective"]["name"] != model.objective:
        raise ValueError("Fitted XGBoost objective disagrees with estimator configuration.")
