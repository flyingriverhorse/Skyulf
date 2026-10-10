"""Immutable latest-feature freshness policy evaluated before fitted preprocessing."""

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import FeatureTrainingSpec, validate_feature_columns

ONLINE_FEATURES_KEY = "online_features"


@dataclass(frozen=True, slots=True, kw_only=True)
class OnlineFeaturePolicy:
    """Require fetched values and bound their age using numeric UTC epoch seconds.

    Each lookup must include its own fitted float64 freshness feature. Native
    serving accepts request feature overrides; this policy validates received
    values and cannot authenticate whether they originated in the online store.
    """

    required_features: tuple[str, ...]
    freshness: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        """Reject mutable, ambiguous and nonfinite policy declarations."""
        validate_feature_columns(self.required_features, "required_features")
        if not isinstance(self.freshness, tuple) or not self.freshness:
            raise ValueError("freshness must contain explicit immutable age rules.")
        for rule in self.freshness:
            _validate_rule(rule)
        names = tuple(name for name, _ in self.freshness)
        validate_feature_columns(names, "freshness")
        if not set(names).issubset(self.required_features):
            raise ValueError("freshness columns must be required features.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize a detached JSON-compatible latest-only policy."""
        return {
            "version": 1,
            "policy": "latest",
            "required_features": list(self.required_features),
            "freshness": [list(rule) for rule in self.freshness],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "OnlineFeaturePolicy":
        """Read only the exact supported metadata format, without coercion."""
        fields = {"version", "policy", "required_features", "freshness"}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError("Online feature policy fields differ from the supported contract.")
        if (
            type(value["version"]) is not int
            or value["version"] != 1
            or value["policy"] != "latest"
        ):
            raise ValueError("Online feature policy requires version 1 and latest policy.")
        _validate_serialized_lists(value)
        return cls(
            required_features=tuple(value["required_features"]),
            freshness=tuple(tuple(rule) for rule in value["freshness"]),
        )

    def validate_schema(self, columns: dict[str, str]) -> None:
        """Require all guarded values to exist in the already fitted raw schema."""
        if not set(self.required_features).issubset(columns):
            raise ValueError("Online required features are missing from fitted raw inputs.")
        if any(columns[name] not in {"float64", "Float64", "double"} for name, _ in self.freshness):
            raise ValueError("Online freshness features must be fitted float64 UTC epoch seconds.")

    def validate_lookup(self, spec: FeatureTrainingSpec, columns: dict[str, str]) -> None:
        """Bind every fetched feature and each lookup group to its own freshness rule."""
        self.validate_schema(columns)
        if set(self.required_features) != set(spec.feature_names):
            raise ValueError("Online required features must exactly cover lookup outputs.")
        timestamps = {name for name, _ in self.freshness}
        if any(not timestamps.intersection(lookup.feature_names) for lookup in spec.lookups):
            raise ValueError("Every lookup must include its own freshness feature.")


def _validate_rule(rule: Any) -> None:
    """Keep freshness declarations immutable and age bounds strictly positive."""
    if not isinstance(rule, tuple) or len(rule) != 2:
        raise ValueError("Each freshness rule must be a column and maximum age pair.")
    age = rule[1]
    if not _finite_number(age) or age <= 0:
        raise ValueError("Freshness maximum age must be a positive finite number of seconds.")


def validate_online_features(
    frame: pd.DataFrame, policy: OnlineFeaturePolicy | None, *, now: float | None = None
) -> None:
    """Reject incomplete, stale or future rows before imputation or row filtering.

    ``now`` is available for standalone deterministic validation only. Saved
    PythonModel prediction always uses the real UTC clock, once per request.
    """
    if policy is None:
        return
    if not frame.columns.is_unique or not set(policy.required_features).issubset(frame.columns):
        raise ValueError("Online required feature columns are missing or duplicated.")
    if frame.loc[:, list(policy.required_features)].isna().any().any():
        raise ValueError("Online required feature values must be present and non-null.")
    instant = time.time() if now is None else now
    if not _finite_number(instant):
        raise ValueError("Online validation clock must be finite UTC epoch seconds.")
    for name, maximum_age in policy.freshness:
        _validate_timestamp(frame[name], name, maximum_age, instant)


def _validate_timestamp(values: pd.Series, name: str, maximum_age: float, now: float) -> None:
    """Validate exact numeric times without accepting strings or booleans."""
    if not pd.api.types.is_numeric_dtype(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
        raise ValueError(f"Online freshness feature {name!r} must contain numeric epoch seconds.")
    numbers = values.to_numpy(dtype=float)
    ages = now - numbers
    if not np.isfinite(numbers).all() or (ages < 0).any() or (ages > maximum_age).any():
        raise ValueError(f"Online freshness feature {name!r} is nonfinite, stale or future.")


def saved_online_policy(
    value: Any, columns: dict[str, str], metadata: Any = None
) -> OnlineFeaturePolicy | None:
    """Validate serialized PythonModel state against raw schema and saved metadata."""
    if value is None:
        if isinstance(metadata, dict) and metadata.get(ONLINE_FEATURES_KEY) is not None:
            raise ValueError("Online policy metadata differs from saved PythonModel state.")
        return None
    policy = OnlineFeaturePolicy.from_dict(value)
    policy.validate_schema(columns)
    if isinstance(metadata, dict) and metadata.get(ONLINE_FEATURES_KEY) != policy.to_dict():
        raise ValueError("Online policy metadata differs from saved PythonModel state.")
    return policy


def _validate_serialized_lists(value: dict[str, Any]) -> None:
    """Require JSON arrays instead of coercing arbitrary iterables into policy rules."""
    if not isinstance(value["required_features"], list) or not isinstance(value["freshness"], list):
        raise ValueError("Online feature policy columns and freshness must be lists.")
    if any(not isinstance(rule, list) for rule in value["freshness"]):
        raise ValueError("Online freshness rules must be lists.")


def _finite_number(value: Any) -> bool:
    """Reject booleans and integers outside finite machine-second representation."""
    if type(value) not in {int, float}:
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False
