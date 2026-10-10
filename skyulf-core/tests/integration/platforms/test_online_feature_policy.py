"""Strict online freshness contracts and pre-preprocessing guard tests."""

import json
from typing import Any

import pandas as pd
import pytest

from skyulf.integrations.databricks.feature_store.config import (
    FeatureLookupSpec,
    FeatureTrainingSpec,
)


def _policy(**kwargs):
    """Use a real policy so malformed metadata cannot bypass construction."""
    from skyulf.integrations.databricks.feature_store.online_policy import OnlineFeaturePolicy

    options: dict[str, Any] = {
        "required_features": ("x", "fresh_at"),
        "freshness": (("fresh_at", 60.0),),
        **kwargs,
    }
    return OnlineFeaturePolicy(**options)


def test_roundtrip_is_immutable_and_strict():
    """Serialized latest policy must detach its values and reject unknown fields."""
    from skyulf.integrations.databricks.feature_store.online_policy import OnlineFeaturePolicy

    policy = _policy()
    saved = json.loads(json.dumps(policy.to_dict()))
    assert OnlineFeaturePolicy.from_dict(saved) == policy
    saved["required_features"].append("other")
    assert policy.required_features == ("x", "fresh_at")
    with pytest.raises(ValueError):
        OnlineFeaturePolicy.from_dict({**policy.to_dict(), "clock": 0})


@pytest.mark.parametrize("age", [True, False, 0, -1, float("nan"), float("inf"), "60"])
def test_invalid_age_rejected(age):
    """Invalid bounds cannot disable expiry or convert booleans into seconds."""
    with pytest.raises(ValueError):
        _policy(freshness=(("fresh_at", age),))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"required_features": ("x", "x")},
        {"freshness": (("fresh_at", 5), ("fresh_at", 10))},
        {"freshness": (("missing", 5),)},
        {"freshness": ()},
    ],
)
def test_invalid_columns_rejected(kwargs):
    """Every freshness rule must belong to distinct explicit fetched columns."""
    with pytest.raises(ValueError):
        _policy(**kwargs)


@pytest.mark.parametrize(
    "value", [None, float("nan"), float("inf"), float("-inf"), 939.99, 1000.01, True, "1000"]
)
def test_invalid_timestamp_fails(value):
    """Missing, stale and future rows cannot reach preprocessing or be silently removed."""
    from skyulf.integrations.databricks.feature_store.online_policy import validate_online_features

    frame = pd.DataFrame({"x": [3.0, 4.0], "fresh_at": [1000.0, value]})
    with pytest.raises(ValueError):
        validate_online_features(frame, _policy(), now=1000.0)


@pytest.mark.parametrize(
    "frame",
    [pd.DataFrame({"fresh_at": [1000.0]}), pd.DataFrame({"x": [None], "fresh_at": [1000.0]})],
)
def test_missing_feature_fails(frame):
    """A fitted imputer must not conceal incomplete feature lookup results."""
    from skyulf.integrations.databricks.feature_store.online_policy import validate_online_features

    with pytest.raises(ValueError, match="required"):
        validate_online_features(frame, _policy(), now=1000.0)


def test_fresh_boundary_keeps_input_unchanged():
    """The inclusive age boundary preserves the original values and row order."""
    from skyulf.integrations.databricks.feature_store.online_policy import validate_online_features

    frame = pd.DataFrame({"x": [3.0, 4.0], "fresh_at": [940.0, 1000.0]})
    before = frame.copy(deep=True)
    validate_online_features(frame, _policy(), now=1000.0)
    pd.testing.assert_frame_equal(frame, before)


def test_lookup_requires_exact_coverage_and_timestamp_for_each_table():
    """One table freshness column must never certify an unrelated table."""
    spec = FeatureTrainingSpec(
        lookups=(
            FeatureLookupSpec(
                table_name="main.features.first",
                lookup_key=("id",),
                feature_names=("x", "fresh_at"),
            ),
            FeatureLookupSpec(
                table_name="main.features.second", lookup_key=("id",), feature_names=("y",)
            ),
        ),
        label=None,
    )
    with pytest.raises(ValueError, match="lookup"):
        _policy().validate_lookup(spec, {"x": "double", "fresh_at": "double", "y": "double"})
    with pytest.raises(ValueError, match="freshness"):
        _policy(required_features=("x", "fresh_at", "y")).validate_lookup(
            spec, {"x": "double", "fresh_at": "double", "y": "double"}
        )


@pytest.mark.parametrize("dtype", ["int64", "float32", "datetime", "string", "Float64"])
def test_timestamp_requires_exact_double(dtype):
    """Only existing ordinary float64 epoch features are supported at this boundary."""
    if dtype == "Float64":
        _policy().validate_schema({"x": "double", "fresh_at": dtype})
    else:
        with pytest.raises(ValueError, match="float64"):
            _policy().validate_schema({"x": "double", "fresh_at": dtype})


def test_unrepresentable_maximum_age_is_rejected():
    """A JSON integer too large for seconds must fail as a policy validation error."""
    with pytest.raises(ValueError, match="finite"):
        _policy(freshness=(("fresh_at", 10**1000),))


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"version": 2},
        {"policy": "training_snapshot"},
        {"required_features": "x"},
        {"freshness": "fresh_at"},
        {"freshness": [{"fresh_at": 60}]},
        {"freshness": [["fresh_at"]]},
    ],
)
def test_malformed_serialized_policy_rejected(change):
    """Reloading cannot coerce malformed persisted policy structures into valid rules."""
    from skyulf.integrations.databricks.feature_store.online_policy import OnlineFeaturePolicy

    with pytest.raises(ValueError):
        OnlineFeaturePolicy.from_dict({**_policy().to_dict(), **change})
