"""Admission, typed thresholds and calibration must preserve the caller's contract."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks.decision_thresholds import manual_thresholds
from skyulf.integrations.databricks.local_cv import LocalCVSpec, validate_fold_membership
from skyulf.integrations.databricks.threshold_training import _calibration_data
from skyulf.preprocessing._target_labels import encoded_label


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_shuffle_admission_uses_actual_twenty_percent_folds(engine):
    """Repeated shuffle validation has three rows even when ten KFold splits would not."""
    frame = pd.DataFrame({"x": range(12), "target": np.arange(12.0)})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    validate_fold_membership(
        frame,
        LocalCVSpec(enabled=True, folds=10, method="shuffle_split"),
        "target",
        "regression",
        None,
    )
    assert len(frame) == 12


@pytest.mark.parametrize("declared", [True, 1.0, "1"])
def test_manual_positive_class_preserves_scalar_type(declared):
    """Numeric equality cannot bind a differently typed user class declaration."""
    with pytest.raises(ValueError, match="class"):
        manual_thresholds({"positive_class": declared, "value": 0.3}, [0, 1])


@pytest.mark.parametrize("declared", [True, 1.0, "1"])
def test_original_positive_class_is_checked_before_encoding(declared):
    """Encoding must not erase a type mismatch before threshold validation."""
    from types import SimpleNamespace

    pipeline = SimpleNamespace(feature_engineer=SimpleNamespace(fitted_steps=[]))
    with pytest.raises(ValueError, match="class"):
        encoded_label(pipeline, declared, [0, 1])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_auto_calibration_retains_external_evaluation_splits(engine):
    """Training calibration cannot silently discard caller-owned evaluation populations."""
    frame = pd.DataFrame({"x": range(40), "target": [0, 1] * 20})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    data = SplitDataset(
        train=frame[:30],
        test=frame[30:36],
        validation=frame[36:],
        train_sample_weight=np.arange(1.0, 31.0),
    )
    config = {"modeling": {"type": "logistic_regression"}}
    fitting, calibration = _calibration_data(
        config,
        data,
        "target",
        {"mode": "auto", "validation_fraction": 0.2, "random_state": 42},
    )
    assert len(fitting.test) == 6
    assert fitting.validation is not None and len(fitting.validation) == 4
    assert len(fitting.train) + len(calibration) == 30
    assert not isinstance(fitting.train, tuple)
    assert not isinstance(data.test, tuple)
    np.testing.assert_array_equal(fitting.train_sample_weight, fitting.train["x"].to_numpy() + 1)
    np.testing.assert_array_equal(fitting.test["x"], data.test["x"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_set_version", "8"),
        ("model_set_name", "catalog.schema.other"),
        ("model_set_branch", "other"),
    ],
)
def test_scoring_enrollment_cannot_replace_another_active_parent(field, value):
    """A reused component version cannot let late scoring overwrite the active parent binding."""
    import json
    import sqlite3

    from skyulf.integrations.databricks.monitoring_store import _inventory_update

    active = {
        "_activation_started_ms": 1000,
        "model_set_name": "catalog.schema.set",
        "model_set_version": "9",
        "model_set_branch": "branch",
    }
    incoming = {**active, field: value}
    row = {"selection": "version:2", "config_json": json.dumps(incoming)}
    update = _inventory_update(row, None, True)
    condition = update.removeprefix("WHEN MATCHED AND ").partition(" THEN UPDATE")[0]
    # Evaluate the generated predicate with the same null-safe comparison semantics.
    with sqlite3.connect(":memory:") as connection:
        connection.create_function(
            "get_json_object", 2, lambda raw, path: json.loads(raw).get(path[2:])
        )
        query = (
            "SELECT "
            + condition.replace("<=>", "IS")
            + " FROM (SELECT ? AS config_json, 'version:2' AS selection) t"
            + " CROSS JOIN (SELECT ? AS config_json, 'version:2' AS selection) s"
        )
        matched = connection.execute(query, (json.dumps(active), json.dumps(incoming))).fetchone()[
            0
        ]
    assert not matched
