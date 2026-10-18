"""Column rules select training features once and preserve that choice during inference."""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf import SkyulfPipeline
from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.core.schema import SkyulfSchema
from skyulf.data.dataset import SplitDataset
from skyulf.inference._fitted_contract import resolve_fitted_step
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.integrations.databricks.training.shared.preprocessing_checks import seal_partition_probe
from skyulf.preprocessing.column_selection import resolve_column_selection
from skyulf.preprocessing.pipeline import FeatureEngineer


@pytest.mark.parametrize("columns", [False, 0, ""])
def test_column_rules_reject_falsey_malformed_explicit_selection(columns):
    """An invalid explicit selection must not silently widen a rule to all feature columns."""
    with pytest.raises(ValueError, match="columns must be a list"):
        resolve_column_selection(
            pd.DataFrame({"x": [1.0], "private": [2.0]}),
            {"columns": columns, "dtype": "numeric"},
            "StandardScaler",
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "rules,expected",
    [
        ({"columns_regex": "^sensor_", "exclude_columns": ["sensor_id"]}, ["sensor_a"]),
        ({"dtype": "numeric", "exclude_columns": ["sensor_id", "other"]}, ["sensor_a"]),
        ({"columns": ["other", "sensor_a"], "columns_regex": "sensor"}, ["sensor_a"]),
    ],
)
def test_pipeline_resolves_rules_once_in_training_order(engine, rules, expected):
    """A broad default must not scale identifiers or unrelated columns when a rule narrows it."""
    frame = pd.DataFrame(
        {"sensor_a": [2.0, 4.0, 8.0], "sensor_id": [10, 20, 30], "other": [7, 9, 11]}
    )
    native = pl.from_pandas(frame) if engine == "polars" else frame
    before = deepcopy(rules)
    engineer = FeatureEngineer(
        [{"name": "scale", "transformer": "StandardScaler", "params": rules}]
    )
    actual, _ = engineer.fit_transform(native)
    step = engineer.fitted_steps[0]
    assert step["artifact"]["columns"] == expected
    assert step["params"]["columns"] == expected
    assert not {"columns_regex", "dtype", "exclude_columns"} & step["params"].keys()
    actual = actual.to_pandas() if hasattr(actual, "to_pandas") else actual
    np.testing.assert_allclose(actual["sensor_id"], frame["sensor_id"])
    np.testing.assert_allclose(actual["other"], frame["other"])
    assert rules == before


@pytest.mark.parametrize(
    "rules,message",
    [
        ({"columns_regex": "missing"}, "matched no columns"),
        ({"columns_regex": "["}, "columns_regex"),
        ({"dtype": "numbers"}, "dtype"),
        ({"exclude_columns": "sensor_a"}, "exclude_columns"),
    ],
)
def test_pipeline_rejects_invalid_or_empty_column_rules(rules, message):
    """An invalid selection must not silently fall back to every numeric feature."""
    engineer = FeatureEngineer(
        [{"name": "scale", "transformer": "StandardScaler", "params": rules}]
    )
    with pytest.raises(ValueError, match=message):
        engineer.fit_transform(pd.DataFrame({"sensor_a": [2.0, 4.0, 8.0]}))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_rule_pipeline_keeps_learned_columns_and_target_boundary(
    engine, tmp_path, monkeypatch
):
    """Saved scoring must never rediscover numeric columns or accidentally select the target."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    frame = pd.DataFrame(
        {"x": np.arange(24, dtype=float), "id": np.arange(24), "target": np.arange(24) * 3}
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"dtype": "numeric", "exclude_columns": ["id"]},
                }
            ],
            "modeling": {"type": "ridge_regression"},
        }
    )
    native = pl.from_pandas(frame) if engine == "polars" else frame
    pipeline.fit(native, target_column="target")
    assert pipeline.feature_engineer.fitted_steps[0]["artifact"]["columns"] == ["x"]
    save_pipeline(pipeline, tmp_path / "rules")
    restored = load_pipeline(tmp_path / "rules")
    query = frame.drop(columns="target").iloc[[9, 1, 7]]
    query = pl.from_pandas(query) if engine == "polars" else query
    expected = pipeline.predict(query)
    result = predict_pipeline(query, restored)
    pieces = [
        predict_pipeline(
            query.slice(i, 1) if isinstance(query, pl.DataFrame) else query.iloc[[i]], restored
        )
        for i in range(3)
    ]
    np.testing.assert_allclose(result["prediction"], expected)
    np.testing.assert_allclose(result["prediction"], pd.concat(pieces)["prediction"])


def test_selection_uses_training_split_only_and_preserves_config():
    """Held-out rows must not influence dtype-based selection or learned scaling statistics."""
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0], "keep": ["a", "b", "c"]})
    test = pd.DataFrame({"x": [100.0], "keep": ["new"]})
    engineer = FeatureEngineer(
        [{"name": "scale", "transformer": "StandardScaler", "params": {"dtype": "numeric"}}]
    )
    result, _ = engineer.fit_transform(SplitDataset(train=train, test=test))
    assert engineer.fitted_steps[0]["artifact"]["mean"] == [2.0]
    assert list(result.test["keep"]) == ["new"]


def test_native_spark_training_cannot_silently_ignore_column_rules(monkeypatch):
    """Unimplemented native Spark selection must fail before any distributed fit or action."""
    monkeypatch.setattr("skyulf.preprocessing.pipeline.use_spark", lambda *args: True)
    engineer = FeatureEngineer(
        [{"name": "scale", "transformer": "StandardScaler", "params": {"dtype": "numeric"}}]
    )
    with pytest.raises(ValueError, match="explicit columns for native Spark"):
        engineer.fit_transform(pd.DataFrame({"x": [1.0, 2.0]}))


def test_numeric_rules_exclude_timedelta_and_match_preview():
    """Duration values must not silently become numeric features only on the pandas engine."""
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "duration": pd.to_timedelta([1, 2, 3], unit="d")})
    config = {"dtype": "numeric"}
    for value in (frame, pl.from_pandas(frame), SkyulfSchema.from_dataframe(frame)):
        assert resolve_column_selection(value, config, "StandardScaler")["columns"] == ["x"]


def test_saved_column_rules_bind_partition_admission_and_reject_recipe_changes(tmp_path):
    """Worker admission must use the training selection without accepting changed rules."""
    frame = pd.DataFrame({"x": np.arange(20, dtype=float), "id": np.arange(20)})
    frame["target"] = frame["x"] * 2 + 1
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"dtype": "numeric", "exclude_columns": ["id"]},
                }
            ],
            "modeling": {"type": "ridge_regression"},
        }
    )
    pipeline.fit(frame, target_column="target")
    save_pipeline(pipeline, tmp_path / "model")
    artifact = load_pipeline(tmp_path / "model")
    artifact = seal_partition_probe(artifact, frame, tmp_path / "model", enabled=True)
    certificate = require_partition_safe_pipeline(artifact)
    assert certificate.steps[0].action == "apply"
    artifact.pipeline.feature_engineer.steps_config[0]["params"]["exclude_columns"] = []
    with pytest.raises(UnsupportedExecutionError, match="configuration disagrees"):
        require_partition_safe_pipeline(artifact)


@pytest.mark.parametrize("tamper", ["missing_binding", "other_option", "no_rules"])
def test_column_binding_cannot_bypass_existing_recipe_validation(tamper):
    """A resolved column list must never authorize unrelated configuration changes."""
    engineer = FeatureEngineer(
        [
            {
                "name": "scale",
                "transformer": "StandardScaler",
                "params": {"columns_regex": "^x", "with_mean": False},
            }
        ]
    )
    engineer.fit_transform(pd.DataFrame({"x": [1.0, 2.0, 4.0], "id": [1, 2, 3]}))
    record = deepcopy(engineer.fitted_steps[0])
    recipe: dict[str, Any] = deepcopy(dict(engineer.steps_config[0]))
    if tamper == "missing_binding":
        record.pop("column_selection_config")
    elif tamper == "other_option":
        recipe["params"]["with_mean"] = True
        record["column_selection_config"] = deepcopy(recipe)
    else:
        recipe["params"] = {"columns": ["x"], "with_mean": False}
        record["column_selection_config"] = deepcopy(recipe)
    with pytest.raises(ValueError):
        resolve_fitted_step(record, recipe)
