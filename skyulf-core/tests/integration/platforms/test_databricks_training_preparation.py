"""Shared training preparation must pin inputs without fitting or publishing models."""

from copy import deepcopy
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks import local_workflow as workflow


def _config(engine, policy, version):
    """Describe a bounded date-free request with an independently selected score model."""
    return {
        "engine": engine,
        "training_table": "workspace.test.source",
        "training_version": version,
        "model_name": "workspace.test.model",
        "model_version": "99",
        "record_key_columns": ["id"],
        "input_columns": ["x"],
        "target_column": "target",
        "max_rows": 100,
        "max_input_mb": 1,
        "pipeline": {"preprocessing": [], "modeling": {"type": "linear_regression"}},
        "score_model_selection": "pinned_version",
        "promotion_policy": policy,
        "quality_threshold": 1.0 if policy == "automatic" else None,
        "metric": "heldout_rmse",
        "min_improvement": 0.0,
        "cv_enabled": True,
        "cv_folds": 3,
    }


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("policy", ["automatic", "manual_approval"])
@pytest.mark.parametrize("version", [None, 4])
@pytest.mark.parametrize("champion", [None, "2"])
def test_preparation_pins_source_and_champion_without_changing_score_selection(
    monkeypatch, engine, policy, version, champion
):
    """Latest selection happens once and training must not replace an independent score pin."""
    config = _config(engine, policy, version)
    original = deepcopy(config)
    spark = Mock()
    history = spark.sql.return_value.select.return_value.orderBy.return_value.first
    history.return_value = {"version": 7}
    resolve = Mock(return_value=champion)
    monkeypatch.setattr(workflow, "controlled_champion_version", resolve)
    spec, cv, selected = workflow.prepare_training(
        spark, config, policy=policy, now=datetime(2026, 9, 1, tzinfo=UTC)
    )
    history.return_value = {"version": 99}
    assert spec.version == (7 if version is None else 4)
    assert spec.table == config["training_table"] and spec.input_columns == ("x",)
    assert cv.enabled and cv.folds == 3
    assert selected == champion and config == original
    assert resolve.call_count == 1
    assert history.call_count == (1 if version is None else 0)


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"quality_threshold": None}, "quality_threshold"),
        ({"cv_type": "stratified_k_fold"}, "classification model"),
    ],
)
def test_invalid_training_preparation_fails_before_source_or_registry(
    monkeypatch, changes, message
):
    """Invalid training policies must fail before any external read or publication."""
    config = {**_config("pandas", "automatic", None), **changes}
    spark = Mock()
    resolve = Mock(side_effect=AssertionError("registry accessed"))
    monkeypatch.setattr(workflow, "controlled_champion_version", resolve)
    with pytest.raises(ValueError, match=message):
        workflow.prepare_training(spark, config, policy="automatic", now=None)
    spark.sql.assert_not_called()
    resolve.assert_not_called()


def test_legacy_automatic_sdk_preserves_its_champion_pin_compatibility(monkeypatch, tmp_path):
    """Sharing preparation must not silently tighten the historical SDK selection contract."""
    config = _config("pandas", "automatic", 4)
    config.pop("score_model_selection")
    config.pop("promotion_policy")
    config.update(model_selection_mode="auto_champion", champion_version="99")
    failure = RuntimeError("fit boundary reached")
    monkeypatch.setattr(workflow, "controlled_champion_version", Mock(return_value="2"))
    monkeypatch.setattr(workflow, "ChallengerLifecycle", Mock())
    fit = Mock(side_effect=failure)
    monkeypatch.setattr(workflow, "train_local_candidate", fit)
    with pytest.warns(DeprecationWarning), pytest.raises(RuntimeError) as caught:
        workflow.run_action(
            None, config, "train", experiment_name="test", artifact_path=tmp_path / "artifact"
        )
    assert caught.value is failure
    assert fit.call_args.kwargs["champion_version"] == "2"
    assert config["champion_version"] == "99"
