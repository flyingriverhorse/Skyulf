"""Real fitted model sets enforce coherent quality activation and rollback."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_mixed_models_auto_release_replacement_failure_and_rollback(
    tmp_path, workflow_config, monkeypatch, engine
):
    """Regression, classification and ensemble gates must jointly control one champion."""
    mlflow = pytest.importorskip("mlflow")
    from test_training_branches import _configs, _data

    from skyulf.integrations.databricks.model_sets import model_set_project as project
    from skyulf.integrations.databricks.model_sets import model_set_release as release
    from skyulf.integrations.databricks.model_sets.model_set_quality import (
        evaluate_model_set_quality,
    )
    from skyulf.integrations.databricks.training import branches as local_branches
    from skyulf.integrations.databricks.training.fitting import candidate as candidate
    from skyulf.integrations.mlflow.lifecycle.model_set_challenger import reject_model_set
    from skyulf.integrations.mlflow.lifecycle.model_set_lifecycle import rollback_model_set
    from skyulf.integrations.mlflow.lifecycle.promotion import (
        AliasChangeReceipt,
        ExclusiveAliasWriterAdmission,
    )
    from skyulf.integrations.mlflow.models.model_set import load_registered_model_set

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    endpoints = {"tracking_uri": uri, "registry_uri": uri}
    client = mlflow.MlflowClient(**endpoints)
    client.create_experiment("auto", artifact_location=(tmp_path / "runs").as_uri())
    data = _data()
    known = data.category.notna()
    data.loc[known, "category"] = (data.loc[known, "x"] >= 30).astype(float)
    monkeypatch.setattr(
        candidate,
        "read_training_snapshot",
        lambda spark, spec: data.loc[:, list(spec.source_columns)].copy(),
    )
    monkeypatch.setattr(project, "approval_frame", lambda *args: data[["id", "x"]].copy())
    spark = Mock()
    spark.read.option.return_value.table.return_value = SimpleNamespace(dtypes=[("id", "bigint")])
    configs = _configs(workflow_config, engine=engine, store=uri)
    for name, config in configs.items():
        config.update(
            cv_enabled=False,
            min_improvement=0.0,
            quality_gates=None,
            quality_threshold=10.0 if name == "category" else 100.0,
        )
    configs["category"]["metric"] = "heldout_log_loss"
    settings = {
        "model_name": "workspace.test.quality_set",
        "promotion_policy": "automatic",
        "composition_config": {"outputs": []},
    }
    base = {**configs["amount"], "max_rows": 100, "max_input_mb": 4}

    def train(number, weak=False, weak_category=False):
        """Register a fresh complete set using the baseline captured before its training."""
        recipes = deepcopy(configs)
        recipes["amount"]["pipeline"]["modeling"] = {
            "type": "ridge_regression",
            "params": {"alpha": 1e6 if weak else 0.1},
        }
        recipes["category"]["pipeline"]["modeling"] = {
            "type": "logistic_regression",
            "params": {"C": 1e-6 if weak or weak_category else 1.0},
        }
        recipes["ensemble"]["pipeline"]["modeling"]["params"]["base_estimator_params"] = {
            "ridge": {"alpha": 1e6 if weak else 0.1}
        }
        pinned, versions = release.pin_model_set_baseline(settings, recipes, endpoints)
        branches = local_branches.prepare_training_branches(
            None, recipes, champion_versions=versions
        )
        outcome = local_branches.train_branches(
            None,
            branches,
            **endpoints,
            experiment_name="auto",
            artifact_path=tmp_path / f"fit{number}",
        )
        candidate = project.package_training_model_set(
            spark, branches, outcome, pinned, composition_source="", **endpoints
        )
        return candidate, pinned, outcome

    first, pinned, outcome = train(1, weak=True)
    initial = release.automatic_model_set_release(spark, first, pinned, base)
    assert initial["alias_change"]["kind"] == "initial"
    for component in outcome.components.values():
        assert client.get_registered_model(component.model_name).aliases == {}
        client.set_registered_model_alias(component.model_name, "champion", component.model_version)
    second, pinned, _ = train(2, weak_category=True)
    quality = evaluate_model_set_quality(
        spark,
        load_registered_model_set(second, **endpoints),
        load_registered_model_set(first, **endpoints),
        expected_champion_version="1",
        max_rows=100,
        max_bytes=4 * 1024 * 1024,
        **endpoints,
    )
    assert quality["passed"]
    assert quality["components"]["category"]["comparison"]["improvement"] == 0.0
    assert set(quality["improved_components"]) == {"amount", "ensemble"}
    promoted = release.automatic_model_set_release(spark, second, pinned, base)
    assert promoted["alias_change"]["kind"] == "promotion"
    assert str(client.get_model_version_by_alias(first.name, "champion").version) == "2"
    third, pinned, _ = train(3, weak_category=True)
    failed = release.automatic_model_set_release(spark, third, pinned, base)
    assert failed["alias_change"] is None
    assert failed["quality"]["failed_components"] == []
    assert failed["quality"]["reason"] == "no_component_improved"
    assert all(
        c["comparison"]["champion_version"] == "2" for c in failed["quality"]["components"].values()
    )
    assert str(client.get_model_version_by_alias(first.name, "champion").version) == "2"
    with pytest.raises(release.ModelSetQualityError):
        release.approve_project_model_set(
            spark, third, base, expected_champion_version="2", policy="manual_approval"
        )
    tags = client.get_model_version(first.name, "3").tags
    assert tags["model_set_quality_status"] == "failed"
    assert str(client.get_model_version_by_alias(first.name, "challenger").version) == "3"
    rejection = reject_model_set(
        third,
        reason="No improvement",
        expected_champion_version="2",
        admission=ExclusiveAliasWriterAdmission(),
        **endpoints,
    )
    assert rejection.kind == "rejection"
    rollback_model_set(
        AliasChangeReceipt(**promoted["alias_change"]),
        expected_current_version="2",
        admission=ExclusiveAliasWriterAdmission(),
        **endpoints,
    )
    assert str(client.get_model_version_by_alias(first.name, "champion").version) == "1"
    assert str(client.get_model_version_by_alias(first.name, "challenger").version) == "3"
    assert all(
        str(client.get_model_version_by_alias(c.model_name, "champion").version) == "1"
        for c in outcome.components.values()
    )
