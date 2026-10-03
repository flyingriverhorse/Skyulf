"""Independent training tasks retain the existing lifecycle integrity contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from test_competition_lifecycle import _competition
from test_databricks_lifecycle_tasks import _call, staged  # noqa: F401 - shared fixture


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("separate_tasks", [False, True])
def test_weighted_competition_preserves_recipe_and_registers(
    staged, monkeypatch, engine, separate_tasks
):
    """Weight evidence must not change recipe identity across training task boundaries."""
    from skyulf.integrations.databricks import local_retraining
    from skyulf.integrations.databricks.training_nodes import run_competition_training
    from skyulf.integrations.mlflow.registry import load_run_local_pipeline

    _, client, config, context, frame = staged
    _competition(config, engine)
    config.update(weight_column="w", reserved_weight_columns=["w"])
    frame["w"] = frame.x + 1.0
    original_fit = local_retraining.fit_candidate
    recipes = []

    def fit_without_mutating_recipe(*args, **kwargs):
        """Retain the real fit and observe the caller-owned frozen configuration."""
        before = deepcopy(kwargs["pipeline_config"])
        fitted = original_fit(*args, **kwargs)
        recipes.append((before, deepcopy(kwargs["pipeline_config"])))
        return fitted

    monkeypatch.setattr(local_retraining, "fit_candidate", fit_without_mutating_recipe)
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    reference = split.reference
    if separate_tasks:
        for name in ("strong", "weak"):
            run_competition_training(
                None,
                name=name,
                context=context,
                tracking_uri=config["tracking_uri"],
                reference=reference,
            )
    else:
        reference = _call(staged, "train", reference).reference
    assert len(recipes) == 2
    assert all(before == after for before, after in recipes)
    selected = _call(staged, "select_best_model", reference)
    assert selected.output["winner"] == "strong"
    registered = _call(staged, "evaluate_register", selected.reference)
    _call(staged, "compare", registered.reference)
    _call(staged, "model_decision", prepared.reference)
    versions = client.search_model_versions(f"name='{config['model_name']}'")
    assert len(versions) == 1
    artifact = load_run_local_pipeline(
        f"runs:/{prepared.reference['run_id']}/model",
        digest=selected.output["model_digest"],
        tracking_uri=config["tracking_uri"],
    )
    weight_evidence = dict(artifact.pipeline.config)["training_weights"]
    assert isinstance(weight_evidence, dict)
    assert weight_evidence["weight_column"] == "w"
    assert "w" not in artifact.manifest.input_columns


@pytest.mark.parametrize("changed_section", ["training_weights", "modeling"])
def test_weighted_competition_rejects_changed_winner_config(staged, monkeypatch, changed_section):
    """Adoption must check weight evidence and model settings against the pinned request."""
    from skyulf.integrations.databricks import training_nodes

    _, client, config, context, frame = staged
    _competition(config)
    config.update(weight_column="w", reserved_weight_columns=["w"])
    frame["w"] = frame.x + 1.0
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    for name in ("strong", "weak"):
        training_nodes.run_competition_training(
            None,
            name=name,
            context=context,
            tracking_uri=config["tracking_uri"],
            reference=split.reference,
        )
    load = training_nodes.load_run_local_pipeline

    def changed_artifact(*args, **kwargs):
        """Alter loaded config to exercise the independent adoption integrity check."""
        artifact = load(*args, **kwargs)
        artifact.pipeline.config[changed_section] = {}
        return artifact

    monkeypatch.setattr(training_nodes, "load_run_local_pipeline", changed_artifact)
    with pytest.raises(ValueError, match="Winning model configuration"):
        _call(staged, "select_best_model", split.reference)
    assert not client.search_registered_models()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_competition_tasks_register_only_complete_winner(staged, engine):
    """Real separate fits must preserve the winner's evidence and ordinary alias lifecycle."""
    from skyulf.integrations.databricks.training_nodes import run_competition_training

    _, client, config, context, _ = staged
    _competition(config, engine)
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    options: dict[str, Any] = {
        "context": context,
        "tracking_uri": config["tracking_uri"],
        "reference": split.reference,
    }
    strong = run_competition_training(None, name="strong", **options)
    assert strong.output["run_id"] != prepared.reference["run_id"]
    from skyulf.integrations.databricks.training_node_output import render_training_node

    report = render_training_node(client, strong.output["run_id"], strong.output)
    assert "Preprocessing and model settings" in report
    assert "Cross-validation and training parameters" in report
    assert "data:image" not in report
    assert not client.search_registered_models()
    with pytest.raises(ValueError, match="already attempted"):
        run_competition_training(None, name="strong", **options)
    weak = run_competition_training(None, name="weak", **options)
    assert weak.output["run_id"] != strong.output["run_id"]
    selected = _call(staged, "select_best_model", split.reference)
    assert selected.output["winner"] == "strong"
    registered = _call(staged, "evaluate_register", selected.reference)
    _call(staged, "compare", registered.reference)
    decided = _call(staged, "model_decision", prepared.reference)
    versions = client.search_model_versions(f"name='{config['model_name']}'")
    assert len(versions) == 1
    assert versions[0].run_id == prepared.reference["run_id"]
    assert decided.output["alias_change"]["new_version"] == "1"


def test_competition_missing_training_blocks_selection(staged):
    """A successful subset cannot silently win when another task has not finished."""
    from skyulf.integrations.databricks.training_nodes import run_competition_training

    _, client, config, context, _ = staged
    _competition(config)
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    run_competition_training(
        None,
        name="strong",
        context=context,
        tracking_uri=config["tracking_uri"],
        reference=split.reference,
    )
    with pytest.raises(ValueError, match="completed receipt"):
        _call(staged, "select_best_model", split.reference)
    assert not client.search_registered_models()


def test_changed_child_receipt_blocks_selection(staged):
    """A modified candidate score must not be accepted by the complete-results join."""
    from skyulf.integrations.databricks._lifecycle_state import PhaseStore
    from skyulf.integrations.databricks.training_nodes import run_competition_training

    _, client, config, context, _ = staged
    _competition(config)
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    for name in ("strong", "weak"):
        run_competition_training(
            None,
            name=name,
            context=context,
            tracking_uri=config["tracking_uri"],
            reference=split.reference,
        )
    store = PhaseStore(config["tracking_uri"], context)
    store.bind(split.reference)
    receipt = store.read("lifecycle/candidate_strong.json")
    receipt["output"]["evaluation"]["mean"] = 1e20
    store.log("lifecycle/candidate_strong.json", receipt)
    with pytest.raises(ValueError, match="digest"):
        _call(staged, "select_best_model", split.reference)
    assert not client.search_registered_models()


def test_named_task_rejects_other_invocation_and_unknown_model(staged):
    """A task name or saved reference cannot escape the frozen invocation's candidate set."""
    from skyulf.integrations.databricks._lifecycle_state import LifecycleContext
    from skyulf.integrations.databricks.training_nodes import run_competition_training

    _, client, config, context, _ = staged
    _competition(config)
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    with pytest.raises(ValueError, match="not a candidate"):
        run_competition_training(
            None,
            name="../other",
            context=context,
            tracking_uri=config["tracking_uri"],
            reference=split.reference,
        )
    with pytest.raises(ValueError, match="invocation"):
        run_competition_training(
            None,
            name="strong",
            context=LifecycleContext("10", "21"),
            tracking_uri=config["tracking_uri"],
            reference=split.reference,
        )
    assert not any(
        "candidate_" in tag for tag in client.get_run(prepared.reference["run_id"]).data.tags
    )


def test_shap_leaf_reads_finished_training_without_mutating_models(staged):
    """The report task remains readable after the training parent has completed."""
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    pytest.importorskip("shap")
    pytest.importorskip("matplotlib")
    from skyulf.integrations.databricks.training_node_notebook import run_shap_notebook

    _, client, config, _, _ = staged
    config["pipeline"]["explainability"] = {
        "method": "shap",
        "max_samples": 3,
        "max_display_samples": 1,
    }
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    trained = _call(staged, "train", split.reference)
    client.set_terminated(trained.reference["run_id"], status="FINISHED")
    values = {
        "workflow_contract": "3",
        "job_id": "10",
        "job_run_id": "20",
        "repair_count": "0",
        "execution_count": "1",
        "tracking_uri": config["tracking_uri"],
        "reference_json": json.dumps(trained.reference),
    }
    displayed = []
    result = run_shap_notebook(
        SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values), notebook=Mock()),
        display_html=displayed.append,
    )
    assert json.loads(result)["status"] == "completed"
    assert any("data:image/png;base64," in html for html in displayed)
    assert not client.search_registered_models()


def test_changed_model_list_requires_graph_refresh():
    """Editing Python declarations alone cannot silently run a stale deployed graph."""
    from skyulf.integrations.databricks.training_node_notebook import validate_model_task_names

    with pytest.raises(ValueError, match="refresh_training_graph"):
        validate_model_task_names({"model_keys_json": '["old"]'}, {"new"})
    assert (
        validate_model_task_names({"model_keys_json": '["ridge", "forest"]'}, {"forest", "ridge"})
        is None
    )
