"""Real tracking and fitted models verify the optional reporting task end to end."""

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("mlflow")
from test_competition_lifecycle import _competition
from test_databricks_lifecycle_tasks import _call, staged  # noqa: F401 - shared fixture
from test_local_branches import _configs, _data, tracked  # noqa: F401 - shared fixture

from skyulf.integrations.databricks.evaluation_chart_task import (
    generate_evaluation_charts,
    render_chart_report,
    run_evaluation_charts_notebook,
)


def _evaluated(staged, competition=False, enabled=True):
    """Use real staged fit/registration/comparison without changing the test source snapshot."""
    from skyulf.integrations.databricks.training_nodes import run_competition_training

    _, _, config, context, _ = staged
    if competition:
        _competition(config)
    config["evaluation_charts"] = {"enabled": enabled, "max_rows": 7}
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    if competition:
        for name in ("strong", "weak"):
            run_competition_training(
                None,
                name=name,
                context=context,
                tracking_uri=config["tracking_uri"],
                reference=split.reference,
            )
        selected = _call(staged, "select_best_model", split.reference)
    else:
        trained = _call(staged, "train", split.reference)
        selected = _call(staged, "select_best_model", trained.reference)
    registered = _call(staged, "evaluate_register", selected.reference)
    return prepared, _call(staged, "compare", registered.reference)


def _assert_chart_run(client, source_id, destination):
    """Images must live in a finished parameter-free child linked to the training evidence."""
    source = client.get_run(source_id)
    chart = client.get_run(destination)
    assert destination != source_id
    assert chart.info.run_name == "Charts"
    assert chart.info.experiment_id == source.info.experiment_id
    assert chart.info.status == "FINISHED"
    assert chart.data.params == {}
    assert chart.data.tags["mlflow.parentRunId"] == source_id
    assert chart.data.tags["skyulf.run_kind"] == "evaluation_charts"
    assert chart.data.tags["mlflow.loggedImages"].lower() == "true"
    assert source.data.tags["skyulf.charts.run_id"] == destination
    assert not source.data.tags.get("mlflow.loggedImages")


@pytest.mark.parametrize("competition", [False, True])
def test_images_are_indexed_after_completed_lifecycle_without_metric_changes(
    staged, competition, capsys
):
    """The task must log native images using saved models, even after the parent is finished."""
    _, client, config, context, _ = staged
    prepared, evaluated = _evaluated(staged, competition)
    _call(staged, "model_decision", prepared.reference)
    parent = prepared.reference["run_id"]
    client.set_tag(parent, "skyulf.charts.run_id", "previous-successful-charts")
    before = client.get_run(parent)
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = {
        **context.identity(),
        "workflow_contract": "3",
        "repair_count": "0",
        "execution_count": "1",
        "tracking_uri": config["tracking_uri"],
        "reference_json": json.dumps(evaluated.reference),
        "workspace_host": "https://workspace.example/",
    }
    capsys.readouterr()
    output = json.loads(run_evaluation_charts_notebook(dbutils))
    printed = capsys.readouterr().out
    run = client.get_run(parent)
    assert output["status"] == "complete"
    destination = output["charts_run_id"]
    _assert_chart_run(client, parent, destination)
    assert run.data.metrics == before.data.metrics
    assert run.data.params == before.data.params
    assert run.info.status == before.info.status
    assert any(item.path.endswith(".png") for item in client.list_artifacts(destination, "images"))
    report = next(iter(output["reports"].values()))
    assert report["sample_rows"] == min(7, report["holdout_rows"])
    assert report["model_digest"]
    assert report["destination_run_id"] == destination
    prefix = "competition_winner" if competition else "single"
    assert f"{prefix}_metric_summary" in report["image_keys"]
    summary = render_chart_report(output, workspace_host="https://workspace.example/")
    assert printed == summary + "\n"
    saved_reference = json.loads(dbutils.jobs.taskValues.set.call_args.kwargs["value"])
    assert saved_reference["run_id"] == parent
    assert saved_reference["phase"] == "generate_charts"
    assert summary.splitlines()[0] == (
        f"MLflow Charts: https://workspace.example/ml/experiments/{run.info.experiment_id}"
        f"/runs/{destination}/model-metrics"
    )
    assert summary.splitlines()[1:] == [f"- {key}" for key in output["image_keys"]]
    if competition:
        assert "competition_leaderboard" in output["image_keys"]
        assert report["run_id"] == parent
        children = client.search_runs(
            [run.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent}'"
        )
        assert [
            child.info.run_id for child in children if child.data.tags.get("mlflow.loggedImages")
        ] == [destination]
    with pytest.raises(ValueError, match="already attempted"):
        generate_evaluation_charts(
            context=context, tracking_uri=config["tracking_uri"], reference=evaluated.reference
        )


def test_disabled_task_does_not_save_holdout_or_images(staged):
    """Opt-out must avoid both input uploads and plotting, even if an old deployed task remains."""
    _, client, config, context, _ = staged
    prepared, evaluated = _evaluated(staged, enabled=False)
    result = generate_evaluation_charts(
        context=context, tracking_uri=config["tracking_uri"], reference=evaluated.reference
    )
    paths = {
        item.path
        for item in client.list_artifacts(prepared.reference["run_id"], "evaluation_charts")
    }
    assert result.output["status"] == "disabled"
    assert paths == {"evaluation_charts/report.json"}
    assert not client.get_run(prepared.reference["run_id"]).data.tags.get("mlflow.loggedImages")
    assert render_chart_report(result.output) == "Evaluation charts are disabled."
    assert not client.search_runs(
        [client.get_run(prepared.reference["run_id"]).info.experiment_id],
        filter_string="tags.`skyulf.run_kind` = 'evaluation_charts'",
    )


def test_chart_failure_does_not_rewrite_successful_training(staged, monkeypatch):
    """A report error must not change promotion or tempt a training retry."""
    from skyulf.integrations.databricks import evaluation_chart_report

    _, client, config, context, _ = staged
    prepared, evaluated = _evaluated(staged)
    _call(staged, "model_decision", prepared.reference)
    parent = prepared.reference["run_id"]
    client.set_tag(parent, "skyulf.charts.run_id", "previous-successful-charts")
    before = client.get_run(parent)

    def fail(*args, **kwargs):
        """Simulate a chart dependency failure after successful model work."""
        raise RuntimeError("plotting unavailable")

    monkeypatch.setattr(evaluation_chart_report, "report_model", fail)
    with pytest.raises(RuntimeError, match="plotting unavailable"):
        generate_evaluation_charts(
            context=context, tracking_uri=config["tracking_uri"], reference=evaluated.reference
        )
    after = client.get_run(parent)
    assert after.info.status == before.info.status
    assert after.data.tags["skyulf.charts.status"] == "failed"
    assert after.data.metrics == before.data.metrics
    assert after.data.params == before.data.params
    assert after.data.tags.get("skyulf.charts.run_id") == before.data.tags.get(
        "skyulf.charts.run_id"
    )
    children = client.search_runs(
        [before.info.experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent}'"
    )
    assert len(children) == 1
    assert children[0].info.status == "FAILED"
    assert children[0].data.params == {}
    assert str(client.get_model_version_by_alias(config["model_name"], "champion").version) == "1"


def test_model_set_images_belong_to_each_independent_child(workflow_config, tracked, monkeypatch):
    """Each model and the set overview need their own parameter-free Charts child."""
    from skyulf.integrations.databricks import branch_tasks, local_retraining, model_set_stages
    from skyulf.integrations.databricks._lifecycle_state import LifecycleContext

    uri, client = tracked
    configs = _configs(workflow_config, store=uri)
    configs = {name: configs[name] for name in ("amount", "category")}
    for config in configs.values():
        config["evaluation_charts"] = {"enabled": True, "max_rows": 9}
    monkeypatch.setattr(
        local_retraining,
        "read_training_snapshot",
        lambda spark, spec: _data().loc[:, list(spec.source_columns)].copy(),
    )
    context = LifecycleContext("10", "20")
    prepared = branch_tasks.initialize_branch_training(
        None,
        configs=configs,
        settings=None,
        composition_source="",
        context=context,
        tracking_uri=uri,
        experiment_name="branches",
    )
    options: dict[str, Any] = {
        "context": context,
        "tracking_uri": uri,
        "reference": prepared.reference,
    }
    for name in configs:
        branch_tasks.run_branch_training(None, name=name, **options)
    registered = branch_tasks.register_branch_set(None, **options)
    options["reference"] = registered.reference
    evaluated = model_set_stages.run_model_set_phase(None, phase="evaluate_model_set", **options)
    options["reference"] = evaluated.reference
    result = generate_evaluation_charts(**options)
    assert result.output["image_keys"] == ["model_set_overview"]
    assert set(result.output["reports"]) == set(configs)
    for report in result.output["reports"].values():
        assert report["sample_rows"] <= 9
        _assert_chart_run(client, report["run_id"], report["destination_run_id"])
    _assert_chart_run(client, result.output["run_id"], result.output["charts_run_id"])
    assert {report["task"] for report in result.output["reports"].values()} == {
        "regression",
        "classification",
    }
    assert result.output["status"] == "complete"
    summary = render_chart_report(result.output, workspace_host="https://workspace.example")
    experiment_id = client.get_run(prepared.reference["run_id"]).info.experiment_id
    for run_id in [result.output["charts_run_id"]] + [
        report["destination_run_id"] for report in result.output["reports"].values()
    ]:
        assert f"/ml/experiments/{experiment_id}/runs/{run_id}/model-metrics" in summary
    for key in result.output["image_keys"] + [
        key for report in result.output["reports"].values() for key in report["image_keys"]
    ]:
        assert summary.splitlines().count(f"- {key}") == 1


def test_chart_upload_failure_does_not_block_registration(staged, monkeypatch):
    """Optional sample storage failure must be disclosed without failing core model work."""
    from skyulf.integrations.databricks import evaluation_chart_data

    _, client, config, context, _ = staged

    def fail(*args, **kwargs):
        """Inject a failure only in chart sample upload, not required lifecycle evidence."""
        raise RuntimeError("optional upload failed")

    monkeypatch.setattr(evaluation_chart_data, "_save_sample_bytes", fail)
    prepared, evaluated = _evaluated(staged)
    decision = _call(staged, "model_decision", prepared.reference)
    result = generate_evaluation_charts(
        context=context, tracking_uri=config["tracking_uri"], reference=evaluated.reference
    )
    assert decision.output["alias_change"]["new_version"] == "1"
    report = result.output["reports"]["single"]
    assert report["image_keys"] == []
    assert "RuntimeError" in report["skipped"][0]
    assert client.get_run(prepared.reference["run_id"]).info.status != "FAILED"
    assert render_chart_report(result.output) == "No evaluation charts were generated."


def test_chart_task_restores_custom_recipe_in_fresh_process(staged, tmp_path, monkeypatch):
    """Saved custom pre-split identities must be restored before validating the reporting spec."""
    from skyulf.integrations.databricks.project import load_project_workflow
    from skyulf.registry import NodeRegistry

    _, _, config, context, frame = staged
    source = (Path(__file__).resolve().parents[1] / "fixtures/custom_recipe.py").read_text(
        encoding="utf-8"
    )
    source += '\ndef build_preprocessing():\n    """Restore the fitted custom feature builder."""\n    return [example_custom_step("x")]\ndef build_pre_split_steps():\n    """Restore fixed training eligibility."""\n    return [example_custom_pre_split("is_test")]\n'
    path = tmp_path / "preprocessing.py"
    path.write_text(source, encoding="utf-8")
    frame["is_test"] = False
    config.update(load_project_workflow(config, path))
    _, evaluated = _evaluated(staged)
    for registry in (NodeRegistry._calculators, NodeRegistry._appliers, NodeRegistry._metadata):
        for key in list(registry):
            if key.startswith("_skyulf_project_"):
                monkeypatch.delitem(registry, key)
    for name in list(sys.modules):
        if name.startswith("_skyulf_project_"):
            monkeypatch.delitem(sys.modules, name)
    path.unlink()
    result = generate_evaluation_charts(
        context=context, tracking_uri=config["tracking_uri"], reference=evaluated.reference
    )
    assert "single_actual_vs_predicted" in result.output["image_keys"]
