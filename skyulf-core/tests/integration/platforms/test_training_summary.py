"""Final reports link saved evidence without exposing samples or changing results."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_training_branches import tracked  # noqa: F401 - real MLflow fixture

from skyulf.integrations.databricks.jobs.shared import job_runtime


class ArtifactClient:
    """Serve real JSON files through the read-only MLflow client boundary."""

    def __init__(self, documents, *, tags=None):
        """Keep each run's artifacts separate so cross-model mistakes are visible."""
        self.documents = documents
        self.tags = tags or {}

    def get_run(self, run_id):
        """Expose only run identity and saved chart destinations."""
        return SimpleNamespace(
            info=SimpleNamespace(experiment_id="42", run_id=run_id),
            data=SimpleNamespace(tags=self.tags.get(run_id, {}), params={}),
        )

    def list_artifacts(self, run_id, path=None):
        """List names without fetching explanation samples or model bytes."""
        return [SimpleNamespace(path=name) for name in self.documents.get(run_id, {})]

    def download_artifacts(self, run_id, path, directory):
        """Write the requested document as MLflow's downloader would."""
        value = self.documents[run_id][path]
        if isinstance(value, Exception):
            raise value
        destination = Path(directory) / Path(path).name
        destination.write_text(json.dumps(value), encoding="utf-8")
        return str(destination)


def _documents(*, probe=False, shap=False):
    """Describe one frozen fit with distinct source and post-filter populations."""
    return {
        "candidate_training_spec.json": {
            "table": "workspace.data.source<x>",
            "version": 7,
            "preprocessing_probe": probe,
        },
        "training_pipeline_config.json": {"explainability": {"method": "shap"}} if shap else {},
        "training_data.json": {
            "train_data_destination": "workspace.data.source<x>",
            "train_data_version": "7",
        },
        "training_filter_evidence.json": {
            "source_rows": 120,
            "training_rows": 80,
            "holdout_rows": 20,
        },
        "pre_split_filters.json": {"excluded_rows": 20},
    }


def _payload():
    """Use the final single-model result without adding report-only machine fields."""
    return {
        "action": "train",
        "score_requested": False,
        "result": {"model_name": "workspace.models.one", "model_version": "3", "run_id": "fit"},
    }


def test_final_notebook_renderer_prepends_saved_evidence_and_keeps_machine_result(monkeypatch):
    """The real final renderer must show the overview without modifying exit JSON."""
    from skyulf.integrations.mlflow.shared import _client

    client = ArtifactClient({"fit": _documents()})
    monkeypatch.setattr(_client, "make_tracking_client", lambda uri: client)
    payload = _payload()
    outcome = SimpleNamespace(output=payload, reference={"run_id": "fit"})
    render = job_runtime._lifecycle_notebook_renderer("complete", outcome, "databricks", False)
    shown = []
    output = job_runtime.notebook_output(
        payload, None, render=render, display_html=shown.append, exit_notebook=False
    )
    assert shown[0].index("Where to look") < shown[0].index("Train completed")
    assert "source&lt;x&gt;" in shown[0] and "80" in shown[0] and "20" in shown[0]
    assert json.loads(output) == payload


@pytest.mark.parametrize("enabled,expected", [(False, "Off"), (True, "Unavailable")])
def test_missing_optional_artifact_uses_frozen_config(enabled, expected):
    """Missing output must not be called disabled when the fit requested it."""
    from skyulf.integrations.databricks.observability.reports.training_summary import (
        render_training_summary,
    )

    html = render_training_summary(
        ArtifactClient({"fit": _documents(probe=enabled, shap=enabled)}), "fit", _payload()
    )
    assert html.count(expected) >= 2
    assert "explainability" in html


def test_summary_links_existing_reports_and_redacts_sample_and_error_fields():
    """The index may expose statuses and thresholds, never SHAP samples or error text."""
    from skyulf.integrations.databricks.observability.reports.training_summary import (
        render_training_summary,
    )

    documents = {
        **_documents(probe=True, shap=True),
        "preprocessing_probe.json": {"status": "requires_context", "error": "private-probe"},
        "explanations.json": {"status": "completed", "shap": {"sample": "private-row"}},
        "explanations.html": "private-html",
        "decision_threshold.json": {
            "mode": "auto",
            "thresholds": [{"class": "yes", "value": 0.35}],
        },
    }
    html = render_training_summary(
        ArtifactClient({"fit": documents}, tags={"fit": {"skyulf.charts.run_id": "charts"}}),
        "fit",
        _payload(),
        workspace_host="https://example.azuredatabricks.net",
    )
    assert "requires_context" in html and "completed" in html and "0.35" in html
    assert "/ml/experiments/42/runs/fit" in html and "/runs/charts/model-metrics" in html
    assert "/explore/data/models/workspace/models/one/version/3" in html
    assert "explanations.html" in html and "decision_threshold.json" in html
    assert 'href="https://example.azuredatabricks.net/ml/experiments/42/runs/fit/artifacts"' in html
    assert "/artifacts?" not in html
    assert "private-" not in html and "<img" not in html


def test_competition_rows_preserve_each_fit_and_link_winner_charts():
    """A winner's parent charts and registered version must not be assigned to losers."""
    from skyulf.integrations.databricks.observability.reports.training_summary import (
        render_training_summary,
    )

    payload = {
        **_payload(),
        "competition": {
            "winner": "tree",
            "leaderboard": [
                {"candidate": "linear", "run_id": "linear-fit"},
                {"candidate": "tree", "run_id": "tree-fit"},
            ],
        },
    }
    client = ArtifactClient(
        {"linear-fit": _documents(), "tree-fit": _documents()},
        tags={"parent": {"skyulf.charts.run_id": "winner-charts"}},
    )
    html = render_training_summary(client, "parent", payload, workspace_host="https://example.net")
    assert html.count("<tbody><tr>") == 1 and html.count("</tr>") == 3
    assert "linear-fit" in html and "tree-fit" in html
    assert html.count("/version/3") == 1 and html.count("winner-charts/model-metrics") == 1


def test_model_set_rows_keep_component_versions_and_set_link():
    """The summary must identify each separately fitted component and its enclosing set."""
    from skyulf.integrations.databricks.observability.reports.training_summary import (
        render_training_summary,
    )

    payload = {
        "components": {
            "demand": {"run_id": "a", "model_name": "workspace.models.a", "model_version": "2"},
            "price": {"run_id": "b", "model_name": "workspace.models.b", "model_version": "4"},
        },
        "model_set_candidate": {"name": "workspace.models.set", "version": "8"},
    }
    html = render_training_summary(
        ArtifactClient({"a": _documents(), "b": _documents()}),
        "parent",
        payload,
        workspace_host="https://example.net",
    )
    assert "demand" in html and "price" in html
    assert "/a/version/2" in html and "/b/version/4" in html and "/set/version/8" in html


def test_missing_parent_chart_metadata_keeps_winner_training_details():
    """A failed optional chart read must not hide the winner's data and probe evidence."""
    from skyulf.integrations.databricks.observability.reports.training_summary import (
        render_training_summary,
    )

    class MissingParent(ArtifactClient):
        """Fail the separate chart-source fetch after the fitting run was read."""

        def get_run(self, run_id):
            """Represent a parent-only authorization or metadata read failure."""
            if run_id == "parent":
                raise RuntimeError("private-response")
            return super().get_run(run_id)

    payload = {
        **_payload(),
        "competition": {"winner": "tree", "leaderboard": [{"candidate": "tree", "run_id": "fit"}]},
    }
    html = render_training_summary(MissingParent({"fit": _documents()}), "parent", payload)
    assert "Training rows: 80" in html and "private-response" not in html


@pytest.mark.parametrize(
    "host", ["javascript:alert(1)", 'https://example.net/" onclick="bad', "//evil.net"]
)
def test_summary_rejects_unsafe_link_origins_and_escapes_names(host):
    """A widget or registry value cannot turn report links into executable markup."""
    from skyulf.integrations.databricks.observability.reports.training_summary import (
        render_training_summary,
    )

    payload = _payload()
    payload["result"]["model_name"] = "<script>alert(1)</script>"
    html = render_training_summary(
        ArtifactClient({"fit": _documents()}), "fit", payload, workspace_host=host
    )
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert 'href="javascript:' not in html and 'href="//' not in html and "onclick=" not in html


def test_unreadable_metadata_remains_unavailable_without_leaking_errors():
    """Optional artifact failures must retain the report and avoid claiming features are off."""
    from skyulf.integrations.databricks.observability.reports.training_summary import (
        render_training_summary,
    )

    documents = {
        "candidate_training_spec.json": RuntimeError("secret-query"),
        "training_pipeline_config.json": RuntimeError("secret-query"),
    }
    html = render_training_summary(ArtifactClient({"fit": documents}), "fit", _payload())
    assert "Unavailable" in html and "Off" not in html and "secret-query" not in html


@pytest.mark.parametrize("artifact", ["decision_threshold.json", "tuning.json"])
def test_threshold_preview_bounds_large_class_labels_and_many_classes(artifact):
    """The compact summary must link full threshold evidence instead of duplicating its size."""
    from skyulf.integrations.databricks.observability.reports.training_summary import (
        render_training_summary,
    )

    thresholds = {"x" * 1024 * 1024: 0.3, **{f"class-{index}": 0.5 for index in range(100)}}
    document = (
        {
            "mode": "manual",
            "thresholds": [{"class": key, "value": value} for key, value in thresholds.items()],
        }
        if artifact == "decision_threshold.json"
        else {"decision_thresholds": thresholds}
    )
    html = render_training_summary(
        ArtifactClient({"fit": {**_documents(), artifact: document}}), "fit", _payload()
    )
    assert len(html) < 5000 and "class-99" not in html
    assert artifact in html and "additional thresholds" in html


def test_real_training_artifacts_populate_summary(workflow_config, tracked, monkeypatch, tmp_path):
    """Producer-written counts, diagnostics and report files must survive actual MLflow retrieval."""
    from dataclasses import asdict

    from test_training_branches import _artifact, _configs, _data

    from skyulf.integrations.databricks.observability.reports.training_summary import (
        render_training_summary,
    )
    from skyulf.integrations.databricks.training import branches
    from skyulf.integrations.databricks.training.fitting import candidate as training
    from skyulf.integrations.mlflow.runs.tracking import TrackingRun

    uri, client = tracked
    configs = _configs(workflow_config, store=uri)
    configs["amount"]["preprocessing_probe"] = True
    configs["amount"]["pipeline"]["explainability"] = {
        "method": "shap",
        "max_samples": 2,
        "max_display_samples": 1,
    }
    monkeypatch.setattr(
        training,
        "read_training_snapshot",
        lambda spark, spec: _data().loc[:, list(spec.source_columns)].copy(),
    )
    branch = branches.prepare_training_branches(None, configs)[0]
    experiment = client.get_experiment_by_name("branches")
    parent = client.create_run(experiment.experiment_id)
    result = branches.train_branch(
        None,
        branch,
        run=TrackingRun(client=client, run_id=parent.info.run_id, enabled=True),
        digest="summary-validation",
        path=tmp_path / "model",
        tracking_uri=uri,
        registry_uri=uri,
        experiment_name="branches",
    )
    evidence = _artifact(client, result.run_id, "training_filter_evidence.json")
    filters = _artifact(client, result.run_id, "pre_split_filters.json")
    probe = _artifact(client, result.run_id, "preprocessing_probe.json")
    shap = _artifact(client, result.run_id, "explanations.json")
    html = render_training_summary(
        client,
        result.run_id,
        {"action": "train", "result": asdict(result)},
        workspace_host="https://example.net",
    )
    client.set_terminated(parent.info.run_id)
    assert f"Training rows: {evidence['training_rows']}" in html
    assert f"Holdout rows: {evidence['holdout_rows']}" in html
    assert f"Pre-split removed: {filters['excluded_rows']}" in html
    assert probe["status"] in html and shap["status"] in html
    assert "explanations.html" in html and f"/version/{result.model_version}" in html
