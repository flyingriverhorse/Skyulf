"""A serving.yml the handoff would reject must stop the run before training, not after the decision."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from test_serving_endpoint_job import endpoint_project  # noqa: F401 - pytest fixture


def _stale(endpoint_project, mode="production"):  # noqa: F811
    """Write a serving.yml with a renamed field, as an older generated project has."""
    document, path, values = endpoint_project
    document["serving_endpoint"]["smoke"] = {"rows": 20}
    path.write_text(yaml.safe_dump(document))
    values.update(bundle_mode=mode)
    return values


def test_active_handoff_settings_reject_stale_fields(endpoint_project):  # noqa: F811
    """The check initialization runs is the same one the post-decision handoff runs."""
    from skyulf.integrations.databricks.jobs.serving_handoff import active_serving_settings

    with pytest.raises(ValueError, match="Unknown serving_endpoint fields"):
        active_serving_settings(_stale(endpoint_project))


def test_inactive_handoff_skips_settings_check(endpoint_project):  # noqa: F811
    """Development runs never hand off, so they keep ignoring endpoint settings."""
    from skyulf.integrations.databricks.jobs.serving_handoff import active_serving_settings

    assert active_serving_settings(_stale(endpoint_project, "development")) is None
    assert active_serving_settings({"deployed_serving_endpoint": "false"}) is None


def test_lifecycle_initialization_rejects_stale_serving_settings(
    endpoint_project,  # noqa: F811
    monkeypatch,
):
    """Single-model and competition runs fail at initialize_run, before any training."""
    from skyulf.integrations.databricks.jobs.shared import job_runtime

    values = {**_stale(endpoint_project), "lifecycle_action": "train"}
    monkeypatch.setattr(job_runtime, "read_notebook_config", lambda values: {})
    monkeypatch.setattr(job_runtime, "validate_workflow_config", lambda config, **kwargs: config)
    monkeypatch.setattr(job_runtime, "validate_monitoring_settings", Mock())
    with pytest.raises(ValueError, match="Unknown serving_endpoint fields"):
        job_runtime._prepared_notebook_request(values, None)


def test_model_set_initialization_rejects_stale_serving_settings(
    endpoint_project,  # noqa: F811
    monkeypatch,
):
    """Model-set runs fail at initialize_models, before branch configs are loaded."""
    from skyulf.integrations.databricks.jobs.training import training_node_notebook as notebook

    values = {**_stale(endpoint_project), "lifecycle_action": "train"}
    monkeypatch.setattr(notebook, "lifecycle_widget_context", lambda values: {})
    monkeypatch.setattr(notebook, "LifecycleContext", Mock())
    monkeypatch.setattr(notebook, "operator_options", Mock())
    load = Mock()
    monkeypatch.setattr(notebook, "load_project_model_set", load)
    dbutils = SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: values))
    with pytest.raises(ValueError, match="Unknown serving_endpoint fields"):
        notebook.run_initialize_models_notebook(None, dbutils)
    load.assert_not_called()
