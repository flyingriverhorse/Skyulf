"""Offline smoke checks validate project files without executing hooks."""

import json

import pytest

from skyulf.integrations.databricks.projects.project_checks import check_project


def _project(tmp_path, workflow_config):
    """Construct project files with a hook that fails immediately if executed."""
    (tmp_path / "config").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "config/workflow.json").write_text(json.dumps(workflow_config), encoding="utf-8")
    (tmp_path / "src/recipe.py").write_text(
        "raise AssertionError('Static smoke executed a project hook')\n", encoding="utf-8"
    )
    (tmp_path / "databricks.yml").write_text("bundle:\n  name: sample\n", encoding="utf-8")
    return tmp_path


def _bindings():
    """Select the fixture's already-resolved UC namespace."""
    return {
        "catalog": "workspace",
        "input_schema": "test",
        "output_schema": "test",
        "metadata_schema": "test",
        "resource_suffix": "",
    }


def test_smoke_reads_syntax_and_config_without_executing_hooks(tmp_path, workflow_config):
    """Even a failing user hook must stay inert in credential-free smoke mode."""
    root = _project(tmp_path, workflow_config)
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = check_project(root, _bindings())
    after = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert result["status"] == "passed"
    assert result["scope"] == "static_config_and_python_syntax"
    assert before == after


def test_smoke_reports_syntax_and_config_errors(tmp_path, workflow_config):
    """Broken source or settings must fail without starting training."""
    root = _project(tmp_path, workflow_config)
    (root / "src/recipe.py").write_text("def broken(\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        check_project(root, _bindings())
    (root / "src/recipe.py").write_text("pass\n", encoding="utf-8")
    settings = {**workflow_config, "engine": "unknown"}
    (root / "config/workflow.json").write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError, match="engine"):
        check_project(root, _bindings())


def test_smoke_validates_real_training_dates_without_running_model_hooks(tmp_path, workflow_config):
    """Scoring substitutes saved dates, but project smoke must reject invalid training dates."""
    settings = {
        **workflow_config,
        "training_window_mode": "fixed_window",
        "monthly_lookback_months": None,
        "window_timezone": None,
        "start": "not-a-date",
        "pipeline": {"preprocessing": [], "modeling": {}},
    }
    root = _project(tmp_path, settings)
    with pytest.raises(ValueError, match="start"):
        check_project(root, _bindings())
