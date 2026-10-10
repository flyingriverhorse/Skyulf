"""Offline smoke checks validate project files without executing hooks."""

import json
import runpy
from pathlib import Path

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


def test_smoke_checks_spark_group_syntax_outside_model_package(tmp_path, workflow_config):
    """Producers need syntax checks but no model-package init or source-budget allocation."""
    root = _project(tmp_path, workflow_config)
    features = root / "src/features"
    groups = features / "groups"
    groups.mkdir(parents=True)
    (features / "__init__.py").write_text("raise AssertionError('must not execute')\n")
    producer = groups / "company.py"
    producer.write_text("raise AssertionError('must not execute')\n#" + "x" * 65536)
    result = check_project(root, _bindings())
    assert result["status"] == "passed"
    assert result["python_files"] == 3
    producer.write_text("def broken(\n")
    with pytest.raises(SyntaxError, match="never closed"):
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


@pytest.mark.parametrize("package", ["features", "composition"])
@pytest.mark.parametrize(
    ("filename", "content", "reason"),
    [
        ("assets.json", '["missing.csv"]', "missing.csv"),
        ("assets.json", '["../outside.csv"]', "canonical relative path"),
        ("requirements.txt", "uninstalled-example>=1.0", "exact distribution==version"),
        ("requirements.txt", "some_pkg==1.0\nsome-pkg==2.0", "Duplicate project dependency"),
        ("nested/recipe.py", "pass\n", "nested/__init__.py"),
    ],
)
def test_smoke_rejects_unpackageable_project_files(
    tmp_path, workflow_config, package, filename, content, reason
):
    """Files that cannot be saved with a model must fail before a cloud run starts."""
    root = _project(tmp_path, workflow_config)
    directory = root / "src" / package
    directory.mkdir()
    (directory / "__init__.py").write_text("raise AssertionError('hook executed')\n")
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        check_project(root, _bindings())
    assert f"src/{package}" in str(caught.value)
    assert reason in str(caught.value)


@pytest.mark.parametrize("package", ["features", "composition"])
def test_smoke_checks_packages_without_importing_or_writing(tmp_path, workflow_config, package):
    """Static checks must accept valid snapshots without importing their pinned dependencies."""
    root = _project(tmp_path, workflow_config)
    directory = root / "src" / package
    directory.mkdir()
    (directory / "__init__.py").write_text("raise AssertionError('hook executed')\n")
    (directory / "requirements.txt").write_text("uninstalled-example==1.0\n")
    (directory / "assets.json").write_text('{"files": ["lookup.csv"]}')
    (directory / "lookup.csv").write_text("id,value\n1,2\n")
    before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = check_project(root, _bindings())
    after = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert result["project_packages"] == [f"src/{package}"]
    assert result["project_hooks_executed"] is False
    assert result["remote_operations"] is False
    assert before == after


@pytest.mark.parametrize("content", ["[]", "{", '{"engine": "unknown"}'])
def test_smoke_config_errors_identify_the_file(tmp_path, workflow_config, content):
    """A configuration failure must name the editable file, including malformed JSON."""
    root = _project(tmp_path, workflow_config)
    (root / "config/workflow.json").write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="config/workflow.json"):
        check_project(root, _bindings())


def _smoke_script(root):
    """Place the actual generated entry point under the fixture project root."""
    template = (
        Path(__file__).parents[3]
        / "templates/databricks/template/{{.project_name}}/src/tools/smoke.py.tmpl"
    )
    script = root / "src/tools/smoke.py"
    script.parent.mkdir()
    script.write_text(
        template.read_text(encoding="utf-8")
        .replace("{{.catalog}}", "workspace")
        .replace("{{.schema}}", "test"),
        encoding="utf-8",
    )
    return script


@pytest.mark.parametrize("failure", ["config", "config_type", "syntax", "encoding", "missing_file"])
def test_smoke_cli_reports_actionable_json_failures(
    tmp_path, workflow_config, monkeypatch, capsys, failure
):
    """Operators and automation need a nonzero exit and a readable file-specific failure."""
    root = _project(tmp_path, workflow_config)
    script = _smoke_script(root)
    expected_path = "config/workflow.json"
    if failure == "config":
        (root / expected_path).write_text("{", encoding="utf-8")
    elif failure == "config_type":
        (root / expected_path).write_text(
            json.dumps({**workflow_config, "training_layout": ["single_model"]}), encoding="utf-8"
        )
    elif failure == "missing_file":
        (root / expected_path).unlink()
    else:
        expected_path = "recipe.py"
        (root / "src/recipe.py").write_bytes(b"\xff" if failure == "encoding" else b"def broken(\n")
    monkeypatch.setattr("sys.argv", [str(script)])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(script), run_name="__main__")
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert caught.value.code == 1
    assert result["status"] == "failed"
    assert expected_path in result["message"]
    assert result["project_hooks_executed"] is False
    assert result["remote_operations"] is False
    assert not output.err
    if failure == "syntax":
        assert result["file"] == str(root / "src/recipe.py")
        assert result["line"] == 1
        assert result["column"] > 0


def test_smoke_cli_success_reports_checked_packages(tmp_path, workflow_config, monkeypatch, capsys):
    """Legacy projects without feature packages still have a successful JSON-only entry point."""
    root = _project(tmp_path, workflow_config)
    script = _smoke_script(root)
    monkeypatch.setattr("sys.argv", [str(script)])
    runpy.run_path(str(script), run_name="__main__")
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["status"] == "passed"
    assert result["project_packages"] == []
    assert not output.err
