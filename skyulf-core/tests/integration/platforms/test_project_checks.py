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


_TUNER = {
    "type": "hyperparameter_tuner",
    "base_model": {"type": "linear_regression", "params": {}},
    "strategy": "random",
    "metric": "rmse",
    "n_trials": 2,
    "random_state": 42,
    "search_space": {"fit_intercept": [True, False]},
}
_HALVING = {
    **_TUNER,
    "base_model": {"type": "random_forest_regressor", "params": {}},
    "strategy": "halving_random",
    "strategy_params": {
        "factor": 3,
        "resource": "n_estimators",
        "min_resources": "10",
        "max_resources": None,
    },
    "search_space": {"max_depth": [None, 5]},
}
_STRATIFIED = {"cv_enabled": True, "cv_folds": 3, "cv_type": "stratified_k_fold"}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"pipeline": {"preprocessing": [], "modeling": _HALVING}},
            "halving max_resources must be auto",
        ),
        (
            {**_STRATIFIED, "pipeline": {"preprocessing": [], "modeling": _TUNER}},
            "Stratified CV requires a classification model",
        ),
        (
            {"training_layout": "model_competition", "metric": "heldout_mape"},
            "Unsupported competition metric heldout_mape for regression",
        ),
    ],
)
def test_smoke_rejects_search_settings_that_preview_rejects(
    tmp_path, workflow_config, changes, message
):
    """Smoke must not pass a search or competition setting that preview and training reject."""
    root = _project(tmp_path, {**workflow_config, **changes})
    with pytest.raises(ValueError) as caught:
        check_project(root, _bindings())
    assert str(caught.value).startswith("config/workflow.json: ")
    assert message in str(caught.value)


def test_smoke_accepts_an_admissible_search(tmp_path, workflow_config):
    """The added search preflight must keep a valid tuner and its hooks inert."""
    settings = {**workflow_config, "pipeline": {"preprocessing": [], "modeling": _TUNER}}
    result = check_project(_project(tmp_path, settings), _bindings())
    assert result["status"] == "passed"
    assert result["project_hooks_executed"] is False


_ENDPOINT = {
    "endpoint_name": "",
    "model_name": "",
    "model_version": "",
    "logging_catalog": "",
    "logging_schema": "",
    "logging_table_prefix": "",
    "logging_mode": "disabled",
}
_PUBLICATION = {
    "source_table": "workspace.features.latest",
    "source_table_id": "0f0e",
    "online_table": "workspace.features.latest_online",
    "online_store": "store",
}


def _online_feature_lookup(**changes):
    """Return an online_feature_lookup section shaped like the generated config/serving.yml."""
    return {
        "enabled": False,
        "exclusive_writer": False,
        "tracking_uri": "databricks",
        "registry_uri": "databricks-uc",
        "endpoint": _ENDPOINT,
        "publications": [{**_PUBLICATION, "publish_mode": "TRIGGERED"}],
        **changes,
    }


def _serving_project(tmp_path, workflow_config, document):
    """Write config/serving.yml as JSON, which the YAML reader accepts unchanged."""
    root = _project(tmp_path, workflow_config)
    (root / "config/serving.yml").write_text(json.dumps(document), encoding="utf-8")
    return root


def test_smoke_accepts_generated_disabled_serving_sections(tmp_path, workflow_config):
    """An unmodified generated project keeps every serving job paused and must pass smoke."""
    document = {
        "online_feature_lookup": _online_feature_lookup(),
        "rollout": {"enabled": False, "exclusive_writer": False, "run_id": ""},
        "online_publication": {"enabled": False, "exclusive_writer": False, "source_table": ""},
    }
    result = check_project(_serving_project(tmp_path, workflow_config, document), _bindings())
    assert result["status"] == "passed"


@pytest.mark.parametrize("include_new", [False, True])
def test_smoke_rejects_old_online_serving_section(tmp_path, workflow_config, include_new):
    """Renamed serving YAML sections must not disappear silently from offline validation."""
    document = {"online_serving": _online_feature_lookup()}
    if include_new:
        document["online_feature_lookup"] = _online_feature_lookup()
    with pytest.raises(ValueError, match="online_serving.*online_feature_lookup"):
        check_project(_serving_project(tmp_path, workflow_config, document), _bindings())


def test_smoke_accepts_complete_enabled_online_feature_lookup_offline(tmp_path, workflow_config):
    """A complete endpoint pin is checked locally; no endpoint or publication is contacted."""
    endpoint = {
        **_ENDPOINT,
        "endpoint_name": "risk-endpoint",
        "model_name": "workspace.models.risk",
        "model_version": "3",
    }
    document = {
        "online_feature_lookup": _online_feature_lookup(enabled=True, exclusive_writer=True)
    }
    document["online_feature_lookup"]["endpoint"] = endpoint
    result = check_project(_serving_project(tmp_path, workflow_config, document), _bindings())
    assert result["remote_operations"] is False


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (
            {"online_feature_lookup": _online_feature_lookup(enabled=True, exclusive_writer=True)},
            "config/serving.yml: online_feature_lookup: ",
        ),
        (
            {"online_feature_lookup": _online_feature_lookup(enabled=True)},
            "config/serving.yml: online_feature_lookup: Serving job requires exclusive_writer=true",
        ),
        (
            {"rollout": {"enabled": "false"}},
            "config/serving.yml: rollout: Serving job requires an explicit enabled boolean.",
        ),
        (
            {"rollout": {"enabled": True, "exclusive_writer": True, "stage_minimum_hours": -1}},
            "config/serving.yml: rollout: Duration hours must be positive and finite.",
        ),
        (
            {
                "online_publication": {
                    "enabled": True,
                    "exclusive_writer": True,
                    **_PUBLICATION,
                    "source_table": "",
                }
            },
            "config/serving.yml: online_publication: Unity Catalog names must be",
        ),
        (
            {"online_publication": {"enabled": True, "exclusive_writer": True}},
            "config/serving.yml: online_publication: missing required setting 'source_table'.",
        ),
        ([], "config/serving.yml: "),
    ],
)
def test_smoke_rejects_enabled_serving_settings_the_jobs_reject(
    tmp_path, workflow_config, document, message
):
    """Serving pins must fail in smoke, not first inside a scheduled serving job."""
    root = _serving_project(tmp_path, workflow_config, document)
    with pytest.raises(ValueError) as caught:
        check_project(root, _bindings())
    assert str(caught.value).startswith(message)
