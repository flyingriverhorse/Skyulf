"""Check generated search catalogs against Core and the native bundle initializer."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"
CLI = shutil.which("databricks")
PROFILE = os.environ.get("SKYULF_BUNDLE_CLI_TEST_PROFILE")


def _builder():
    """Load the generator without relying on the caller's working directory."""
    spec = importlib.util.spec_from_file_location(
        "bundle_model_spaces", ROOT / "build_model_spaces.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_space_catalog_is_fresh_and_deterministic():
    """Committed catalogs must follow current Core defaults and stable ordering."""
    builder = _builder()
    rendered = builder.render_model_spaces()
    assert rendered == builder.render_model_spaces()
    assert rendered == (ROOT / "library/model_search_space.tmpl").read_text(encoding="utf-8")


def test_catalog_generator_needs_no_installed_packages(tmp_path):
    """Maintainers can refresh catalogs with Python's standard library alone."""
    result = subprocess.run(
        [sys.executable, "-S", str(ROOT / "build_model_spaces.py"), "--check"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_catalog_check_detects_stale_output_without_rewriting(tmp_path, monkeypatch):
    """The freshness gate must fail safely when its committed output is stale."""
    builder = _builder()
    rendered = builder.render_model_spaces()
    output = tmp_path / "library/model_search_space.tmpl"
    output.parent.mkdir()
    output.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "render_model_spaces", lambda: rendered)
    monkeypatch.setattr(sys, "argv", ["build_model_spaces.py", "--check"])
    assert builder.main() == 1
    assert output.read_text(encoding="utf-8") == "stale"


def _go_map(values):
    """Encode nested string maps using the CLI's native map and pair helpers."""
    pairs = []
    for key, value in values.items():
        encoded = _go_map(value) if isinstance(value, dict) else json.dumps(value)
        pairs.append(f"(pair {json.dumps(key)} {encoded})")
    return "(map " + " ".join(pairs) + ")"


@pytest.fixture(scope="module")
def rendered_spaces(tmp_path_factory):
    """Render all catalog variants once using the installed Databricks CLI."""
    if not CLI or not PROFILE:
        pytest.skip("Set SKYULF_BUNDLE_CLI_TEST_PROFILE to opt into local CLI generation.")
    from skyulf.modeling.hyperparameters import DEFAULT_SEARCH_SPACES

    cases = {}
    for model in DEFAULT_SEARCH_SPACES:
        if model.startswith(("voting_", "stacking_")):
            continue
        for strategy in ("grid", "random", "optuna", "halving_grid", "halving_random"):
            cases[f"{model}:{strategy}"] = {
                "model": model,
                "task": "regression",
                "strategy": strategy,
                "resource": "n_estimators",
                "values": {},
                "ensemble_prefix": "",
            }
    for task, suffix, learners, final in (
        (
            "classification",
            "classifier",
            ["svc", "random_forest", "lightgbm"],
            "logistic_regression",
        ),
        ("regression", "regressor", ["ridge", "random_forest", "xgboost"], "elasticnet"),
    ):
        for family in ("voting", "stacking"):
            for calibrated in ("false", "true"):
                for strategy in ("grid", "random"):
                    prefix = "branch_1_ensemble_"
                    values = {
                        prefix + task + "_base_count": "2",
                        prefix + "calibrate": calibrated,
                        prefix + task + "_final": final,
                    }
                    values.update(
                        {prefix + task + f"_base_{i}": name for i, name in enumerate(learners, 1)}
                    )
                    cases[f"{family}_{suffix}:{calibrated}:{strategy}"] = {
                        "model": f"{family}_{suffix}",
                        "task": task,
                        "strategy": strategy,
                        "resource": "",
                        "values": values,
                        "ensemble_prefix": prefix,
                    }
    registry = _builder()._load_registry()
    for task, suffix, learners in (
        ("classification", "classifier", registry._BASE_KEY_TO_REGISTRY_CLF),
        ("regression", "regressor", registry._BASE_KEY_TO_REGISTRY_REG),
    ):
        prefix = "competition_ensemble_8_"
        values = {prefix + task + "_base_count": str(len(learners)), prefix + "calibrate": "true"}
        values.update({prefix + task + f"_base_{i}": name for i, name in enumerate(learners, 1)})
        cases[f"all_bases:{task}"] = {
            "model": f"voting_{suffix}",
            "task": task,
            "strategy": "halving_random",
            "resource": "random_forest__estimator__n_estimators"
            if task == "classification"
            else "random_forest__n_estimators",
            "values": values,
            "ensemble_prefix": prefix,
        }
    temporary = tmp_path_factory.mktemp("catalog_cli")
    template = temporary / "source"
    (template / "template").mkdir(parents=True)
    (template / "library").mkdir()
    shutil.copyfile(
        ROOT / "library/model_search_space.tmpl", template / "library/model_search_space.tmpl"
    )
    (template / "databricks_template_schema.json").write_text(
        json.dumps(
            {
                "properties": {
                    "project_name": {
                        "type": "string",
                        "default": "catalog",
                        "description": "Catalog verification",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (temporary / "inputs.json").write_text('{"project_name":"catalog"}', encoding="utf-8")
    entries = [
        json.dumps(key) + ': {{template "model_search_space" ' + _go_map(value) + "}}"
        for key, value in cases.items()
    ]
    (template / "template/spaces.json.tmpl").write_text(
        "{" + ",\n".join(entries) + "}", encoding="utf-8"
    )
    result = subprocess.run(
        [
            str(CLI),
            "bundle",
            "init",
            str(template),
            "--config-file",
            str(temporary / "inputs.json"),
            "--output-dir",
            str(temporary / "output"),
            "--profile",
            str(PROFILE),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return cases, json.loads((temporary / "output/spaces.json").read_text(encoding="utf-8"))


def test_cli_catalog_matches_all_ordinary_core_spaces(rendered_spaces):
    """Every strategy must retain Core values and reserve halving's resource axis."""
    from skyulf.modeling.hyperparameters import get_default_search_space

    cases, spaces = rendered_spaces
    for name, case in cases.items():
        if case["ensemble_prefix"]:
            continue
        expected = dict(get_default_search_space(case["model"], case["strategy"]))
        if case["strategy"].startswith("halving_"):
            expected.pop(case["resource"], None)
        assert spaces[name] == expected, name


def test_cli_ensemble_catalog_respects_composition_and_calibration(rendered_spaces):
    """Only selected learners are searched while structural parameters stay fixed."""
    from skyulf.modeling.hyperparameters import build_ensemble_search_space

    cases, spaces = rendered_spaces
    for name, case in cases.items():
        prefix = case["ensemble_prefix"]
        if not prefix:
            continue
        task = case["task"]
        values = case["values"]
        expected = build_ensemble_search_space(
            case["model"],
            [
                values[prefix + task + f"_base_{i}"]
                for i in range(1, int(values[prefix + task + "_base_count"]) + 1)
            ],
            values[prefix + task + "_final"] if case["model"].startswith("stacking_") else "",
            strategy=case["strategy"],
            problem_type=task,
            calibrate_base_models=values[prefix + "calibrate"] == "true",
        )
        expected = {key: value for key, value in expected.items() if "__" in key}
        if case["strategy"].startswith("halving_"):
            expected.pop(case["resource"], None)
        assert spaces[name] == expected, name
