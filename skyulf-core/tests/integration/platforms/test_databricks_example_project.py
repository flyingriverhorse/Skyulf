"""Keep the prepared NYC taxi examples current and separate from normal generated projects."""

import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from test_databricks_bundle_generation import CLI, OFFLINE_CLI, PROFILE, _generate_project

ROOT = Path(__file__).resolve().parents[3] / "templates/databricks"
TEMPLATE = ROOT / "template/{{.project_name}}"
EXAMPLES = ROOT / "examples/nyctaxi"
BUILDER = runpy.run_path(str(ROOT / "build_example_project.py"))
PROJECTS = list(BUILDER["PROJECTS"])
LAYOUTS = ["single_model", "model_competition", "multi_target"]
CLI_ONLY = pytest.mark.skipif(not (PROFILE or OFFLINE_CLI) or not CLI, reason="Opt into CLI")


@CLI_ONLY
@pytest.mark.parametrize("layout", LAYOUTS)
def test_normal_generation_excludes_the_prepared_examples(tmp_path, layout):
    """Example files and recipes must never appear in an operator's own project."""
    project = _generate_project(
        tmp_path,
        training_layout=layout,
        source_table_name="real_customer_feed",
        record_key_columns="order_id",
        input_columns="amount",
        target_column="risk_score",
    )
    assert not (project / "examples").exists()
    assert not list(project.rglob("trip_rules.py"))
    training = yaml.safe_load((project / "config/training.yml").read_text())
    for entry in training["models"].values():
        config = training["defaults"] | entry
        assert config.get("preprocessing_recipe", "default") == "default"
    for arguments in (["smoke.py"], ["preview.py", "--action", "train"]):
        result = subprocess.run(
            [sys.executable, str(project / "src/tools" / arguments[0]), *arguments[1:]],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@CLI_ONLY
@pytest.mark.parametrize("name", PROJECTS)
def test_prepared_example_matches_regeneration_and_previews(tmp_path, name):
    """A stored example must equal template output plus its own files, and its recipes must run."""
    regenerated = BUILDER["generate_example"](tmp_path, name)
    assert BUILDER["project_files"](regenerated) == BUILDER["project_files"](
        BUILDER["example_root"](name) / name
    )
    assert yaml.safe_load((regenerated / "config/training.yml").read_text())["defaults"][
        "preprocessing_probe"
    ]
    for arguments in (
        ["smoke.py"],
        ["preview.py", "--action", "train"],
        ["preview.py", "--action", "score"],
    ):
        result = subprocess.run(
            [sys.executable, str(regenerated / "src/tools" / arguments[0]), *arguments[1:]],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", PROJECTS)
def test_example_copies_of_static_template_files_are_current(name):
    """Without the CLI, CI still catches a template edit that a stored example missed."""
    own = set(BUILDER["PROJECTS"][name]["sources"])
    compared = 0
    for source in TEMPLATE.rglob("*"):
        relative = source.relative_to(TEMPLATE).as_posix()
        copy = BUILDER["example_root"](name) / name / relative
        if (
            not source.is_file()
            or source.suffix in {".tmpl", ".pyc"}
            or "__pycache__" in source.parts
            or relative in own
            or not copy.is_file()
        ):
            continue
        text = source.read_bytes().replace(b"\r\n", b"\n")
        if b"{{" in text:
            continue
        assert copy.read_bytes().replace(b"\r\n", b"\n") == text, relative
        compared += 1
    assert compared >= 20


def test_examples_add_no_setup_question():
    """Reading the samples must not change the user's quick or detailed wizard path."""
    properties = json.loads((ROOT / "databricks_template_schema.json").read_text())["properties"]
    assert "include_example" not in properties
    for path in EXAMPLES.glob("*.json"):
        assert set(json.loads(path.read_text())) <= set(properties), path.name


def test_loader_creates_every_column_the_examples_read():
    """The SQL loader must be create-only, keep change data feed, and supply every answered column."""
    sql = (EXAMPLES / "load_nyctaxi.sql").read_text(encoding="utf-8")
    code = "\n".join(line for line in sql.splitlines() if not line.lstrip().startswith("--"))
    assert "OR REPLACE" not in code
    assert "IF NOT EXISTS" not in code.replace("CREATE SCHEMA IF NOT EXISTS", "")
    tables = {}
    for table, columns in re.findall(
        r"'\.(trips_\w+)'\)\s+TBLPROPERTIES[^;]*?SELECT (.*?)\nFROM", code, re.S
    ):
        tables[table] = {column.split(" AS ")[-1].strip() for column in columns.split(",")}
    assert set(tables) == {"trips_train", "trips_score", "trips_labels"}
    assert code.count("delta.enableChangeDataFeed = true") == 4
    match = re.search(r"'\.zone_features'\) \((.*?)\n\)", code, re.S)
    assert match, "load_nyctaxi.sql must declare zone_features with its columns"
    declared = match.group(1)
    zone_columns = {line.split()[0] for line in declared.splitlines() if line.strip()}
    assert "PRIMARY KEY (pickup_zip)" in declared and "pickup_zip STRING NOT NULL" in declared
    for path in EXAMPLES.glob("*.json"):
        answers = json.loads(path.read_text())
        needed = {answers["record_key_columns"]}
        for key, value in answers.items():
            if key.endswith("input_columns"):
                needed |= {column.strip() for column in value.split(",")}
            elif key.endswith("target_column"):
                needed.add(value)
        training = yaml.safe_load((EXAMPLES / path.stem / "config/training.yml").read_text())
        lookups = training["defaults"].get("feature_lookup", {}).get("lookups", [])
        looked_up = {name for lookup in lookups for name in lookup["feature_names"]}
        keys = {name for lookup in lookups for name in lookup["lookup_key"]}
        assert looked_up <= zone_columns, path.name
        assert answers["source_table_name"] == "trips_train"
        assert answers.get("score_source_table_name") == "trips_score"
        assert (needed - looked_up) | keys <= tables["trips_train"] & tables["trips_score"]
