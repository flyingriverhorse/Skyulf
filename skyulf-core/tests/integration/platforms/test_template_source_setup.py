"""Source and policy answers must become target-bound, executable configuration."""

import json
import re

import pytest
import yaml
from test_databricks_bundle_generation import _generate_project, _initialize_project

from skyulf.integrations.databricks.observability.monitoring.performance.performance_policy import (
    validate_performance_policy,
)


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("mode", ["full_snapshot", "from_now"])
def test_start_choice_reaches_each_generated_workflow(tmp_path, layout, mode):
    """All scoring layouts must consume the same explicit first-run selection policy."""
    from skyulf.integrations.databricks.lifecycle.workflow import _scoring_config
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    project = _generate_project(tmp_path, training_layout=layout, initial_scoring=mode)
    config = read_workflow_config(project / "config/training.yml")
    assert config["initial_scoring"] == mode
    assert _scoring_config(config).initial_scoring == mode
    assert "initial_scoring" in (project / "SOURCES.md").read_text()


@pytest.mark.parametrize(
    "table,expected",
    [
        ("outcomes", "${var.catalog}.${var.input_schema}.outcomes"),
        ("truth.prod.outcomes", "truth.prod.outcomes"),
    ],
)
def test_label_answers_bind_to_target_or_keep_explicit_namespace(tmp_path, table, expected):
    """Moving a project between targets must not send labels to a baked-in test namespace."""
    project = _generate_project(
        tmp_path, label_table_name=table, label_available_at_column="known_at"
    )
    variables = yaml.safe_load((project / "deployment/variables.yml").read_text())["variables"]
    assert variables["monitoring_label_table"]["default"] == expected
    assert variables["monitoring_result_available_at_column"]["default"] == "known_at"


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_performance_policy_covers_each_component_with_matching_metric(tmp_path, layout):
    """Mixed target sets need separate model identities and metric directions."""
    project = _generate_project(
        tmp_path,
        training_layout=layout,
        monitoring_catalog="cat",
        monitoring_schema="monitoring",
        label_table_name="outcomes",
        label_available_at_column="known_at",
        performance_policy_mode="report",
        branch_1_task="regression",
        branch_2_task="classification",
    )
    variables = yaml.safe_load((project / "deployment/variables.yml").read_text())["variables"]
    policies = json.loads(variables["monitoring_performance_policies"]["default"])
    assert len(policies) == (2 if layout == "multi_target" else 1)
    assert all(name.startswith("${var.catalog}.${var.metadata_schema}.") for name in policies)
    assert all(name.endswith("${var.resource_suffix}") for name in policies)
    for policy in policies.values():
        assert validate_performance_policy(policy) == policy
        assert policy["mode"] == "report"
        assert policy["baseline"] == {"kind": "training_holdout", "model_version": "1"}
    if layout == "multi_target":
        assert {policy["metric"] for policy in policies.values()} == {"wape", "accuracy"}


@pytest.mark.parametrize(
    "answers",
    [
        {"label_table_name": "outcomes"},
        {"label_available_at_column": "known_at"},
        {"performance_policy_mode": "report"},
        {
            "performance_policy_mode": "retrain",
            "label_table_name": "outcomes",
            "label_available_at_column": "known_at",
        },
    ],
)
def test_incomplete_label_or_performance_binding_fails_at_generation(tmp_path, answers):
    """An enabled policy must not generate a job with no labels or monitoring store."""
    result, _ = _initialize_project(tmp_path, **answers)
    assert result.returncode != 0
    assert "ERROR_" in result.stderr + result.stdout


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_separate_outcomes_connects_training_and_monitoring(tmp_path, layout):
    """A shared feature source must learn from the same explicitly selected actual outcomes."""
    from skyulf.integrations.databricks.projects.yaml_config import read_workflow_config

    project = _generate_project(
        tmp_path,
        source_layout="separate_outcomes",
        training_layout=layout,
        label_table_name="outcomes",
        label_available_at_column="known_at",
    )
    document = yaml.safe_load((project / "config/training.yml").read_text())
    assert document["defaults"]["training_label_table"] == "{catalog}.{input_schema}.outcomes"
    for model in (
        document["models"].values() if layout == "multi_target" else [document["defaults"]]
    ):
        assert model["filter_unavailable_results"] is True
        assert model["result_available_at_column"] == "known_at"
    config = read_workflow_config(project / "config/training.yml")
    assert config["training_table"] == config["score_source_table"]
    assert "SOURCES.md" in (project / "README.md").read_text()
    assert not (project / "config/sources.sql").exists()


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("include", ["false", "true"])
def test_source_sql_is_opt_in_without_hiding_checks(tmp_path, layout, include):
    """Existing-table projects need source guidance and checks without an unsolicited DDL file."""
    project = _generate_project(tmp_path, training_layout=layout, include_source_sql=include)
    assert (project / "config/sources.sql").is_file() == (include == "true")
    guide = (project / "SOURCES.md").read_text()
    assert "--action sources" in guide
    assert "Existing tables" in guide
    assert ("[config/sources.sql](config/sources.sql)" in guide) == (include == "true")
    assert "include_source_sql" in (project / "README.md").read_text()
    assert (project / "src/tools/preview.py").is_file()


@pytest.mark.parametrize(
    "answers",
    [
        {"source_layout": "separate_outcomes"},
        {
            "source_layout": "separate_outcomes",
            "score_source_table_name": "other",
            "label_table_name": "outcomes",
            "label_available_at_column": "known_at",
        },
        {
            "source_layout": "separate_all",
            "label_table_name": "outcomes",
            "label_available_at_column": "known_at",
        },
    ],
)
def test_source_layout_rejects_incomplete_or_contradictory_tables(tmp_path, answers):
    """The chosen source arrangement must describe the files actually generated."""
    result, _ = _initialize_project(tmp_path, **answers)
    assert result.returncode != 0
    assert "ERROR_source_layout" in result.stderr + result.stdout


@pytest.mark.parametrize("label", ["observations", "OBSERVATIONS", "cat.data.observations"])
def test_source_sql_creates_a_shared_label_source_only_once(tmp_path, label):
    """A single table serving both roles must include its availability column without duplicate DDL."""
    project = _generate_project(
        tmp_path,
        source_table_name="observations",
        label_table_name=label,
        catalog="cat",
        schema="data",
        label_available_at_column="known_at",
        cv_group_column="customer",
        include_source_sql="true",
    )
    sql = (project / "config/sources.sql").read_text()
    assert sql.count("CREATE TABLE {catalog}.{input_schema}.observations") == 1
    assert "`known_at` TIMESTAMP" in sql
    assert "`customer` TYPE_FOR_customer" in sql


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
@pytest.mark.parametrize("sources", ["single_table", "separate_outcomes", "separate_all"])
def test_source_sql_places_availability_on_the_table_that_owns_outcomes(tmp_path, layout, sources):
    """Separate outcomes must not require label metadata on measurements or drop other inputs."""
    project = _generate_project(
        tmp_path,
        training_layout=layout,
        source_layout=sources,
        source_table_name="measurements",
        score_source_table_name="new_measurements" if sources == "separate_all" else "",
        label_table_name="outcomes",
        label_available_at_column="known_at",
        record_key_columns="id",
        input_columns="measurement",
        target_column="value",
        event_column="event_time",
        cv_group_column="group_id",
        result_available_at_column="training_known_at",
        branch_1_input_columns="measurement",
        branch_2_input_columns="measurement",
        branch_1_target_column="first_value",
        branch_2_target_column="second_value",
        branch_1_event_column="event_time",
        branch_2_event_column="event_time",
        branch_1_cv_group_column="group_id",
        branch_2_cv_group_column="group_id",
        branch_1_result_available_at_column="first_known_at",
        branch_2_result_available_at_column="second_known_at",
        include_source_sql="true",
    )
    sql = (project / "config/sources.sql").read_text()
    tables = dict(re.findall(r"CREATE TABLE (\S+) \((.*?)\) USING DELTA", sql, re.S))
    columns = {table: set(re.findall(r"`([^`]+)`", body)) for table, body in tables.items()}
    targets = {"first_value", "second_value"} if layout == "multi_target" else {"value"}
    availability = (
        {"first_known_at", "second_known_at"} if layout == "multi_target" else {"training_known_at"}
    )
    training = {"id", "measurement", "event_time", "group_id"}
    if sources != "separate_outcomes":
        training |= targets | availability
    assert columns["{catalog}.{input_schema}.measurements"] == training
    assert columns["{catalog}.{input_schema}.outcomes"] == {"id", "known_at"} | targets
    assert "`known_at` TIMESTAMP" in tables["{catalog}.{input_schema}.outcomes"]
    if sources == "separate_all":
        assert columns["{catalog}.{input_schema}.new_measurements"] == {"id", "measurement"}
    assert len(tables) == (3 if sources == "separate_all" else 2)


@pytest.mark.parametrize(
    "layout,labels",
    [
        ("separate_outcomes", "OBSERVATIONS"),
        ("separate_all", "score"),
        ("separate_outcomes", "cat.data.observations"),
    ],
)
def test_separate_layouts_reject_outcomes_aliasing_an_input(tmp_path, layout, labels):
    """A named separate outcome role cannot resolve to the input table itself."""
    result, _ = _initialize_project(
        tmp_path,
        source_layout=layout,
        catalog="cat",
        schema="data",
        source_table_name="observations",
        score_source_table_name="score" if layout == "separate_all" else "",
        label_table_name=labels,
        label_available_at_column="known_at",
    )
    assert result.returncode != 0
    assert "ERROR_source_layout_requires_distinct_outcomes_table" in result.stderr + result.stdout
