"""Producer projects provision one shared store and reuse it without owner-only DDL."""

import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_monitoring_model_set import component
from test_monitoring_registration import settings, workflow

from skyulf.integrations.databricks.observability.monitoring import monitoring_store as store
from skyulf.integrations.databricks.observability.monitoring.performance.performance_actions import (
    ACTION_SCHEMA,
)


class StoreSpark:
    """Model object creation, metadata reads and enrollment writes without a cluster."""

    def __init__(self):
        """Start with no shared namespace objects."""
        self.objects = {}
        self.statements = []
        self.catalog = SimpleNamespace(
            tableExists=lambda name: name in self.objects, dropTempView=lambda name: None
        )
        self.race_owner = None

    def createDataFrame(self, rows, schema):
        """Keep schema comparisons exact and support the real typed merge path."""
        return SimpleNamespace(
            schema=SimpleNamespace(simpleString=lambda: schema),
            createOrReplaceTempView=lambda name: None,
        )

    def table(self, name):
        """Provide an empty inventory after the first scheduled bootstrap."""
        frame = Mock()
        frame.schema.simpleString.return_value = self.objects[name]["schema"]
        frame.select.return_value.limit.return_value.collect.return_value = []
        return frame

    def sql(self, statement):
        """Record DDL and simulate another creator winning CREATE IF NOT EXISTS."""
        self.statements.append(statement)
        result = Mock()
        result.collect.return_value = []
        if statement.startswith("SHOW TBLPROPERTIES"):
            match = re.search(r"SHOW TBLPROPERTIES ([^ ]+)", statement)
            assert match is not None, "Property inspection must identify a concrete object"
            name = match[1].replace("`", "")
            key = "owner" if store.PROPERTY in statement else "isolation"
            result.first.return_value = {"value": self.objects[name].get(key)}
        elif statement.startswith(("CREATE TABLE", "CREATE VIEW")):
            match = re.search(r"IF NOT EXISTS ([^ ]+)", statement)
            assert match is not None, "Bootstrap must use conditional object creation"
            name = match[1].replace("`", "")
            schema = (
                statement.split(" (", 1)[1].split(") USING", 1)[0] if " (" in statement else "view"
            )
            self.objects.setdefault(
                name,
                {
                    "schema": schema,
                    "owner": self.race_owner or store.OWNER,
                    "isolation": "Serializable",
                },
            )
        return result


def ensure(spark):
    """Exercise the production bootstrap API once it is installed."""
    operation = getattr(store, "ensure_monitoring_store", None)
    assert callable(operation), "Producer bootstrap API is missing"
    return operation(spark, "ops", "monitoring")


def existing_store():
    """Represent a compatible shared store owned by another project principal."""
    spark = StoreSpark()
    schemas = {
        "model_inventory": store.INVENTORY_SCHEMA,
        "monitoring_results": store.RESULT_SCHEMA,
        "performance_actions": ACTION_SCHEMA,
        **dict.fromkeys(store.monitoring_views("ops.monitoring"), "view"),
    }
    spark.objects = {
        f"ops.monitoring.{name}": {"schema": ddl, "owner": store.OWNER, "isolation": "Serializable"}
        for name, ddl in schemas.items()
    }
    return spark


def mutations(spark):
    """Exclude read-only metadata inspection from the side-effect assertions."""
    return [sql for sql in spark.statements if not sql.startswith("SHOW")]


def test_fresh_store_is_created_once_and_second_producer_needs_no_ddl():
    """Two model projects must share six objects without replacing each other's views."""
    spark = StoreSpark()
    assert ensure(spark) == "ops.monitoring"
    assert len(spark.objects) == 6
    assert not any("REPLACE" in sql or "ALTER" in sql for sql in mutations(spark))
    spark.statements.clear()
    assert ensure(spark) == "ops.monitoring"
    assert mutations(spark) == []


def test_partial_store_creates_only_missing_objects():
    """An additive bootstrap cannot require ownership of already compatible shared views."""
    spark = existing_store()
    del spark.objects["ops.monitoring.metric_history"]
    ensure(spark)
    ddl = mutations(spark)
    assert len([sql for sql in ddl if sql.startswith("CREATE VIEW")]) == 1
    assert all("REPLACE" not in sql and "CREATE TABLE" not in sql for sql in ddl)


@pytest.mark.parametrize("defect", ["foreign", "schema", "actions_schema", "isolation"])
def test_incompatible_existing_store_fails_before_any_ddl(defect):
    """Missing sibling objects must not permit writes before all compatibility checks pass."""
    spark = existing_store()
    del spark.objects["ops.monitoring.metric_history"]
    name = "performance_actions" if defect == "actions_schema" else "model_inventory"
    key = {
        "foreign": "owner",
        "schema": "schema",
        "actions_schema": "schema",
        "isolation": "isolation",
    }[defect]
    spark.objects[f"ops.monitoring.{name}"][key] = "incompatible"
    with pytest.raises(ValueError):
        ensure(spark)
    assert mutations(spark) == []


def test_concurrent_foreign_creation_is_rejected_before_enrollment():
    """A CREATE IF NOT EXISTS race must not relabel or populate an unrelated table."""
    spark = StoreSpark()
    spark.race_owner = "another.owner"
    with pytest.raises(ValueError, match="ownership"):
        ensure(spark)
    assert not any(sql.startswith(("ALTER", "MERGE")) for sql in mutations(spark))


@pytest.mark.parametrize("layout", ["single_model", "model_competition"])
def test_scoring_producer_bootstraps_before_first_inventory_merge(layout):
    """Both producer layouts must work without a separate monitoring deployment."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        register_scoring_monitor,
    )

    spark = StoreSpark()
    payload = {
        "result": {"selected_model_name": "models.risk.model", "selected_model_version": "2"}
    }
    result = register_scoring_monitor(spark, workflow(training_layout=layout), settings(), payload)
    assert result is not None
    assert result["inventory_table"] == "ops.monitoring.model_inventory"
    assert len(spark.objects) == 6
    assert mutations(spark)[-1].startswith("MERGE")


def test_existing_shared_store_accepts_producer_without_creation_or_replacement():
    """A second project needs enrollment write access without becoming the view owner."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        register_scoring_monitor,
    )

    spark = existing_store()
    payload = {
        "result": {"selected_model_name": "models.risk.model", "selected_model_version": "2"}
    }
    register_scoring_monitor(spark, workflow(), settings(), payload)
    assert len(mutations(spark)) == 1
    assert mutations(spark)[0].startswith("MERGE")


@pytest.mark.parametrize("changes", [{"monitoring_deployment_mode": "development"}, {}])
def test_unconfigured_or_development_producer_never_provisions(changes):
    """An inactive producer must not acquire accidental central storage dependencies."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        register_scoring_monitor,
    )

    spark = StoreSpark()
    assert register_scoring_monitor(spark, {}, changes, {}) is None
    assert spark.statements == []


def test_activation_bootstraps_before_first_inventory_merge():
    """The activated version must appear before its first scoring run."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        register_deployed_monitor,
    )

    spark = StoreSpark()
    payload = {
        "action": "approve",
        "result": {"kind": "promotion", "model_name": "models.risk.model", "new_version": "2"},
    }
    result = register_deployed_monitor(
        spark, workflow(), settings(), payload, activation_started_ms=123
    )
    assert result is not None
    assert result["inventory_table"] == "ops.monitoring.model_inventory"
    assert len(spark.objects) == 6


@pytest.mark.parametrize("invalid", ["activation", "version", "settings", "receipt"])
def test_invalid_activation_never_bootstraps(invalid):
    """All lifecycle validation must precede provisioning a new central namespace."""
    from skyulf.integrations.databricks.observability.monitoring.monitoring_registration import (
        register_deployed_monitor,
    )

    spark = StoreSpark()
    values = (
        settings(monitoring_drift_thresholds='{"psi": 0}') if invalid == "settings" else settings()
    )
    receipt = {"kind": "promotion", "model_name": "models.risk.model", "new_version": "2"}
    if invalid == "version":
        receipt["new_version"] = "bad"
    if invalid == "receipt":
        receipt["model_name"] = "other.risk.model"
    with pytest.raises(ValueError):
        register_deployed_monitor(
            spark,
            workflow(),
            values,
            {"action": "approve", "result": receipt},
            activation_started_ms=0 if invalid == "activation" else 123,
        )
    assert mutations(spark) == []


@pytest.mark.parametrize("invalid_second", [False, True])
def test_model_set_validates_all_components_before_shared_bootstrap(invalid_second):
    """A malformed later component cannot leave behind a partially enrolled model set."""
    from skyulf.integrations.databricks.model_sets.monitoring_model_set import register_set_monitors

    spark = StoreSpark()
    parent = {"model_name": "models.risk.set", "prediction_table": "outputs.risk.set_scores"}
    resolved = SimpleNamespace(name=parent["model_name"], version="9")
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            components=[
                component("revenue", "2"),
                component("cost", "bad" if invalid_second else "4"),
            ]
        )
    )
    if invalid_second:
        with pytest.raises(ValueError):
            register_set_monitors(spark, workflow(), settings(), parent, resolved, artifact)
        assert mutations(spark) == []
    else:
        result = register_set_monitors(spark, workflow(), settings(), parent, resolved, artifact)
        assert result is not None
        assert len(result["configs"]) == 2
        assert len([sql for sql in mutations(spark) if sql.startswith("MERGE")]) == 2


def test_fresh_scheduled_project_creates_empty_store_without_models():
    """An enabled schedule must succeed before the first producer activation."""
    from skyulf.integrations.databricks.jobs.monitoring.spark_monitoring_job import (
        run_project_monitoring_notebook,
    )

    spark = StoreSpark()
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings()
    assert run_project_monitoring_notebook(spark, dbutils) == {"status": "no_active_models"}
    assert len(spark.objects) == 6


@pytest.mark.parametrize(
    "changes", [{"monitoring_deployment_mode": "development"}, {"monitoring_enabled": "false"}]
)
def test_inactive_schedule_does_not_bootstrap(changes):
    """A paused schedule or development target must never create shared infrastructure."""
    from skyulf.integrations.databricks.jobs.monitoring.spark_monitoring_job import (
        run_project_monitoring_notebook,
    )

    spark = StoreSpark()
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings(**changes)
    assert run_project_monitoring_notebook(spark, dbutils) == {"status": "disabled"}
    assert spark.statements == []


@pytest.mark.parametrize(
    "changes",
    [
        {"monitoring_environment": ""},
        {"monitoring_project": "invalid project"},
        {"as_of": "not-a-time"},
        {"as_of": "2026-10-05T00:00:00"},
        {"monitoring_revisit_windows": "0"},
        {"monitoring_revisit_windows": "101"},
        {"monitoring_action": "invalid"},
        {"monitoring_request": "invalid-json"},
        {"monitoring_request": '{"namespace": "other.store"}'},
    ],
)
def test_invalid_schedule_or_request_never_bootstraps(changes):
    """Bad schedule settings and foreign scoring receipts cannot provision shared objects."""
    from skyulf.integrations.databricks.jobs.monitoring.spark_monitoring_job import (
        run_project_monitoring_notebook,
    )

    spark = StoreSpark()
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings(**changes)
    with pytest.raises(ValueError):
        run_project_monitoring_notebook(spark, dbutils)
    assert mutations(spark) == []


@pytest.mark.parametrize("raw", ["null", "[]", '"receipt"', "123", "true"])
def test_non_object_request_cannot_be_treated_as_scheduled_observation(raw):
    """Only an absent receipt authorizes the scheduled bootstrap path."""
    from skyulf.integrations.databricks.jobs.monitoring.spark_monitoring_job import (
        run_project_monitoring_notebook,
    )

    spark = StoreSpark()
    dbutils = Mock()
    dbutils.widgets.getAll.return_value = settings(monitoring_request=raw)
    with pytest.raises(ValueError, match="object"):
        run_project_monitoring_notebook(spark, dbutils)
    assert mutations(spark) == []
