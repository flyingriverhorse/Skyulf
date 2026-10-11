"""Typed SQL calls preserve the pinned endpoint's model boundary."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("mlflow")
pytest.importorskip("databricks.sdk")

from skyulf.integrations.databricks import serving  # noqa: E402


@pytest.fixture
def endpoint():
    """A complete endpoint schema distinguishes fitted types from wire strings."""
    spec = serving.PinnedEndpointSpec("risk-v7", "main.ml.risk", "7", "main", "logs", "risk")
    return serving.PinnedEndpointPlan(
        spec=spec,
        config={},
        input_columns=("amount", "count", "active", "segment"),
        input_schema=(
            ("amount", "float64"),
            ("count", "Int64"),
            ("active", "boolean"),
            ("segment", "object"),
        ),
        output_schema=(("prediction", "float64"), ("eligible", "bool"), ("reason", "string")),
    )


def test_function_has_named_inputs_exact_outputs_and_nullable_transport(endpoint):
    """SQL must call the admitted package using its saved transport and full response."""
    plan = serving.build_serving_sql_function(endpoint, "main.api.score_risk_v7")
    assert plan.create_sql == (
        "CREATE FUNCTION `main`.`api`.`score_risk_v7`(\n"
        "  `amount` DOUBLE,\n  `count` BIGINT,\n  `active` BOOLEAN,\n  `segment` STRING\n)\n"
        "RETURNS STRUCT<`prediction`: DOUBLE, `eligible`: BOOLEAN, `reason`: STRING>\n"
        "LANGUAGE SQL\nNOT DETERMINISTIC\n"
        "COMMENT 'Skyulf model models:/main.ml.risk/7; endpoint risk-v7'\n"
        "RETURN ai_query(\n  endpoint => 'risk-v7',\n  request => named_struct(\n"
        "    'amount', COALESCE(`amount`, CAST('NaN' AS DOUBLE)),\n"
        "    'count', CAST(`count` AS STRING),\n"
        "    'active', CAST(`active` AS STRING),\n    'segment', `segment`\n  ),\n"
        "  returnType => 'STRUCT<`prediction`: DOUBLE, `eligible`: BOOLEAN, `reason`: STRING>',\n"
        "  failOnError => true\n)"
    )


def test_call_uses_names_and_qualified_columns_in_saved_order(endpoint):
    """A differently ordered source table must not shift features by position."""
    plan = serving.build_serving_sql_function(endpoint, "main.api.score_risk_v7")
    assert plan.call_sql(table_alias="source") == (
        "`main`.`api`.`score_risk_v7`(`amount` => `source`.`amount`, "
        "`count` => `source`.`count`, `active` => `source`.`active`, "
        "`segment` => `source`.`segment`)"
    )
    assert "`amount` => `amount`" in plan.call_sql()


def test_capture_mode_has_typed_error_envelope(endpoint):
    """Endpoint errors may be inspected without silently treating them as predictions."""
    plan = serving.build_serving_sql_function(
        endpoint, "main.api.score_risk_v7", fail_on_error=False
    )
    assert plan.return_type == (
        "STRUCT<response: STRUCT<`prediction`: DOUBLE, `eligible`: BOOLEAN, "
        "`reason`: STRING>, errorMessage: STRING>"
    )
    assert "failOnError => false" in plan.create_sql
    assert "returnType => 'STRUCT<`prediction`: DOUBLE" in plan.create_sql


@pytest.mark.parametrize("name", ["score", "main.score", "a.b.c.d", "a.b.c;DROP TABLE x", "a..c"])
def test_function_requires_unambiguous_uc_name(endpoint, name):
    """DDL must never interpolate caller SQL or resolve the current catalog implicitly."""
    with pytest.raises(ValueError, match="function_name"):
        serving.build_serving_sql_function(endpoint, name)


@pytest.mark.parametrize(
    "schema",
    [
        (),
        (("x", "date"),),
        (("x", "double; DROP TABLE t"),),
        (("x", "float64"), ("X", "float64")),
        (("bad'name", "float64"),),
    ],
)
@pytest.mark.parametrize("side", ["input", "output"])
def test_invalid_schema_fails_before_sql(endpoint, schema, side):
    """Unsupported and ambiguous schemas must fail locally instead of degrading SQL types."""
    changes = {f"{side}_schema": schema}
    if side == "input":
        changes["input_columns"] = tuple(name for name, _ in schema)
    with pytest.raises(ValueError, match="schema|identifier|dtype"):
        serving.build_serving_sql_function(replace(endpoint, **changes), "main.api.score")


@pytest.mark.parametrize(
    "dtype,sql_type,encoded",
    [
        ("int32", "INT", False),
        ("int64", "BIGINT", False),
        ("Int32", "INT", True),
        ("Int64", "BIGINT", True),
        ("bool", "BOOLEAN", False),
        ("Boolean", "BOOLEAN", True),
        ("utf8", "STRING", False),
    ],
)
def test_primitive_types_use_only_required_wire_casts(endpoint, dtype, sql_type, encoded):
    """SQL must not stringify regular inputs or round nullable 64-bit integers via floats."""
    endpoint = replace(endpoint, input_columns=("select",), input_schema=(("select", dtype),))
    plan = serving.build_serving_sql_function(endpoint, "main.api.score")
    assert f"`select` {sql_type}" in plan.create_sql
    expression = "CAST(`select` AS STRING)" if encoded else "`select`"
    assert f"'select', {expression}\n" in plan.create_sql


@pytest.mark.parametrize(
    "dtype,sql_type",
    [("float32", "FLOAT"), ("Float32", "FLOAT"), ("float64", "DOUBLE"), ("Float64", "DOUBLE")],
)
def test_sql_float_nulls_reach_fitted_imputation_instead_of_becoming_zero(
    endpoint, dtype, sql_type
):
    """Native ai_query turns numeric SQL nulls into zero unless missing floats are explicit."""
    endpoint = replace(endpoint, input_columns=("value",), input_schema=(("value", dtype),))
    ddl = serving.build_serving_sql_function(endpoint, "main.api.score").create_sql
    assert f"'value', COALESCE(`value`, CAST('NaN' AS {sql_type}))" in ddl
    assert "CAST(`value` AS STRING)" not in ddl


@pytest.mark.parametrize("error_mode", ["false", 0, None])
def test_inconsistent_input_names_and_non_boolean_error_mode_rejected(endpoint, error_mode):
    """A request must agree with the inspected columns and use an explicit error mode."""
    with pytest.raises(ValueError, match="input_columns"):
        serving.build_serving_sql_function(replace(endpoint, input_columns=("wrong",)), "a.b.c")
    with pytest.raises(TypeError, match="fail_on_error"):
        serving.build_serving_sql_function(endpoint, "a.b.c", fail_on_error=error_mode)
    plan = serving.build_serving_sql_function(endpoint, "a.b.c")
    with pytest.raises(ValueError, match="table_alias"):
        plan.call_sql(table_alias="t; --")


def test_creation_checks_readiness_before_collecting_create_only_ddl(endpoint, monkeypatch):
    """No existing function may be silently reused or overwritten."""
    from skyulf.integrations.databricks.serving import sql_functions

    plan = serving.build_serving_sql_function(endpoint, "main.api.score")
    calls = Mock()
    monkeypatch.setattr(sql_functions, "require_pinned_endpoint_ready", calls.ready)
    spark, client = SimpleNamespace(sql=calls.sql), object()
    serving.create_serving_sql_function(spark, client, plan)
    assert [call[0] for call in calls.mock_calls] == ["ready", "sql", "sql().collect"]
    calls.ready.assert_called_once_with(client, endpoint)
    calls.sql.assert_called_once_with(plan.create_sql)
    assert "OR REPLACE" not in plan.create_sql and "IF NOT EXISTS" not in plan.create_sql


@pytest.mark.parametrize(
    "failure",
    [ValueError("not ready"), ValueError("config mismatch"), PermissionError("CAN QUERY denied")],
)
def test_endpoint_failures_prevent_function_creation(endpoint, monkeypatch, failure):
    """A denied, stale or unavailable endpoint must never get a new SQL wrapper."""
    from skyulf.integrations.databricks.serving import sql_functions

    monkeypatch.setattr(sql_functions, "require_pinned_endpoint_ready", Mock(side_effect=failure))
    spark = SimpleNamespace(sql=Mock())
    plan = serving.build_serving_sql_function(endpoint, "main.api.score")
    with pytest.raises(type(failure), match=str(failure)):
        serving.create_serving_sql_function(spark, object(), plan)
    spark.sql.assert_not_called()


@pytest.mark.parametrize(
    "failure", [RuntimeError("ROUTINE_ALREADY_EXISTS"), PermissionError("CREATE FUNCTION denied")]
)
def test_sql_failures_are_not_retried_or_suppressed(endpoint, monkeypatch, failure):
    """DDL failures must stay actionable without an automatic destructive replacement."""
    from skyulf.integrations.databricks.serving import sql_functions

    monkeypatch.setattr(sql_functions, "require_pinned_endpoint_ready", Mock())
    spark = SimpleNamespace(sql=Mock(side_effect=failure))
    plan = serving.build_serving_sql_function(endpoint, "main.api.score")
    with pytest.raises(type(failure), match=str(failure)):
        serving.create_serving_sql_function(spark, object(), plan)
    spark.sql.assert_called_once()
