"""Typed Unity Catalog SQL wrappers for admitted custom-model serving endpoints."""

from dataclasses import dataclass
from typing import Any

from ...mlflow.shared._model_metadata import normalized_dtype
from ...mlflow.shared._nullable_transport import transport_spec
from .contracts import PinnedEndpointPlan, is_uc_identifier
from .endpoints import require_pinned_endpoint_ready

_SQL_TYPES = {
    "int32": "INT",
    "int64": "BIGINT",
    "float32": "FLOAT",
    "float64": "DOUBLE",
    "bool": "BOOLEAN",
    "string": "STRING",
}


def _identifier(value: str, label: str) -> str:
    """Quote a supported identifier without accepting SQL expressions or fragments."""
    if not is_uc_identifier(value):
        raise ValueError(f"{label} must be a simple SQL identifier (letters, digits, underscore).")
    return f"`{value}`"


def _function_name(value: str) -> str:
    """Require a fully qualified UC function independent of session defaults."""
    if not isinstance(value, str) or len(value.split(".")) != 3:
        raise ValueError("function_name must be a concrete catalog.schema.function UC name.")
    return ".".join(_identifier(part, "function_name") for part in value.split("."))


def _sql_schema(schema: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    """Validate scalar fields before embedding them in DDL and response type literals."""
    if not schema or len({name.casefold() for name, _ in schema}) != len(schema):
        raise ValueError("SQL schema must contain nonempty, case-insensitively unique fields.")
    fields = []
    for name, dtype in schema:
        quoted = _identifier(name, "schema field identifier")
        sql_type = _SQL_TYPES.get(normalized_dtype(dtype))
        if sql_type is None:
            raise ValueError(f"SQL serving cannot preserve schema dtype {dtype!r}.")
        fields.append((quoted, sql_type))
    return tuple(fields)


def _struct_type(fields: tuple[tuple[str, str], ...]) -> str:
    """Render the complete named pyfunc output, including model-set/scoring fields."""
    return "STRUCT<" + ", ".join(f"{name}: {dtype}" for name, dtype in fields) + ">"


def _request_fields(endpoint: PinnedEndpointPlan) -> str:
    """Preserve missing floats and exact nullable integer/boolean transport."""
    transport = transport_spec(endpoint.input_schema)
    encoded = transport["columns"] if transport else {}
    fields = []
    for name, dtype in endpoint.input_schema:
        expression = _request_expression(name, dtype, name in encoded)
        fields.append(f"    '{name}', {expression}")
    return ",\n".join(fields)


def _request_expression(name: str, dtype: str, encoded: bool) -> str:
    """Keep ai_query numeric null conversion from bypassing fitted imputation."""
    if encoded:
        return f"CAST(`{name}` AS STRING)"
    normalized = normalized_dtype(dtype)
    if normalized in {"float32", "float64"}:
        sql_type = _SQL_TYPES[normalized]
        return f"COALESCE(`{name}`, CAST('NaN' AS {sql_type}))"
    return f"`{name}`"


@dataclass(frozen=True, slots=True)
class ServingSQLFunctionPlan:
    """A create-only SQL wrapper using one inspected endpoint input/output schema.

    Use ``build_serving_sql_function`` to prepare a plan. Model version pinning
    is checked when deploying; the SQL function itself calls an endpoint name.
    Administrators must keep that release endpoint on the selected version.
    """

    endpoint: PinnedEndpointPlan
    function_name: str
    fail_on_error: bool = True

    def __post_init__(self) -> None:
        """Reject incomplete schemas and ambiguous configuration before rendering SQL."""
        _function_name(self.function_name)
        _sql_schema(self.endpoint.input_schema)
        _sql_schema(self.endpoint.output_schema)
        if self.endpoint.input_columns != tuple(name for name, _ in self.endpoint.input_schema):
            raise ValueError("input_columns differ from the inspected input schema.")
        if not isinstance(self.fail_on_error, bool):
            raise TypeError("fail_on_error must be a boolean.")

    @property
    def response_type(self) -> str:
        """Return the parsed per-row prediction type, without the HTTP predictions array."""
        return _struct_type(_sql_schema(self.endpoint.output_schema))

    @property
    def return_type(self) -> str:
        """Include the platform's error envelope only when explicitly requested."""
        if self.fail_on_error:
            return self.response_type
        return f"STRUCT<response: {self.response_type}, errorMessage: STRING>"

    @property
    def create_sql(self) -> str:
        """Render DDL which fails if a function already exists; never replace or reuse it."""
        parameters = ",\n".join(
            f"  {name} {dtype}" for name, dtype in _sql_schema(self.endpoint.input_schema)
        )
        spec = self.endpoint.spec
        error_mode = str(self.fail_on_error).lower()
        return (
            f"CREATE FUNCTION {_function_name(self.function_name)}(\n{parameters}\n)\n"
            f"RETURNS {self.return_type}\nLANGUAGE SQL\nNOT DETERMINISTIC\n"
            f"COMMENT 'Skyulf model {spec.model_uri}; endpoint {spec.endpoint_name}'\n"
            f"RETURN ai_query(\n  endpoint => '{spec.endpoint_name}',\n"
            f"  request => named_struct(\n{_request_fields(self.endpoint)}\n  ),\n"
            f"  returnType => '{self.response_type}',\n  failOnError => {error_mode}\n)"
        )

    def call_sql(self, *, table_alias: str | None = None) -> str:
        """Render a named-argument call against matching model-input column names.

        ``table_alias`` is an optional simple SQL alias, not a table expression.
        The caller owns the surrounding SELECT, source selection and any writes.
        """
        prefix = "" if table_alias is None else _identifier(table_alias, "table_alias") + "."
        arguments = ", ".join(
            f"`{name}` => {prefix}`{name}`" for name in self.endpoint.input_columns
        )
        return f"{_function_name(self.function_name)}({arguments})"


def build_serving_sql_function(
    endpoint: PinnedEndpointPlan,
    function_name: str,
    *,
    fail_on_error: bool = True,
) -> ServingSQLFunctionPlan:
    """Build a typed SQL function from a prepared pinned endpoint, without any writes.

    Requires named scalar schemas; supported field names use letters, digits
    and underscores. The function accepts fitted input types and encodes nullable
    integer/boolean wire columns. No preprocessing is fitted or moved into SQL.
    With ``fail_on_error=False``, endpoint errors appear as ``errorMessage``;
    SQL, permission and other non-endpoint failures can still fail the query.
    """
    return ServingSQLFunctionPlan(endpoint, function_name, fail_on_error)


def create_serving_sql_function(spark: Any, client: Any, plan: ServingSQLFunctionPlan) -> None:
    """Create the function after checking the exact endpoint version and readiness.

    Pass a Databricks Spark session and SDK client for the same workspace.
    Creation requires UC USE CATALOG, USE SCHEMA, CREATE FUNCTION and endpoint
    CAN QUERY. The SDK readiness check additionally requires endpoint visibility.
    Existing functions, SQL errors and permission failures propagate unchanged;
    no grants, retries, endpoint mutations or prediction-table writes occur.
    The check is point-in-time and cannot lock the endpoint against later edits.
    """
    require_pinned_endpoint_ready(client, plan.endpoint)
    spark.sql(plan.create_sql).collect()
