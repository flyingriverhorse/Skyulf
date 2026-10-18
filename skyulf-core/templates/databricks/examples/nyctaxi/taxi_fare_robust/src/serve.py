"""Explicit raw-input endpoint example, called from a notebook; see SERVE_EXAMPLE.md."""

from pathlib import Path

import yaml
from databricks.sdk import WorkspaceClient

from skyulf.integrations.databricks.serving import (
    PinnedEndpointSpec,
    build_serving_sql_function,
    create_pinned_endpoint,
    create_serving_sql_function,
    prepare_pinned_endpoint,
)


def prepare(config_path):
    """Inspect the configured registered package without creating cloud resources."""
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    endpoint = prepare_pinned_endpoint(
        PinnedEndpointSpec(**config["endpoint"]),
        tracking_uri="databricks",
        registry_uri="databricks-uc",
    )
    return build_serving_sql_function(endpoint, config["sql_function"])


def create(spark, config_path):
    """Create one new endpoint and SQL function using the inspected settings."""
    function = prepare(config_path)
    print(function.create_sql)
    client = WorkspaceClient()
    create_pinned_endpoint(client, function.endpoint)
    create_serving_sql_function(spark, client, function)
    print("Ready:", function.endpoint.spec.endpoint_name)
    return function
