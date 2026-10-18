"""Install production shared views in the existing SQLite dashboard fixtures."""

from skyulf.integrations.databricks.observability.monitoring.monitoring_store import (
    monitoring_views,
)


def install_shared_views(connection):
    """Execute the real SQL rules while fixtures supply base tables and scalar functions."""
    views = monitoring_views("ops.monitoring")
    for name in ("complete_observations", "monitoring_identities", "serving_windows"):
        query = views[name].replace("`ops`.`monitoring`.", "")
        connection.execute(f"CREATE VIEW {name} AS {query}")
