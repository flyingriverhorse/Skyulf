"""Generate the optional feature job after editing config/features.yml."""

from pathlib import Path

from skyulf.integrations.databricks.features.graph import refresh_feature_graph

if __name__ == "__main__":
    groups = refresh_feature_graph(Path(__file__).resolve().parents[2])
    print("Feature groups:", ", ".join(groups) or "disabled (ready-table workflow)")
    print("Validate and redeploy the Bundle to apply this graph.")
