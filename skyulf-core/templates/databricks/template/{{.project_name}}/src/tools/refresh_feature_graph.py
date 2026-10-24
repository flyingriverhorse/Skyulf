"""Generate the optional feature job after editing config/features.yml."""

from pathlib import Path

from skyulf.integrations.databricks.features.graph import (
    feature_targets_snippet,
    missing_feature_controls,
    refresh_feature_graph,
    stale_feature_targets,
)

if __name__ == "__main__":
    project = Path(__file__).resolve().parents[2]
    groups = refresh_feature_graph(project)
    if (project / "resources/features.job.yml").is_file():
        print("Feature groups:", ", ".join(groups) or "none (base_transform only)")
    else:
        print("Feature job: disabled (ready-table workflow)")
    missing = missing_feature_controls(project)
    if missing:
        print("\nAdd the feature job's run_as/permissions to deployment/targets.yml,")
        print("merging these keys into the existing targets:\n")
        print(feature_targets_snippet(missing))
    for name in stale_feature_targets(project):
        print(f"Remove targets.{name}.resources.jobs.features from deployment/targets.yml.")
    print("Validate and redeploy the Bundle to apply this graph.")
