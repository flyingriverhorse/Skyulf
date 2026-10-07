"""Move generated model settings to config/training.yml and config/inference.yml.

Run once from any directory with the project's installed Skyulf environment.
Original declarations are retained in .skyulf-yaml-backup. Custom model factories
require manual migration; Python feature functions stay in src/features.
After migration run smoke.py, preview.py and refresh_training_graph.py, then
validate and redeploy the Bundle. Existing fitted models remain self-contained.
"""

import json
from pathlib import Path

from skyulf.integrations.databricks.projects.yaml_migration import migrate_project_yaml

if __name__ == "__main__":
    print(json.dumps(migrate_project_yaml(Path(__file__).resolve().parents[2]), indent=2))
