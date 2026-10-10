"""Public integration imports must not depend on another caller's import order."""

import subprocess
import sys


def test_promotion_can_import_before_branch_orchestration():
    """Standalone approval tools must load without first importing the Databricks package."""
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from skyulf.integrations.mlflow.lifecycle.promotion import controlled_champion_version; "
                "from skyulf.integrations.databricks import TrainingBranch, train_branches; "
                "assert callable(controlled_champion_version) and callable(train_branches); "
                "assert TrainingBranch.__name__ == 'TrainingBranch'"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
