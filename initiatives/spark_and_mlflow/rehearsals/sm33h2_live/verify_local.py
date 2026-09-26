"""Exercise the live harness lifecycle against local MLflow, never a cloud API.

Only Delta snapshot reads are substituted with the deterministic fixture. The
registered model artifacts, evaluation, approval, rejection and rollback are real.
Run this after the H2 runtime is stable; cloud score writes remain unverified here.
"""

import json
import runpy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import mlflow
import pandas as pd

from skyulf.integrations.databricks import local_retraining, local_workflow


def main():
    """Check both engines using isolated SQLite registries and local artifact stores."""
    path = Path(__file__).with_name("rehearsal.py")
    harness = runpy.run_path(str(path), run_name="h2_rehearsal_local_check")
    namespace = harness["train_engine"].__globals__
    original_config = namespace["config"]
    original_client = mlflow.MlflowClient
    # Retain inspectable evidence; MLflow keeps SQLite connections open on Windows.
    with TemporaryDirectory(
        prefix="h2-harness-", dir=Path.cwd() / ".cache", delete=False
    ) as directory:
        root = Path(directory)
        uri = f"sqlite:///{(root / 'registry.db').as_posix()}"
        client = original_client(tracking_uri=uri, registry_uri=uri)
        client.create_experiment("h2-harness", artifact_location=(root / "artifacts").as_uri())
        namespace["EXPERIMENT"] = "h2-harness"

        def local_config(engine, **kwargs):
            """Bind all harness model operations to the isolated local registry."""
            values = original_config(engine, **kwargs)
            values.update(tracking_uri=uri, registry_uri=uri, model_name=f"h2_harness_{engine}")
            return values

        def local_client(*args, **kwargs):
            """Replace the harness's explicit Databricks client with this local client."""
            return original_client(tracking_uri=uri, registry_uri=uri)

        namespace["config"] = local_config
        namespace["mlflow"] = SimpleNamespace(MlflowClient=local_client)
        frame = pd.DataFrame(
            namespace["rows"](0, 240), columns=["record_id", "x", "z", "age", "target"]
        )
        results = {}
        with (
            patch.object(local_retraining, "read_training_snapshot", return_value=frame),
            patch.object(local_workflow, "read_training_snapshot", return_value=frame),
        ):
            for engine in ("pandas", "polars"):
                result = harness["train_engine"](None, engine)
                assert len(result["candidates"]) == 3
                assert result["rollback"]["new_version"] == "1"
                results[engine] = {
                    "versions": [candidate["version"] for candidate in result["candidates"]],
                    "rollback_version": result["rollback"]["new_version"],
                }
        print(
            json.dumps(
                {
                    "local_lifecycle": results,
                    "cloud_execution": False,
                    "local_evidence": str(root),
                }
            )
        )


if __name__ == "__main__":
    main()
