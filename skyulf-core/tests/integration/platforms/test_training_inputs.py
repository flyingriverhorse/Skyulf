"""Training input roles remain readable through MLflow's bounded input tags."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")

from skyulf.integrations.databricks.training.shared.training_inputs import log_training_inputs


@pytest.mark.parametrize(
    "columns,inline",
    [
        (("customer_feature_00",), True),
        (("x" * 496,), True),
        (("x" * 497,), False),
        (tuple(f"customer_feature_{index:02}" for index in range(30)), False),
    ],
    ids=["short", "500-characters", "501-characters", "wide"],
)
def test_real_mlflow_training_inputs_keep_complete_feature_roles(tmp_path, columns, inline):
    """Wide feature names must log successfully with verifiable, complete role evidence."""
    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
    experiment = client.create_experiment(
        "input-roles", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    run_id = client.create_run(experiment).info.run_id
    frame = pd.DataFrame({column: [1.0, 2.0] for column in columns} | {"target": [1.0, 2.0]})
    fitted = SimpleNamespace(
        spec=SimpleNamespace(
            table="main.data.training",
            version=0,
            dataset_id="input-role-proof",
            input_columns=columns,
            target_column="target",
            weight_column=None,
        ),
        source_frame=frame,
        training_frame=frame,
        holdout=frame,
        evidence={},
    )
    run = SimpleNamespace(
        client=client,
        run_id=run_id,
        log_inputs=lambda inputs: client.log_inputs(run_id, inputs),
    )

    log_training_inputs(run, fitted, {"input_columns": list(columns)}, {})

    inputs = client.get_run(run_id).inputs.dataset_inputs
    assert len(inputs) == 2
    provenance = json.loads(
        Path(client.download_artifacts(run_id, "dataset_inputs.json")).read_text()
    )
    serialized = json.dumps(list(columns))
    for item in inputs:
        tags = {tag.key: tag.value for tag in item.tags}
        assert all(len(value) <= 500 for value in tags.values())
        assert {column["name"] for column in json.loads(item.dataset.schema)["mlflow_colspec"]} == {
            *columns,
            "target",
        }
        if inline:
            assert json.loads(tags["skyulf.feature_columns"]) == list(columns)
        else:
            assert tags["skyulf.feature_columns_count"] == str(len(columns))
            assert tags["skyulf.feature_columns_artifact"] == "dataset_inputs.json#/feature_columns"
            assert (
                tags["skyulf.feature_columns_sha256"]
                == hashlib.sha256(serialized.encode()).hexdigest()
            )
            assert provenance["feature_columns"] == list(columns)
