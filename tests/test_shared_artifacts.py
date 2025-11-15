import json
from pathlib import Path

import joblib

from core.feature_engineering.modeling.shared import (
    CrossValidationConfig,
    _build_metadata_update,
    _persist_training_artifact,
    _write_transformer_debug_snapshot,
)


def test_persist_training_artifact_writes_joblib(tmp_path):
    artifact_root = tmp_path / "artifacts"
    artifact_data = {"model": {"coef": [1, 2, 3]}, "version": 7}

    artifact_path = _persist_training_artifact(
        str(artifact_root),
        pipeline_id="pipe-123",
        job_id="job-456",
        version=7,
        artifact_data=artifact_data,
    )

    artifact_file = Path(artifact_path)
    assert artifact_file.exists()
    assert joblib.load(artifact_file) == artifact_data


def test_write_transformer_debug_snapshot_creates_json(tmp_path):
    class DummyScaler:
        pass

    transformers = [
        {
            "node_id": "node-1",
            "transformer_name": "standard",
            "column_name": "feature_a",
            "transformer": DummyScaler(),
            "metadata": {"method": "standard"},
        }
    ]
    transformer_plan = [
        {
            "node_id": "node-1",
            "transformers": [{"transformer_name": "standard", "column_name": "feature_a"}],
        }
    ]

    _write_transformer_debug_snapshot(transformers, transformer_plan, tmp_path)

    snapshot = json.loads((tmp_path / "fitted_params_debug_latest.json").read_text(encoding="utf-8"))
    assert snapshot["transformer_plan"] == transformer_plan
    assert snapshot["transformers_overview"][0]["node_id"] == "node-1"
    assert snapshot["transformers_overview"][0]["metadata"] == {"method": "standard"}


def test_build_metadata_update_includes_dataset():
    cv_config = CrossValidationConfig(
        enabled=True,
        strategy="kfold",
        folds=4,
        shuffle=True,
        random_state=123,
        refit_strategy="train_plus_validation",
    )
    dataset_meta = {"rows": 1000, "source": "unit-test"}

    metadata = _build_metadata_update(
        resolved_problem_type="classification",
        target_column="label",
        feature_columns=["f1", "f2"],
        cv_config=cv_config,
        dataset_meta=dataset_meta,
    )

    assert metadata["resolved_problem_type"] == "classification"
    assert metadata["dataset"] == dataset_meta
    assert metadata["cross_validation"]["folds"] == 4
    assert metadata["cross_validation"]["refit_strategy"] == "train_plus_validation"
