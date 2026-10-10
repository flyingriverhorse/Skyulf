"""Weight provenance survives training while scoring and monitoring use clean features."""

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from test_databricks_lifecycle_tasks import _call, staged  # noqa: F401 - shared real-store fixture

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.fitting.candidate import (
    TrainingSpec,
    split_labeled_snapshot,
)
from skyulf.integrations.databricks.training.shared.training_evidence import (
    build_training_evidence,
    evidence_digest,
    validate_training_evidence,
)
from skyulf.integrations.databricks.training.shared.training_parameters import (
    log_training_parameters,
)


def _spec(weighted=True):
    """Use captured source that must never execute during saved-artifact replay."""
    source = "raise RuntimeError('mutable weight hook executed')\n"
    return TrainingSpec(
        table="workspace.test.source",
        version=4,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=100,
        max_bytes=100_000,
        weight_column="training_weight" if weighted else None,
        reserved_weight_columns=("training_weight",) if weighted else (),
        weights_python_source=source if weighted else None,
        weights_python_sha256=hashlib.sha256(source.encode()).hexdigest() if weighted else None,
    )


def _frame():
    """Make row weights distinctive enough to expose changed pairings in replay."""
    return pd.DataFrame(
        {
            "id": range(20),
            "x": range(20),
            "target": [i % 2 for i in range(20)],
            "training_weight": np.arange(20, dtype=float) + 1,
        }
    )


def test_weight_receipt_freezes_summary_and_rejects_changed_pairing():
    """A replay with identical keys but different weights must invalidate its saved receipt."""
    spec = _spec()
    frame = _frame()
    _, heldout, _ = split_labeled_snapshot(frame, spec)
    spec = replace(
        spec,
        holdout_key_sha256=heldout.attrs["holdout_key_sha256"],
        survivor_key_sha256=heldout.attrs["survivor_key_sha256"],
    )
    receipt = build_training_evidence(spec, heldout, project_source_sha256=None)
    assert receipt["training_weights"] == heldout.attrs["training_weights"]
    frozen = deepcopy(receipt)
    heldout.attrs["training_weights"]["sum"] += 1
    assert receipt == frozen
    spec = replace(spec, training_evidence_sha256=evidence_digest(receipt))
    frame["training_weight"] = frame["training_weight"].iloc[::-1].to_numpy()
    _, replayed, _ = split_labeled_snapshot(frame, spec)
    with pytest.raises(ValueError, match="Replayed training filter evidence"):
        validate_training_evidence(receipt, spec, project_source_sha256=None, heldout=replayed)


def test_unweighted_receipt_keeps_original_fields():
    """Inactive weights must not add receipt fields that would invalidate legacy digests."""
    spec = _spec(False)
    _, heldout, _ = split_labeled_snapshot(_frame(), spec)
    receipt = build_training_evidence(spec, heldout, project_source_sha256=None)
    assert "training_weights" not in receipt
    assert receipt["version"] == 1


@pytest.mark.parametrize("class_weight", [None, "balanced"])
def test_saved_weight_summary_and_configured_class_weight_are_logged(tmp_path, class_weight):
    """Non-native class weights must remain visible even when absent from estimator params."""
    spec = _spec()
    train, heldout, _ = split_labeled_snapshot(_frame(), spec)
    summary = heldout.attrs["training_weights"]
    config = {
        "preprocessing": [],
        "modeling": {"type": "gaussian_nb", "params": {"class_weight": class_weight}},
        "training_weights": summary,
    }
    weights = train.pop("training_weight").to_numpy()
    artifact = fit_workflow(
        config,
        SplitDataset(train=train, test=train.head(0), train_sample_weight=weights),
        target_column="target",
        artifact_path=tmp_path / "model",
        max_rows=100,
        max_bytes=100_000,
    )
    assert artifact.manifest.input_columns == ("x",)
    assert artifact.manifest.feature_columns == ("x",)
    assert cast(dict[str, Any], artifact.pipeline.config)["training_weights"] == summary
    assert len(artifact.pipeline.predict(train[["x"]])) == len(train)
    run = Mock()
    log_training_parameters(run, artifact, spec, config)
    logged = run.client.log_dict.call_args.args[1]
    assert logged["configured_class_weight"] == class_weight
    assert logged["training_weights"] == summary
    params = run.log_params.call_args.args[0]
    assert params["training_weights.weight_column"] == "training_weight"
    assert params["training_weights.weights_python_sha256"] == spec.weights_python_sha256
    assert all(
        not isinstance(value, (list, np.ndarray)) for value in logged["training_weights"].values()
    )


def test_template_has_no_separate_weight_hook():
    """New bundles keep the weight declaration beside the selected model settings."""
    template = Path(__file__).resolve().parents[3] / "templates/databricks"
    assert not (template / "template/{{.project_name}}/src/modeling").exists()
    assert "weight_column:" in (template / "library/training_settings.tmpl").read_text()


@pytest.mark.parametrize("layout", ["single_model", "model_competition", "multi_target"])
def test_cli_generates_optional_weight_declaration_for_each_layout(tmp_path, layout):
    """Default initialization omits inactive weights while retaining unweighted runtime behavior."""
    from test_databricks_bundle_generation import CLI, PROFILE, _generate_project

    if not CLI or not PROFILE:
        pytest.skip("Set SKYULF_BUNDLE_CLI_TEST_PROFILE to opt into installed CLI generation.")
    project = _generate_project(tmp_path, training_layout=layout)
    from skyulf.integrations.databricks.projects.yaml_config import read_training_config
    from skyulf.integrations.databricks.projects.yaml_models import model_entries

    document = read_training_config(project / "config")
    assert document is not None
    assert all(entry.get("weight_column") is None for entry in model_entries(document).values())
    assert not (project / "src/modeling/weights.py").exists()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_weighted_lifecycle_preserves_snapshot_and_cleans_monitoring(staged, monkeypatch, engine):
    """Frozen Parquet retains weights while verified drift references and scoring exclude them."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.databricks.observability.monitoring import monitoring_reference
    from skyulf.integrations.databricks.observability.monitoring.monitoring_config import (
        MonitorConfig,
    )
    from skyulf.integrations.databricks.training.fitting import candidate as candidate

    _, client, config, _, frame = staged
    explained_columns = []
    explain = candidate.log_training_explanations

    def record_explanation_inputs(run, artifact, training_frame):
        """Observe the real explanation boundary before its optional SHAP execution."""
        explained_columns.append(list(training_frame.columns))
        return explain(run, artifact, training_frame)

    monkeypatch.setattr(candidate, "log_training_explanations", record_explanation_inputs)
    config["pipeline"]["explainability"] = {
        "method": "shap",
        "max_samples": 4,
        "max_features": 2,
        "max_display_samples": 2,
    }
    captured = _spec()
    config.update(
        engine=engine,
        promotion_policy="manual_approval",
        weight_column=captured.weight_column,
        reserved_weight_columns=list(captured.reserved_weight_columns),
        weights_python_source=captured.weights_python_source,
        weights_python_sha256=captured.weights_python_sha256,
    )
    frame["training_weight"] = np.arange(len(frame), dtype=float) + 1
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    trained = _call(staged, "train", split.reference)
    selected = _call(staged, "select_best_model", trained.reference)
    registered = _call(staged, "evaluate_register", selected.reference)
    run_id = registered.reference["run_id"]
    source_path = client.download_artifacts(run_id, "lifecycle/data/source.parquet")
    train_path = client.download_artifacts(run_id, "lifecycle/data/train.parquet")
    frozen_source = pd.read_parquet(source_path)
    frozen_train = pd.read_parquet(train_path)
    assert frozen_source["training_weight"].tolist() == frame["training_weight"].tolist()
    assert frozen_train["training_weight"].tolist() == (frozen_train["x"] + 1).tolist()
    receipt = json.loads(
        Path(client.download_artifacts(run_id, "training_filter_evidence.json")).read_text()
    )
    assert receipt["training_weights"]["count"] == len(frozen_train)
    assert all(not isinstance(value, list) for value in receipt["training_weights"].values())
    monitor = MonitorConfig(
        environment="test",
        project="weighted_reference",
        model_name=config["model_name"],
        model_version="1",
        source_table=config["training_table"],
        prediction_table="workspace.test.predictions",
    )
    monkeypatch.setattr(
        monitoring_reference, "read_training_snapshot", lambda spark, spec: frame.copy()
    )
    artifact, _, reference, _ = monitoring_reference.load_monitoring_reference(
        None, monitor, tracking_uri=config["tracking_uri"], registry_uri=config["registry_uri"]
    )
    assert (
        cast(dict[str, Any], artifact.pipeline.config)["training_weights"]
        == receipt["training_weights"]
    )
    assert artifact.manifest.input_columns == ("x",)
    assert artifact.manifest.feature_columns == ("x",)
    assert explained_columns == [["x", "target"]]
    assert "training_weight" not in reference.columns
    assert len(artifact.pipeline.predict(reference[["x"]])) == len(reference)
    frame["training_weight"] *= 2
    with pytest.raises(ValueError, match="Replayed training filter evidence"):
        monitoring_reference.load_monitoring_reference(
            None, monitor, tracking_uri=config["tracking_uri"], registry_uri=config["registry_uri"]
        )
