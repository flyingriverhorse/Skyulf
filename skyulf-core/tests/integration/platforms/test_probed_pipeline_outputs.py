"""Whole-pipeline admission must expose preprocessing noise amplified by the model."""

import pandas as pd
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.pipeline_scoring import score_pipeline
from skyulf.integrations.databricks.projects.project import load_project_workflow
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.shared.preprocessing_checks import seal_partition_probe


def _noise_artifact(tmp_path, scale, business=False, eligibility=False):
    """Fit a real decision boundary after a custom transform with tiny batch-dependent noise."""
    root = tmp_path / "features"
    root.mkdir()
    (root / "__init__.py").write_text(
        "from skyulf.preprocessing import column_step\n"
        "def tiny_noise(df):\n"
        f"    return df['x'] + {scale!r} * getattr(df['x'], 'mean')()\n"
        "def build_preprocessing(recipe='default'):\n"
        "    return [column_step('noise', tiny_noise, output='x', replace=True, inference_context='row')]\n"
        f"# {tmp_path.name}\n",
        encoding="utf-8",
    )
    if business:
        source = root / "__init__.py"
        source.write_text(
            source.read_text() + "import pandas as pd\n"
            "def flag(frame, predictions, params):\n"
            "    return (predictions['prediction'] > 0).astype('int64').to_frame('flag')\n"
            "def build_scoring():\n"
            "    return {'eligibility': [], 'outputs': [{'name': 'flag', 'version': '1', "
            "'function': 'flag', 'params': {}, 'columns': [{'name': 'flag', 'dtype': 'int64'}]}]}\n",
            encoding="utf-8",
        )
    if eligibility:
        source = root / "__init__.py"
        source.write_text(
            source.read_text().replace(
                "'eligibility': []",
                "'eligibility': [{'name': 'eligible', 'version': '1', "
                "'function': 'eligible', 'params': {}}]",
            )
            + "def eligible(frame, params):\n"
            "    return pd.Series('too_large', index=frame.index).where(frame['x'] > 10)\n",
            encoding="utf-8",
        )
    config = load_project_workflow(
        {
            "pipeline": {
                "preprocessing": [],
                "modeling": {
                    "type": "decision_tree_classifier",
                    "params": {"max_depth": 1, "random_state": 42},
                },
            }
        },
        root,
    )["pipeline"]
    if business:
        config["modeling"] = {"type": "linear_regression"}
    config["admit_probed_custom_steps"] = True
    train = pd.DataFrame({"x": [-1.0, 1.0] * 16, "target": [0, 1] * 16})
    if business:
        train["target"] = train["x"]
    path = tmp_path / "artifact"
    artifact = fit_workflow(
        config,
        SplitDataset(train=train, test=train.head(0)),
        target_column="target",
        artifact_path=path,
        max_rows=100,
        max_bytes=100000,
    )
    holdout = pd.DataFrame({"x": [-2.5e-14, 1.0] * 8})
    return artifact, holdout, path


def test_small_preprocessing_noise_cannot_hide_downstream_label_flip(tmp_path):
    """Stage-local float agreement is insufficient when a tree amplifies the tolerated noise."""
    artifact, holdout, path = _noise_artifact(tmp_path, 1e-13)
    whole = predict_pipeline(holdout, artifact)
    singleton = pd.concat([predict_pipeline(holdout.iloc[i : i + 1], artifact) for i in range(16)])
    assert whole.prediction.tolist() != singleton.prediction.tolist()
    with pytest.raises(
        ValueError, match="Pipeline partition probe failed chunks:1: output_mismatch"
    ):
        seal_partition_probe(artifact, holdout, path, enabled=True)


def test_reviewed_model_with_custom_step_seals_raw_pipeline_checks(tmp_path):
    """Reviewed estimators still need final raw-input parity when preprocessing is empirical."""
    artifact, holdout, path = _noise_artifact(tmp_path, 0.0)
    sealed = seal_partition_probe(artifact, holdout, path, enabled=True)
    evidence = sealed.pipeline.config.get("partition_probe", {})
    assert "model" not in evidence
    assert evidence["pipeline"] == {
        "rows": 16,
        "scored_rows": 16,
        "checks": [
            {"name": name, "status": "passed"}
            for name in ("full", "repeat", "chunks:1", "chunks:7")
        ],
    }
    assert require_partition_safe_pipeline(sealed).steps[0].action == "apply_probed"


@pytest.fixture(params=["custom", "builtin", "model"])
def sealed_path(tmp_path, request):
    """Exercise all worker-envelope routes using real saved and sealed artifacts."""
    if request.param == "custom":
        artifact, holdout, path = _noise_artifact(tmp_path, 0.0)
    elif request.param == "builtin":
        from tests.integration.platforms.test_probed_builtin_steps import _alias_artifact

        artifact, holdout, path = _alias_artifact(tmp_path)
    else:
        from tests.integration.platforms.test_probed_models import _artifact

        artifact, holdout, path = _artifact(tmp_path)
    seal_partition_probe(artifact, holdout, path, enabled=True)
    return path


@pytest.mark.parametrize(
    "tamper",
    [
        "missing",
        "check",
        "rows",
        "row_type",
        "scored_missing",
        "scored_type",
        "scored_small",
        "scored_large",
    ],
)
def test_workers_require_complete_raw_pipeline_observations(sealed_path, tamper):
    """Component evidence cannot authorize a pipeline when final raw-input parity is missing."""
    artifact = load_pipeline(sealed_path)
    evidence = artifact.pipeline.config.get("partition_probe", {})
    assert "scored_rows" in evidence["pipeline"]
    if tamper == "missing":
        evidence.pop("pipeline", None)
    else:
        detail = evidence.setdefault("pipeline", {"rows": evidence["rows"], "checks": []})
        if tamper == "scored_missing":
            detail.pop("scored_rows")
        elif tamper.startswith("scored_"):
            detail["scored_rows"] = {
                "scored_type": True,
                "scored_small": 15,
                "scored_large": evidence["rows"] + 1,
            }[tamper]
        elif tamper == "check":
            detail["checks"] = [{"name": "full", "status": "passed"}]
        else:
            detail["rows"] = True if tamper == "row_type" else evidence["rows"] - 1
    with pytest.raises(UnsupportedExecutionError, match="pipeline partition probe"):
        require_partition_safe_pipeline(artifact)


def test_project_output_label_amplification_is_checked_end_to_end(tmp_path):
    """Business labels derived from tolerated regression noise must still match exactly."""
    artifact, holdout, path = _noise_artifact(tmp_path, 1e-13, business=True)
    whole = score_pipeline(holdout, artifact)
    singleton = pd.concat([score_pipeline(holdout.iloc[i : i + 1], artifact) for i in range(16)])
    assert whole.flag.tolist() == [1] * 16
    assert singleton.flag.tolist() == [0, 1] * 8
    with pytest.raises(
        ValueError, match="Pipeline partition probe failed chunks:1: output_mismatch"
    ):
        seal_partition_probe(artifact, holdout, path, enabled=True)


@pytest.mark.parametrize("mode", ["input_mutation", "state_mutation"])
def test_raw_pipeline_probe_guards_inputs_and_retained_state(tmp_path, monkeypatch, mode):
    """Composition-level mutation must not hide behind successful per-component probes."""
    from skyulf.inference import probed_models

    artifact, holdout, _ = _noise_artifact(tmp_path, 0.0)
    original = probed_models.score_pipeline

    def mutate(frame, current):
        """Make a tiny input change or retained configuration change after real scoring."""
        result = original(frame, current)
        if mode == "input_mutation":
            frame.iloc[0, 0] += 1e-13
        else:
            current.pipeline.config["mutation"] = 1
        return result

    monkeypatch.setattr(probed_models, "score_pipeline", mutate)
    with pytest.raises(ValueError, match=mode):
        probed_models.probe_pipeline(artifact, holdout)


@pytest.mark.parametrize("predicted", [0, 10, 15])
def test_final_pipeline_sample_needs_actual_predicted_outcomes(tmp_path, predicted):
    """Later eligible rule samples must not make an all-excluded raw pipeline probe sufficient."""
    scale = 1e-13 if predicted == 0 else 0.0
    artifact, eligible, path = _noise_artifact(tmp_path, scale, business=True, eligibility=True)
    holdout = pd.concat(
        [eligible.head(predicted), pd.DataFrame({"x": [100.0] * (256 - predicted)}), eligible],
        ignore_index=True,
    )
    whole = score_pipeline(eligible, artifact)
    singleton = pd.concat([score_pipeline(eligible.iloc[i : i + 1], artifact) for i in range(16)])
    assert whole.flag.tolist() == ([1] * 16 if predicted == 0 else [0, 1] * 8)
    assert singleton.flag.tolist() == [0, 1] * 8
    with pytest.raises(ValueError, match="actually predicted outcomes"):
        seal_partition_probe(artifact, holdout, path, enabled=True)
