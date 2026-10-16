"""Empirical admission needs enough rows reaching each step after earlier exclusions."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.integration.platforms.test_probed_custom_steps import HOLDOUT, _project

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import predict_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.shared.preprocessing_checks import seal_partition_probe


def _artifact(tmp_path, kind="custom", mode="row", second_filter=False):
    """Fit real saved preprocessing with an outlier filter before the empirical step."""
    config: dict[str, Any]
    if kind == "custom":
        config = _project(tmp_path, mode=mode, kind="column")["pipeline"]
    else:
        config = {
            "preprocessing": [
                {"name": "robust", "transformer": "RobustScaler", "params": {"columns": ["x"]}}
            ],
            "modeling": {"type": "decision_tree_regressor", "params": {"random_state": 42}},
        }
    config["admit_probed_custom_steps"] = True
    filters = [
        {
            "name": "bound",
            "transformer": "ManualBounds",
            "params": {"bounds": {"x": {"upper": 10.0}}},
        }
    ]
    if second_filter:
        filters.append(
            {
                "name": "second",
                "transformer": "ManualBounds",
                "params": {"bounds": {"x": {"upper": 3.0}}},
            }
        )
    config["preprocessing"] = filters + config["preprocessing"]
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "target": [3.0, 5.0, 7.0, 9.0]})
    path = tmp_path / "artifact"
    fitted = fit_workflow(
        config,
        SplitDataset(train=train, test=train.head(0)),
        target_column="target",
        artifact_path=path,
        max_rows=100,
        max_bytes=100000,
    )
    return fitted, path


def _holdout(survivors):
    """Keep a twenty-row raw sample while varying the downstream coverage."""
    values = np.concatenate([np.linspace(1.0, 4.0, survivors), np.arange(100.0, 120.0 - survivors)])
    return pd.DataFrame({"x": values, "target": values * 2.0 + 1.0})


def test_empty_downstream_probe_cannot_admit_real_batch_dependent_custom_step(tmp_path):
    """A filter that excludes the sample must not hide nonlocal custom inference."""
    artifact, path = _artifact(tmp_path, mode="batch_mean")
    query = pd.DataFrame({"x": [1.0, 4.0]})
    whole = predict_pipeline(query, artifact).prediction
    chunks = pd.concat(
        [predict_pipeline(query.iloc[i : i + 1], artifact) for i in range(2)]
    ).prediction
    np.testing.assert_allclose(whole.to_numpy(dtype=float), [3.0, 9.0])
    np.testing.assert_allclose(chunks.to_numpy(dtype=float), [4.5, 7.5])
    with pytest.raises(ValueError, match="actual input rows"):
        seal_partition_probe(artifact, HOLDOUT, path, enabled=True)


@pytest.mark.parametrize("kind", ["custom", "builtin"])
@pytest.mark.parametrize("survivors", [0, 10, 15])
def test_admission_rejects_too_few_rows_entering_each_step(tmp_path, kind, survivors):
    """Raw sample size cannot substitute for actual observations of a downstream step."""
    artifact, path = _artifact(tmp_path, kind)
    with pytest.raises(ValueError, match="actual input rows"):
        seal_partition_probe(artifact, _holdout(survivors), path, enabled=True)


@pytest.mark.parametrize("kind", ["custom", "builtin"])
def test_diagnostic_and_sealed_entry_record_actual_entering_rows(tmp_path, kind):
    """Sixteen surviving observations remain sufficient and are bound into evidence."""
    artifact, path = _artifact(tmp_path, kind)
    holdout = _holdout(16)
    report = probe_fitted_preprocessing(artifact, holdout[["x"]])
    assert report["sample_rows"] == 20
    assert [step["input_rows"] for step in report["steps"]] == [20, 16]
    sealed = seal_partition_probe(artifact, holdout, path, enabled=True)
    evidence = sealed.pipeline.config.get("partition_probe", {})
    assert evidence["rows"] == 20
    assert evidence["pipeline"]["scored_rows"] == 16
    entry = evidence["steps"][-1] if kind == "custom" else evidence["builtins"][-1]
    assert entry["input_rows"] == 16
    assert require_partition_safe_pipeline(sealed).steps


@pytest.mark.parametrize("kind", ["custom", "builtin"])
@pytest.mark.parametrize("count", [None, True, 0, 10, 15, 21, 257, "16"])
def test_worker_rejects_missing_or_invalid_per_step_row_count(tmp_path, kind, count):
    """Saved evidence must retain a strict bounded count for the actual step sample."""
    artifact, path = _artifact(tmp_path, kind)
    sealed = seal_partition_probe(artifact, _holdout(16), path, enabled=True)
    evidence = sealed.pipeline.config.get("partition_probe", {})
    entry = evidence["steps"][-1] if kind == "custom" else evidence["builtins"][-1]
    if count is None:
        entry.pop("input_rows", None)
    else:
        entry["input_rows"] = count
    with pytest.raises(UnsupportedExecutionError, match="actual input rows"):
        require_partition_safe_pipeline(sealed)


def test_multiple_filters_measure_rows_at_each_actual_boundary(tmp_path):
    """Coverage must track every exclusion, not just the first outlier in the pipeline."""
    artifact, path = _artifact(tmp_path, second_filter=True)
    frame = pd.DataFrame({"x": [1.0] * 10 + [4.0] * 10 + [100.0] * 12})
    with pytest.raises(ValueError, match="actual input rows"):
        seal_partition_probe(artifact, frame, path, enabled=True)
