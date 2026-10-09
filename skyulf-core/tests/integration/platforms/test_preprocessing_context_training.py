"""Saved models retain real training-only skips and inspection passthroughs after reload."""

import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline.seal import artifact_digest
from skyulf.registry import NodeRegistry

MARKERS = {"TrainTestSplitter", "Split", "feature_target_split"}
SKIPPED = {"DropMissingRows", "Oversampling", "Undersampling"}
RECIPES = {
    "DropMissingRows": {"subset": ["x"], "how": "any"},
    "Oversampling": {"method": "random_over", "random_state": 42},
    "Undersampling": {"method": "random_under_sampling", "random_state": 42},
    "DatasetProfile": {},
    "DataSnapshot": {"n_rows": 2},
    "TrainTestSplitter": {"test_size": 0.25, "random_state": 42, "target_column": "target"},
    "Split": {"test_size": 0.25, "random_state": 42, "target_column": "target"},
    "feature_target_split": {"target_column": "target"},
}


def _save_models(directory, engine):
    """Exercise each actual training path before recording predictions and fitted state."""
    requests = {}
    for node, params in RECIPES.items():
        training = pd.DataFrame({"x": np.arange(32, dtype=float), "target": [0, 0, 0, 1] * 8})
        sample = pd.DataFrame({"x": [1.0, 3.0, 9.0]})
        steps = [{"name": "reviewed", "transformer": node, "params": params}]
        if node == "DropMissingRows":
            training.loc[10, "x"], sample.loc[1, "x"] = np.nan, np.nan
            steps.append(
                {
                    "name": "fill_prediction_nulls",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["x"], "strategy": "mean"},
                }
            )
        if engine == "polars":
            training, sample = pl.from_pandas(training), pl.from_pandas(sample)
        pipeline = SkyulfPipeline(
            {
                "preprocessing": steps,
                "modeling": {
                    "type": "random_forest_classifier",
                    "params": {"n_estimators": 4, "max_depth": 3, "n_jobs": 1, "random_state": 42},
                },
            }
        )
        data = (
            training
            if node in {"TrainTestSplitter", "Split"}
            else SplitDataset(train=training[8:], test=training[:8])
        )
        pipeline.fit(data, target_column="target")
        records = pipeline.feature_engineer.fitted_steps
        assert not records if node in MARKERS else records[0]["type"] == node
        expected = pipeline.predict(sample)
        digest = artifact_digest(records)
        save_local_pipeline(pipeline, directory / node)
        requests[node] = (sample, expected, digest)
    (directory / "requests.pkl").write_bytes(pickle.dumps(requests))


def _reload_models(directory):
    """Fail if any saved model retrains, resamples, filters or splits prediction requests."""

    def forbidden(*args, **kwargs):
        """Make any training-only work during load or inference immediately observable."""
        raise AssertionError("Training-only execution during prediction")

    for node in NodeRegistry.list_transformers():
        calculator: Any = NodeRegistry.get_calculator(node)
        calculator.fit = forbidden
    for node in MARKERS | SKIPPED:
        applier: Any = NodeRegistry.get_applier(node)
        applier.apply = forbidden
    statuses = {}
    for node, (sample, expected, digest) in pickle.loads(
        (directory / "requests.pkl").read_bytes()
    ).items():
        artifact = load_local_pipeline(directory / node)
        actual = artifact.pipeline.predict(sample)
        np.testing.assert_array_equal(actual, expected)
        assert len(actual) == len(sample)
        report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 2))
        assert report["status"] == "passed" and report["feature_schema"] == "passed", report
        assert not {step["node_type"] for step in report["steps"]}.intersection(MARKERS)
        if node in MARKERS:
            assert report["steps"] == []
        else:
            first = report["steps"][0]
            assert first["node_type"] == node
            if node in SKIPPED:
                assert first["status"] == "skipped" and first["action"] == "skip_preserve_rows"
                assert first["checks"] == [] and first["state_validation"] == "unavailable"
                assert first["context"] == ("row" if node == "DropMissingRows" else "global")
                assert first["row_effect"] == ("expand" if node == "Oversampling" else "filter")
            else:
                assert first["status"] == "passed" and first["context"] == "row"
                assert first["state_validation"] == "node_owned"
        chunks = [
            sample.iloc[i : i + 1] if isinstance(sample, pd.DataFrame) else sample.slice(i, 1)
            for i in range(len(sample))
        ]
        np.testing.assert_array_equal(
            actual, np.concatenate([artifact.pipeline.predict(chunk) for chunk in chunks])
        )
        assert artifact_digest(artifact.pipeline.feature_engineer.fitted_steps) == digest
        statuses[node] = report["status"]
    return statuses


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_training_only_contexts_preserve_reloaded_model_predictions(tmp_path, engine):
    """Saved inference keeps every requested row and never invokes discarded training steps."""
    pytest.importorskip("imblearn")
    _save_models(tmp_path, engine)
    code = "import json,runpy,sys; from pathlib import Path; m=runpy.run_path(sys.argv[1]); print(json.dumps(m['_reload_models'](Path(sys.argv[2]))))"
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == dict.fromkeys(RECIPES, "passed")
