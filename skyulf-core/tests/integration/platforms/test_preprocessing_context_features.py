"""Replay real feature, selection, detector and text models after disabling all fit calls."""

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

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline.seal import artifact_digest
from skyulf.registry import NodeRegistry

RECIPES = {
    "FeatureGenerationNode": {
        "operations": [
            {
                "operation_type": "group_agg",
                "method": "mean",
                "input_columns": ["group"],
                "secondary_columns": ["x"],
                "output_column": "group_mean",
            }
        ]
    },
    "ModelBasedSelection": {
        "columns": ["x", "constant"],
        "method": "select_from_model",
        "estimator": "linear_regression",
        "problem_type": "regression",
    },
    "feature_selection": {"columns": ["x", "constant"], "method": "variance"},
    "IQR": {"columns": ["x"], "multiplier": 1.5},
    "ZScore": {"columns": ["x"], "threshold": 3.0},
    "EllipticEnvelope": {"columns": ["x"], "contamination": 0.1, "random_state": 42},
    "count_vectorizer": {"columns": ["text"], "drop_original": True},
    "hashing_vectorizer": {
        "columns": ["text"],
        "n_features": 8,
        "norm": "none",
        "alternate_sign": False,
        "drop_original": True,
    },
    "tfidf_vectorizer": {"columns": ["text"], "drop_original": True},
    "tokenizer": {"columns": ["text"], "drop_original": True, "add_token_count": True},
    "H3Index": {"lat_col": "lat", "lon_col": "lon", "resolution": 9},
}
DETECTORS = {"IQR", "ZScore", "EllipticEnvelope"}


def _frames(node):
    """Keep every tested model fitted on nonempty real preprocessing artifacts."""
    training = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 32)})
    sample = pd.DataFrame({"x": [-0.3, 0.0, 0.3]})
    if node in {"count_vectorizer", "hashing_vectorizer", "tfidf_vectorizer", "tokenizer"}:
        training = pd.DataFrame({"text": ["red blue", "blue green", "green red", "red red"] * 8})
        sample = pd.DataFrame({"text": ["red unseen", None, "blue"]})
    elif node == "H3Index":
        training = pd.DataFrame(
            {"lat": [40.6, 51.5, 48.8, 41.0] * 8, "lon": [-73.7, -0.1, 2.3, 29.0] * 8}
        )
        sample = pd.DataFrame({"lat": [40.6, 51.5, 50.0], "lon": [-73.7, -0.1, 8.0]})
    elif node in {"ModelBasedSelection", "feature_selection"}:
        training["constant"], sample["constant"] = 1.0, 1.0
    elif node == "FeatureGenerationNode":
        training["group"], sample["group"] = [1.0, 2.0] * 16, [1.0, 2.0, 1.0]
    training["target"] = np.linspace(0.0, 3.0, len(training))
    return training, sample


def _save(directory, engine, nodes):
    """Save fitted predictions and state digests before creating an isolated process."""
    requests = {}
    for node in nodes:
        training, sample = _frames(node)
        if engine == "polars":
            training, sample = pl.from_pandas(training), pl.from_pandas(sample)
        steps = [{"name": "reviewed", "transformer": node, "params": RECIPES[node]}]
        if node == "tokenizer":
            steps.append(
                {
                    "name": "tokens_to_numbers",
                    "transformer": "LabelEncoder",
                    "params": {"columns": ["text__tokens"]},
                }
            )
        if node == "H3Index":
            steps.append(
                {
                    "name": "cells_to_numbers",
                    "transformer": "LabelEncoder",
                    "params": {"columns": ["h3_index"]},
                }
            )
        pipeline = SkyulfPipeline(
            {
                "preprocessing": steps,
                "modeling": {
                    "type": "random_forest_regressor",
                    "params": {"n_estimators": 4, "max_depth": 3, "n_jobs": 1, "random_state": 42},
                },
            }
        )
        pipeline.fit(SplitDataset(train=training[4:], test=training[:4]), target_column="target")
        assert pipeline.feature_engineer.fitted_steps[0]["artifact"], node
        predictions = pipeline.predict(sample)
        digest = artifact_digest(pipeline.feature_engineer.fitted_steps)
        save_local_pipeline(pipeline, directory / node)
        requests[node] = (sample, predictions, digest)
    (directory / "requests.pkl").write_bytes(pickle.dumps(requests))


def _replay(directory):
    """Reuse learned transformations and reject filtering requests after genuine reload."""

    def forbidden(*args, **kwargs):
        """Any preprocessing fit during loading or scoring breaks the saved-state contract."""
        raise AssertionError("Unexpected fit")

    for node in NodeRegistry.list_transformers():
        calculator: Any = NodeRegistry.get_calculator(node)
        calculator.fit = forbidden
    statuses = {}
    for node, (sample, expected, digest) in pickle.loads(
        (directory / "requests.pkl").read_bytes()
    ).items():
        artifact = load_local_pipeline(directory / node)
        actual = artifact.pipeline.predict(sample)
        np.testing.assert_array_equal(actual, expected)
        assert np.isfinite(actual).all() and len(actual) == len(sample)
        report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 2))
        step = report["steps"][0]
        assert step["state_validation"] == "node_owned" and step["context"] == "row", report
        assert step["row_effect"] == ("filter" if node in DETECTORS else "preserve"), report
        expected_checks = [
            {"name": name, "status": "passed"}
            for name in ("full", "repeat", "chunks:1", "chunks:2", "reverse", "empty")
        ]
        assert step["checks"] == expected_checks, report
        assert report["status"] == "passed", report
        assert report["feature_schema"] == "passed", report
        if node in DETECTORS:
            outlier = pd.DataFrame({"x": [100.0]})
            if isinstance(sample, pl.DataFrame):
                outlier = pl.from_pandas(outlier)
            with pytest.raises(ValueError, match="row"):
                artifact.pipeline.predict(outlier)
        chunks = [
            sample.iloc[i : i + 1] if isinstance(sample, pd.DataFrame) else sample.slice(i, 1)
            for i in range(len(sample))
        ]
        np.testing.assert_array_equal(
            actual, np.concatenate([artifact.pipeline.predict(c) for c in chunks])
        )
        assert artifact_digest(artifact.pipeline.feature_engineer.fitted_steps) == digest
        with pytest.raises(UnsupportedExecutionError):
            require_partition_safe_pipeline(artifact)
        statuses[node] = step["status"]
    return statuses


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("geo", [False, True])
def test_feature_owners_preserve_saved_model_predictions(tmp_path, engine, geo):
    """Fresh-process prediction must preserve learned behavior and row-loss rejection."""
    if geo:
        pytest.importorskip("h3")
    nodes = ["H3Index"] if geo else [node for node in RECIPES if node != "H3Index"]
    _save(tmp_path, engine, nodes)
    code = "import json,runpy,sys; from pathlib import Path; m=runpy.run_path(sys.argv[1]); print(json.dumps(m['_replay'](Path(sys.argv[2]))))"
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)) == set(nodes)
