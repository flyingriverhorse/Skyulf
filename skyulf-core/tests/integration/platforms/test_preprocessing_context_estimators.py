"""Real saved encoder, imputer and power pipelines replay without calculator fitting."""

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
    "DummyEncoder": {"columns": ["cat"], "drop_first": False},
    "HashEncoder": {"columns": ["cat"], "n_features": 8},
    "LabelEncoder": {"columns": ["cat"], "missing_code": -1},
    "OrdinalEncoder": {"columns": ["cat"]},
    "TargetEncoder": {"columns": ["cat"], "target_type": "continuous", "smooth": 0.0},
    "WOEEncoder": {"columns": ["cat"], "regularization": 0.5},
    "KNNImputer": {"columns": ["x", "z"], "n_neighbors": 2},
    "IterativeImputer": {"columns": ["x", "z"], "max_iter": 5, "random_state": 0},
    "GeneralTransformation": {"transformations": [{"column": "x", "method": "square"}]},
    "PowerTransformer": {"columns": ["x"], "method": "yeo-johnson"},
}


def _frames(node):
    """Keep supervised encoders genuinely fitted and model inputs free from labels."""
    if "Encoder" in node:
        training = pd.DataFrame({"cat": ["a", "a", "b", "b", None, "c", "a", "c", "b", "a"]})
        sample = pd.DataFrame({"cat": ["a", "new", None, "c"]})
    else:
        training = pd.DataFrame({"x": [0.0, 1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]})
        sample = pd.DataFrame({"x": [1.0, 2.0, None, 7.0]})
        if "Imputer" in node:
            training["z"] = [0.0, 2.0, None, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
            sample["z"] = [None, 4.0, 6.0, 14.0]
    training["target"] = [0.0, 1.0] * 5
    sample.index = [0, 0, 1, 1]
    return training, sample


def _save(directory, engine):
    """Persist ten fitted models and expected whole-request predictions before reload."""
    evidence = {}
    for node, config in RECIPES.items():
        training, sample = _frames(node)
        if engine == "polars":
            training, sample = pl.from_pandas(training), pl.from_pandas(sample)
        pipeline = SkyulfPipeline(
            {
                "preprocessing": [{"name": "reviewed", "transformer": node, "params": config}],
                "modeling": {
                    "type": "random_forest_regressor",
                    "params": {"n_estimators": 4, "max_depth": 3, "n_jobs": 1, "random_state": 42},
                },
            }
        )
        pipeline.fit(SplitDataset(train=training[2:], test=training[:2]), target_column="target")
        assert pipeline.feature_engineer.fitted_steps[0]["artifact"], node
        predictions = pipeline.predict(sample)
        digest = artifact_digest(pipeline.feature_engineer.fitted_steps)
        save_local_pipeline(pipeline, directory / node)
        evidence[node] = (sample, predictions, digest)
    (directory / "requests.pkl").write_bytes(pickle.dumps(evidence))


def _replay(directory):
    """Load actual artifacts after disabling fit and respect contexts in diagnostics."""

    def forbidden(*args, **kwargs):
        """Any calculator invocation after reload violates fitted-state reuse."""
        raise AssertionError("Unexpected fit")

    for node in RECIPES:
        calculator: Any = NodeRegistry.get_calculator(node)
        calculator.fit = forbidden
    requests = pickle.loads((directory / "requests.pkl").read_bytes())
    statuses = {}
    for node, (sample, expected, digest) in requests.items():
        artifact = load_local_pipeline(directory / node)
        assert artifact_digest(artifact.pipeline.feature_engineer.fitted_steps) == digest
        actual = artifact.pipeline.predict(sample)
        np.testing.assert_array_equal(actual, expected)
        assert len(actual) == len(sample) and np.isfinite(actual).all()
        report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 3))
        step = report["steps"][0]
        assert step["state_validation"] == "node_owned", report
        if node in ("IterativeImputer", "PowerTransformer"):
            assert step["context"] == "global" and step["status"] == "requires_context", report
        elif node == "HashEncoder":
            assert step["context"] == "row" and step["status"] == "failed", report
            failed = [check for check in step["checks"] if check["status"] == "failed"]
            assert failed and all(
                check["name"] == "empty" and check["reason"] == "output_mismatch"
                for check in failed
            ), report
        else:
            assert step["context"] == "row" and report["status"] == "passed", report
        if step["context"] == "row":
            parts = [
                sample.iloc[i : i + 1] if isinstance(sample, pd.DataFrame) else sample.slice(i, 1)
                for i in range(len(sample))
            ]
            np.testing.assert_array_equal(
                actual, np.concatenate([artifact.pipeline.predict(part) for part in parts])
            )
        assert artifact_digest(artifact.pipeline.feature_engineer.fitted_steps) == digest
        with pytest.raises(UnsupportedExecutionError):
            require_partition_safe_pipeline(artifact)
        statuses[node] = step["status"]
    return statuses


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ten_fitted_owners_preserve_saved_predictions(tmp_path, engine):
    """A fresh process must preserve learned model behavior without discarding context limits."""
    _save(tmp_path, engine)
    code = "import json,runpy,sys; from pathlib import Path; m=runpy.run_path(sys.argv[1]); print(json.dumps(m['_replay'](Path(sys.argv[2]))))"
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)) == set(RECIPES)
