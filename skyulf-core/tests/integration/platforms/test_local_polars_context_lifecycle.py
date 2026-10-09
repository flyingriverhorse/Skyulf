"""Reviewed existing families retain local context through real saved-model replay."""

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

RECIPES = {
    "SimpleImputer": {"columns": ["z"], "strategy": "mean"},
    "GroupImputer": {"columns": ["x"], "group_by": "group", "strategy": "mean"},
    "ClipValues": {"bounds": {"x": {"lower": 0, "upper": 10}}},
    "MinMaxScaler": {"columns": ["x", "z"]},
    "FeatureInteraction": {"columns": ["x", "z"], "degree": 2},
    "OneHotEncoder": {"columns": ["group"], "max_categories": None},
}


def _replay(directory):
    """Use only saved artifacts in a fresh process, with every calculator disabled."""

    def forbidden(*args, **kwargs):
        """Model loading, inspection and prediction must not fit preprocessing again."""
        raise AssertionError("Unexpected fit")

    for node in RECIPES:
        calculator: Any = NodeRegistry.get_calculator(node)
        calculator.fit = forbidden
    sample = pickle.loads((directory / "sample.pkl").read_bytes())
    artifact = load_local_pipeline(directory / "model")
    before = artifact_digest(artifact.pipeline.feature_engineer.fitted_steps)
    report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 2))
    assert report["status"] == "passed", report
    assert {step["node_type"] for step in report["steps"]} == set(RECIPES)
    assert all(
        step["context"] == "row" and step["status"] == "passed" for step in report["steps"]
    ), report
    predictions = artifact.pipeline.predict(sample)
    chunks = [
        sample.iloc[i : i + 1] if isinstance(sample, pd.DataFrame) else sample.slice(i, 1)
        for i in range(len(sample))
    ]
    singles = np.concatenate([artifact.pipeline.predict(part) for part in chunks])
    np.testing.assert_array_equal(predictions, singles)
    assert len(predictions) == len(sample) and np.isfinite(predictions).all()
    assert artifact_digest(artifact.pipeline.feature_engineer.fitted_steps) == before
    return report


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_existing_contexts_reload_and_predict_without_fit(tmp_path, engine):
    """All six owners must report reviewed context after real fit/save and fresh-process load."""
    training = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "z": [1.0, 2.0, 3.0, 4.0] * 2,
            "group": ["a", "b"] * 4,
            "target": range(8),
        }
    )
    sample = pd.DataFrame(
        {
            "x": [None, None, -5.0, 100.0],
            "z": [None, 10.0, 0.0, 2.0],
            "group": ["a", "new", None, "b"],
        },
        index=[7, 2, 2, 0],
    )
    if engine == "polars":
        training, sample = pl.from_pandas(training), pl.from_pandas(sample)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": node, "transformer": node, "params": config}
                for node, config in RECIPES.items()
            ],
            "modeling": {
                "type": "random_forest_regressor",
                "params": {"n_estimators": 2, "max_depth": 2, "random_state": 42, "n_jobs": 1},
            },
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training[:0]), target_column="target")
    save_local_pipeline(pipeline, tmp_path / "model")
    (tmp_path / "sample.pkl").write_bytes(pickle.dumps(sample))
    code = "import json,runpy,sys; from pathlib import Path; module=runpy.run_path(sys.argv[1]); print(json.dumps(module['_replay'](Path(sys.argv[2]))))"
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "passed" and report["admission"] == "diagnostic_only"
