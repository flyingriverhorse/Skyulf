"""Saved local context declarations survive fresh processes without learning again."""

import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    "CustomBinning": {
        "columns": ["x"],
        "bins": [-100.0, 0.0, 10.0, 1000.0],
        "label_format": "range",
    },
    "ValueReplacement": {"columns": ["x"], "mapping": {"2": 12.0}},
    "DropMissingColumns": {"missing_threshold": 0.5},
    "MissingIndicator": {"columns": ["x"]},
    "CorrelationThreshold": {"columns": ["x", "correlated"], "threshold": 0.9},
    "UnivariateSelection": {
        "columns": ["x", "z"],
        "k": 1,
        "score_func": "f_regression",
        "problem_type": "regression",
    },
    "VarianceThreshold": {"columns": ["x", "constant"]},
    "MaxAbsScaler": {"columns": ["x", "constant"]},
    "RobustScaler": {"columns": ["x", "constant"]},
    "GeoDistance": {"lat1_col": "lat1", "lon1_col": "lon1", "lat2_col": "lat2", "lon2_col": "lon2"},
}


def _training():
    """Use exact scaler divisors; unit tests pin nonbinary floating-point diagnostics."""
    return pd.DataFrame(
        {
            "x": [0.0, 2.0, 2.0, 4.0, 4.0, 8.0],
            "z": [4.0, 2.0, 6.0, 0.0, 3.0, 1.0],
            "correlated": [0.0, 4.0, 4.0, 8.0, 8.0, 16.0],
            "constant": [0.0] * 6,
            "lat1": [40.0] * 6,
            "lon1": [20.0] * 6,
            "lat2": [41.0] * 6,
            "lon2": [21.0] * 6,
            "target": [0.0, 3.0, 4.0, 7.0, 8.0, 11.0],
        }
    )


def _sample():
    """Request values differ from training and include null and unseen numerical values."""
    frame = _training().drop(columns="target").iloc[:5].copy()
    frame["x"] = [None, 2.0, -10.0, 200.0, 6.0]
    frame["constant"] = [8.0, 0.0, -9.0, 1.0, None]
    frame.loc[0, "lat1"] = None
    frame.index = [8, 3, 3, 9, 1]
    return frame


def _save_cases(directory, engine):
    """Export real fitted pipelines, their schemas and exact diagnostic inputs."""
    samples = {}
    for node, params in RECIPES.items():
        frame, sample = _training(), _sample()
        if node == "DropMissingColumns":
            frame["drop"] = float("nan")
            sample["drop"] = 100.0
        steps = [{"name": "reviewed", "transformer": node, "params": params}]
        if node == "CustomBinning":
            steps.append(
                {
                    "name": "encode",
                    "transformer": "OneHotEncoder",
                    "params": {"columns": ["x_binned"], "max_categories": None},
                }
            )
        if engine == "polars":
            frame, sample = pl.from_pandas(frame), pl.from_pandas(sample)
        pipeline = SkyulfPipeline(
            {"preprocessing": steps, "modeling": {"type": "linear_regression"}}
        )
        pipeline.fit(SplitDataset(train=frame, test=frame[:0]), target_column="target")
        save_local_pipeline(pipeline, directory / node)
        samples[node] = sample
    (directory / "samples.pkl").write_bytes(pickle.dumps(samples))


def _replay_saved(directory):
    """Load trusted test artifacts and probe with every involved calculator disabled."""

    def forbidden(*args, **kwargs):
        """Fresh inference must never learn statistics or reconstruct a recipe."""
        raise AssertionError("Unexpected fit during saved preprocessing replay")

    for node in [*RECIPES, "OneHotEncoder"]:
        calculator: Any = NodeRegistry.get_calculator(node)
        calculator.fit = forbidden
    samples = pickle.loads((directory / "samples.pkl").read_bytes())
    reports = {}
    for node, sample in samples.items():
        artifact = load_local_pipeline(directory / node)
        before = artifact_digest(artifact.pipeline.feature_engineer.fitted_steps)
        report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(2, 3))
        assert report["status"] == "passed", (node, report)
        step = report["steps"][0]
        assert step["context"] == "row", (node, step)
        assert step["state_validation"] == "node_owned", (node, step)
        assert all(check["status"] == "passed" for check in step["checks"]), (node, step)
        assert artifact_digest(artifact.pipeline.feature_engineer.fitted_steps) == before
        assert report["admission"] == "diagnostic_only"
        with pytest.raises(UnsupportedExecutionError):
            require_partition_safe_pipeline(artifact)
        reports[node] = report
    return reports


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_ten_local_contexts_reload_without_fit_in_a_fresh_process(tmp_path, engine):
    """Installed owners and saved state must suffice for every reviewed local family."""
    _save_cases(tmp_path, engine)
    code = (
        "import json, runpy, sys; from pathlib import Path; "
        "module = runpy.run_path(sys.argv[1]); "
        "print(json.dumps(module['_replay_saved'](Path(sys.argv[2]))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    reports = json.loads(result.stdout)
    assert set(reports) == set(RECIPES)
    assert all(report["status"] == "passed" for report in reports.values())


def test_nullable_bins_and_object_replacement_reload_without_fit(tmp_path):
    """Stable pandas output types must survive real model training and fresh-process replay."""
    training = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "value": pd.Series([0.0, 1.0, 2.0, 3.0], dtype=object),
            "y": [0, 0, 1, 1],
        }
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "replace",
                    "transformer": "ValueReplacement",
                    "params": {"columns": ["value"], "mapping": {0.0: 0.5}},
                },
                {
                    "name": "bin",
                    "transformer": "CustomBinning",
                    "params": {"columns": ["x"], "bins": [0.0, 2.0, 4.0], "drop_original": True},
                },
            ],
            "modeling": {
                "type": "random_forest_regressor",
                "params": {"n_estimators": 4, "random_state": 42},
            },
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training[:0]), target_column="y")
    save_local_pipeline(pipeline, tmp_path / "model")
    code = """
import json, sys
import numpy as np
import pandas as pd
from skyulf.registry import NodeRegistry
from skyulf.inference.local_pipeline import load_local_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
def forbidden(*args, **kwargs):
    # Loaded artifacts must use saved transforms, never fit again.
    raise AssertionError('Unexpected fit during saved replay')
for node in ('CustomBinning', 'ValueReplacement'):
    NodeRegistry.get_calculator(node).fit = forbidden
artifact = load_local_pipeline(sys.argv[1])
sample = pd.DataFrame({'x': [0.0, 3.0, None, 9.0],
                       'value': pd.Series([0.0, 3.0, None, 9.0], dtype=object)})
report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 3))
assert report['status'] == 'passed', report
predictions = artifact.pipeline.predict(sample)
singletons = np.concatenate([artifact.pipeline.predict(sample.iloc[i:i+1]) for i in range(len(sample))])
assert np.isfinite(predictions).all()
np.testing.assert_array_equal(predictions, singletons)
print(json.dumps(report))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "model")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["steps"]) == 2
