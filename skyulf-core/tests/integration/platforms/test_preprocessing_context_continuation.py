"""Additional saved local preprocessing owners survive real model reload and prediction."""

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
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline.seal import artifact_digest
from skyulf.registry import NodeRegistry

RECIPES = {
    "Casting": {"column_types": {"x": "float64"}},
    "AliasReplacement": {"columns": ["text"], "alias_type": "boolean"},
    "InvalidValueReplacement": {"columns": ["x"], "rule": "negative", "replacement": 0.0},
    "TextCleaning": {
        "columns": ["text"],
        "operations": [{"op": "trim"}, {"op": "case", "mode": "lower"}],
    },
    "GeneralBinning": {
        "columns": ["x"],
        "strategy": "uniform",
        "n_bins": 2,
        "drop_original": True,
        "output_suffix": "",
    },
    "KBinsDiscretizer": {
        "columns": ["x"],
        "strategy": "uniform",
        "n_bins": 2,
        "drop_original": True,
        "output_suffix": "",
    },
    "SimpleTransformation": {"transformations": [{"column": "x", "method": "square"}]},
    "DateFeatures": {"columns": ["date"], "features": ["day", "hour"], "drop_original": True},
    "PolynomialFeaturesNode": {"columns": ["x", "z"], "degree": 2},
    "ManualBounds": {"bounds": {"x": {"lower": 1.0, "upper": 7.0}}},
    "Winsorize": {"columns": ["x"], "lower_percentile": 25, "upper_percentile": 75},
}


def _frames(node):
    """Provide fitting and genuinely different scoring inputs for each saved recipe."""
    training = pd.DataFrame({"x": [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]})
    sample = pd.DataFrame({"x": [None, -5.0, 1.0, 3.0, 7.0, 40.0]})
    if node == "Casting":
        training["x"] = ["0", "1", "2", "4", "6", "8"]
        sample["x"] = [None, "invalid", "1", "3.5", "7", "40"]
    elif node in ("AliasReplacement", "TextCleaning"):
        training["text"] = [" YES! ", "no", "true", " NO ", "yes", "unseen"]
        sample["text"] = [None, "new", " No ", "true", "yes", " YES! "]
    elif node == "DateFeatures":
        training = pd.DataFrame({"date": [f"2024-01-0{i}T0{i}:30:00Z" for i in range(1, 7)]})
        sample = pd.DataFrame({"date": [f"2024-01-0{i}T0{i + 2}:30:00+02:00" for i in range(1, 7)]})
    elif node == "PolynomialFeaturesNode":
        training["z"] = [4.0, 3.0, 2.0, 1.0, 0.0, -2.0]
        sample = pd.DataFrame({"x": [2.0, 0.0, -3.0, 8.0], "z": [-1.0, 4.0, 0.0, 2.0]})
    elif node == "ManualBounds":
        sample["x"] = [None, 1.0, 2.0, 3.0, 6.0, 7.0]
    training["target"] = [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]
    sample.index = [i // 2 for i in range(len(sample))]
    return training, sample


def _save(directory, engine):
    """Save fitted real models, with text encoding handled by the existing node."""
    samples = {}
    for node, params in RECIPES.items():
        training, sample = _frames(node)
        steps = [{"name": "reviewed", "transformer": node, "params": params}]
        if node in ("AliasReplacement", "TextCleaning"):
            steps.append(
                {
                    "name": "encode",
                    "transformer": "OneHotEncoder",
                    "params": {
                        "columns": ["text"],
                        "max_categories": None,
                        "handle_unknown": "ignore",
                    },
                }
            )
        if engine == "polars":
            training, sample = pl.from_pandas(training), pl.from_pandas(sample)
        pipeline = SkyulfPipeline(
            {
                "preprocessing": steps,
                "modeling": {
                    "type": "random_forest_regressor",
                    "params": {
                        "n_estimators": 4,
                        "max_depth": 3,
                        "n_jobs": 1,
                        "random_state": 42,
                    },
                },
            }
        )
        pipeline.fit(SplitDataset(train=training[2:], test=training[:2]), target_column="target")
        save_pipeline(pipeline, directory / node)
        samples[node] = sample
    (directory / "samples.pkl").write_bytes(pickle.dumps(samples))


def _replay(directory):
    """Disable calculators before loading and exercising the actual saved prediction path."""

    def forbidden(*args, **kwargs):
        """Inference must not relearn preprocessing or re-execute a recipe builder."""
        raise AssertionError("Unexpected calculator fit")

    for node in [*RECIPES, "OneHotEncoder"]:
        calculator: Any = NodeRegistry.get_calculator(node)
        calculator.fit = forbidden
    samples = pickle.loads((directory / "samples.pkl").read_bytes())
    reports = {}
    for node, sample in samples.items():
        artifact = load_pipeline(directory / node)
        before = artifact_digest(artifact.pipeline.feature_engineer.fitted_steps)
        report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 3))
        assert report["status"] == "passed", (node, report)
        step = report["steps"][0]
        assert step["context"] == "row", (node, step)
        assert step["status"] == "passed"
        if node == "PolynomialFeaturesNode":
            assert (
                next(check for check in step["checks"] if check["name"] == "empty")["status"]
                == "passed"
            )
        predictions = artifact.pipeline.predict(sample)
        chunks = [
            sample.iloc[i : i + 1] if isinstance(sample, pd.DataFrame) else sample.slice(i, 1)
            for i in range(len(sample))
        ]
        singletons = np.concatenate([artifact.pipeline.predict(chunk) for chunk in chunks])
        assert len(predictions) == len(sample)
        assert np.isfinite(predictions).all()
        np.testing.assert_array_equal(predictions, singletons)
        if node == "ManualBounds":
            outside = pd.DataFrame({"x": [100.0]})
            if isinstance(sample, pl.DataFrame):
                outside = pl.from_pandas(outside)
            with pytest.raises(ValueError, match="row"):
                artifact.pipeline.predict(outside)
        assert artifact_digest(artifact.pipeline.feature_engineer.fitted_steps) == before
        with pytest.raises(UnsupportedExecutionError):
            require_partition_safe_pipeline(artifact)
        reports[node] = report
    return reports


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_eleven_owners_load_probe_and_predict_without_relearning(tmp_path, engine):
    """A fresh process must retain real preprocessing and model behavior for both engines."""
    _save(tmp_path, engine)
    code = (
        "import json, runpy, sys; from pathlib import Path; "
        "module = runpy.run_path(sys.argv[1]); print(json.dumps(module['_replay'](Path(sys.argv[2]))))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    reports = json.loads(result.stdout)
    assert set(reports) == set(RECIPES)
    assert all(report["status"] == "passed" for report in reports.values())
