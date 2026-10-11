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

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.partition_safety import require_partition_safe_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.pipeline import _apply_prediction_step
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
    artifact = load_pipeline(directory / "model")
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
    save_pipeline(pipeline, tmp_path / "model")
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


def _replay_integer_group(directory):
    """Inspect exact saved preprocessing independently of the model's numeric conversion."""

    def forbidden(*args, **kwargs):
        """Reloaded group values must come from the saved training artifact."""
        raise AssertionError("Unexpected fit")

    calculator: Any = NodeRegistry.get_calculator("GroupImputer")
    calculator.fit = forbidden
    sample, value = pickle.loads((directory / "sample.pkl").read_bytes())
    artifact = load_pipeline(directory / "model")
    records = artifact.pipeline.feature_engineer.fitted_steps
    before = artifact_digest(records)
    state = records[0]["artifact"]
    assert state["fill_values"]["x"] == value
    assert type(state["fill_values"]["x"]) is int
    result = _apply_prediction_step(sample, records[0])
    assert result["x"].to_list() == [value, 1, value, value]
    assert result["x"].dtype == sample["x"].dtype
    report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 2))
    assert report["status"] == "passed", report
    assert report["steps"][0]["context"] == "row"
    predictions = artifact.pipeline.predict(sample)
    singles = np.concatenate(
        [artifact.pipeline.predict(sample[i : i + 1]) for i in range(len(sample))]
    )
    np.testing.assert_array_equal(predictions, singles)
    assert np.isfinite(predictions).all()
    assert artifact_digest(records) == before
    return report


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype,value", [("Int64", 2**53 + 1), ("UInt64", 2**64 - 1)])
def test_integer_group_state_survives_fresh_process_reload(tmp_path, engine, dtype, value):
    """Saving a real pipeline must retain exact integer modes in both native engines."""
    training = pd.DataFrame(
        {
            "g": [1, 1, 1, 2],
            "x": pd.Series([value, None, value, 1], dtype=dtype),
            "target": [0.0, 1.0, 2.0, 3.0],
        }
    )
    sample = pd.DataFrame(
        {"g": [1, 2, 3, 1], "x": pd.Series([None, None, None, value], dtype=dtype)}
    )
    if engine == "polars":
        training, sample = pl.from_pandas(training), pl.from_pandas(sample)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "group",
                    "transformer": "GroupImputer",
                    "params": {"columns": ["x"], "group_by": "g", "strategy": "mode"},
                }
            ],
            "modeling": {
                "type": "random_forest_regressor",
                "params": {"n_estimators": 2, "max_depth": 2, "random_state": 42, "n_jobs": 1},
            },
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training[:0]), target_column="target")
    save_pipeline(pipeline, tmp_path / "model")
    (tmp_path / "sample.pkl").write_bytes(pickle.dumps((sample, value)))
    code = "import json,runpy,sys; from pathlib import Path; module=runpy.run_path(sys.argv[1]); print(json.dumps(module['_replay_integer_group'](Path(sys.argv[2]))))"
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "passed"


def _replay_imputation_statistic(directory):
    """Replay saved statistics and check the strategy's existing worker admission."""
    node, strategy, sample = pickle.loads((directory / "sample.pkl").read_bytes())

    def forbidden(*args, **kwargs):
        """Loading and diagnostics must never recompute statistics from request rows."""
        raise AssertionError("Unexpected fit")

    calculator: Any = NodeRegistry.get_calculator(node)
    calculator.fit = forbidden
    artifact = load_pipeline(directory / "model")
    records = artifact.pipeline.feature_engineer.fitted_steps
    before = artifact_digest(records)
    output = _apply_prediction_step(sample, records[0])
    expected = {
        ("SimpleImputer", "median"): [4.0] * 4,
        ("SimpleImputer", "mode"): [1.0] * 4,
        ("GroupImputer", "median"): [2.5, 9.0, 4.0, 9.0],
        ("GroupImputer", "mode"): [1.0, 9.0, 1.0, 9.0],
    }[node, strategy]
    assert output["x"].to_list() == expected
    report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 2))
    assert report["status"] == "passed", report
    assert report["steps"][0]["context"] == "row"
    assert report["steps"][0]["state_validation"] == "node_owned"
    predictions = artifact.pipeline.predict(sample)
    singles = np.concatenate(
        [artifact.pipeline.predict(sample[i : i + 1]) for i in range(len(sample))]
    )
    np.testing.assert_array_equal(predictions, singles)
    if strategy == "mode" and isinstance(sample, pd.DataFrame):
        certificate = require_partition_safe_pipeline(artifact)
        assert certificate.fitted_engine == "pandas"
        assert len(certificate.steps) == 1 and certificate.steps[0].node_type == node
        config = dict(json.loads(certificate.steps[0].config_json)["items"])
        assert config["strategy"] == {"kind": "string", "value": "most_frequent"}
    else:
        with pytest.raises(UnsupportedExecutionError):
            require_partition_safe_pipeline(artifact)
    assert np.isfinite(predictions).all()
    assert artifact_digest(records) == before
    return report


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["SimpleImputer", "GroupImputer"])
@pytest.mark.parametrize("strategy", ["median", "mode"])
def test_imputation_statistic_context_survives_saved_model_reload(tmp_path, engine, node, strategy):
    """Median and mode replay saved rows while only pandas modes retain worker admission."""
    training = pd.DataFrame(
        {"g": [1, 1, 2, 2] * 2, "x": [1.0, 4.0, 9.0, None] * 2, "target": range(8)}
    )
    sample = pd.DataFrame({"g": [1, 2, 3, 2], "x": [np.nan] * 4})
    sample.index = [7, 2, 2, 0]
    if engine == "polars":
        training, sample = pl.from_pandas(training), pl.from_pandas(sample)
    params = {"columns": ["x"], "strategy": strategy}
    if node == "GroupImputer":
        params["group_by"] = "g"
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [{"name": strategy, "transformer": node, "params": params}],
            "modeling": {
                "type": "random_forest_regressor",
                "params": {"n_estimators": 2, "max_depth": 2, "random_state": 42, "n_jobs": 1},
            },
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training[:0]), target_column="target")
    save_pipeline(pipeline, tmp_path / "model")
    (tmp_path / "sample.pkl").write_bytes(pickle.dumps((node, strategy, sample)))
    code = "import json,runpy,sys; from pathlib import Path; module=runpy.run_path(sys.argv[1]); print(json.dumps(module['_replay_imputation_statistic'](Path(sys.argv[2]))))"
    result = subprocess.run(
        [sys.executable, "-c", code, str(Path(__file__).resolve()), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "passed"


def _replay_capped_onehot(directory):
    """Inspect saved infrequent categories without fitting or admitting a worker pipeline."""

    def forbidden(*args, **kwargs):
        """Rare categories must come from training instead of each prediction partition."""
        raise AssertionError("Unexpected fit")

    calculator: Any = NodeRegistry.get_calculator("OneHotEncoder")
    calculator.fit = forbidden
    sample = pickle.loads((directory / "sample.pkl").read_bytes())
    artifact = load_pipeline(directory / "model")
    records = artifact.pipeline.feature_engineer.fitted_steps
    before = artifact_digest(records)
    encoder = records[0]["artifact"]["encoder_object"]
    assert len(encoder.infrequent_categories_[0]) > 0
    if encoder.drop == "first":
        assert encoder.drop_idx_[0] > 0
    output = _apply_prediction_step(sample, records[0])
    assert output["group_infrequent_sklearn"].to_list() == [1, 0, 0, 0]
    report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 2))
    assert report["status"] == "passed", report
    assert report["steps"][0]["context"] == "row"
    assert report["steps"][0]["state_validation"] == "node_owned"
    predictions = artifact.pipeline.predict(sample)
    singles = np.concatenate(
        [artifact.pipeline.predict(sample[i : i + 1]) for i in range(len(sample))]
    )
    np.testing.assert_array_equal(predictions, singles)
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(artifact)
    assert np.isfinite(predictions).all()
    assert artifact_digest(records) == before
    return report


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("options", [{}, {"max_categories": 3, "drop_first": True}])
def test_capped_onehot_local_context_survives_saved_model_reload(tmp_path, engine, options):
    """Default and explicit caps retain real rare-category grouping through fresh-process replay."""
    groups = [f"c{i:02}" for i in range(22)] + ["c10"] * 4 + ["c20"] * 3
    training = pd.DataFrame({"group": groups, "target": range(len(groups))})
    sample = pd.DataFrame({"group": ["c00", "c10", "c20", "unseen"]}, index=[7, 2, 2, 0])
    if engine == "polars":
        training, sample = pl.from_pandas(training), pl.from_pandas(sample)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "capped",
                    "transformer": "OneHotEncoder",
                    "params": {"columns": ["group"], **options},
                }
            ],
            "modeling": {
                "type": "random_forest_regressor",
                "params": {"n_estimators": 2, "max_depth": 2, "random_state": 42, "n_jobs": 1},
            },
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training[:0]), target_column="target")
    save_pipeline(pipeline, tmp_path / "model")
    (tmp_path / "sample.pkl").write_bytes(pickle.dumps(sample))
    code = "import json,runpy,sys; from pathlib import Path; module=runpy.run_path(sys.argv[1]); print(json.dumps(module['_replay_capped_onehot'](Path(sys.argv[2]))))"
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


def _replay_missing_onehot(directory):
    """Replay saved missing-value identity while retaining the local-only execution boundary."""

    def forbidden(*args, **kwargs):
        """Saved missing categories must not be relearned from inference partitions."""
        raise AssertionError("Unexpected fit")

    calculator: Any = NodeRegistry.get_calculator("OneHotEncoder")
    calculator.fit = forbidden
    sample = pickle.loads((directory / "sample.pkl").read_bytes())
    artifact = load_pipeline(directory / "model")
    records = artifact.pipeline.feature_engineer.fitted_steps
    before = artifact_digest(records)
    state = records[0]["artifact"]
    assert state["missing_encoding_version"] == 1
    encoder = state["encoder_object"]
    if encoder.max_categories is not None:
        assert len(encoder.infrequent_categories_[0]) > 0
    output = _apply_prediction_step(sample, records[0])
    token = "__mlops_missing__"
    prefix = "__mlops_literal__:"
    columns = [f"group_{token}", f"group_{prefix}{token}", f"group_{prefix}{prefix}{token}"]
    np.testing.assert_array_equal(
        output[columns].to_numpy(), [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]
    )
    report = probe_fitted_preprocessing(artifact, sample, chunk_sizes=(1, 2))
    assert report["status"] == "passed", report
    assert report["steps"][0]["context"] == "row"
    assert report["steps"][0]["state_validation"] == "node_owned"
    predictions = artifact.pipeline.predict(sample)
    singles = np.concatenate(
        [artifact.pipeline.predict(sample[i : i + 1]) for i in range(len(sample))]
    )
    np.testing.assert_array_equal(predictions, singles)
    with pytest.raises(UnsupportedExecutionError):
        require_partition_safe_pipeline(artifact)
    assert np.isfinite(predictions).all()
    assert artifact_digest(records) == before
    return report


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("options", [{"max_categories": None}, {}])
def test_missing_onehot_local_context_survives_saved_model_reload(tmp_path, engine, options):
    """Real nulls and escaped literals remain distinct after saved capped and uncapped replay."""
    special = [None, "__mlops_missing__", "__mlops_literal__:__mlops_missing__"]
    groups = special * 3 + [f"c{i:02}" for i in range(22)]
    training = pd.DataFrame({"group": groups, "target": range(len(groups))})
    sample = pd.DataFrame({"group": [*special, "unseen"]}, index=[7, 2, 2, 0])
    if engine == "polars":
        training, sample = pl.from_pandas(training), pl.from_pandas(sample)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "missing",
                    "transformer": "OneHotEncoder",
                    "params": {"columns": ["group"], "include_missing": True, **options},
                }
            ],
            "modeling": {
                "type": "random_forest_regressor",
                "params": {"n_estimators": 2, "max_depth": 2, "random_state": 42, "n_jobs": 1},
            },
        }
    )
    pipeline.fit(SplitDataset(train=training, test=training[:0]), target_column="target")
    save_pipeline(pipeline, tmp_path / "model")
    (tmp_path / "sample.pkl").write_bytes(pickle.dumps(sample))
    code = "import json,runpy,sys; from pathlib import Path; module=runpy.run_path(sys.argv[1]); print(json.dumps(module['_replay_missing_onehot'](Path(sys.argv[2]))))"
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
