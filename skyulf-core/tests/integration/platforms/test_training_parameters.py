"""Experiments expose the actual fitted recipe and active holdout controls."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.fitting.candidate import TrainingSpec
from skyulf.integrations.databricks.training.shared.training_parameters import (
    _parameter_value,
    log_training_parameters,
)
from skyulf.integrations.mlflow.runs.tracking import TrackingRun


@pytest.fixture
def fitted_parameters(tmp_path, request):
    """Fit real fixed models so defaults and constructor normalization are observable."""
    engine, task = getattr(request, "param", ("pandas", "regression"))
    classification = task == "classification"
    config = {
        "preprocessing": [
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
        ],
        "feature_recipes": {"preprocessing": "numeric", "pre_split": "eligible"},
        "modeling": {
            "type": "logistic_regression" if classification else "ridge_regression",
            "params": {"C": 0.7} if classification else {"alpha": 0.7},
        },
    }
    frame = pd.DataFrame(
        {"x": range(24), "target": [i % 2 if classification else i * 2 for i in range(24)]}
    )
    native = pl.from_pandas(frame) if engine == "polars" else frame
    artifact = fit_workflow(
        config,
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=tmp_path / "model",
        max_rows=50,
        max_bytes=100_000,
    )
    spec = TrainingSpec(
        table="workspace.test.source",
        version=1,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="target",
        max_rows=50,
        max_bytes=100_000,
        test_size=0.25,
        random_state=7,
        stratify=classification,
        pre_split_steps=(
            {"name": "eligible", "transformer": "DropMissingRows", "params": {"subset": ["x"]}},
        ),
    )
    return artifact, spec, config


@pytest.mark.parametrize(
    "fitted_parameters",
    [
        (engine, task)
        for engine in ("pandas", "polars")
        for task in ("classification", "regression")
    ],
    indirect=True,
)
def test_fixed_model_defaults_and_recipe_are_visible_in_mlflow(fitted_parameters, tmp_path):
    """Fixed and tuned models need equally useful parameter comparisons across engines."""
    mlflow = pytest.importorskip("mlflow")
    artifact, spec, config = fitted_parameters
    client = mlflow.tracking.MlflowClient(tracking_uri=f"sqlite:///{tmp_path.as_posix()}/runs.db")
    experiment = client.create_experiment("parameters", artifact_location=tmp_path.as_uri())
    created = client.create_run(experiment)
    run = TrackingRun(client=client, run_id=created.info.run_id, enabled=True)
    log_training_parameters(run, artifact, spec, config)
    client.set_terminated(created.info.run_id)
    params = client.get_run(created.info.run_id).data.params
    assert params["model_type"] == config["modeling"]["type"]
    assert params["model_params.fit_intercept"] == "true"
    field = "C" if spec.stratify else "alpha"
    assert params[f"model_params.{field}"] == "0.7"
    assert params["split_random_state"] == "7"
    assert params["split_test_size"] == "0.25"
    assert params["split_stratify"] == str(spec.stratify).lower()
    assert params["preprocessing_recipe"] == "numeric"
    assert params["pre_split_recipe"] == "eligible"
    assert json.loads(params["preprocessing_steps"]) == ["scale"]
    assert json.loads(params["pre_split_steps"]) == ["eligible"]
    assert "training_parameters.json" in {
        item.path for item in client.list_artifacts(created.info.run_id)
    }


def test_temporal_parameters_show_boundaries_without_inactive_random_controls(fitted_parameters):
    """Temporal runs must not advertise a random holdout ratio or random seed."""
    artifact, spec, config = fitted_parameters
    temporal = replace(
        spec,
        split_strategy="temporal",
        test_size=None,
        random_state=None,
        stratify=None,
        event_column="event_time",
        start=datetime(2026, 1, 1, tzinfo=UTC),
        holdout_start=datetime(2026, 2, 1, tzinfo=UTC),
        cutoff=datetime(2026, 3, 1, tzinfo=UTC),
    )
    run = Mock()
    log_training_parameters(run, artifact, temporal, config)
    params = run.log_params.call_args.args[0]
    assert params["split_strategy"] == "temporal"
    assert params["split_holdout_start"] == "2026-02-01T00:00:00+00:00"
    assert params["split_event_column"] == "event_time"
    assert not {"split_test_size", "split_random_state", "split_stratify"} & params.keys()


def test_large_step_lists_remain_complete_in_artifact(fitted_parameters):
    """Display bounds must preserve complete ordered step names in the saved summary."""
    artifact, spec, config = fitted_parameters
    config["preprocessing"] = [{"name": f"step_{index:04d}"} for index in range(100)]
    run = Mock()
    log_training_parameters(run, artifact, spec, config)
    params = run.log_params.call_args.args[0]
    summary = run.client.log_dict.call_args.args[1]
    assert params["preprocessing_steps"].startswith("See training_parameters.json:")
    assert summary["preprocessing_steps"] == [f"step_{index:04d}" for index in range(100)]


def test_constructor_summary_describes_non_json_values_without_object_reprs():
    """Numpy defaults and estimator objects cannot leak unstable reprs or invalid JSON."""
    value = _parameter_value(
        {"missing": float("nan"), "weights": np.array([1, 2]), "model": object()}
    )
    assert value == {
        "missing": {"nonfinite": "nan"},
        "weights": [1, 2],
        "model": {"python_type": "builtins.object"},
    }
    assert json.loads(json.dumps(value, allow_nan=False)) == value
