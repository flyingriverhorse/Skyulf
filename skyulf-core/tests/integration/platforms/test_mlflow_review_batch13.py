"""Real MLflow transport and asynchronous logging retain Skyulf's explicit contracts."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")
from mlflow.utils.async_logging.run_operations import RunOperations  # noqa: E402

from skyulf.inference.local_pipeline import (  # noqa: E402
    load_local_pipeline,
    predict_local_pipeline,
    save_local_pipeline,
)
from skyulf.integrations.mlflow import local_model  # noqa: E402
from skyulf.integrations.mlflow.local_model import log_local_model  # noqa: E402
from skyulf.integrations.mlflow.tracking import TrackingConfig, TrackingRun, track_run  # noqa: E402
from skyulf.pipeline import SkyulfPipeline  # noqa: E402


@pytest.fixture
def tracking_store(tmp_path, monkeypatch):
    """Keep the real SQLite store and every artifact under this test's owned directory."""
    previous = mlflow.get_tracking_uri()
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "false")
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient(tracking_uri=uri)
    client.create_experiment("batch13", artifact_location=(tmp_path / "artifacts").as_uri())
    config = TrackingConfig(enabled=True, tracking_uri=uri, experiment_name="batch13")
    try:
        yield client, config
    finally:
        mlflow.set_tracking_uri(previous)


def _nullable_model(tmp_path, config, dtype):
    """Save and load through actual MLflow pyfunc signature enforcement."""
    values = [False, True] * 6 if dtype == "boolean" else list(range(12))
    frame = pd.DataFrame({"x": pd.Series(values, dtype=dtype), "target": np.arange(12.0)})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "impute",
                    "transformer": "SimpleImputer",
                    "params": {"columns": ["x"], "strategy": "most_frequent"},
                }
            ],
            "modeling": {"type": "decision_tree_regressor", "params": {"max_depth": 2}},
        }
    )
    pipeline.fit(frame, target_column="target")
    local_path = tmp_path / "local"
    save_local_pipeline(pipeline, local_path)
    local = load_local_pipeline(local_path)
    with track_run(config, run_name="nullable") as run:
        assert run.run_id is not None
        uri = log_local_model(
            local_path, run_id=run.run_id, artifact_path="model", tracking_uri=config.tracking_uri
        )
    return mlflow.pyfunc.load_model(uri), local


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
@pytest.mark.parametrize("missing", [False, True])
def test_nullable_pyfunc_preserves_declared_input(
    tmp_path, tracking_store, monkeypatch, dtype, missing
):
    """MLflow must replay fitted nullable inputs without changing nulls or caller indices."""
    _, config = tracking_store
    model, local = _nullable_model(tmp_path, config, dtype)
    values: list[bool | int | float | None] = (
        [False, True, False]
        if dtype == "boolean"
        else [2**53 + 1, 7, 9]
        if dtype == "Int64"
        else [3.25, 7.5, 9.75]
    )
    if missing:
        values[1] = None
    query = pd.DataFrame({"x": pd.array(values, dtype=dtype)}, index=[19, 3, 19])
    before = query.copy(deep=True)
    expected = predict_local_pipeline(query, local)
    if missing and dtype in {"Int64", "boolean"}:
        # Scalar MLflow signatures reject these nulls before calling the adapter.
        with pytest.raises(mlflow.exceptions.MlflowException, match="Failed to enforce schema"):
            model.predict(query)
        pd.testing.assert_frame_equal(query, before)
        return
    observer = Mock(wraps=local_model.score_local_pipeline)
    monkeypatch.setattr(local_model, "score_local_pipeline", observer)
    actual = model.predict(query)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(observer.call_args.args[0], query)
    if not missing:
        numpy_query = query.astype(
            {"x": {"Int64": "int64", "Float64": "float64", "boolean": "bool"}[dtype]}
        )
        pd.testing.assert_frame_equal(model.predict(numpy_query), expected)
        pd.testing.assert_frame_equal(observer.call_args.args[0], query)
    pd.testing.assert_frame_equal(query, before)


@pytest.mark.parametrize(
    "method,values",
    [
        ("log_metrics", {"score": 1.0}),
        ("log_params", {"kind": "test"}),
        ("set_tags", {"kind": "test"}),
    ],
)
@pytest.mark.parametrize("policy", ["raise", "warn"])
def test_future_failures_follow_tracking_policy(
    tracking_store, monkeypatch, method, values, policy
):
    """A returned asynchronous operation cannot hide its failure from the run policy."""
    client, config = tracking_store
    config = TrackingConfig(
        enabled=True,
        tracking_uri=config.tracking_uri,
        experiment_name="batch13",
        failure_policy=policy,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:

        def fail():
            """Reproduce an asynchronous backend failure on an actual worker future."""
            raise RuntimeError("controlled asynchronous failure")

        operation = RunOperations([executor.submit(fail)])
        with track_run(config, run_name="future-policy") as run:
            backend_method = {
                "log_metrics": "log_metric",
                "log_params": "log_param",
                "set_tags": "set_tag",
            }[method]
            monkeypatch.setattr(run.client, backend_method, lambda *args, **kwargs: operation)
            if policy == "raise":
                with pytest.raises(
                    mlflow.exceptions.MlflowException, match="controlled asynchronous"
                ):
                    getattr(run, method)(values)
            else:
                getattr(run, method)(values)
            assert run.tracking_error is not None
            assert "controlled asynchronous" in run.tracking_error
        assert client.get_run(run.run_id).info.status == "FINISHED"


def test_real_async_parameter_failure_marks_run_failed(tracking_store, monkeypatch):
    """Actual async MLflow parameter conflicts must escape and fail the owning run."""
    client, config = tracking_store
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "true")
    with (
        pytest.raises(mlflow.exceptions.MlflowException, match="Changing param values"),
        track_run(config, run_name="real-async") as run,
    ):
        client.log_param(run.run_id, "fixed", "original", synchronous=True)
        run.log_params({"fixed": "changed"})
    assert client.get_run(run.run_id).info.status == "FAILED"


@pytest.mark.parametrize(
    "path",
    [
        "../escape.json",
        r"..\escape.json",
        "nested/../../escape.json",
        r"nested\..\..\escape.json",
        "/escape.json",
        r"\escape.json",
        r"C:\escape.json",
        "C:/escape.json",
        "C:escape.json",
        r"\\host\share\escape.json",
    ],
)
def test_config_rejects_nonrelative_paths_before_client_calls(path):
    """Both OS path grammars must be checked before a client can write outside its artifact root."""
    client = Mock()
    run = TrackingRun(client=client, run_id="run", enabled=True)
    with pytest.raises(ValueError, match="artifact_file"):
        run.log_config({"safe": True}, artifact_file=path)
    assert not client.mock_calls


def test_config_keeps_safe_nested_artifacts(tracking_store, tmp_path):
    """Portable relative subdirectories retain real config bytes and the explicit digest."""
    client, config = tracking_store
    with track_run(config, run_name="safe-config") as run:
        run.log_config({"value": 7}, artifact_file="nested/config.json")
    download = client.download_artifacts(run.run_id, "nested/config.json", str(tmp_path))
    assert '"value": 7' in Path(download).read_text(encoding="utf-8")
    assert client.get_run(run.run_id).data.params["config_sha256"]


@pytest.mark.parametrize(
    "dtype,bad", [("Int64", [1.0, 2.0]), ("boolean", [1, 0]), ("Float64", ["1", "2"])]
)
def test_nullable_pyfunc_does_not_cast_incompatible_transport(tmp_path, tracking_store, dtype, bad):
    """The adapter must retain signature and raw artifact rejection of incompatible inputs."""
    _, config = tracking_store
    model, local = _nullable_model(tmp_path, config, dtype)
    query = pd.DataFrame({"x": bad})
    with pytest.raises((mlflow.exceptions.MlflowException, ValueError)):
        model.predict(query)
    with pytest.raises(ValueError, match="dtype"):
        predict_local_pipeline(query, local)


def test_async_success_finishes_before_tracking_returns(tracking_store, monkeypatch):
    """Successful async logging must be durable before the owning run finishes."""
    client, config = tracking_store
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_LOGGING", "true")
    with track_run(config, run_name="real-async-success") as run:
        run.log_metrics({"score": 1.25})
        run.log_params({"kind": "complete"})
        run.set_tags({"owner": "batch13"})
    stored = client.get_run(run.run_id)
    assert stored.info.status == "FINISHED"
    assert stored.data.metrics["score"] == 1.25
    assert stored.data.params["kind"] == "complete"
    assert stored.data.tags["owner"] == "batch13"
