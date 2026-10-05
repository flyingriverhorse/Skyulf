"""Contract tests for the optional MLflow tracking adapter."""

import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyulf.integrations.mlflow.runs import tracking


def test_experiment_creation_recovers_forced_race() -> None:
    """Both first-use workers must track in the experiment created by the winner."""
    mlflow = pytest.importorskip("mlflow")
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    experiment = None

    def lookup(name):
        """Force both initial reads to observe the missing experiment."""
        existing = experiment
        if existing is None:
            barrier.wait(timeout=10)
        return existing

    def create(name):
        """Allow one create and return MLflow's race error to the loser."""
        nonlocal experiment
        with lock:
            if experiment is not None:
                error = mlflow.exceptions.MlflowException("already exists")
                error.error_code = "RESOURCE_ALREADY_EXISTS"
                raise error
            experiment = SimpleNamespace(experiment_id="7")
            return "7"

    client = SimpleNamespace(get_experiment_by_name=lookup, create_experiment=create)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(tracking.get_or_create_experiment, client, "shared") for _ in range(2)
        ]
        assert [future.result(timeout=15) for future in futures] == ["7", "7"]


@pytest.mark.parametrize("code", ["PERMISSION_DENIED", "INTERNAL_ERROR", "RESOURCE_ALREADY_EXISTS"])
def test_experiment_creation_does_not_hide_failure(code) -> None:
    """Only an existing experiment after a create race may replace a creation failure."""
    mlflow = pytest.importorskip("mlflow")
    error = mlflow.exceptions.MlflowException("failed")
    error.error_code = code
    client = Mock()
    client.get_experiment_by_name.return_value = None
    client.create_experiment.side_effect = error
    with pytest.raises(mlflow.exceptions.MlflowException) as caught:
        tracking.get_or_create_experiment(client, "shared")
    assert caught.value is error
    assert client.get_experiment_by_name.call_count == (
        2 if code == "RESOURCE_ALREADY_EXISTS" else 1
    )


@pytest.mark.parametrize("failure_type", [ConnectionError, TimeoutError])
def test_experiment_creation_propagates_transport_failure(failure_type) -> None:
    """Network failures must propagate without attempting a speculative lookup."""
    pytest.importorskip("mlflow")
    error = failure_type("unavailable")
    client = Mock()
    client.get_experiment_by_name.return_value = None
    client.create_experiment.side_effect = error
    with pytest.raises(failure_type) as caught:
        tracking.get_or_create_experiment(client, "shared")
    assert caught.value is error
    client.get_experiment_by_name.assert_called_once_with("shared")


def test_disabled_tracking_never_constructs_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline training must not import or construct an MLflow client."""

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("MLflow client was constructed")

    monkeypatch.setattr(tracking, "make_tracking_client", forbidden)
    config = tracking.TrackingConfig()
    with tracking.track_run(config, run_name="offline") as run:
        run.log_metrics({"rmse": 0.5})
        run.log_params({"engine": "pandas"})
        run.set_tags({"mode": "local"})

    assert run.enabled is False
    assert run.run_id is None
    assert run.tracking_error is None


def test_tracking_config_rejects_invalid_failure_policy() -> None:
    """Only explicit raise or warn policies can control tracking failures."""
    with pytest.raises(ValueError, match="failure_policy"):
        tracking.TrackingConfig(failure_policy="ignore")  # ty: ignore[invalid-argument-type]


def test_enabled_tracking_reports_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabling tracking fails explicitly when the optional client cannot be loaded."""

    def missing_client(uri: str | None) -> object:
        raise ModuleNotFoundError("No module named 'mlflow'")

    monkeypatch.setattr(tracking, "make_tracking_client", missing_client)
    config = tracking.TrackingConfig(enabled=True)
    with (
        pytest.raises(ModuleNotFoundError, match="mlflow"),
        tracking.track_run(config, run_name="requires-mlflow"),
    ):
        pass


@pytest.mark.skipif(
    importlib.util.find_spec("mlflow") is None,
    reason="MLflow is an optional test dependency",
)
def test_enabled_tracking_logs_a_real_run_and_explicit_config(tmp_path: Path) -> None:
    """An enabled run stores metrics, tags, and an explicitly requested config artifact."""
    db_path = (tmp_path / "tracking.db").resolve()
    uri = f"sqlite:///{db_path.as_posix()}"
    config = tracking.TrackingConfig(
        enabled=True,
        tracking_uri=uri,
        experiment_name="skyulf-sm12-real",
    )

    with tracking.track_run(config, run_name="real-run") as run:
        run.log_metrics({"rmse": 0.5})
        run.log_params({"engine": "pandas"})
        run.set_tags({"mode": "local"})
        run.log_config({"engine": "pandas", "rows": 10})
        run_id = run.run_id

    assert run_id
    client = tracking.make_tracking_client(uri)
    stored = client.get_run(run_id)
    assert stored.info.status == "FINISHED"
    assert stored.data.metrics["rmse"] == 0.5
    assert stored.data.params["engine"] == "pandas"
    assert stored.data.tags["mode"] == "local"
    assert stored.data.params["config_sha256"]
    artifacts = {item.path for item in client.list_artifacts(run_id)}
    assert "config.json" in artifacts


@pytest.mark.skipif(
    importlib.util.find_spec("mlflow") is None,
    reason="MLflow is an optional test dependency",
)
def test_exception_marks_enabled_run_failed(tmp_path: Path) -> None:
    """A failed training body must terminate its own MLflow run as FAILED."""
    db_path = (tmp_path / "tracking.db").resolve()
    config = tracking.TrackingConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{db_path.as_posix()}",
        experiment_name="skyulf-sm12-failure",
    )

    with (
        pytest.raises(RuntimeError, match="training failed"),
        tracking.track_run(config, run_name="failed-run") as run,
    ):
        run.log_params({"step": "fit"})
        raise RuntimeError("training failed")

    client = tracking.make_tracking_client(config.tracking_uri)
    experiment = client.get_experiment_by_name(config.experiment_name)
    stored = client.search_runs([experiment.experiment_id])[0]
    assert stored.info.status == "FAILED"


@pytest.mark.skipif(
    importlib.util.find_spec("mlflow") is None,
    reason="MLflow is an optional test dependency",
)
def test_run_ids_are_isolated_for_concurrent_contexts(tmp_path: Path) -> None:
    """Concurrent contexts use distinct client-bound run IDs without global active-run state."""
    db_path = (tmp_path / "tracking.db").resolve()
    config = tracking.TrackingConfig(
        enabled=True,
        tracking_uri=f"sqlite:///{db_path.as_posix()}",
        experiment_name="skyulf-sm12-concurrent",
    )
    barrier = threading.Barrier(2)
    run_ids: list[str] = []

    def worker(name: str) -> None:
        with tracking.track_run(config, run_name=name) as run:
            barrier.wait(timeout=10)
            run.set_tags({"worker": name})
            run_id = run.run_id
            assert run_id
            run_ids.append(run_id)

    first = threading.Thread(target=worker, args=("one",))
    second = threading.Thread(target=worker, args=("two",))
    first.start()
    second.start()
    first.join(timeout=20)
    second.join(timeout=20)

    assert not first.is_alive() and not second.is_alive()
    assert len(run_ids) == 2
    assert len(set(run_ids)) == 2


@pytest.mark.skipif(
    importlib.util.find_spec("mlflow") is None,
    reason="MLflow is an optional test dependency",
)
def test_tracking_does_not_close_callers_active_run(tmp_path: Path) -> None:
    """A nested Skyulf run must leave a caller-owned fluent run active."""
    import mlflow  # ty: ignore[unresolved-import]

    db_path = (tmp_path / "tracking.db").resolve()
    uri = f"sqlite:///{db_path.as_posix()}"
    config = tracking.TrackingConfig(
        enabled=True,
        tracking_uri=uri,
        experiment_name="skyulf-sm12-caller-run",
    )
    client = tracking.make_tracking_client(uri)
    experiment_id = tracking.get_or_create_experiment(client, config.experiment_name)
    mlflow.set_tracking_uri(uri)

    with mlflow.start_run(experiment_id=experiment_id, run_name="caller") as caller:
        with tracking.track_run(config, run_name="skyulf"):
            active = mlflow.active_run()
            assert active is not None
            assert active.info.run_id == caller.info.run_id
        active = mlflow.active_run()
        assert active is not None
        assert active.info.run_id == caller.info.run_id


def test_warn_policy_preserves_body_result_when_logging_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warn mode exposes tracking loss while leaving the training body successful."""

    class BrokenClient:
        def create_run(self, **kwargs: object) -> object:
            return type("Created", (), {"info": type("Info", (), {"run_id": "warn-run"})()})()

        def log_metric(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("tracking unavailable")

        def set_terminated(self, *args: object, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(tracking, "make_tracking_client", lambda uri: BrokenClient())
    config = tracking.TrackingConfig(enabled=True, failure_policy="warn")
    with tracking.track_run(config, run_name="warn") as run:
        result = 42
        run.log_metrics({"rmse": 0.5})

    assert result == 42
    assert run.run_id == "warn-run"
    assert run.tracking_error == "tracking unavailable"
