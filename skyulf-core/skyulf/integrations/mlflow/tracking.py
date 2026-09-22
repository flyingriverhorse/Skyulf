"""Explicit, optional MLflow tracking lifecycle for Skyulf jobs.

The adapter uses ``MlflowClient`` directly instead of MLflow's process-global
fluent active-run state. This keeps independent jobs isolated and avoids
closing a run owned by a caller. MLflow is imported only after tracking is
enabled.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Configure optional MLflow tracking without changing local execution."""

    enabled: bool = False
    tracking_uri: str | None = None
    experiment_name: str | None = None
    failure_policy: Literal["raise", "warn"] = "raise"

    def __post_init__(self) -> None:
        """Reject ambiguous configuration before a client or network call exists."""
        if type(self.enabled) is not bool:
            raise TypeError("enabled must be a bool.")
        if self.tracking_uri is not None and (
            type(self.tracking_uri) is not str or not self.tracking_uri.strip()
        ):
            raise ValueError("tracking_uri must be a non-empty string or None.")
        if self.experiment_name is not None and (
            type(self.experiment_name) is not str or not self.experiment_name.strip()
        ):
            raise ValueError("experiment_name must be a non-empty string or None.")
        if self.failure_policy not in ("raise", "warn"):
            raise ValueError("failure_policy must be 'raise' or 'warn'.")


@dataclass(slots=True)
class TrackingRun:
    """Client-bound run handle exposed by :func:`track_run`."""

    client: Any = field(repr=False, default=None)
    run_id: str | None = None
    enabled: bool = False
    failure_policy: Literal["raise", "warn"] = "raise"
    tracking_error: str | None = None

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Log explicitly selected numeric metrics for this run."""
        if not self.enabled:
            return
        for key, value in _items(metrics, "metrics"):
            self._call(self.client.log_metric, self.run_id, key, float(value))

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log explicitly selected parameters, preserving MLflow's string values."""
        if not self.enabled:
            return
        for key, value in _items(params, "params"):
            self._call(self.client.log_param, self.run_id, key, str(value))

    def set_tags(self, tags: Mapping[str, object]) -> None:
        """Set explicitly selected run tags."""
        if not self.enabled:
            return
        for key, value in _items(tags, "tags"):
            self._call(self.client.set_tag, self.run_id, key, str(value))

    def log_config(
        self, config: Mapping[str, object], *, artifact_file: str = "config.json"
    ) -> None:
        """Store an explicit config artifact and its digest without auto-logging data."""
        if not self.enabled:
            return
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping.")
        if type(artifact_file) is not str or not artifact_file.strip():
            raise ValueError("artifact_file must be a non-empty string.")
        try:
            payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("config must be JSON serializable.") from exc
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self._call(self.client.log_dict, self.run_id, dict(config), artifact_file)
        self.log_params({"config_sha256": digest})

    def _call(self, operation: Any, *args: object, **kwargs: object) -> Any:
        """Run one client operation and apply the configured failure policy."""
        if not self.enabled:
            return None
        try:
            return operation(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - tracking policy must contain client failures
            self.tracking_error = str(exc) or type(exc).__name__
            if self.failure_policy == "raise":
                raise
            return None

    def _terminate(self, status: str) -> None:
        """Terminate this run without touching any process-global active run."""
        if self.enabled:
            self._call(self.client.set_terminated, self.run_id, status=status)


@contextmanager
def track_run(config: TrackingConfig, *, run_name: str) -> Iterator[TrackingRun]:
    """Yield an isolated no-op or MLflow run and close it with the body status."""
    if not isinstance(config, TrackingConfig):
        raise TypeError("config must be TrackingConfig.")
    if type(run_name) is not str or not run_name.strip():
        raise ValueError("run_name must be a non-empty string.")
    if not config.enabled:
        yield TrackingRun(failure_policy=config.failure_policy)
        return

    try:
        client = _make_client(config.tracking_uri)
        experiment_id = _get_or_create_experiment(client, config.experiment_name)
        created = client.create_run(experiment_id=experiment_id, run_name=run_name)
        run = TrackingRun(
            client=client,
            run_id=created.info.run_id,
            enabled=True,
            failure_policy=config.failure_policy,
        )
    except Exception as exc:  # noqa: BLE001 - configured policy controls optional service failure
        if config.failure_policy == "raise":
            raise
        run = TrackingRun(
            failure_policy=config.failure_policy,
            tracking_error=str(exc) or type(exc).__name__,
        )
        yield run
        return

    try:
        yield run
    except BaseException:
        with suppress(Exception):  # Preserve the original training exception.
            run._terminate("FAILED")
        raise
    else:
        run._terminate("FINISHED")


def _make_client(tracking_uri: str | None) -> Any:
    """Construct an MLflow client lazily, keeping the base import dependency-free."""
    from mlflow import (  # noqa: PLC0415 - optional dependency is lazy by design  # ty: ignore[unresolved-import]
        MlflowClient,  # ty: ignore[unresolved-import]
    )

    return MlflowClient(tracking_uri=tracking_uri)


def _get_or_create_experiment(client: Any, experiment_name: str | None) -> str:
    """Resolve an experiment through the supplied client without global MLflow state."""
    if experiment_name is None:
        return "0"
    existing = client.get_experiment_by_name(experiment_name)
    if existing is not None:
        return existing.experiment_id
    return client.create_experiment(experiment_name)


def _items(values: Mapping[str, Any], label: str) -> list[tuple[str, Any]]:
    """Validate explicit log mappings and return stable key/value pairs."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping.")
    result: list[tuple[str, Any]] = []
    for key, value in values.items():
        if type(key) is not str or not key:
            raise TypeError(f"{label} keys must be non-empty strings.")
        result.append((key, value))
    return result
