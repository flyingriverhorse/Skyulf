"""Package-internal lazy MLflow clients and experiment resolution.

Registry and tracking factories intentionally keep their different URI contracts.
Importing this module does not import MLflow or change its global configuration.
"""

from typing import Any


def require_mlflow() -> Any:
    """Import the optional MLflow package and expose a typed dependency failure."""
    try:
        import mlflow  # noqa: PLC0415  # ty: ignore[unresolved-import]
    except ImportError as exc:
        from .registry import RegistryDependencyError  # noqa: PLC0415 - lazy error type

        raise RegistryDependencyError(
            "MLflow registry support requires the optional 'mlflow' extra."
        ) from exc
    return mlflow


def make_registry_client(mlflow: Any, tracking_uri: str | None, registry_uri: str | None) -> Any:
    """Create a client with explicit tracking and registry stores."""
    return mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)


def make_tracking_client(tracking_uri: str | None) -> Any:
    """Construct an MLflow client lazily, keeping the base import dependency-free."""
    from mlflow import (  # noqa: PLC0415 - optional dependency is lazy by design  # ty: ignore[unresolved-import]
        MlflowClient,  # ty: ignore[unresolved-import]
    )

    return MlflowClient(tracking_uri=tracking_uri)


def get_or_create_experiment(client: Any, experiment_name: str | None) -> str:
    """Resolve an experiment through the supplied client without global MLflow state."""
    if experiment_name is None:
        return "0"
    existing = client.get_experiment_by_name(experiment_name)
    if existing is not None:
        return existing.experiment_id
    from mlflow.exceptions import (  # noqa: PLC0415 - optional dependency loaded on enabled tracking  # ty: ignore[unresolved-import]
        MlflowException,  # ty: ignore[unresolved-import]
    )

    try:
        return client.create_experiment(experiment_name)
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        existing = client.get_experiment_by_name(experiment_name)
        if existing is None:
            raise
        return existing.experiment_id
