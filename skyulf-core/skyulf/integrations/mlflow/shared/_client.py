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
        from ..registration.registry import (  # noqa: PLC0415 - lazy error type
            RegistryDependencyError,
        )

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


# Databricks groups experiments by this tag; this value lists them under "Machine learning".
EXPERIMENT_KIND_TAG = "mlflow.experimentKind"
EXPERIMENT_KIND = "custom_model_development"


def get_or_create_experiment(client: Any, experiment_name: str | None) -> str:
    """Resolve an experiment through the supplied client without global MLflow state.

    New experiments get the ``custom_model_development`` kind tag; an existing one
    gets it only when it has no kind yet.
    """
    if experiment_name is None:
        return "0"
    from mlflow.exceptions import (  # noqa: PLC0415 - optional dependency loaded on enabled tracking  # ty: ignore[unresolved-import]
        MlflowException,  # ty: ignore[unresolved-import]
    )

    existing = client.get_experiment_by_name(experiment_name)
    if existing is not None:
        _tag_experiment_kind(client, existing, MlflowException)
        return existing.experiment_id
    try:
        return client.create_experiment(
            experiment_name, tags={EXPERIMENT_KIND_TAG: EXPERIMENT_KIND}
        )
    except MlflowException as exc:
        if exc.error_code != "RESOURCE_ALREADY_EXISTS":
            raise
        existing = client.get_experiment_by_name(experiment_name)
        if existing is None:
            raise
        return existing.experiment_id


def _tag_experiment_kind(client: Any, experiment: Any, error_type: type[Exception]) -> None:
    """Add the kind tag to an older experiment; lacking permission only skips the tag."""
    tags = getattr(experiment, "tags", None)
    if not isinstance(tags, dict) or EXPERIMENT_KIND_TAG in tags:
        return
    try:
        client.set_experiment_tag(experiment.experiment_id, EXPERIMENT_KIND_TAG, EXPERIMENT_KIND)
    except error_type as exc:
        if getattr(exc, "error_code", None) != "PERMISSION_DENIED":
            raise
