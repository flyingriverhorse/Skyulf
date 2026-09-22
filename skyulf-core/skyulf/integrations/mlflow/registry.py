"""Explicit MLflow model registry publication and version resolution.

The adapter keeps registry selection separate from prediction. Callers publish
an already logged model explicitly, resolve an alias or version once, and carry
the resulting concrete ``models:/.../<version>`` URI through the rest of a job.
MLflow remains an optional dependency and is imported only when an operation is
requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "RegistryAccessError",
    "RegistryDependencyError",
    "RegistryError",
    "RegistryModelNotFoundError",
    "RegistryOperationError",
    "ResolvedModel",
    "register_model",
    "resolve_model",
]


class RegistryError(RuntimeError):
    """Base class for typed MLflow registry failures."""


class RegistryDependencyError(RegistryError):
    """The optional MLflow package is unavailable in the current environment."""


class RegistryModelNotFoundError(RegistryError):
    """The requested registered model or concrete version does not exist."""


class RegistryAccessError(RegistryError):
    """The registry rejected the operation because credentials or permissions are insufficient."""


class RegistryOperationError(RegistryError):
    """The registry returned an error that is neither a missing resource nor an access failure."""


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """An immutable model reference after alias selection has been pinned."""

    name: str
    version: str
    model_uri: str
    signature: Any | None
    digest: str | None


def resolve_model(
    name: str,
    *,
    alias: str | None = None,
    version: str | int | None = None,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> ResolvedModel:
    """Resolve exactly one alias or version and return a concrete model URI.

    Alias resolution is performed once. The returned ``model_uri`` contains the
    resolved version, so later inference does not follow a moving alias.
    ``tracking_uri`` and ``registry_uri`` are explicit client settings; neither
    operation uses MLflow's process-global active run.
    """
    _validate_reference(name, alias, version, tracking_uri, registry_uri)
    mlflow = _require_mlflow()
    client = _make_client(mlflow, tracking_uri, registry_uri)
    try:
        model_version = (
            client.get_model_version_by_alias(name, alias)
            if alias is not None
            else client.get_model_version(name, str(version))
        )
    except Exception as exc:  # noqa: BLE001 - translate MLflow's backend errors at the boundary
        raise _translate_error(exc, name=name, version=str(version or alias)) from exc

    concrete_version = str(model_version.version)
    model_uri = f"models:/{name}/{concrete_version}"
    # The registry response carries the source URI that was recorded at
    # publication time.  Resolving that source avoids relying on MLflow's
    # process-global registry store when a caller supplied explicit stores.
    source_uri = getattr(model_version, "source", None)
    artifact_uri = source_uri if isinstance(source_uri, str) and source_uri else model_uri
    try:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        model = mlflow.models.Model.load(Path(local_path))
    except Exception as exc:  # noqa: BLE001 - artifact metadata is another registry boundary
        raise _translate_error(exc, name=name, version=concrete_version) from exc
    metadata = model.metadata or {}
    digest = metadata.get("skyulf_bundle_digest")
    return ResolvedModel(
        name=name,
        version=concrete_version,
        model_uri=model_uri,
        signature=model.signature,
        digest=digest if isinstance(digest, str) else None,
    )


def register_model(
    model_uri: str,
    name: str,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> Any:
    """Publish a run artifact as a new registered-model version explicitly.

    ``model_uri`` must be a ``runs:/<run_id>/<artifact_path>`` URI produced by
    the packaging step. This function never changes an alias and is never called
    implicitly during prediction. It returns MLflow's created model-version
    object so callers can persist its concrete version and status.
    """
    _validate_registry_options(name, tracking_uri, registry_uri)
    run_id = _parse_runs_uri(model_uri)
    mlflow = _require_mlflow()
    client = _make_client(mlflow, tracking_uri, registry_uri)
    try:
        client.create_registered_model(name)
    except Exception as exc:  # noqa: BLE001 - existing registration is the only benign result
        if _error_code(exc) not in {"RESOURCE_ALREADY_EXISTS", "ALREADY_EXISTS"}:
            raise _translate_error(exc, name=name, version="new") from exc
    try:
        return client.create_model_version(name=name, source=model_uri, run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - translate MLflow's backend errors at the boundary
        raise _translate_error(exc, name=name, version="new") from exc


def _require_mlflow() -> Any:
    """Import the optional MLflow package and expose a typed dependency failure."""
    try:
        import mlflow  # noqa: PLC0415  # ty: ignore[unresolved-import]
    except ImportError as exc:
        raise RegistryDependencyError(
            "MLflow registry support requires the optional 'mlflow' extra."
        ) from exc
    return mlflow


def _make_client(mlflow: Any, tracking_uri: str | None, registry_uri: str | None) -> Any:
    """Create a client with explicit tracking and registry stores."""
    return mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)


def _validate_reference(
    name: str,
    alias: str | None,
    version: str | int | None,
    tracking_uri: str | None,
    registry_uri: str | None,
) -> None:
    """Validate selector cardinality and UC-qualified names before any network call."""
    _validate_registry_options(name, tracking_uri, registry_uri)
    if (alias is None) == (version is None):
        raise ValueError("Provide exactly one of alias or version.")
    if alias is not None and (type(alias) is not str or not alias.strip()):
        raise ValueError("alias must be a non-empty string.")
    if version is not None and (
        isinstance(version, bool) or type(version) not in (str, int) or not str(version).strip()
    ):
        raise ValueError("version must be a non-empty string or positive integer.")
    if version is not None and str(version).isdigit() and int(version) <= 0:
        raise ValueError("version must be a positive integer.")


def _validate_registry_options(
    name: str, tracking_uri: str | None, registry_uri: str | None
) -> None:
    """Validate registry names and URI options without constructing an MLflow client."""
    if type(name) is not str or not name.strip() or "/" in name:
        raise ValueError("name must be a non-empty model name without '/'.")
    parts = name.split(".")
    if any(not part for part in parts) or len(parts) not in (1, 3):
        raise ValueError("name must be model or catalog.schema.model.")
    if _is_unity_catalog(registry_uri) and len(parts) != 3:
        raise ValueError("Unity Catalog names must use catalog.schema.model.")
    for value, label in ((tracking_uri, "tracking_uri"), (registry_uri, "registry_uri")):
        if value is not None and (type(value) is not str or not value.strip()):
            raise ValueError(f"{label} must be a non-empty string or None.")


def _is_unity_catalog(registry_uri: str | None) -> bool:
    """Identify the explicit MLflow Unity Catalog registry scheme."""
    return registry_uri is not None and registry_uri.startswith("databricks-uc")


def _parse_runs_uri(model_uri: str) -> str:
    """Extract and validate the source run ID from one packaging URI."""
    if type(model_uri) is not str or not model_uri.startswith("runs:/"):
        raise ValueError("model_uri must be a runs:/<run_id>/<artifact_path> URI.")
    value = model_uri.removeprefix("runs:/")
    run_id, separator, artifact_path = value.partition("/")
    if not run_id or not separator or not artifact_path:
        raise ValueError("model_uri must include a run ID and artifact path.")
    return run_id


def _error_code(exc: Exception) -> str:
    """Read an MLflow error code without importing optional exception classes."""
    value = getattr(exc, "error_code", "")
    return str(getattr(value, "value", value)).upper()


def _translate_error(exc: Exception, *, name: str, version: str) -> RegistryError:
    """Translate backend errors into stable caller-facing registry categories."""
    code = _error_code(exc)
    if code in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"}:
        return RegistryModelNotFoundError(f"Model '{name}' version '{version}' was not found.")
    if code in {"PERMISSION_DENIED", "UNAUTHENTICATED", "UNAUTHORIZED"}:
        return RegistryAccessError(f"Access denied for model '{name}' version '{version}'.")
    return RegistryOperationError(f"MLflow registry operation failed for model '{name}'.")
