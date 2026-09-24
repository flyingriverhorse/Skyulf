"""Explicit MLflow model registry publication and version resolution.

The adapter keeps registry selection separate from prediction. Callers publish
an already logged model explicitly, resolve an alias or version once, and carry
the resulting concrete ``models:/.../<version>`` URI through the rest of a job.
MLflow remains an optional dependency and is imported only when an operation is
requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...inference.bundle import InferenceBundle
    from ...inference.local_pipeline import LocalPipelineArtifact

__all__ = [
    "RegistryAccessError",
    "RegistryDependencyError",
    "RegistryError",
    "RegistryModelNotFoundError",
    "RegistryOperationError",
    "ResolvedModel",
    "load_registered_bundle",
    "load_registered_local_pipeline",
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


def load_registered_bundle(
    resolved: ResolvedModel,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> InferenceBundle:
    """Load a trusted packaged bundle using a previously pinned registry version.

    The client uses explicit stores without changing MLflow's global URIs or
    active run. The concrete name/version selects the package even if an alias
    has since moved. Its pyfunc artifact map locates the bundle; package and
    bundle digests must agree with the resolved identity.

    This operation deserializes pickle. Only use trusted registry artifacts;
    digest checks do not authenticate producers. Local MLflow stores are covered
    by integration tests; remote Unity Catalog execution requires deployment
    validation.
    """
    if not isinstance(resolved, ResolvedModel):
        raise TypeError("resolved must be a ResolvedModel.")
    _validate_registry_options(resolved.name, tracking_uri, registry_uri)
    if (
        not isinstance(resolved.version, str)
        or not resolved.version.isascii()
        or not resolved.version.isdigit()
        or int(resolved.version) <= 0
        or resolved.model_uri != f"models:/{resolved.name}/{resolved.version}"
    ):
        raise ValueError("resolved must identify a concrete positive model version.")
    if not isinstance(resolved.digest, str) or not resolved.digest.strip():
        raise ValueError("resolved must include the Skyulf bundle digest.")
    mlflow = _require_mlflow()
    client = _make_client(mlflow, tracking_uri, registry_uri)
    try:
        version = client.get_model_version(resolved.name, resolved.version)
        source = getattr(version, "source", None)
        artifact_uri = source if isinstance(source, str) and source else resolved.model_uri
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        model = mlflow.models.Model.load(Path(local_path))
    except Exception as exc:  # noqa: BLE001 - translate registry and artifact transport failures
        raise _translate_error(exc, name=resolved.name, version=resolved.version) from exc
    metadata = model.metadata or {}
    if metadata.get("skyulf_bundle_digest") != resolved.digest:
        raise ValueError(
            "Packaged Skyulf bundle digest is missing or differs from resolved digest."
        )
    bundle_path = _packaged_bundle_path(Path(local_path), model.flavors)
    from ...inference.bundle import (  # noqa: PLC0415 - lazy bundle dependency
        load_bundle,
    )

    bundle = load_bundle(bundle_path)
    if bundle.semantic_digest != resolved.digest:
        raise ValueError("Loaded Skyulf bundle digest differs from resolved digest.")
    return bundle


def load_registered_local_pipeline(
    resolved: ResolvedModel,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> LocalPipelineArtifact:
    """Load a trusted local pipeline from one concrete registered-model version.

    The package's declared artifact path and both digests are checked before the
    fitted pipeline is returned. A digest is an integrity check, not a signature.
    """
    if not isinstance(resolved, ResolvedModel):
        raise TypeError("resolved must be a ResolvedModel.")
    _validate_registry_options(resolved.name, tracking_uri, registry_uri)
    if (
        not isinstance(resolved.version, str)
        or not resolved.version.isascii()
        or not resolved.version.isdigit()
        or int(resolved.version) <= 0
        or resolved.model_uri != f"models:/{resolved.name}/{resolved.version}"
    ):
        raise ValueError("resolved must identify a concrete positive model version.")
    if not isinstance(resolved.digest, str) or not resolved.digest.strip():
        raise ValueError("resolved must include the Skyulf local pipeline digest.")
    mlflow = _require_mlflow()
    client = _make_client(mlflow, tracking_uri, registry_uri)
    try:
        version = client.get_model_version(resolved.name, resolved.version)
        source = getattr(version, "source", None)
        artifact_uri = source if isinstance(source, str) and source else resolved.model_uri
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
        )
        model = mlflow.models.Model.load(Path(local_path))
    except Exception as exc:  # noqa: BLE001 - translate registry and transport failures
        raise _translate_error(exc, name=resolved.name, version=resolved.version) from exc
    metadata = model.metadata or {}
    if (
        metadata.get("skyulf_artifact_kind") != "local_pipeline"
        or metadata.get("skyulf_execution_scope") != "whole_frame_local"
        or metadata.get("skyulf_local_pipeline_digest") != resolved.digest
    ):
        raise ValueError(
            "Packaged Skyulf local pipeline metadata differs from resolved digest or scope."
        )
    artifact_path = _packaged_artifact_path(Path(local_path), model.flavors, "local_pipeline")
    from ...inference.local_pipeline import (  # noqa: PLC0415 - lazy pickle dependency
        load_local_pipeline,
    )

    artifact = load_local_pipeline(artifact_path)
    if (
        artifact.manifest.pipeline_sha256 != resolved.digest
        or artifact.manifest.fitted_engine != metadata.get("skyulf_fitted_engine")
    ):
        raise ValueError("Loaded Skyulf local pipeline identity differs from resolved package.")
    return artifact


def _packaged_bundle_path(package: Path, flavors: dict[str, Any]) -> Path:
    """Find the declared bundle directory and reject paths escaping the package."""
    return _packaged_artifact_path(package, flavors, "bundle")


def _packaged_artifact_path(package: Path, flavors: dict[str, Any], key: str) -> Path:
    """Locate a declared artifact directory without allowing package traversal."""
    try:
        relative = flavors["python_function"]["artifacts"][key]["path"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"MLflow model is missing the Skyulf {key} artifact path.") from exc
    if (
        not isinstance(relative, str)
        or not relative.strip()
        or Path(relative).is_absolute()
        or PureWindowsPath(relative).anchor
        or ".." in PureWindowsPath(relative).parts
    ):
        raise ValueError(
            "Skyulf bundle artifact path must be relative and contained in the package."
        )
    root = package.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("Skyulf bundle artifact path must be contained in the package.")
    if not candidate.is_dir():
        raise ValueError("MLflow model is missing the Skyulf bundle artifact directory.")
    return candidate


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
        if (
            alias is not None
            and _error_code(exc) == "INVALID_PARAMETER_VALUE"
            and "alias" in str(exc).lower()
            and "not found" in str(exc).lower()
        ):
            raise RegistryModelNotFoundError(
                f"Model '{name}' alias '{alias}' was not found."
            ) from exc
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
    digest = metadata.get("skyulf_bundle_digest") or metadata.get("skyulf_local_pipeline_digest")
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
