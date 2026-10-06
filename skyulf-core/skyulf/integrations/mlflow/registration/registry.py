"""Explicit MLflow model registry publication and version resolution.

The adapter keeps registry selection separate from prediction. Callers publish
an already logged model explicitly, resolve an alias or version once, and carry
the resulting concrete ``models:/.../<version>`` URI through the rest of a job.
MLflow remains an optional dependency and is imported only when an operation is
requested.
"""

import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from skyulf.integrations.mlflow.shared._client import make_registry_client, require_mlflow

if TYPE_CHECKING:
    from ....inference.bundle import InferenceBundle
    from ....inference.local_pipeline import LocalPipelineArtifact

__all__ = [
    "RegistryAccessError",
    "RegistryDependencyError",
    "RegistryError",
    "RegistryModelNotFoundError",
    "RegistryOperationError",
    "ResolvedModel",
    "load_registered_bundle",
    "load_registered_local_pipeline",
    "load_run_local_pipeline",
    "register_model",
    "resolve_model",
]


def digest_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize historical digest keys at the artifact read boundary."""
    return metadata | {
        key: metadata.get(key, metadata.get(f"skyulf_{key}"))
        for key in ("bundle_digest", "local_pipeline_digest", "model_set_digest")
    }


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


def download_registered_package(
    mlflow: Any, client: Any, name: str, version: str, tracking_uri: str | None
) -> str:
    """Download the full pinned registry copy using its credential transport."""
    return mlflow.artifacts.download_artifacts(
        artifact_uri=_registered_package_uri(client, name, version),
        tracking_uri=tracking_uri,
        registry_uri=client._registry_uri,
    )


def _registered_package_uri(client: Any, name: str, version: str) -> str:
    """Select a pinned artifact URI without downloading any package contents.

    MLflow exposes no public getter for a client's resolved registry URI. Read
    its bound value once instead of consulting mutable global defaults. UC and
    workspace registries require their models transport to obtain scoped tokens;
    a bare cloud-storage download URI does not carry those credentials. OSS
    stores need their explicit download URI because MLflow's OSS models resolver
    can consult the global registry instead of the supplied registry URI.
    """
    registry_uri = client._registry_uri
    scheme = urlparse(registry_uri).scheme or registry_uri
    return (
        f"models:/{name}/{version}"
        if scheme in {"databricks", "databricks-uc"}
        else client.get_model_version_download_uri(name, version)
    )


def _artifact_uri(package_uri: str, relative: str) -> str:
    """Append a validated artifact path while retaining store query parameters."""
    parsed = urlparse(package_uri)
    return parsed._replace(path=parsed.path.rstrip("/") + "/" + relative).geturl()


def _download_registered_entry(
    mlflow: Any,
    client: Any,
    resolved: ResolvedModel,
    tracking_uri: str | None,
    uri: str,
    destination: Path,
) -> Path:
    """Translate remote file transport errors using the pinned registry identity."""
    try:
        return Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=uri,
                dst_path=str(destination),
                tracking_uri=tracking_uri,
                registry_uri=client._registry_uri,
            )
        )
    except Exception as exc:  # noqa: BLE001 - artifact transport boundary
        raise translate_error(exc, name=resolved.name, version=resolved.version) from exc


def _registered_metadata(
    mlflow: Any,
    client: Any,
    resolved: ResolvedModel,
    tracking_uri: str | None,
    root: Path,
) -> tuple[str, Any]:
    """Read pinned MLmodel metadata with the existing typed registry failures."""
    try:
        package_uri = _registered_package_uri(client, resolved.name, resolved.version)
    except Exception as exc:  # noqa: BLE001 - explicit OSS registry lookup boundary
        raise translate_error(exc, name=resolved.name, version=resolved.version) from exc
    metadata_path = _download_registered_entry(
        mlflow,
        client,
        resolved,
        tracking_uri,
        _artifact_uri(package_uri, "MLmodel"),
        root,
    )
    try:
        return package_uri, mlflow.models.Model.load(metadata_path)
    except Exception as exc:  # noqa: BLE001 - preserve metadata error translation
        raise translate_error(exc, name=resolved.name, version=resolved.version) from exc


def _payload_metadata(model: Any, key: str, digest: str | None) -> dict[str, Any]:
    """Validate direct-loader identity before resolving or downloading its payload."""
    metadata = digest_metadata(model.metadata or {})
    expected = {f"{key}_digest": digest}
    if key != "bundle":
        expected.update(skyulf_artifact_kind=key, skyulf_execution_scope="whole_frame_local")
    errors = {
        "bundle": "Packaged Skyulf bundle digest is missing or differs from resolved digest.",
        "local_pipeline": "Packaged Skyulf local pipeline metadata differs from resolved digest or scope.",
        "model_set": "Packaged model set kind, scope or digest differs from resolved identity.",
    }
    if any(metadata.get(name) != value for name, value in expected.items()):
        raise ValueError(errors[key])
    return metadata


@contextmanager
def downloaded_registered_payload(
    mlflow: Any,
    client: Any,
    resolved: ResolvedModel,
    tracking_uri: str | None,
    key: str,
) -> Iterator[tuple[Path, Any]]:
    """Fetch only metadata and its contained payload for direct Skyulf loaders.

    Failed loads remove their owned temporary root. Successful roots persist,
    matching MLflow downloads: model-set artifacts retain these paths for later
    scoring. Pyfunc and Spark callers continue downloading the complete package.
    """
    root = Path(tempfile.mkdtemp(prefix="skyulf-registry-payload-")).resolve()
    try:
        package_uri, model = _registered_metadata(mlflow, client, resolved, tracking_uri, root)
        _payload_metadata(model, key, resolved.digest)
        destination = _packaged_artifact_destination(root, model.flavors, key)
        relative = destination.relative_to(root).as_posix()
        if any(char in relative for char in "%?#:\\") or any(ord(char) < 32 for char in relative):
            raise ValueError("Skyulf artifact path must be a contained, unambiguous URI path.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # MLflow keeps the full artifact prefix for models:/, but only the
        # basename for direct download URIs returned by OSS registries.
        download_root = root if urlparse(package_uri).scheme == "models" else destination.parent
        try:
            downloaded = _download_registered_entry(
                mlflow,
                client,
                resolved,
                tracking_uri,
                _artifact_uri(package_uri, relative),
                download_root,
            )
        except RegistryModelNotFoundError as exc:
            raise ValueError(
                "MLflow model is missing the Skyulf bundle artifact directory."
            ) from exc
        if downloaded.resolve() != destination or not destination.is_dir():
            raise ValueError("Downloaded Skyulf artifact directory differs from its declared path.")
        yield root, model
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)
        raise


def load_registered_bundle(
    resolved: ResolvedModel,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> "InferenceBundle":
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
    validate_registry_options(resolved.name, tracking_uri, registry_uri)
    validate_concrete_version(resolved)
    if not isinstance(resolved.digest, str) or not resolved.digest.strip():
        raise ValueError("resolved must include the Skyulf bundle digest.")
    mlflow = require_mlflow()
    client = make_registry_client(mlflow, tracking_uri, registry_uri)
    with downloaded_registered_payload(mlflow, client, resolved, tracking_uri, "bundle") as (
        local_path,
        model,
    ):
        bundle_path = _packaged_bundle_path(local_path, model.flavors)
        from ....inference.bundle import (  # noqa: PLC0415 - lazy bundle dependency
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
) -> "LocalPipelineArtifact":
    """Load a trusted local pipeline from one concrete registered-model version.

    The package's declared artifact path and both digests are checked before the
    fitted pipeline is returned. A digest is an integrity check, not a signature.
    """
    if not isinstance(resolved, ResolvedModel):
        raise TypeError("resolved must be a ResolvedModel.")
    validate_registry_options(resolved.name, tracking_uri, registry_uri)
    validate_concrete_version(resolved)
    if not isinstance(resolved.digest, str) or not resolved.digest.strip():
        raise ValueError("resolved must include the Skyulf local pipeline digest.")
    mlflow = require_mlflow()
    client = make_registry_client(mlflow, tracking_uri, registry_uri)
    with downloaded_registered_payload(
        mlflow, client, resolved, tracking_uri, "local_pipeline"
    ) as (
        local_path,
        model,
    ):
        return load_local_package(local_path, model, resolved.digest)


def load_run_local_pipeline(
    model_uri: str,
    *,
    digest: str,
    tracking_uri: str | None = None,
) -> "LocalPipelineArtifact":
    """Load a trusted unregistered run package with the registered loader's checks."""
    _parse_runs_uri(model_uri)
    if not isinstance(digest, str) or not digest.strip():
        raise ValueError("Run artifact requires a concrete local pipeline digest.")
    mlflow = require_mlflow()
    local_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        tracking_uri=tracking_uri,
    )
    model = mlflow.models.Model.load(Path(local_path))
    return load_local_package(Path(local_path), model, digest)


def load_local_package(local_path: Path, model: Any, digest: str) -> "LocalPipelineArtifact":
    """Validate shared run and registry metadata, contained paths and fitted identity."""
    metadata = _payload_metadata(model, "local_pipeline", digest)
    artifact_path = packaged_artifact_path(Path(local_path), model.flavors, "local_pipeline")
    from ....inference.local_pipeline import (  # noqa: PLC0415 - lazy pickle dependency
        load_local_pipeline,
    )

    artifact = load_local_pipeline(artifact_path)
    if (
        artifact.manifest.pipeline_sha256 != digest
        or artifact.manifest.fitted_engine != metadata.get("skyulf_fitted_engine")
    ):
        raise ValueError("Loaded Skyulf local pipeline identity differs from resolved package.")
    return artifact


def _packaged_bundle_path(package: Path, flavors: dict[str, Any]) -> Path:
    """Find the declared bundle directory and reject paths escaping the package."""
    return packaged_artifact_path(package, flavors, "bundle")


def packaged_artifact_path(package: Path, flavors: dict[str, Any], key: str) -> Path:
    """Locate a declared artifact directory without allowing package traversal."""
    candidate = _packaged_artifact_destination(package, flavors, key)
    if not candidate.is_dir():
        raise ValueError("MLflow model is missing the Skyulf bundle artifact directory.")
    return candidate


def _packaged_artifact_destination(package: Path, flavors: dict[str, Any], key: str) -> Path:
    """Validate the declared package-relative path before any payload transfer."""
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
    Only the ``MLmodel`` metadata file is downloaded; fitted package integrity
    is verified by the separate artifact loaders when the model is consumed.
    ``tracking_uri`` and ``registry_uri`` are explicit client settings; neither
    operation uses MLflow's process-global active run.
    """
    _validate_reference(name, alias, version, tracking_uri, registry_uri)
    mlflow = require_mlflow()
    client = make_registry_client(mlflow, tracking_uri, registry_uri)
    model_version = _resolve_version(client, name, alias, version)

    concrete_version = str(model_version.version)
    model_uri = f"models:/{name}/{concrete_version}"
    try:
        package_uri = urlparse(_registered_package_uri(client, name, concrete_version))
        metadata_uri = package_uri._replace(path=package_uri.path.rstrip("/") + "/MLmodel").geturl()
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=metadata_uri,
            tracking_uri=tracking_uri,
            registry_uri=client._registry_uri,
        )
        model = mlflow.models.Model.load(Path(local_path))
    except Exception as exc:  # noqa: BLE001 - artifact metadata is another registry boundary
        raise translate_error(exc, name=name, version=concrete_version) from exc
    metadata = digest_metadata(model.metadata or {})
    digest = (
        metadata.get("bundle_digest")
        or metadata.get("local_pipeline_digest")
        or metadata.get("model_set_digest")
    )
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
    tags: dict[str, str] | None = None,
) -> Any:
    """Publish a run artifact as a new registered-model version explicitly.

    ``model_uri`` must be a ``runs:/<run_id>/<artifact_path>`` URI produced by
    the packaging step. This function never changes an alias and is never called
    implicitly during prediction. It returns MLflow's created model-version
    object so callers can persist its concrete version and status.
    """
    validate_registry_options(name, tracking_uri, registry_uri)
    run_id = _parse_runs_uri(model_uri)
    mlflow = require_mlflow()
    client = make_registry_client(mlflow, tracking_uri, registry_uri)
    try:
        client.create_registered_model(name)
    except Exception as exc:  # noqa: BLE001 - existing registration is the only benign result
        if error_code(exc) not in {"RESOURCE_ALREADY_EXISTS", "ALREADY_EXISTS"}:
            raise translate_error(exc, name=name, version="new") from exc
    try:
        options = {"tags": tags} if tags is not None else {}
        return client.create_model_version(name=name, source=model_uri, run_id=run_id, **options)
    except Exception as exc:  # noqa: BLE001 - translate MLflow's backend errors at the boundary
        raise translate_error(exc, name=name, version="new") from exc


def _validate_reference(
    name: str,
    alias: str | None,
    version: str | int | None,
    tracking_uri: str | None,
    registry_uri: str | None,
) -> None:
    """Validate selector cardinality and UC-qualified names before any network call."""
    validate_registry_options(name, tracking_uri, registry_uri)
    if (alias is None) == (version is None):
        raise ValueError("Provide exactly one of alias or version.")
    if alias is not None and (type(alias) is not str or not alias.strip()):
        raise ValueError("alias must be a non-empty string.")
    if version is not None and (
        type(version) not in (str, int)
        or not str(version).isascii()
        or not str(version).isdigit()
        or int(version) <= 0
    ):
        raise ValueError("version must be a positive integer or positive ASCII decimal string.")


def validate_registry_options(
    name: str, tracking_uri: str | None, registry_uri: str | None
) -> None:
    """Validate registry names and URI options without constructing an MLflow client."""
    if type(name) is not str or not name.strip() or "/" in name:
        raise ValueError("name must be a non-empty model name without '/'.")
    parts = name.split(".")
    if any(not part for part in parts) or len(parts) not in (1, 3):
        raise ValueError("name must be model or catalog.schema.model.")
    _validate_store_uris(tracking_uri, registry_uri)
    if _is_unity_catalog(registry_uri) and len(parts) != 3:
        raise ValueError("Unity Catalog names must use catalog.schema.model.")


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


def error_code(exc: Exception) -> str:
    """Read an MLflow error code without importing optional exception classes."""
    value = getattr(exc, "error_code", "")
    return str(getattr(value, "value", value)).upper()


def translate_error(exc: Exception, *, name: str, version: str) -> RegistryError:
    """Translate backend errors into stable caller-facing registry categories."""
    code = error_code(exc)
    if code in {"RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"}:
        return RegistryModelNotFoundError(f"Model '{name}' version '{version}' was not found.")
    if code in {"PERMISSION_DENIED", "UNAUTHENTICATED", "UNAUTHORIZED"}:
        return RegistryAccessError(f"Access denied for model '{name}' version '{version}'.")
    return RegistryOperationError(f"MLflow registry operation failed for model '{name}'.")


def validate_concrete_version(resolved: ResolvedModel) -> None:
    """Require a positive pinned version whose URI agrees with its identity."""
    if (
        not isinstance(resolved.version, str)
        or not resolved.version.isascii()
        or not resolved.version.isdigit()
        or int(resolved.version) <= 0
        or resolved.model_uri != f"models:/{resolved.name}/{resolved.version}"
    ):
        raise ValueError("resolved must identify a concrete positive model version.")


def _validate_store_uris(tracking_uri: str | None, registry_uri: str | None) -> None:
    """Reject malformed explicit store URIs before constructing a client."""
    for value, label in ((tracking_uri, "tracking_uri"), (registry_uri, "registry_uri")):
        if value is not None and (type(value) is not str or not value.strip()):
            raise ValueError(f"{label} must be a non-empty string or None.")


def _resolve_version(client: Any, name: str, alias: str | None, version: str | int | None) -> Any:
    """Resolve one registry selector and translate missing alias errors."""
    try:
        model_version = (
            client.get_model_version_by_alias(name, alias)
            if alias is not None
            else client.get_model_version(name, str(version))
        )
    except Exception as exc:  # noqa: BLE001 - translate MLflow's backend errors at the boundary
        if (
            alias is not None
            and error_code(exc) == "INVALID_PARAMETER_VALUE"
            and "alias" in str(exc).lower()
            and "not found" in str(exc).lower()
        ):
            raise RegistryModelNotFoundError(
                f"Model '{name}' alias '{alias}' was not found."
            ) from exc
        raise translate_error(exc, name=name, version=str(version or alias)) from exc
    return model_version
