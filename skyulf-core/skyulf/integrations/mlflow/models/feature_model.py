"""Package unchanged Skyulf pyfunc models through native Feature Engineering."""

import os
import shutil
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4

import mlflow  # ty: ignore[unresolved-import]
import numpy as np

from ....inference.fitted_pipeline import load_pipeline
from ....inference.model_set import load_model_set
from ...databricks.feature_store.config import FeatureTrainingSpec
from ...databricks.feature_store.lifecycle_config import (
    deserialize_feature_spec,
    parse_feature_binding,
    serialize_feature_spec,
)
from ...databricks.feature_store.online_policy import ONLINE_FEATURES_KEY, OnlineFeaturePolicy
from ...databricks.feature_store.runtime import feature_engineering_client
from ..shared._client import make_tracking_client
from ..shared._model_metadata import mlflow_dtype, normalized_dtype, scrub_local_artifact_uri
from ..shared._nullable_transport import TRANSPORT_KEY, transport_spec
from ._feature_package_spec import validate_native_training_set, validate_saved_feature_spec
from .pipeline_model import (
    pipeline_model_save_options,
    validate_model_destination,
)

FEATURE_STORE_KEY = "skyulf_feature_store"
RAW_MODEL_PATH_KEY = "skyulf_feature_store_raw_model_path"
FEATURE_SPEC_DIGEST_KEY = "skyulf_feature_store_spec_sha256"
_RUN_LOCK = threading.RLock()


def _contained(root: Path, relative: Any) -> Path:
    """Require portable relative paths within the downloaded package."""
    if not isinstance(relative, str) or not relative or PureWindowsPath(relative).anchor:
        raise ValueError("Feature model paths must be relative and contained.")
    if ".." in PureWindowsPath(relative).parts or any(mark in relative for mark in ("%", "?", "#")):
        raise ValueError("Feature model paths must be relative and contained.")
    candidate = (root / relative).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError("Feature model paths must be contained in the package.")
    return candidate


def _raw_model_path(root: Path, outer: Any) -> Path:
    """Derive the SDK raw model from the wrapper's actual loader data path."""
    flavor = outer.flavors.get("python_function", {})
    if flavor.get("loader_module") != "databricks.feature_store.mlflow_model":
        raise ValueError("Feature model must use the native Feature Engineering loader.")
    data = _contained(root, flavor.get("data"))
    raw = _contained(root, (data / "raw_model").relative_to(root).as_posix())
    if not (raw / "MLmodel").is_file():
        raise ValueError("Feature model is missing its raw MLflow model.")
    return raw


def _validate_envelope(outer: Any, raw: Any) -> None:
    """Keep the fitted identity, runtime certificate and named output contract intact."""
    metadata = outer.metadata or {}
    raw_metadata = raw.metadata or {}
    if raw_metadata.get("skyulf_artifact_kind") not in {"local_pipeline", "model_set"}:
        raise ValueError("Feature model must contain a fitted Skyulf model.")
    envelope_keys = {FEATURE_STORE_KEY, RAW_MODEL_PATH_KEY, FEATURE_SPEC_DIGEST_KEY}
    if {key: value for key, value in metadata.items() if key not in envelope_keys} != raw_metadata:
        raise ValueError("Feature model outer and raw metadata differ.")
    if raw_metadata.get(TRANSPORT_KEY) is not None:
        raise ValueError(
            "Feature lookup cannot preserve nullable primitive transport before Arrow."
        )
    _validate_signatures(outer, raw)


def _validate_signatures(outer: Any, raw: Any) -> None:
    """Retain all named output columns and typed prediction parameters."""
    if raw.signature is None or outer.signature is None:
        raise ValueError("Feature model requires named raw and outer signatures.")
    if raw.signature.inputs is None:
        raise ValueError("Feature model requires a named raw input signature.")
    if raw.signature.outputs != outer.signature.outputs:
        raise ValueError("Feature model outer and raw output signatures differ.")
    expected = [] if raw.signature.params is None else raw.signature.params.to_dict()
    actual = [] if outer.signature.params is None else outer.signature.params.to_dict()
    if any(parameter not in actual for parameter in expected):
        raise ValueError("Feature model outer signature is missing Skyulf prediction params.")


def _outer_worker_wheels(root: Path, raw: Path, *, copy: bool = False) -> None:
    """Preserve SDK outer environment pins to the same exact nested worker wheel."""
    requirements = (raw / "requirements.txt").read_text(encoding="utf-8").splitlines()
    outer_requirements = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    for requirement in requirements:
        if not requirement.startswith("code/"):
            continue
        source, target = _contained(raw, requirement), _contained(root, requirement)
        if source.suffix != ".whl" or requirement not in outer_requirements:
            raise ValueError("Feature model outer worker requirements differ from the raw model.")
        if copy:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if not target.is_file() or target.read_bytes() != source.read_bytes():
            raise ValueError("Feature model outer worker wheel differs from the raw model.")


def feature_package_models(local_path: str | Path) -> tuple[Any, Any, Path]:
    """Validate a complete native feature envelope before exposing its raw package."""
    root = Path(local_path).resolve()
    outer = mlflow.models.Model.load(root)
    metadata = outer.metadata or {}
    if FEATURE_STORE_KEY not in metadata:
        raise ValueError("Feature model is missing its immutable lookup binding.")
    binding = parse_feature_binding(metadata[FEATURE_STORE_KEY])
    raw_path = _raw_model_path(root, outer)
    if _contained(root, metadata.get(RAW_MODEL_PATH_KEY)) != raw_path:
        raise ValueError("Feature model raw path differs from its native loader layout.")
    raw = mlflow.models.Model.load(raw_path)
    _validate_envelope(outer, raw)
    _outer_worker_wheels(root, raw_path)
    digest = validate_saved_feature_spec(
        raw_path.parent / "feature_spec.yaml",
        deserialize_feature_spec(binding["lookup_spec"]),
        {column.name: column.type.name for column in raw.signature.inputs},
    )
    if metadata.get(FEATURE_SPEC_DIGEST_KEY) != digest:
        raise ValueError("Feature model spec digest differs from saved lookup instructions.")
    _validate_online_contract(raw, deserialize_feature_spec(binding["lookup_spec"]))
    return outer, raw, raw_path


@contextmanager
def _active_logging_run(run_id: str, tracking_uri: str | None) -> Iterator[Any]:
    """Bind SDK fluent logging explicitly and restore the caller's active run and URI."""
    with _RUN_LOCK:
        previous_uri = mlflow.get_tracking_uri()
        uri = previous_uri if tracking_uri is None else tracking_uri
        active = mlflow.active_run()
        if active is not None and (active.info.run_id != run_id or uri != previous_uri):
            raise ValueError("Feature logging requires the matching active run and tracking URI.")
        tracking = make_tracking_client(uri)
        status = tracking.get_run(run_id).info.status
        if active is not None:
            yield tracking
            return
        environment_run = os.environ.get("MLFLOW_RUN_ID")
        try:
            mlflow.set_tracking_uri(uri)
            mlflow.start_run(run_id=run_id)
            yield tracking
        finally:
            try:
                _end_owned_run(run_id, status)
            finally:
                mlflow.set_tracking_uri(previous_uri)
                _restore_environment_run(environment_run)


def _end_owned_run(run_id: str, status: str) -> None:
    """Detach only the run introduced by this context while preserving its status."""
    current = mlflow.active_run()
    if current is not None and current.info.run_id == run_id:
        mlflow.end_run(status=status)


def _restore_environment_run(previous: str | None) -> None:
    """Undo MLflow end_run's removal of the caller's environment run selector."""
    if previous is None:
        os.environ.pop("MLFLOW_RUN_ID", None)
    else:
        os.environ["MLFLOW_RUN_ID"] = previous


def _validate_contract(
    training_set: Any,
    lookup_spec: FeatureTrainingSpec,
    lookup_binding: dict[str, Any],
    columns: list[tuple[str, str]],
) -> dict[str, Any]:
    """Reject unsafe transport or different native lineage before invoking the SDK."""
    binding = parse_feature_binding(lookup_binding)
    if binding["lookup_spec"] != serialize_feature_spec(lookup_spec):
        raise ValueError("Feature lookup binding differs from lookup_spec.")
    if transport_spec(columns) is not None:
        raise ValueError(
            "Feature lookup cannot preserve nullable primitive transport before Arrow."
        )
    validate_native_training_set(
        training_set,
        lookup_spec,
        {name: mlflow_dtype(normalized_dtype(dtype)).name for name, dtype in columns},
    )
    return binding


def _finish_feature_package(
    local: Path,
    options: dict[str, Any],
    binding: dict[str, Any],
) -> None:
    """Enrich the SDK outer model while verifying its preserved raw fitted package."""
    outer = mlflow.models.Model.load(local)
    raw_path = _raw_model_path(local, outer)
    raw = mlflow.models.Model.load(raw_path)
    if raw.metadata != options["metadata"] or raw.signature != options["signature"]:
        raise ValueError("Feature Engineering changed the saved Skyulf raw model contract.")
    spec = deserialize_feature_spec(binding["lookup_spec"])
    digest = validate_saved_feature_spec(
        raw_path.parent / "feature_spec.yaml",
        spec,
        {column.name: column.type.name for column in raw.signature.inputs},
    )
    key = options["metadata"]["skyulf_artifact_kind"]
    scrub_local_artifact_uri(raw_path, key)
    _outer_worker_wheels(local, raw_path, copy=True)
    outer.metadata = {
        **options["metadata"],
        FEATURE_STORE_KEY: binding,
        RAW_MODEL_PATH_KEY: raw_path.relative_to(local).as_posix(),
        FEATURE_SPEC_DIGEST_KEY: digest,
    }
    outer.save(str(local / "MLmodel"))
    feature_package_models(local)


def _log_feature_options(
    options: dict[str, Any],
    *,
    training_set: Any,
    binding: dict[str, Any],
    run_id: str,
    artifact_path: str,
    tracking_uri: str | None,
    client: Any,
) -> str:
    """Use the official pyfunc flavor boundary then publish a complete run artifact."""
    kwargs = {name: value for name, value in options.items() if name != "python_model"}
    signature = options["signature"]
    params = (
        {}
        if signature.params is None
        else {
            param.name: np.int64(param.default) if param.dtype.name == "long" else param.default
            for param in signature.params.params
        }
    )
    with _active_logging_run(run_id, tracking_uri) as tracking:
        sdk_path = f"skyulf_feature_source_{uuid4().hex}"
        logged = feature_engineering_client(client).log_model(
            model=options["python_model"],
            flavor=mlflow.pyfunc,
            training_set=training_set,
            artifact_path=sdk_path,
            output_schema=signature.outputs,
            params=params,
            **kwargs,
        )
        uri = getattr(logged, "model_uri", None) or f"runs:/{run_id}/{sdk_path}"
        with tempfile.TemporaryDirectory(prefix="skyulf-feature-copy-") as directory:
            local = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=uri, dst_path=directory, tracking_uri=tracking_uri
                )
            )
            _finish_feature_package(local, options, binding)
            tracking.log_artifacts(run_id, str(local), artifact_path=artifact_path)
    return f"runs:/{run_id}/{artifact_path}"


def log_feature_pipeline_model(
    local_artifact_path: str | Path,
    *,
    training_set: Any,
    lookup_spec: FeatureTrainingSpec,
    lookup_binding: dict[str, Any],
    run_id: str,
    artifact_path: str,
    tracking_uri: str | None = None,
    client: Any = None,
    online_policy: OnlineFeaturePolicy | None = None,
) -> str:
    """Log an existing fitted pandas/Polars pipeline with native point-in-time lookup lineage.

    The native TrainingSet must expose exactly the fitted inputs plus its label.
    Nullable integer/boolean transport is rejected until the SDK provides a
    post-lookup encoder. An online_policy checks received feature completeness
    and freshness before preprocessing without replacing training snapshots.
    Native request overrides remain possible. Cloud acceptance is separate.
    """
    validate_model_destination(run_id, artifact_path, tracking_uri)
    path = Path(local_artifact_path).resolve()
    artifact = load_pipeline(path)
    columns = list(
        zip(artifact.manifest.input_columns, artifact.manifest.input_dtypes, strict=True)
    )
    binding = _validate_contract(training_set, lookup_spec, lookup_binding, columns)
    if online_policy is not None:
        online_policy.validate_lookup(lookup_spec, dict(columns))
    with tempfile.TemporaryDirectory(prefix="skyulf-feature-local-") as directory:
        options = pipeline_model_save_options(
            artifact, path, Path(directory), online_policy=online_policy
        )
        return _log_feature_options(
            options,
            training_set=training_set,
            binding=binding,
            run_id=run_id,
            artifact_path=artifact_path,
            tracking_uri=tracking_uri,
            client=client,
        )


def log_feature_model_set(
    local_artifact_path: str | Path,
    *,
    training_set: Any,
    lookup_spec: FeatureTrainingSpec,
    lookup_binding: dict[str, Any],
    run_id: str,
    artifact_path: str,
    tracking_uri: str | None = None,
    client: Any = None,
    online_policy: OnlineFeaturePolicy | None = None,
) -> str:
    """Log a complete model set against its compatible union lookup contract.

    Optional online_policy guards the union before any component preprocessing;
    it does not change the original training snapshot or authenticate overrides.
    """
    from .model_set import model_set_save_options  # noqa: PLC0415 - avoid registry import cycle

    validate_model_destination(run_id, artifact_path, tracking_uri)
    artifact = load_model_set(local_artifact_path)
    columns = [(column.name, column.dtype) for column in artifact.manifest.input_schema]
    binding = _validate_contract(training_set, lookup_spec, lookup_binding, columns)
    if online_policy is not None:
        online_policy.validate_lookup(lookup_spec, dict(columns))
    with tempfile.TemporaryDirectory(prefix="skyulf-feature-set-") as directory:
        options = model_set_save_options(artifact, Path(directory), online_policy=online_policy)
        return _log_feature_options(
            options,
            training_set=training_set,
            binding=binding,
            run_id=run_id,
            artifact_path=artifact_path,
            tracking_uri=tracking_uri,
            client=client,
        )


def copy_feature_package(
    local_path: str | Path,
    *,
    run_id: str,
    artifact_path: str,
    tracking_uri: str | None = None,
) -> str:
    """Adopt a verified complete native package without rebuilding its training lineage."""
    validate_model_destination(run_id, artifact_path, tracking_uri)
    feature_package_models(local_path)
    tracking = make_tracking_client(tracking_uri)
    tracking.get_run(run_id)
    tracking.log_artifacts(run_id, str(Path(local_path).resolve()), artifact_path=artifact_path)
    return f"runs:/{run_id}/{artifact_path}"


def _validate_online_contract(raw: Any, spec: FeatureTrainingSpec) -> None:
    """Bind opt-in policy metadata to the executable raw model configuration."""
    value = (raw.metadata or {}).get(ONLINE_FEATURES_KEY)
    config = raw.flavors.get("python_function", {}).get("config") or {}
    if config.get(ONLINE_FEATURES_KEY) != value:
        raise ValueError("Online policy metadata differs from saved model configuration.")
    if value is not None:
        policy = OnlineFeaturePolicy.from_dict(value)
        policy.validate_lookup(
            spec, {column.name: column.type.name for column in raw.signature.inputs}
        )
