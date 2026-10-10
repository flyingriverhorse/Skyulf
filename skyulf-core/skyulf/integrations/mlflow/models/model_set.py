"""Optional MLflow packaging for a complete pinned, locally executable model set."""

import tempfile
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import mlflow  # ty: ignore[unresolved-import]
import pandas as pd

from skyulf.integrations.mlflow.shared._client import make_tracking_client
from skyulf.integrations.mlflow.shared._model_metadata import (
    column_schema,
    normalized_dtype,
    scrub_local_artifact_uri,
)

from ....inference.fitted_pipeline import (
    load_pipeline as load_local_pipeline,
)
from ....inference.fitted_pipeline import (
    read_bounded_artifact,
)
from ....inference.model_set import ModelSetArtifact, load_model_set
from ....inference.model_set_scoring import model_set_output_schema, predict_model_set
from ....inference.project_code import MAX_PROJECT_SOURCE_BYTES
from ....inference.project_dependencies import (
    parse_project_requirements,
    source_project_requirements,
)
from ..registration.registry import (
    ResolvedModel,
    downloaded_registered_payload,
    packaged_artifact_path,
    unwrap_feature_package,
    validate_concrete_version,
    validate_registry_options,
)
from ..shared._nullable_transport import (
    TRANSPORT_KEY,
    decode_frame,
    restore_nullable_dtypes,
    transport_spec,
    validated_transport,
)
from ..spark._spark_environment import pyfunc_environment
from ..spark._spark_output import (
    prepare_spark_output,
    require_spark_output,
    score_prediction_batches,
    spark_output_params,
)
from ..spark.spark_model import (
    SAFETY_KEY,
    SOURCE_KEY,
    optional_partition_certificate,
    validate_worker_certificate,
)
from .pipeline_model import pip_requirements, validate_local_destination


class SkyulfModelSetPythonModel(mlflow.pyfunc.PythonModel):
    """Execute the frozen components and composition shipped in one package."""

    def __init__(
        self,
        input_transport: dict[str, Any] | None = None,
        safety_certificate: dict[str, Any] | None = None,
        source_sha256: str | None = None,
    ) -> None:
        """Defer loading fitted assets until MLflow supplies package context."""
        self._artifact: ModelSetArtifact | None = None
        self._input_transport = deepcopy(input_transport)
        self._safety_certificate = deepcopy(safety_certificate)
        self._source_sha256 = source_sha256

    def __getstate__(self) -> dict[str, Any]:
        """Exclude process-local artifact paths and cached fitted objects."""
        return {**self.__dict__, "_artifact": None}

    def load_context(self, context: Any) -> None:
        """Validate all packaged components and rules before prediction."""
        try:
            path = context.artifacts["model_set"]
        except (AttributeError, KeyError) as exc:
            raise ValueError("MLflow model is missing its model set artifact.") from exc
        self._artifact = load_model_set(path)
        self.input_transport()
        validate_worker_certificate(
            self._artifact,
            getattr(self, "_safety_certificate", None),
            getattr(self, "_source_sha256", None),
        )

    def input_transport(self) -> dict[str, Any] | None:
        """Return the artifact-validated nullable input contract without mutable aliases."""
        if self._artifact is None:
            raise RuntimeError("SkyulfModelSetPythonModel.load_context() was not called.")
        return validated_transport(
            getattr(self, "_input_transport", None),
            ((column.name, column.dtype) for column in self._artifact.manifest.input_schema),
        )

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Score one bounded whole frame using explicit, preserved record keys."""
        del context
        if self._artifact is None:
            raise RuntimeError("SkyulfModelSetPythonModel.load_context() was not called.")
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("Skyulf model set pyfunc requires a pandas DataFrame.")
        spark_output = require_spark_output(params, getattr(self, "_safety_certificate", None))
        model_input = decode_frame(model_input, self.input_transport())
        model_input = restore_nullable_dtypes(
            model_input,
            ((column.name, column.dtype) for column in self._artifact.manifest.input_schema),
        )
        result = score_prediction_batches(
            model_input, partial(predict_model_set, artifact=self._artifact), params, spark_output
        )
        return prepare_spark_output(result, model_set_output_schema(self._artifact), spark_output)


def log_model_set(
    local_artifact_path: str | Path,
    *,
    run_id: str,
    artifact_path: str,
    tracking_uri: str | None = None,
) -> str:
    """Log a complete model set without selecting aliases or retaining producer paths."""
    validate_local_destination(run_id, artifact_path, tracking_uri)
    artifact = load_model_set(local_artifact_path)
    client = make_tracking_client(tracking_uri)
    client.get_run(run_id)
    with tempfile.TemporaryDirectory(prefix="skyulf-set-mlflow-") as directory:
        model_path = Path(directory) / "model"
        mlflow.pyfunc.save_model(
            path=str(model_path),
            mlflow_model=mlflow.models.Model(run_id=run_id, artifact_path=artifact_path),
            **model_set_save_options(artifact, Path(directory)),
        )
        scrub_local_artifact_uri(model_path, "model_set")
        client.log_artifacts(run_id, str(model_path), artifact_path=artifact_path)
    return f"runs:/{run_id}/{artifact_path}"


def model_set_save_options(artifact: ModelSetArtifact, directory: Path) -> dict[str, Any]:
    """Build the identical set wrapper and immutable evidence for every logger."""
    transport = transport_spec(
        (column.name, column.dtype) for column in artifact.manifest.input_schema
    )
    certificate = optional_partition_certificate(artifact)
    options, source_sha256 = pyfunc_environment(
        directory, _set_requirements(artifact), spark_certified=bool(certificate)
    )
    return {
        **options,
        "python_model": SkyulfModelSetPythonModel(transport, certificate, source_sha256),
        "artifacts": {"model_set": str(artifact.directory)},
        "signature": _signature(artifact, spark_certified=certificate is not None),
        "metadata": {
            "skyulf_artifact_kind": "model_set",
            "skyulf_execution_scope": "whole_frame_local",
            "model_set_digest": artifact.manifest.set_sha256,
            **({TRANSPORT_KEY: transport} if transport else {}),
            **({SAFETY_KEY: certificate, SOURCE_KEY: source_sha256} if certificate else {}),
        },
    }


def _signature(artifact: ModelSetArtifact, *, spark_certified: bool = False) -> Any:
    """Require an exact MLflow scalar representation for every input and output."""
    from mlflow.models import ModelSignature  # noqa: PLC0415  # ty: ignore[unresolved-import]

    transport = transport_spec(
        (column.name, column.dtype) for column in artifact.manifest.input_schema
    )
    encoded = transport["columns"] if transport else {}

    options = {"params": spark_output_params()} if spark_certified else {}
    return ModelSignature(
        inputs=column_schema(
            (c.name, "string" if c.name in encoded else normalized_dtype(c.dtype))
            for c in artifact.manifest.input_schema
        ),
        outputs=column_schema(
            (c.name, normalized_dtype(c.dtype)) for c in model_set_output_schema(artifact)
        ),
        **options,
    )


def _set_requirements(artifact: ModelSetArtifact) -> list[str]:
    """Merge component and captured composition pins without producer URLs or conflicts."""
    requirements: dict[str, str] = {}
    for component in artifact.manifest.components:
        pins = pip_requirements(
            load_local_pipeline(artifact.directory / "components" / component.branch)
        )
        _merge_requirements(requirements, pins)
    source = read_bounded_artifact(artifact.directory / "composition.py", MAX_PROJECT_SOURCE_BYTES)
    _merge_requirements(requirements, source_project_requirements(source.decode("utf-8")))
    return list(requirements.values())


def _merge_requirements(requirements: dict[str, str], pins: Iterable[str]) -> None:
    """Require every source of a distribution dependency to agree on one exact pin."""
    for raw in pins:
        normalized = parse_project_requirements(raw)[0]
        key = normalized.split("==")[0]
        if key in requirements and requirements[key] != normalized:
            raise ValueError(f"Conflicting model set requirement pins for {key}.")
        requirements[key] = normalized


def load_registered_model_set(
    resolved: ResolvedModel,
    *,
    tracking_uri: str | None = None,
    registry_uri: str | None = None,
) -> ModelSetArtifact:
    """Load one concrete trusted set and verify package kind and complete digest."""
    if not isinstance(resolved, ResolvedModel):
        raise TypeError("resolved must be a ResolvedModel.")
    validate_registry_options(resolved.name, tracking_uri, registry_uri)
    validate_concrete_version(resolved)
    if not isinstance(resolved.digest, str) or not resolved.digest.strip():
        raise ValueError("Resolved model set requires a digest.")
    client = mlflow.MlflowClient(tracking_uri=tracking_uri, registry_uri=registry_uri)
    with downloaded_registered_payload(mlflow, client, resolved, tracking_uri, "model_set") as (
        local,
        model,
    ):
        local, model, feature_lookup_json = unwrap_feature_package(local, model)
        artifact = load_model_set(packaged_artifact_path(local, model.flavors, "model_set"))
        if artifact.manifest.set_sha256 != resolved.digest:
            raise ValueError("Loaded model set digest differs from resolved identity.")
        return replace(artifact, feature_lookup_json=feature_lookup_json)
