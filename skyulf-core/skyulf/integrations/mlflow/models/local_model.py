"""Optional MLflow pyfunc adapter for fitted local pipeline artifacts.

The artifact is for whole-frame local batches. A pyfunc model here does not
certify row-local HTTP serving or Spark partition safety.
"""

import inspect
import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import mlflow  # ty: ignore[unresolved-import]
import pandas as pd

import skyulf.integrations.mlflow.shared._model_metadata as model_metadata
from skyulf.integrations.mlflow.shared._client import make_tracking_client
from skyulf.integrations.mlflow.shared._model_metadata import (
    mlflow_dtype,
    scrub_local_artifact_uri,
)

from ....inference.local_pipeline import (
    LocalPipelineArtifact,
    load_local_pipeline,
)
from ....inference.local_scoring import score_local_pipeline, scoring_output_schema
from ..shared._nullable_transport import (
    TRANSPORT_KEY,
    decode_frame,
    encode_frame,
    restore_nullable_dtypes,
    transport_spec,
    validated_transport,
)
from ..shared._nullable_transport import (  # noqa: F401 - public compatibility re-export
    prepare_pyfunc_input as prepare_pyfunc_input,
)
from ..spark._spark_environment import snapshot_worker_environment
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


class SkyulfLocalPythonModel(mlflow.pyfunc.PythonModel):
    """Load one fitted local pipeline and retain its recorded execution engine."""

    def __init__(
        self,
        input_transport: dict[str, Any] | None = None,
        safety_certificate: dict[str, Any] | None = None,
        source_sha256: str | None = None,
    ) -> None:
        """Start unloaded until MLflow provides the saved artifact path."""
        self._artifact: LocalPipelineArtifact | None = None
        self._input_transport = deepcopy(input_transport)
        self._safety_certificate = deepcopy(safety_certificate)
        self._source_sha256 = source_sha256

    def __getstate__(self) -> dict[str, Any]:
        """Reload saved project classes through context in each fresh process."""
        return {**self.__dict__, "_artifact": None}

    def load_context(self, context: Any) -> None:
        """Validate the trusted pipeline artifact before the first prediction."""
        try:
            artifact_path = context.artifacts["local_pipeline"]
        except (AttributeError, KeyError) as exc:
            raise ValueError("MLflow model is missing the local pipeline artifact.") from exc
        self._artifact = load_local_pipeline(artifact_path)
        self.input_transport()
        validate_worker_certificate(
            self._artifact,
            getattr(self, "_safety_certificate", None),
            getattr(self, "_source_sha256", None),
        )

    def input_transport(self) -> dict[str, Any] | None:
        """Return a detached transport contract checked against the loaded artifact."""
        if self._artifact is None:
            raise RuntimeError("SkyulfLocalPythonModel.load_context() was not called.")
        manifest = self._artifact.manifest
        return validated_transport(
            getattr(self, "_input_transport", None),
            zip(manifest.input_columns, manifest.input_dtypes, strict=True),
        )

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Apply the saved fit engine to MLflow's named pandas input columns."""
        del context
        if self._artifact is None:
            raise RuntimeError("SkyulfLocalPythonModel.load_context() was not called.")
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("Skyulf local pyfunc requires a pandas DataFrame.")
        spark_output = require_spark_output(params, getattr(self, "_safety_certificate", None))
        model_input = decode_frame(model_input, self.input_transport())
        model_input = _restore_nullable_dtypes(model_input, self._artifact)
        result = score_prediction_batches(
            model_input,
            partial(score_local_pipeline, artifact=self._artifact),
            params,
            spark_output,
        )
        return prepare_spark_output(result, scoring_output_schema(self._artifact), spark_output)


def log_local_model(
    local_artifact_path: str | Path,
    *,
    run_id: str,
    artifact_path: str,
    tracking_uri: str | None = None,
) -> str:
    """Log a fitted local pipeline under one explicit MLflow run and artifact path."""
    validate_local_destination(run_id, artifact_path, tracking_uri)
    local_path = Path(local_artifact_path).resolve()
    artifact = load_local_pipeline(local_path)
    client = make_tracking_client(tracking_uri)
    client.get_run(run_id)
    with tempfile.TemporaryDirectory(prefix="skyulf-local-mlflow-") as directory:
        model_path = Path(directory) / "model"
        mlflow.pyfunc.save_model(
            path=str(model_path),
            mlflow_model=mlflow.models.Model(run_id=run_id, artifact_path=artifact_path),
            **local_model_save_options(artifact, local_path, Path(directory)),
        )
        scrub_local_artifact_uri(model_path, "local_pipeline")
        client.log_artifacts(run_id, str(model_path), artifact_path=artifact_path)
    return f"runs:/{run_id}/{artifact_path}"


def local_model_save_options(
    artifact: LocalPipelineArtifact, local_path: Path, directory: Path
) -> dict[str, Any]:
    """Share the exact fitted pyfunc, schema and worker evidence across loggers."""
    transport = transport_spec(
        zip(artifact.manifest.input_columns, artifact.manifest.input_dtypes, strict=True)
    )
    certificate = optional_partition_certificate(artifact)
    requirements = pip_requirements(artifact)
    options: dict[str, Any] = {}
    if "uv_project_path" in inspect.signature(mlflow.pyfunc.save_model).parameters:
        options["uv_project_path"] = str(directory)
    source_sha256 = None
    if certificate:
        code_paths, requirements, source_sha256 = snapshot_worker_environment(
            directory, requirements
        )
        options["code_paths"] = code_paths
    return {
        **options,
        "python_model": SkyulfLocalPythonModel(transport, certificate, source_sha256),
        "artifacts": {"local_pipeline": str(local_path)},
        "signature": _signature(artifact, spark_certified=certificate is not None),
        "input_example": _input_example(artifact),
        "pip_requirements": requirements,
        "metadata": {
            "skyulf_artifact_kind": "local_pipeline",
            "skyulf_fitted_engine": artifact.manifest.fitted_engine,
            "skyulf_execution_scope": "whole_frame_local",
            "local_pipeline_digest": artifact.manifest.pipeline_sha256,
            **({TRANSPORT_KEY: transport} if transport else {}),
            **({SAFETY_KEY: certificate, SOURCE_KEY: source_sha256} if certificate else {}),
        },
    }


def normalized_dtype(dtype: str) -> str:
    """Map fitted pandas/Polars labels to the supported MLflow scalar vocabulary."""
    return model_metadata.normalized_dtype(dtype)


def _signature(artifact: LocalPipelineArtifact, *, spark_certified: bool = False) -> Any:
    """Declare ordered raw inputs and prediction/probability output columns."""
    from mlflow.models import (  # noqa: PLC0415 - optional dependency boundary  # ty: ignore[unresolved-import]
        ModelSignature,
    )
    from mlflow.types import (  # noqa: PLC0415 - optional dependency boundary  # ty: ignore[unresolved-import]
        ColSpec,
        Schema,
    )

    manifest = artifact.manifest
    transport = transport_spec(zip(manifest.input_columns, manifest.input_dtypes, strict=True))
    encoded = transport["columns"] if transport else {}
    inputs = Schema(
        [
            ColSpec(
                mlflow_dtype("string" if name in encoded else normalized_dtype(dtype)), name=name
            )
            for name, dtype in zip(manifest.input_columns, manifest.input_dtypes, strict=True)
        ]
    )
    outputs = Schema(
        [
            ColSpec(mlflow_dtype(column.dtype), name=column.name)
            for column in scoring_output_schema(artifact)
        ]
    )
    options = {"params": spark_output_params()} if spark_certified else {}
    return ModelSignature(inputs=inputs, outputs=outputs, **options)


def _input_example(artifact: LocalPipelineArtifact) -> pd.DataFrame:
    """Build a synthetic typed row without retaining training data."""
    values: dict[str, pd.Series] = {}
    for name, dtype in zip(
        artifact.manifest.input_columns, artifact.manifest.input_dtypes, strict=True
    ):
        normalized = normalized_dtype(dtype)
        if normalized == "string":
            values[name] = pd.Series(["example"], dtype="object")
        elif normalized == "bool":
            values[name] = pd.Series([False], dtype="bool")
        elif normalized.startswith(("int", "uint")):
            values[name] = pd.Series([0], dtype=normalized)
        elif normalized.startswith("float"):
            values[name] = pd.Series([0.0], dtype=normalized)
        else:
            raise ValueError(f"Unsupported MLflow local input dtype: {dtype}.")
    transport = transport_spec(
        zip(artifact.manifest.input_columns, artifact.manifest.input_dtypes, strict=True)
    )
    return encode_frame(pd.DataFrame(values), transport)


def pip_requirements(artifact: LocalPipelineArtifact) -> list[str]:
    """Pin the fitted runtime and optional MLflow flavor for reproducible loading."""
    return list(
        dict.fromkeys(
            [
                *artifact.manifest.project_requirements,
                *(
                    f"{name}=={value}"
                    for name, value in artifact.manifest.requirements
                    if name != "python"
                ),
                f"mlflow=={mlflow.__version__}",
            ]
        )
    )


def validate_local_destination(run_id: str, artifact_path: str, tracking_uri: str | None) -> None:
    """Validate the explicit run and upload destination before reading the artifact."""
    if type(run_id) is not str or not run_id.strip():
        raise ValueError("run_id must be a non-empty string.")
    if type(artifact_path) is not str or not artifact_path.strip():
        raise ValueError("artifact_path must be a non-empty string.")
    if any(marker in artifact_path for marker in ("#", "?")):
        raise ValueError("artifact_path cannot contain URI delimiters.")
    if tracking_uri is not None and (type(tracking_uri) is not str or not tracking_uri.strip()):
        raise ValueError("tracking_uri must be a non-empty string or None.")


def _restore_nullable_dtypes(frame: pd.DataFrame, artifact: LocalPipelineArtifact) -> pd.DataFrame:
    """Restore exact pandas extension types erased by MLflow scalar signatures.

    Only the matching NumPy storage type is accepted; integer inputs never
    pass through a float conversion. MLflow can reject nullable integer or
    boolean nulls in legacy packages before this adapter is reached. New
    packages use the explicit nullable transport before this restoration.
    """
    if artifact.manifest.fitted_engine != "pandas":
        return frame
    return restore_nullable_dtypes(
        frame, zip(artifact.manifest.input_columns, artifact.manifest.input_dtypes, strict=True)
    )
