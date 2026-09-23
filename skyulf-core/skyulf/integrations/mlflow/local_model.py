"""Optional MLflow pyfunc adapter for fitted local pipeline artifacts.

The artifact is for whole-frame local batches. A pyfunc model here does not
certify row-local HTTP serving or Spark partition safety.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any

import mlflow  # ty: ignore[unresolved-import]
import pandas as pd

from ...inference._manifest import label_dtype
from ...inference.local_pipeline import (
    LocalPipelineArtifact,
    load_local_pipeline,
    predict_local_pipeline,
)
from .model import _make_client, _mlflow_dtype, _scrub_local_artifact_uri


class SkyulfLocalPythonModel(mlflow.pyfunc.PythonModel):
    """Load one fitted local pipeline and retain its recorded execution engine."""

    def __init__(self) -> None:
        """Start unloaded until MLflow provides the saved artifact path."""
        self._artifact: LocalPipelineArtifact | None = None

    def load_context(self, context: Any) -> None:
        """Validate the trusted pipeline artifact before the first prediction."""
        try:
            artifact_path = context.artifacts["local_pipeline"]
        except (AttributeError, KeyError) as exc:
            raise ValueError("MLflow model is missing the local pipeline artifact.") from exc
        self._artifact = load_local_pipeline(artifact_path)

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Apply the saved fit engine to MLflow's named pandas input columns."""
        del context, params
        if self._artifact is None:
            raise RuntimeError("SkyulfLocalPythonModel.load_context() was not called.")
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("Skyulf local pyfunc requires a pandas DataFrame.")
        return predict_local_pipeline(model_input, self._artifact)


def log_local_model(
    local_artifact_path: str | Path,
    *,
    run_id: str,
    artifact_path: str,
    tracking_uri: str | None = None,
) -> str:
    """Log a fitted local pipeline under one explicit MLflow run and artifact path."""
    if type(run_id) is not str or not run_id.strip():
        raise ValueError("run_id must be a non-empty string.")
    if type(artifact_path) is not str or not artifact_path.strip():
        raise ValueError("artifact_path must be a non-empty string.")
    if any(marker in artifact_path for marker in ("#", "?")):
        raise ValueError("artifact_path cannot contain URI delimiters.")
    if tracking_uri is not None and (type(tracking_uri) is not str or not tracking_uri.strip()):
        raise ValueError("tracking_uri must be a non-empty string or None.")
    local_path = Path(local_artifact_path).resolve()
    artifact = load_local_pipeline(local_path)
    client = _make_client(tracking_uri)
    client.get_run(run_id)
    with tempfile.TemporaryDirectory(prefix="skyulf-local-mlflow-") as directory:
        model_path = Path(directory) / "model"
        save_options: dict[str, Any] = {}
        if "uv_project_path" in inspect.signature(mlflow.pyfunc.save_model).parameters:
            save_options["uv_project_path"] = directory
        mlflow.pyfunc.save_model(
            path=str(model_path),
            python_model=SkyulfLocalPythonModel(),
            artifacts={"local_pipeline": str(local_path)},
            signature=_signature(artifact),
            input_example=_input_example(artifact),
            pip_requirements=_pip_requirements(artifact),
            metadata={
                "skyulf_artifact_kind": "local_pipeline",
                "skyulf_fitted_engine": artifact.manifest.fitted_engine,
                "skyulf_execution_scope": "whole_frame_local",
                "skyulf_local_pipeline_digest": artifact.manifest.pipeline_sha256,
            },
            mlflow_model=mlflow.models.Model(run_id=run_id, artifact_path=artifact_path),
            **save_options,
        )
        _scrub_local_artifact_uri(model_path, "local_pipeline")
        client.log_artifacts(run_id, str(model_path), artifact_path=artifact_path)
    return f"runs:/{run_id}/{artifact_path}"


def _normalized_dtype(dtype: str) -> str:
    """Map fitted pandas/Polars labels to the supported MLflow scalar vocabulary."""
    normalized = dtype.lower()
    return {"object": "string", "str": "string", "utf8": "string", "boolean": "bool"}.get(
        normalized, normalized
    )


def _signature(artifact: LocalPipelineArtifact) -> Any:
    """Declare ordered raw inputs and prediction/probability output columns."""
    from mlflow.models import (  # noqa: PLC0415 - optional dependency boundary  # ty: ignore[unresolved-import]
        ModelSignature,
    )
    from mlflow.types import (  # noqa: PLC0415 - optional dependency boundary  # ty: ignore[unresolved-import]
        ColSpec,
        Schema,
    )

    manifest = artifact.manifest
    inputs = Schema(
        [
            ColSpec(_mlflow_dtype(_normalized_dtype(dtype)), name=name)
            for name, dtype in zip(manifest.input_columns, manifest.input_dtypes, strict=True)
        ]
    )
    prediction_dtype = "float64" if manifest.task == "regression" else label_dtype(manifest.classes)
    outputs = [ColSpec(_mlflow_dtype(prediction_dtype), name="prediction")]
    if manifest.task == "classification":
        outputs.extend(
            ColSpec(_mlflow_dtype("float64"), name=f"probability_{position}")
            for position in range(len(manifest.classes))
        )
    return ModelSignature(inputs=inputs, outputs=Schema(outputs))


def _input_example(artifact: LocalPipelineArtifact) -> pd.DataFrame:
    """Build a synthetic typed row without retaining training data."""
    values: dict[str, pd.Series] = {}
    for name, dtype in zip(
        artifact.manifest.input_columns, artifact.manifest.input_dtypes, strict=True
    ):
        normalized = _normalized_dtype(dtype)
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
    return pd.DataFrame(values)


def _pip_requirements(artifact: LocalPipelineArtifact) -> list[str]:
    """Pin the fitted runtime and optional MLflow flavor for reproducible loading."""
    return [
        *(f"{name}=={value}" for name, value in artifact.manifest.requirements if name != "python"),
        f"mlflow=={mlflow.__version__}",
    ]
