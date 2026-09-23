"""MLflow pyfunc packaging for trusted Skyulf inference bundles.

This module is optional. Importing ``skyulf.integrations.mlflow`` does not load
it; callers must install the ``mlflow`` extra before importing this module.
The adapter transports the existing bundle directory and delegates prediction
to :func:`skyulf.inference.bundle.predict_local`.
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any

import mlflow  # ty: ignore[unresolved-import]
import pandas as pd

from ...inference.bundle import InferenceBundle, load_bundle, predict_local, save_bundle


class SkyulfPythonModel(mlflow.pyfunc.PythonModel):
    """Load one immutable bundle artifact and reuse its local prediction contract."""

    def __init__(self) -> None:
        """Create an unloaded adapter; MLflow supplies the artifact context later."""
        self._bundle: InferenceBundle | None = None

    def load_context(self, context: Any) -> None:
        """Load and validate the transported bundle from MLflow's artifact context."""
        try:
            bundle_path = context.artifacts["bundle"]
        except (AttributeError, KeyError) as exc:
            raise ValueError("MLflow model is missing the Skyulf bundle artifact.") from exc
        self._bundle = load_bundle(Path(bundle_path))

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        """Predict after MLflow normalizes named columns against the saved signature."""
        del context, params
        if self._bundle is None:
            raise RuntimeError("SkyulfPythonModel.load_context() was not called.")
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError(
                "SkyulfPythonModel requires a pandas DataFrame with named columns; "
                "positional NumPy input is unsupported."
            )
        return predict_local(model_input, self._bundle)


def log_model(
    bundle: InferenceBundle,
    *,
    run_id: str | None,
    artifact_path: str,
    tracking_uri: str | None = None,
) -> str:
    """Log a bundle as an MLflow pyfunc model under the explicit run ID.

    The bundle is first written to a bounded temporary directory and passed as
    an MLflow artifact. The serialized pyfunc directory is then uploaded with
    an explicit ``MlflowClient`` and ``run_id``; the caller's process-global
    active run is never used to select the destination. ``tracking_uri`` is
    optional when the process-wide MLflow URI is already configured, but should
    be supplied when the run was created by a client-bound ``track_run``.
    """
    run_id = _validate_arguments(bundle, run_id, artifact_path, tracking_uri)
    signature = _signature(bundle)
    client = _make_client(tracking_uri)
    client.get_run(run_id)
    with tempfile.TemporaryDirectory(prefix="skyulf-mlflow-") as directory:
        bundle_path = Path(directory) / "bundle"
        model_path = Path(directory) / "model"
        save_bundle(bundle, bundle_path)
        # Newer MLflow versions auto-copy the caller's uv project by default.
        # An explicit empty project directory keeps packaging independent of
        # the producer's cwd, without changing process-global settings.
        save_options: dict[str, Any] = {}
        if "uv_project_path" in inspect.signature(mlflow.pyfunc.save_model).parameters:
            save_options["uv_project_path"] = directory
        mlflow.pyfunc.save_model(
            path=str(model_path),
            python_model=SkyulfPythonModel(),
            artifacts={"bundle": str(bundle_path)},
            signature=signature,
            input_example=_input_example(bundle),
            pip_requirements=_pip_requirements(bundle),
            metadata=_metadata(bundle),
            mlflow_model=mlflow.models.Model(run_id=run_id, artifact_path=artifact_path),
            **save_options,
        )
        _scrub_local_artifact_uri(model_path)
        client.log_artifacts(run_id, str(model_path), artifact_path=artifact_path)
    return f"runs:/{run_id}/{artifact_path}"


def _validate_arguments(
    bundle: InferenceBundle,
    run_id: str | None,
    artifact_path: str,
    tracking_uri: str | None,
) -> str:
    """Reject missing bundle identity before serializing or contacting MLflow."""
    if not isinstance(bundle, InferenceBundle):
        raise TypeError("bundle must be an InferenceBundle.")
    if type(run_id) is not str or not run_id.strip():
        raise ValueError("run_id must be a non-empty string.")
    if type(artifact_path) is not str or not artifact_path.strip():
        raise ValueError("artifact_path must be a non-empty string.")
    if any(reserved in artifact_path for reserved in ("#", "?")):
        raise ValueError("artifact_path cannot contain '#' or '?' because they are URI delimiters.")
    if tracking_uri is not None and (type(tracking_uri) is not str or not tracking_uri.strip()):
        raise ValueError("tracking_uri must be a non-empty string or None.")
    return run_id


def _make_client(tracking_uri: str | None) -> Any:
    """Construct the MLflow client used to validate and upload one run artifact."""
    from mlflow import (  # noqa: PLC0415 - optional module boundary  # ty: ignore[unresolved-import]
        MlflowClient,
    )

    return MlflowClient(tracking_uri=tracking_uri)


def _scrub_local_artifact_uri(model_path: Path, artifact_key: str = "bundle") -> None:
    """Remove the temporary producer path from the portable MLflow metadata."""
    model = mlflow.models.Model.load(str(model_path))
    flavor = model.flavors[mlflow.pyfunc.FLAVOR_NAME]
    artifacts = flavor.get("artifacts", {})
    saved_artifact = artifacts.get(artifact_key)
    if isinstance(saved_artifact, dict) and "uri" in saved_artifact:
        saved_artifact["uri"] = artifact_key
    model.save(str(model_path / "MLmodel"))


def _metadata(bundle: InferenceBundle) -> dict[str, str]:
    """Record contract identity without copying rows, credentials or sessions."""
    return {
        "skyulf_bundle_digest": bundle.semantic_digest,
        "skyulf_input_stage": bundle.input_stage,
        "skyulf_task": bundle.manifest.task,
        "skyulf_feature_order": ",".join(bundle.feature_order),
    }


def _pip_requirements(bundle: InferenceBundle) -> list[str]:
    """Pin the bundle runtime and the MLflow flavor used to load this model."""
    requirements = [
        f"{name}=={value}" for name, value in bundle.manifest.requirements if name != "python"
    ]
    requirements.append(f"mlflow=={mlflow.__version__}")
    return requirements


def _input_example(bundle: InferenceBundle) -> pd.DataFrame:
    """Build one synthetic typed row from the manifest, never from training data."""
    return pd.DataFrame(
        {
            column.name: pd.Series([_example_value(column.dtype)], dtype=column.dtype)
            for column in bundle.manifest.input_schema
        }
    )


def _example_value(dtype: str) -> bool | int | float:
    """Return a safe zero-like scalar for an input schema dtype."""
    if dtype == "bool":
        return False
    if dtype.startswith(("int", "uint")):
        return 0
    if dtype.startswith("float"):
        return 0.0
    raise ValueError(f"Unsupported bundle input dtype: {dtype}.")


def _signature(bundle: InferenceBundle) -> Any:
    """Create an explicit MLflow tabular signature from the frozen bundle schema."""
    from mlflow.models import (  # noqa: PLC0415 - optional module boundary  # ty: ignore[unresolved-import]
        ModelSignature,
    )
    from mlflow.types import (  # noqa: PLC0415 - optional module boundary  # ty: ignore[unresolved-import]
        ColSpec,
        Schema,
    )

    inputs = Schema(
        [
            ColSpec(_mlflow_dtype(column.dtype), name=column.name)
            for column in bundle.manifest.input_schema
        ]
    )
    outputs = Schema(
        [
            ColSpec(_mlflow_dtype(column.dtype), name=column.name)
            for column in bundle.manifest.output_schema
        ]
    )
    return ModelSignature(inputs=inputs, outputs=outputs)


def _mlflow_dtype(dtype: str) -> Any:
    """Map the bundle's primitive dtype vocabulary to MLflow tabular types."""
    from mlflow.types import (  # noqa: PLC0415 - optional module boundary  # ty: ignore[unresolved-import]
        DataType,
    )

    mapping = {
        "bool": DataType.boolean,
        "int32": DataType.integer,
        "int64": DataType.long,
        "float32": DataType.float,
        "float64": DataType.double,
        "string": DataType.string,
    }
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise ValueError(
            "MLflow column signatures cannot preserve this bundle dtype exactly: "
            f"{dtype}. Use int32/int64, float32/float64, bool or string."
        ) from exc
