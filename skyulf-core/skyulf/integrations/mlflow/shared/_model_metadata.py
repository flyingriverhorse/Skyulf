"""Package-internal MLflow signature and portable artifact metadata helpers."""

from pathlib import Path
from typing import Any


def normalized_dtype(dtype: str) -> str:
    """Map fitted scalar aliases to the shared MLflow signature vocabulary."""
    normalized = dtype.lower()
    return {"object": "string", "str": "string", "utf8": "string", "boolean": "bool"}.get(
        normalized, normalized
    )


def mlflow_dtype(dtype: str) -> Any:
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


def scrub_local_artifact_uri(model_path: Path, artifact_key: str = "bundle") -> None:
    """Remove the temporary producer path from the portable MLflow metadata."""
    import mlflow  # noqa: PLC0415 - optional metadata operation  # ty: ignore[unresolved-import]

    model = mlflow.models.Model.load(str(model_path))
    flavor = model.flavors[mlflow.pyfunc.FLAVOR_NAME]
    artifacts = flavor.get("artifacts", {})
    saved_artifact = artifacts.get(artifact_key)
    if isinstance(saved_artifact, dict) and "uri" in saved_artifact:
        saved_artifact["uri"] = artifact_key
    model.save(str(model_path / "MLmodel"))
