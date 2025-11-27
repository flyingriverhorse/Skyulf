"""Model export utilities (ONNX and MLflow-compatible packaging).

These functions are optional and will gracefully no-op if the required
packages (onnx, skl2onnx, mlflow) are not installed. This allows the
platform to keep a small default footprint, while enabling interop when
users opt in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def export_to_onnx(
    estimator: Any,
    X_example: Any,
    output_path: str | Path,
    *,
    target_opset: Optional[int] = None,
) -> Dict[str, Any]:
    """Export a scikit-learn estimator/pipeline to ONNX if dependencies exist.

    Returns a dict with keys: success (bool), path (str|None), message (str).
    """
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import numpy as np

        # Infer input shape for conversion
        if hasattr(X_example, "shape"):
            n_features = int(X_example.shape[1])
        else:
            # Try to coerce to numpy
            X_arr = np.asarray(X_example)
            n_features = int(X_arr.shape[1])

        initial_types = [("input", FloatTensorType([None, n_features]))]
        onnx_model = convert_sklearn(
            estimator,
            initial_types=initial_types,
            target_opset=target_opset,
        )

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "wb") as f:
            f.write(onnx_model.SerializeToString())
        return {"success": True, "path": str(out), "message": "Exported to ONNX"}

    except ImportError:
        return {
            "success": False,
            "path": None,
            "message": (
                "ONNX export not available. Please install 'onnx' and 'skl2onnx' to enable."
            ),
        }
    except Exception as e:
        return {"success": False, "path": None, "message": f"ONNX export failed: {e}"}


def export_to_mlflow(
    estimator: Any,
    artifact_path: str | Path,
    *,
    conda_env: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save a model in an MLflow-compatible format, if mlflow is available.

    Returns a dict with keys: success (bool), path (str|None), message (str).
    """
    try:
        import mlflow
        import mlflow.sklearn

        out = Path(artifact_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        mlflow.sklearn.save_model(sk_model=estimator, path=str(out), conda_env=conda_env)
        return {"success": True, "path": str(out), "message": "Saved MLflow artifact"}
    except ImportError:
        return {
            "success": False,
            "path": None,
            "message": "MLflow not installed. Install 'mlflow' to enable packaging.",
        }
    except Exception as e:
        return {"success": False, "path": None, "message": f"MLflow export failed: {e}"}


from dataclasses import dataclass

@dataclass
class ExportBundleResult:
    manifest_payload: Dict[str, Any]
    artefact_entries: List[Dict[str, Any]]
    output_directory: Path
    manifest_path: Path

def export_project_bundle(
    *,
    artifact_path: str | Path,
    output_directory: str | Path,
    job_id: str,
    pipeline_id: Optional[str] = None,
    job_metadata: Optional[Dict[str, Any]] = None,
) -> ExportBundleResult:
    """Export training job artifacts as a zip bundle."""
    import shutil
    import tempfile

    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Dummy implementation
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    
    return ExportBundleResult(
        manifest_payload={},
        artefact_entries=[],
        output_directory=out_dir,
        manifest_path=manifest_path
    )



__all__ = ["export_to_onnx", "export_to_mlflow", "export_project_bundle"]
