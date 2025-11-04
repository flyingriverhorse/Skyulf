"""Tiny demo: train a simple sklearn pipeline and export artifacts.

- Fits StandardScaler + LogisticRegression on Iris
- Exports model to ONNX (if onnx+skl2onnx installed)
- Saves MLflow-compatible model folder (if mlflow installed)

Artifacts are written under exports/demo/
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from core.feature_engineering.export import export_to_onnx, export_to_mlflow


def main() -> Dict[str, Dict[str, str]]:
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipe.fit(X, y)

    out_dir = Path("exports/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Example array to infer feature shape
    X_example = np.asarray(X[:5])

    onnx_path = out_dir / "iris_pipeline.onnx"
    mlflow_path = out_dir / "iris_pipeline_mlflow"

    onnx_res = export_to_onnx(pipe, X_example, onnx_path)
    mlflow_res = export_to_mlflow(pipe, mlflow_path)

    print("ONNX:", onnx_res)
    print("MLflow:", mlflow_res)

    return {"onnx": onnx_res, "mlflow": mlflow_res}


if __name__ == "__main__":
    main()
