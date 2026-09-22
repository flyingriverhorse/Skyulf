"""Worker-local model loading and optional-runtime isolation contracts."""

import base64
import importlib
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from skyulf.data.dataset import SplitDataset
from skyulf.inference.bundle import build_bundle
from skyulf.pipeline import SkyulfPipeline


def _classification_bundle():
    """Build a real string-label bundle without involving Spark training."""
    train = pd.DataFrame(
        {"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], "label": ["no", "no", "no", "yes", "yes", "yes"]}
    )
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="label")
    return build_bundle(pipeline, input_stage="raw", feature_order=("x",))


def test_classification_worker_loads_once_and_bounds_model_calls(monkeypatch):
    """One iterator load must be reused across rows and both prediction outputs."""
    module = importlib.import_module("skyulf.inference.spark")
    bundle = _classification_bundle()
    original_load = module.load_model
    loaded = []
    predict_shapes = []
    probability_shapes = []

    def recorded_load(payload, manifest):
        """Observe the real estimator while preserving its class semantics."""
        model = original_load(payload, manifest)
        original_predict = model.predict
        original_probability = model.predict_proba

        def recorded_predict(values):
            """Model calls must receive bounded feature chunks, never individual rows."""
            predict_shapes.append(values.shape)
            return original_predict(values)

        def recorded_probability(values):
            """Probability calls must use the same bounded chunks as labels."""
            probability_shapes.append(values.shape)
            return original_probability(values)

        model.predict = recorded_predict
        model.predict_proba = recorded_probability
        loaded.append(model)
        return model

    monkeypatch.setattr(module, "load_model", recorded_load)
    worker = module._prediction_iterator(
        bundle.model_payload, bundle.manifest, ("id",), batch_rows=2
    )
    batch = pd.DataFrame({"id": range(5), "x": [-2.0, -1.0, 0.1, 1.0, 2.0]})
    output = pd.concat(worker(iter([batch, batch.head(0), batch.iloc[:1]])), ignore_index=True)

    assert len(loaded) == 1
    assert predict_shapes == [(2, 1), (2, 1), (1, 1), (1, 1)]
    assert probability_shapes == predict_shapes
    assert output.columns.tolist() == ["id", "prediction", *bundle.probability_columns]
    assert output.id.tolist() == [0, 1, 2, 3, 4, 0]
    assert set(output.prediction) <= {"no", "yes"}
    np.testing.assert_allclose(output[list(bundle.probability_columns)].sum(axis=1), 1.0)


def test_worker_missing_dependency_error_survives_separate_process():
    """A worker runtime mismatch must fail clearly before unpickling the model."""
    bundle = _classification_bundle()
    encoded = base64.b64encode(pickle.dumps(bundle)).decode("ascii")
    code = r"""
import base64
import pickle
import sys

bundle = pickle.loads(base64.b64decode(sys.argv[1]))
import skyulf.inference._manifest as metadata
import skyulf.inference.spark as module

requirements = bundle.manifest.requirements
metadata.runtime_requirements = lambda: tuple(
    (name, "0.0.0" if name == "scikit-learn" else version)
    for name, version in requirements
)
worker = module._prediction_iterator(
    bundle.model_payload, bundle.manifest, ("id",), batch_rows=2
)
try:
    list(worker(iter(())))
except ValueError as error:
    assert "runtime mismatch" in str(error)
else:
    raise AssertionError("worker accepted an incompatible dependency")
"""
    result = subprocess.run(
        [sys.executable, "-c", code, encoded], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
