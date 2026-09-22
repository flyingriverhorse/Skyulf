"""Exercise the standalone platform probe without claiming a live Databricks gate."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def smoke_module():
    """Load the same script that will be uploaded as a Databricks Python task."""
    path = Path(__file__).parents[2] / "examples" / "databricks_batch_smoke.py"
    spec = importlib.util.spec_from_file_location("skyulf_platform_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_gold_bundle_has_independent_expected_predictions(smoke_module, engine):
    """A known numerical oracle catches matching but incorrect local/Spark predictions."""
    from skyulf.inference import predict_local

    bundle, gold = smoke_module.build_gold_bundle(engine)
    predicted = predict_local(gold, bundle)
    np.testing.assert_allclose(predicted.prediction, [-2.0, 3.0, 5.0], atol=1e-10)
    assert bundle.input_stage == "raw"


def test_parity_rejects_duplicate_or_missing_keys(smoke_module):
    """Equal prediction values must not conceal lost or duplicated source identities."""
    with pytest.raises(AssertionError, match="keys"):
        smoke_module.check_predictions([(101, -2.0), (101, 3.0), (103, 5.0)])


def test_registry_to_both_spark_modes(delta_spark, smoke_module, tmp_path):
    """A downloaded registered model must reach actual Python workers in both FE modes."""
    mlflow = pytest.importorskip("mlflow")
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    registry = f"sqlite:///{(tmp_path / 'registry.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=registry)
    client.create_experiment("sm16-test", artifact_location=(tmp_path / "artifacts").as_uri())
    report = smoke_module.run_smoke(
        delta_spark,
        model_prefix="skyulf_sm16_test",
        experiment_name="sm16-test",
        tracking_uri=uri,
        registry_uri=registry,
        training_engine="polars",
    )
    assert report["stage"] == "registry_spark_parity"
    assert report["model_version"] == "1"
    assert report["model_name"].startswith("skyulf_sm16_test_")
    assert report["checks"] == {
        "local_gold": True,
        "native_features": True,
        "python_pipeline": True,
    }
    assert report["platform_gate_complete"] is False
    assert "delta_publication" in report["remaining_gates"]
