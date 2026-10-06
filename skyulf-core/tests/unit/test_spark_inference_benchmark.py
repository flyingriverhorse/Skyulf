"""Benchmark evidence must prove complete, correct, materialized predictions."""

import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest


def benchmark_module():
    """Load the standalone notebook without requiring a local Spark installation."""
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_spark_inference.py"
    spec = importlib.util.spec_from_file_location("spark_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_measurement_requires_every_prediction():
    """Timing results are useful only when every expected key was actually scored."""
    module = benchmark_module()
    result = module.validate_measurement(
        {
            "rows": 100,
            "keys": 100,
            "min_id": 0,
            "max_id": 99,
            "invalid": 0,
            "max_error": 1e-10,
            "prediction_sum": 12.0,
        },
        expected_rows=100,
        elapsed_seconds=2.0,
    )
    assert result["rows_per_second"] == 50.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", 99),
        ("keys", 99),
        ("min_id", 1),
        ("max_id", 100),
        ("invalid", 1),
        ("max_error", 0.01),
        ("max_error", float("nan")),
        ("max_error", None),
        ("prediction_sum", float("inf")),
    ],
)
def test_invalid_evidence_is_never_reported_as_throughput(field, value):
    """Partial, duplicated, missing or wrong predictions must fail benchmark acceptance."""
    module = benchmark_module()
    evidence = {
        "rows": 100,
        "keys": 100,
        "min_id": 0,
        "max_id": 99,
        "invalid": 0,
        "max_error": 1e-10,
        "prediction_sum": 12.0,
    }
    evidence[field] = value
    with pytest.raises(ValueError, match="Invalid benchmark"):
        module.validate_measurement(evidence, expected_rows=100, elapsed_seconds=2.0)


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timer_cannot_create_plausible_throughput(seconds):
    """A broken timer cannot silently produce an apparently successful benchmark."""
    module = benchmark_module()
    evidence = {
        "rows": 100,
        "keys": 100,
        "min_id": 0,
        "max_id": 99,
        "invalid": 0,
        "max_error": 0.0,
        "prediction_sum": 12.0,
    }
    with pytest.raises(ValueError, match="Invalid benchmark"):
        module.validate_measurement(evidence, expected_rows=100, elapsed_seconds=seconds)


def test_failed_repeat_preserves_first_action_and_error(tmp_path):
    """A later worker failure must not erase measurements already completed."""
    module = benchmark_module()
    report = {"cases": [], "complete": False}
    destination = tmp_path / "report.json"

    def failing_case(on_progress):
        """Simulate success followed by a worker error in the second action."""
        on_progress({"mode": "pyfunc", "stage": "repeat_1", "actions": [{"seconds": 2.0}]})
        raise RuntimeError("worker failed")

    with pytest.raises(RuntimeError, match="worker failed"):
        module.record_case(report, destination, {"mode": "pyfunc"}, failing_case)
    saved = json.loads(destination.read_text())
    assert saved["complete"] is False
    assert saved["cases"][0]["actions"] == [{"seconds": 2.0}]
    assert saved["cases"][0]["failure"] == {"type": "RuntimeError", "message": "worker failed"}


def test_preparation_failure_records_case_identity(tmp_path):
    """An error before prediction must still identify the attempted configuration."""
    module = benchmark_module()
    destination = tmp_path / "report.json"

    def rejected_case(on_progress):
        """Represent an unsupported distributed request."""
        raise ValueError("unsupported")

    with pytest.raises(ValueError, match="unsupported"):
        module.record_case(
            {"cases": [], "complete": False},
            destination,
            {"mode": "pyfunc", "rows": 5000000},
            rejected_case,
        )
    saved = json.loads(destination.read_text())
    assert saved["cases"][0]["rows"] == 5000000
    assert saved["cases"][0]["stage"] == "preparation"
    assert saved["cases"][0]["failure"]["type"] == "ValueError"


@pytest.mark.parametrize("enabled,python_rss", [(False, 123), (True, 0)])
def test_unobserved_process_metrics_are_not_zero_memory(enabled, python_rss):
    """Disabled or unobserved process counters cannot become zero-memory claims."""
    module = benchmark_module()
    result = module.normalize_executor_metrics(
        {
            "JVMHeapMemory": 42,
            "ProcessTreePythonRSSMemory": python_rss,
            "ProcessTreeJVMRSSMemory": 0,
            "ProcessTreeOtherRSSMemory": 0,
        },
        process_metrics_enabled=enabled,
    )
    assert result["JVMHeapMemory"] == 42
    assert result["ProcessTreePythonRSSMemory"] is None


def test_observed_process_metrics_remain_exact():
    """Enabled nonzero counters retain their byte values without unit conversion."""
    module = benchmark_module()
    result = module.normalize_executor_metrics(
        {"ProcessTreePythonRSSMemory": 123, "ProcessTreeJVMRSSMemory": 456},
        process_metrics_enabled=True,
    )
    assert result["ProcessTreePythonRSSMemory"] == 123


@pytest.mark.parametrize("predictions", [[1.0], [1.0, None]])
def test_worker_probe_rejects_incomplete_predictions(monkeypatch, predictions):
    """A worker timing must fail when prediction rows are missing or null, even with -O."""
    module = benchmark_module()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w"):
        pass
    monkeypatch.setattr(module, "package_archive", lambda uri: archive.getvalue())
    model = SimpleNamespace(predict=lambda frame: pd.DataFrame({"prediction": predictions}))
    monkeypatch.setitem(
        sys.modules, "mlflow", SimpleNamespace(pyfunc=SimpleNamespace(load_model=lambda _: model))
    )
    monkeypatch.setitem(
        sys.modules,
        "resource",
        SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=1)),
    )
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(
            Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=1))
        ),
    )
    spark = Mock()

    def consume(probe, schema):
        """Execute the worker closure locally without a Spark installation."""
        return SimpleNamespace(collect=lambda: list(probe(iter([object()]))))

    spark.range.return_value.mapInPandas.side_effect = consume
    with pytest.raises(ValueError, match="Invalid worker benchmark predictions"):
        module.worker_load_probe(spark, {"model_uri": "trusted-model", "columns": ["x"]}, 2)
