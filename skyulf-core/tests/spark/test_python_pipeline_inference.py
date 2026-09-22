"""Worker-side Python feature engineering must preserve local predictions."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import skyulf.inference as inference
from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.inference._manifest import checksum, semantic_digest
from skyulf.inference.bundle import build_bundle, predict_local
from skyulf.inference.spark import _validate_python_pipeline
from skyulf.preprocessing.pipeline import FeatureEngineer


@pytest.fixture
def regression_bundle(fitted_regression_pipeline):
    """Build a raw regression bundle from each local training engine."""
    return build_bundle(fitted_regression_pipeline, input_stage="raw", feature_order=("x", "z"))


def _predict(frame, bundle, **kwargs):
    """Call the distributed runner with the explicit Spark contract."""
    return inference.predict_spark(
        frame,
        bundle,
        frame_spec=kwargs.pop("frame_spec", FrameSpec(row_keys=("id",))),
        options=kwargs.pop("options", ExecutionOptions("spark", python_batch_rows=2)),
        mode=kwargs.pop("mode", "python_pipeline"),
        **kwargs,
    )


@pytest.mark.parametrize("partitions,arrow_rows", [(1, 1), (3, 5)])
def test_python_pipeline_matches_local_for_each_training_engine(
    spark, regression_bundle, arrow_rows, partitions
):
    """Worker Python FE must match predict_local across partition and Arrow boundaries."""
    raw = pd.DataFrame({"x": [150.0, 80.0, np.nan, 110.0], "z": [9.0, 3.0, 4.0, np.nan]})
    expected = predict_local(raw, regression_bundle)
    frame = spark.createDataFrame(
        [(10, 150.0, 9.0), (11, 80.0, 3.0), (12, None, 4.0), (13, 110.0, float("nan"))],
        "id long, x double, z double",
    ).repartition(partitions)
    old_limit = spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", str(arrow_rows))
    try:
        rows = _predict(frame, regression_bundle).orderBy("id").collect()
    finally:
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", old_limit)
    np.testing.assert_allclose([row.prediction for row in rows], expected.prediction, rtol=1e-10)
    assert [row.id for row in rows] == [10, 11, 12, 13]


def test_python_pipeline_applies_frozen_feature_state_once_per_worker_iterator(
    regression_bundle, monkeypatch
):
    """A worker iterator restores state once and transforms each nonempty batch once."""
    module = __import__("skyulf.inference.spark", fromlist=["FeatureEngineer"])
    original_restore = module.FeatureEngineer.from_state
    restored = []
    transformed = []

    def record_restore(*args, **kwargs):
        """Record worker state restoration while retaining the real implementation."""
        engineer = original_restore(*args, **kwargs)
        restored.append(engineer)
        original_transform = engineer.transform

        def record_transform(data, **transform_kwargs):
            """Record each frozen batch application without changing its result."""
            transformed.append(len(data))
            return original_transform(data, **transform_kwargs)

        engineer.transform = record_transform
        return engineer

    monkeypatch.setattr(module.FeatureEngineer, "from_state", record_restore)
    worker = module._python_pipeline_prediction_iterator(
        regression_bundle.feature_state,
        regression_bundle.model_payload,
        regression_bundle.manifest,
        ("id",),
        2,
        8 * 1024 * 1024,
    )
    batches = iter(
        [
            pd.DataFrame({"id": [1, 2], "x": [150.0, 80.0], "z": [9.0, 3.0]}),
            pd.DataFrame({"id": [3], "x": [110.0], "z": [4.0]}),
        ]
    )
    result = pd.concat(worker(batches), ignore_index=True)
    assert len(restored) == 1
    assert transformed == [2, 1]
    assert result.columns.tolist() == ["id", "prediction"]


def test_python_pipeline_never_fits_or_materializes_source(spark, regression_bundle, monkeypatch):
    """Distributed Python inference must not fit or collect the Spark source frame."""
    frame = spark.createDataFrame([(1, 150.0, 9.0)], "id long, x double, z double")

    def forbidden(*args, **kwargs):
        """Fail if inference attempts training or full-frame materialization."""
        pytest.fail("inference attempted fitting or driver materialization")

    monkeypatch.setattr(type(frame), "toPandas", forbidden)
    monkeypatch.setattr(type(frame), "toLocalIterator", forbidden)
    monkeypatch.setattr(
        __import__("skyulf.inference.spark", fromlist=["FeatureEngineer"]).FeatureEngineer,
        "fit_transform",
        forbidden,
    )
    rows = _predict(frame, regression_bundle).collect()
    assert len(rows) == 1


def test_python_pipeline_rejects_unsupported_context_before_action(
    spark, regression_bundle, monkeypatch
):
    """Unknown portable pipeline state must be rejected before Spark validation actions."""
    frame = spark.createDataFrame([(1, 150.0, 9.0)], "id long, x double, z double")
    payload = regression_bundle.feature_state.replace(b"SimpleImputer", b"RollingWindowStep")
    manifest = regression_bundle.manifest.model_copy(
        update={"fe_sha256": checksum(payload), "fe_semantic_digest": "invalid"}
    )
    manifest = manifest.model_copy(update={"semantic_digest": semantic_digest(manifest)})
    unsupported = replace(regression_bundle, feature_state=payload, manifest=manifest)
    for action in ("collect", "count", "first", "take", "toPandas", "toLocalIterator"):
        monkeypatch.setattr(
            type(frame), action, lambda *args, **kwargs: pytest.fail("Spark action")
        )
    with pytest.raises(
        (ValueError, UnsupportedExecutionError), match="portable|semantic|unsupported"
    ):
        _predict(frame, unsupported)


def test_python_pipeline_rejects_nonportable_imputer_strategy_before_action(regression_bundle):
    """Median imputer state cannot silently enter the mean/constant worker path."""
    engineer = FeatureEngineer.from_state(regression_bundle.feature_state)
    engineer.fitted_steps[0]["params"]["strategy"] = "median"
    with pytest.raises(UnsupportedExecutionError, match="mean or constant"):
        _validate_python_pipeline(engineer)


def test_native_mode_still_uses_native_feature_path(spark, regression_bundle):
    """The native mode remains available after adding the Python pipeline mode."""
    frame = spark.createDataFrame([(1, 150.0, 9.0)], "id long, x double, z double")
    rows = inference.predict_spark(
        frame,
        regression_bundle,
        frame_spec=FrameSpec(row_keys=("id",)),
        options=ExecutionOptions("spark", python_batch_rows=2),
        mode="native_features",
    ).collect()
    assert rows[0].id == 1
