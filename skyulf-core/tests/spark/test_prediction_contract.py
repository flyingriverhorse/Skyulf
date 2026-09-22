"""Classification output contracts for distributed inference."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.data.dataset import SplitDataset
from skyulf.inference.bundle import build_bundle, predict_local
from skyulf.inference.spark import predict_spark
from skyulf.pipeline import SkyulfPipeline


def _binary_pipeline() -> tuple[SkyulfPipeline, pd.DataFrame]:
    """Fit a binary string-label model whose class order is observable in output."""
    train = pd.DataFrame(
        {"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0], "label": ["no", "no", "no", "yes", "yes", "yes"]}
    )
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="label")
    return pipeline, pd.DataFrame({"x": [-2.5, -0.25, 0.25, 2.5]})


def _multiclass_pipeline() -> tuple[SkyulfPipeline, pd.DataFrame]:
    """Fit a three-class integer-label model to cover non-binary probability output."""
    train = pd.DataFrame(
        {
            "x": [-3.0, -2.5, -2.0, 0.0, 0.2, -0.2, 2.0, 2.5, 3.0],
            "z": [0.0, 0.2, -0.2, 2.0, 2.2, 1.8, 0.0, -0.2, 0.2],
            "label": [10, 10, 10, 20, 20, 20, 30, 30, 30],
        }
    )
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    pipeline.fit(SplitDataset(train=train, test=train.head(0)), target_column="label")
    return pipeline, pd.DataFrame({"x": [-2.0, 0.0, 2.0], "z": [0.0, 2.0, 0.0]})


def _spark_output(
    spark, bundle, query: pd.DataFrame, mode: str, *, partitions: int = 2, arrow_rows: int = 2
):
    """Run one classification bundle through a small, repartitioned Spark frame."""
    rows = [
        (index, *values) for index, values in enumerate(query.itertuples(index=False, name=None))
    ]
    schema = "id long, " + ", ".join(f"{name} double" for name in query.columns)
    frame = spark.createDataFrame(rows, schema).repartition(partitions)
    previous = spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", str(arrow_rows))
    try:
        return (
            predict_spark(
                frame,
                bundle,
                frame_spec=FrameSpec(row_keys=("id",)),
                options=ExecutionOptions("spark", python_batch_rows=2),
                mode=mode,
            )
            .orderBy("id")
            .collect()
        )
    finally:
        spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", previous)


@pytest.mark.parametrize("mode", ["native_features", "python_pipeline"])
@pytest.mark.parametrize("partitions,arrow_rows", [(1, 1), (3, 5)])
def test_binary_string_labels_and_probability_order_match_local(
    spark, mode, partitions, arrow_rows
):
    """Distributed class labels and probability positions must preserve the bundle manifest."""
    pipeline, query = _binary_pipeline()
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x",))
    expected = predict_local(query, bundle).reset_index(drop=True)
    assert pipeline.model_estimator is not None and pipeline.model_estimator.model is not None
    model = pipeline.model_estimator.model
    rows = _spark_output(spark, bundle, query, mode, partitions=partitions, arrow_rows=arrow_rows)

    assert bundle.classes == ("no", "yes")
    assert bundle.positive_label == "yes"
    assert bundle.classes == tuple(model.classes_)
    assert tuple(row.asDict() for row in rows)
    actual = pd.DataFrame([row.asDict() for row in rows]).drop(columns="id")
    assert list(actual.columns) == ["prediction", *bundle.probability_columns]
    np.testing.assert_array_equal(actual.prediction.to_numpy(), expected.prediction.to_numpy())
    np.testing.assert_allclose(
        actual[list(bundle.probability_columns)].to_numpy(),
        model.predict_proba(query.to_numpy()),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", ["native_features", "python_pipeline"])
def test_multiclass_integer_labels_match_local(spark, mode):
    """Multiclass integer predictions and class-ordered probabilities must remain aligned."""
    pipeline, query = _multiclass_pipeline()
    bundle = build_bundle(pipeline, input_stage="raw", feature_order=("x", "z"))
    assert pipeline.model_estimator is not None and pipeline.model_estimator.model is not None
    model = pipeline.model_estimator.model
    rows = _spark_output(spark, bundle, query, mode)
    actual = pd.DataFrame([row.asDict() for row in rows]).drop(columns="id")

    assert bundle.classes == (10, 20, 30)
    assert bundle.positive_label is None
    assert bundle.classes == tuple(model.classes_)
    assert actual.prediction.dtype.kind in "iu"
    np.testing.assert_array_equal(actual.prediction.to_numpy(), model.predict(query.to_numpy()))
    np.testing.assert_allclose(
        actual[list(bundle.probability_columns)].to_numpy(),
        model.predict_proba(query.to_numpy()),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("mode", ["native_features", "python_pipeline"])
def test_saved_threshold_decisions_match_local(spark, mode):
    """A frozen tuned threshold must have the same precedence in local and Spark inference."""
    pipeline, query = _binary_pipeline()
    pipeline.optimize_thresholds(
        query, np.array(["no", "yes", "yes", "yes"]), accuracy_score, grid_points=5
    )
    bundle = build_bundle(
        pipeline, input_stage="raw", feature_order=("x",), use_tuned_thresholds=True
    )
    expected = predict_local(query, bundle).reset_index(drop=True)
    rows = _spark_output(spark, bundle, query, mode)
    actual = pd.DataFrame([row.asDict() for row in rows]).drop(columns="id")

    assert bundle.manifest.thresholds.source == "pipeline_override"
    assert bundle.manifest.thresholds.values != (0.5, 0.5)
    assert pipeline.model_estimator is not None and pipeline.model_estimator.model is not None
    default_predictions = pipeline.model_estimator.model.predict(query.to_numpy())
    assert np.any(actual.prediction.to_numpy() != default_predictions)
    np.testing.assert_array_equal(actual.prediction.to_numpy(), expected.prediction.to_numpy())
    np.testing.assert_allclose(
        actual[list(bundle.probability_columns)].to_numpy(),
        expected[list(bundle.probability_columns)].to_numpy(),
        rtol=1e-10,
        atol=1e-12,
    )
