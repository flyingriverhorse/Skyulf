"""Saved scoring preserves positional outcomes for fitted outlier filters."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.inference.pipeline_scoring import score_pipeline, scoring_output_schema
from skyulf.pipeline import SkyulfPipeline

RECIPES = [
    ("IQR", {"columns": ["x"], "multiplier": 1.5}),
    ("ZScore", {"columns": ["x"], "threshold": 2.0}),
    ("EllipticEnvelope", {"columns": ["x"], "contamination": 0.1}),
    ("ManualBounds", {"bounds": {"x": {"lower": -3.0, "upper": 3.0}}}),
]


def _artifact(
    tmp_path, engine, recipe, *, scoring=False, steps=None, strings=False, tuned=False, model=None
):
    """Fit real learned thresholds and save the same artifact workers consume."""
    values = np.random.default_rng(81).normal(size=120)
    data = pd.DataFrame({"x": values, "target": (values > 0).astype(int)})
    if strings:
        data["target"] = data["target"].map({0: "normal", 1: "premium"})
    node, params = recipe
    config = {
        "preprocessing": steps
        or [
            {"name": "bounds", "transformer": node, "params": params},
        ],
        "modeling": model or {"type": "decision_tree_classifier", "params": {"max_depth": 3}},
    }
    if scoring:
        config.update(
            project_python_source="""import pandas as pd
def eligibility(frame, params):
    return pd.Series(['policy' if x < -50 else None for x in frame.x], index=frame.index)
def outputs(frame, predictions, params):
    assert frame.index.equals(pd.RangeIndex(len(frame)))
    assert frame.x.abs().lt(4).all()
    return pd.DataFrame({'value': predictions.prediction + 10}, index=frame.index)
""",
            project_scoring={
                "eligibility": [
                    {"name": "policy", "version": "1", "function": "eligibility", "params": {}}
                ],
                "outputs": [
                    {
                        "name": "value",
                        "version": "1",
                        "function": "outputs",
                        "params": {},
                        "columns": [{"name": "value", "dtype": "int64"}],
                    }
                ],
            },
        )
    pipeline = SkyulfPipeline(config)
    train, test = data.iloc[:90], data.iloc[90:]
    if engine == "polars":
        train, test = pl.from_pandas(train), pl.from_pandas(test)
    pipeline.fit(SplitDataset(train=train, test=test), target_column="target")
    if tuned:
        assert pipeline.model_estimator is not None
        assert pipeline.model_estimator.model is not None
        pipeline._tuned_thresholds = dict.fromkeys(pipeline.model_estimator.model.classes_, 0.7)
    destination = tmp_path / "artifact"
    save_pipeline(pipeline, destination, use_tuned_thresholds=tuned)
    return load_pipeline(destination)


@pytest.mark.parametrize("recipe", RECIPES, ids=[item[0] for item in RECIPES])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_saved_outlier_scoring_preserves_rows_and_chunk_parity(tmp_path, engine, recipe):
    """Outlier rows retain null results instead of shifting other predictions."""
    artifact = _artifact(tmp_path, engine, recipe)
    frame = pd.DataFrame({"x": [0.0, 100.0, 0.1, -100.0]}, index=[7, 7, 2, 7])
    actual = score_pipeline(frame, artifact)
    chunks = pd.concat([score_pipeline(frame.iloc[i : i + 1], artifact) for i in range(4)])
    assert_frame_equal(actual, chunks)
    assert actual.index.tolist() == frame.index.tolist()
    assert actual.scoring_status.tolist() == ["predicted", "excluded", "predicted", "excluded"]
    assert actual.exclusion_reason.fillna("").tolist() == [
        "",
        "outlier:bounds",
        "",
        "outlier:bounds",
    ]
    assert actual.iloc[[1, 3]][["prediction", "probability_0", "probability_1"]].isna().all().all()
    assert actual.iloc[[0, 2]].prediction.notna().all()
    assert tuple(actual.columns) == tuple(column.name for column in scoring_output_schema(artifact))
    assert_frame_equal(actual, predict_pipeline(frame, artifact))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_outliers_merge_with_project_eligibility_and_skip_output_rules(tmp_path, engine):
    """Earlier policy reasons survive and callbacks see only successfully predicted rows."""
    artifact = _artifact(tmp_path, engine, RECIPES[-1], scoring=True)
    frame = pd.DataFrame({"x": [-100.0, 100.0, 0.0]}, index=[1, 1, 1])
    actual = score_pipeline(frame, artifact)
    assert actual.exclusion_reason.fillna("").tolist() == ["policy", "outlier:bounds", ""]
    assert actual.scoring_status.tolist() == ["excluded", "excluded", "predicted"]
    assert actual.value.iloc[:2].isna().all()
    assert actual.value.iloc[2] in (10, 11)


def test_all_outliers_and_empty_request_skip_estimator(tmp_path, monkeypatch):
    """Empty or wholly excluded requests remain typed outcomes without estimator work."""
    artifact = _artifact(tmp_path, "pandas", RECIPES[-1])

    def forbidden(*args, **kwargs):
        """Fail if a request without eligible rows reaches the estimator."""
        pytest.fail("Estimator invoked for excluded rows")

    monkeypatch.setattr(artifact.pipeline.model_estimator.applier, "predict", forbidden)
    frame = pd.DataFrame({"x": [100.0, -100.0]})
    actual = score_pipeline(frame, artifact)
    empty = score_pipeline(frame.head(0), artifact)
    assert actual.scoring_status.tolist() == ["excluded", "excluded"]
    assert actual.prediction.isna().all()
    assert_frame_equal(actual.head(0), empty)


def test_spark_transport_declares_outlier_predictions_nullable(tmp_path):
    """Spark transport must carry missing integer labels as SQL nulls."""
    from skyulf.integrations.mlflow.spark._spark_output import (
        nullable_outputs,
        prepare_spark_output,
    )

    artifact = _artifact(tmp_path, "pandas", RECIPES[-1])
    actual = score_pipeline(pd.DataFrame({"x": [100.0, 0.0]}), artifact)
    assert nullable_outputs(artifact)
    transported = prepare_spark_output(
        actual, scoring_output_schema(artifact), True, nullable=nullable_outputs(artifact)
    )
    assert str(transported.prediction.dtype) == "float64"
    assert np.isnan(transported.prediction.iloc[0])
    assert transported.prediction.iloc[1] == actual.prediction.iloc[1]


@pytest.mark.parametrize(
    "params",
    [
        {"multiplier": 0.5},
        {"multiplier": 3.0},
        {"threshold": 1.0},
        {"threshold": 5.0},
        {"bounds": {"x": {"lower": -1.0}}},
        {"bounds": {"x": {"upper": 1.0}}},
        {"bounds": {"x": {"lower": None, "upper": None}}},
    ],
)
def test_threshold_sweep_matches_fitted_filter_positions(tmp_path, params):
    """Inference exclusions must exactly reuse each fitted filter's inclusive bounds."""
    node = (
        "IQR" if "multiplier" in params else "ZScore" if "threshold" in params else "ManualBounds"
    )
    recipe = (node, {**params, **({"columns": ["x"]} if node != "ManualBounds" else {})})
    artifact = _artifact(tmp_path, "pandas", recipe)
    record = artifact.pipeline.feature_engineer.fitted_steps[0]
    frame = pd.DataFrame({"x": [-100.0, -1.0, 0.0, 1.0, 100.0]}, index=[9] * 5)
    _, kept = record["applier"].apply((frame, np.arange(len(frame))), record["artifact"])
    actual = score_pipeline(frame, artifact)
    assert np.array_equal(np.flatnonzero(actual.scoring_status.eq("predicted")), kept)
    assert actual.loc[actual.scoring_status.eq("excluded"), "prediction"].isna().all()


def test_ordered_transforms_and_multiple_outliers_keep_first_reason(tmp_path):
    """Each outlier decision sees preceding feature edits without overwriting earlier reasons."""
    steps = [
        {
            "name": "positive",
            "transformer": "ManualBounds",
            "params": {"bounds": {"x": {"lower": -2.0}}},
        },
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
        {
            "name": "scaled_limit",
            "transformer": "ManualBounds",
            "params": {"bounds": {"x": {"upper": 3.0}}},
        },
    ]
    artifact = _artifact(tmp_path, "pandas", RECIPES[-1], steps=steps)
    actual = score_pipeline(pd.DataFrame({"x": [-100.0, 100.0, 0.0]}, index=[4, 4, 4]), artifact)
    assert actual.exclusion_reason.fillna("").tolist() == [
        "outlier:positive",
        "outlier:scaled_limit",
        "",
    ]
    assert actual.prediction.notna().tolist() == [False, False, True]


def test_project_output_probe_uses_only_predicted_outlier_rows(tmp_path):
    """Admission callbacks receive the same nonexcluded inputs as serving callbacks."""
    from skyulf.inference.probed_rules import _probe_outputs
    from skyulf.inference.project_code import load_project_module

    artifact = _artifact(tmp_path, "pandas", RECIPES[-1], scoring=True)
    module = load_project_module(artifact.pipeline.config["project_python_source"])
    frame = pd.DataFrame({"x": [0.0, 100.0] * 20})
    details = _probe_outputs(
        artifact, artifact.pipeline.config["project_scoring"], frame, module, module
    )
    assert details[0]["status"] == "passed", details


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_holdout_metrics_skip_exclusions_but_observer_retains_all_rows(tmp_path, engine):
    """Metric coverage is explicit while the monitoring observer retains excluded outcomes."""
    from skyulf.inference.pipeline_evaluation import evaluate_holdout

    artifact = _artifact(tmp_path, engine, RECIPES[-1])
    heldout = pd.DataFrame({"x": [-100.0, -1.0, 1.0, 100.0], "target": [0, 0, 1, 1]})
    observed = []
    native = pl.from_pandas(heldout) if engine == "polars" else heldout
    metrics = evaluate_holdout(
        artifact,
        native,
        target_column="target",
        on_predictions=lambda actual, predicted: observed.append((actual, predicted)),
    )
    assert metrics["heldout_total_rows"] == 4
    assert metrics["heldout_scored_rows"] == 2
    assert metrics["heldout_excluded_rows"] == 2
    assert metrics["heldout_accuracy"] == 1.0
    assert len(observed[0][0]) == len(observed[0][1]) == 4
    assert observed[0][1].scoring_status.tolist() == [
        "excluded",
        "predicted",
        "predicted",
        "excluded",
    ]


def test_holdout_rejects_insufficient_survivors_before_observer(tmp_path):
    """An almost-empty accepted subset cannot become model-quality evidence."""
    from skyulf.inference.pipeline_evaluation import evaluate_holdout

    artifact = _artifact(tmp_path, "pandas", RECIPES[-1])
    observed = []
    with pytest.raises(ValueError, match="at least two"):
        evaluate_holdout(
            artifact,
            pd.DataFrame({"x": [100.0, 0.0], "target": [1, 0]}),
            target_column="target",
            on_predictions=lambda *args: observed.append(args),
        )
    assert observed == []


def test_comparison_rejects_different_masks_even_with_equal_counts(tmp_path):
    """Equal coverage counts must not conceal candidate and champion scoring different rows."""
    pytest.importorskip("mlflow")
    from skyulf.integrations.mlflow.lifecycle.validation import _comparison_metrics

    candidate = _artifact(
        tmp_path / "candidate", "pandas", ("ManualBounds", {"bounds": {"x": {"lower": -2.0}}})
    )
    champion = _artifact(
        tmp_path / "champion", "pandas", ("ManualBounds", {"bounds": {"x": {"upper": 2.0}}})
    )
    heldout = pd.DataFrame({"x": [-100.0, -1.0, 1.0, 100.0], "target": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="same.*rows|different.*rows"):
        _comparison_metrics(candidate, champion, heldout, "target")
    equal, repeated = _comparison_metrics(candidate, candidate, heldout, "target")
    assert equal == repeated
    assert equal["heldout_scored_rows"] == 3


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_string_labels_tuned_thresholds_and_pyfunc_preserve_exclusions(tmp_path, engine):
    """The endpoint pyfunc and direct saved scorer share tuned string-label outcomes."""
    pytest.importorskip("mlflow")
    from types import SimpleNamespace

    from skyulf.integrations.mlflow.models.pipeline_model import SkyulfPipelinePythonModel

    artifact = _artifact(tmp_path, engine, RECIPES[-1], strings=True, tuned=True)
    frame = pd.DataFrame({"x": [-1.0, 100.0, 1.0]}, index=[3, 3, 3])
    expected = score_pipeline(frame, artifact)
    pyfunc = SkyulfPipelinePythonModel()
    pyfunc.load_context(SimpleNamespace(artifacts={"local_pipeline": str(tmp_path / "artifact")}))
    assert_frame_equal(pyfunc.predict(None, frame), expected)
    assert expected.prediction.tolist() == ["normal", pd.NA, "premium"]
    assert str(expected.prediction.dtype) == "string"


@pytest.mark.parametrize("scoring", [False, True])
def test_probed_model_uses_outlier_survivors_and_typed_empty_schema(tmp_path, scoring):
    """An empirically admitted model must probe eligible features and its actual empty output."""
    from skyulf.inference.probed_models import probe_model

    artifact = _artifact(
        tmp_path,
        "pandas",
        RECIPES[-1],
        scoring=scoring,
        model={"type": "hist_gradient_boosting_classifier", "params": {"max_iter": 8}},
    )
    holdout = pd.DataFrame({"x": [0.0, 0.5, -0.5, 100.0] * 8})
    evidence = probe_model(artifact, holdout)
    assert evidence["rows"] == 24
    assert all(check["status"] == "passed" for check in evidence["checks"])


def test_reused_presplit_exclusion_reason_precedes_outlier_filter(tmp_path):
    """The already-published pre-split exclusion contract retains its first reason."""
    artifact = _artifact(tmp_path, "pandas", RECIPES[-1], scoring=True)
    config = artifact.pipeline.config["project_scoring"]
    config["pre_split"] = {
        "steps": [
            {
                "name": "raw_bound",
                "transformer": "ManualBounds",
                "params": {"bounds": {"x": {"lower": -50.0}}},
            }
        ],
        "target_column": "target",
        "engine": "pandas",
        "skipped_target_steps": [],
    }
    actual = score_pipeline(pd.DataFrame({"x": [-100.0, 100.0, 0.0]}), artifact)
    assert "raw_bound" in actual.exclusion_reason.iloc[0]
    assert actual.exclusion_reason.iloc[1] == "outlier:bounds"
    assert pd.isna(actual.exclusion_reason.iloc[2])


def test_direct_array_prediction_keeps_explicit_row_loss_error(tmp_path):
    """Callers needing exclusion reasons must use saved scoring rather than unkeyed arrays."""
    artifact = _artifact(tmp_path, "pandas", RECIPES[-1])
    with pytest.raises(ValueError, match="changed row count"):
        artifact.pipeline.predict(pd.DataFrame({"x": [100.0, 0.0]}))


def test_sealed_outlier_model_binds_raw_and_surviving_probe_counts(tmp_path):
    """An outlier model certificate must distinguish raw sample size from scored sample size."""
    from skyulf.inference.partition_safety import require_partition_safe_pipeline
    from skyulf.inference.probed_models import require_probed_model
    from skyulf.integrations.databricks.training.shared.preprocessing_checks import (
        seal_partition_probe,
    )

    artifact = _artifact(
        tmp_path,
        "pandas",
        RECIPES[-1],
        model={"type": "hist_gradient_boosting_classifier", "params": {"max_iter": 8}},
    )
    sample = pd.DataFrame({"x": [0.0, 0.5, -0.5, 100.0] * 8})
    sealed = seal_partition_probe(artifact, sample, tmp_path / "artifact", enabled=True)
    certificate = require_partition_safe_pipeline(sealed)
    evidence = require_probed_model(sealed)
    assert evidence["rows"] == 24
    assert evidence["input_rows"] == 32
    assert certificate.steps[0].action == "exclude_outliers"
    for key, value in [("input_rows", 31), ("rows", 33), ("rows", True)]:
        previous = evidence[key]
        evidence[key] = value
        with pytest.raises(ValueError):
            require_probed_model(sealed)
        evidence[key] = previous
