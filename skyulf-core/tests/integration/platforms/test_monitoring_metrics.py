"""Monitoring reports use saved, keyed observations and finite evidence."""

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import polars as pl
import pytest

from skyulf.integrations.databricks.observability.monitoring.local.monitoring_metrics import (
    build_monitoring_report,
)

AS_OF = datetime(2026, 10, 1, tzinfo=UTC)


def frame(engine: str, rows: list[dict]) -> pd.DataFrame | pl.DataFrame:
    """Keep the same input records for both supported frame engines."""
    return (
        pd.DataFrame(rows) if engine == "pandas" else pl.DataFrame(rows, infer_schema_length=None)
    )


def report(engine: str, *, reference=None, current=None, predictions=None, labels=None, **kwargs):
    """Apply the declared source contract to small hand-checked records."""
    reference = (
        reference
        if reference is not None
        else [{"value": 1, "kind": "a"}, {"value": 2, "kind": "b"}]
    )
    current = (
        current
        if current is not None
        else [
            {"id": "one", "value": 1, "kind": "a"},
            {"id": "two", "value": 2, "kind": "b"},
        ]
    )
    predictions = (
        predictions
        if predictions is not None
        else [
            {"id": "one", "prediction": "no"},
            {"id": "two", "prediction": "yes"},
        ]
    )
    task = kwargs.pop("task", "classification")
    classes = kwargs.pop("classes", ("no", "yes"))
    return build_monitoring_report(
        frame(engine, reference),
        frame(engine, current),
        frame(engine, predictions),
        None if labels is None else frame(engine, labels),
        feature_columns=("value", "kind"),
        record_key_columns=("id",),
        target_column="outcome",
        result_available_at_column="available_at",
        as_of=AS_OF,
        task=task,
        classes=classes,
        **kwargs,
    )


def metric(result: dict, category: str, name: str, column: str = "outcome") -> dict:
    """Read one public metric rather than depending on list ordering."""
    return next(
        item
        for item in result["metrics"]
        if item["category"] == category
        and item["metric_name"] == name
        and item["column_name"] == column
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_identical_features_are_healthy_without_labels(engine):
    """Missing labels cannot conceal healthy measured feature drift."""
    reference = [{"value": index % 2 + 1, "kind": "ab"[index % 2]} for index in range(40)]
    current = [{"id": str(index), **row} for index, row in enumerate(reference)]
    predictions = [
        {"id": str(index), "prediction": "yes" if index % 2 else "no"} for index in range(40)
    ]
    result = report(engine, reference=reference, current=current, predictions=predictions)
    assert result["status"] == "healthy"
    assert (result["reference_rows"], result["current_rows"], result["scored_rows"]) == (40, 40, 40)
    assert result["labeled_rows"] == 0 and result["label_coverage"] == 0
    assert metric(result, "performance", "accuracy")["status"] == "unavailable"
    json.dumps(result, allow_nan=False)
    assert any("label" in note.lower() for note in result["notes"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_numeric_and_categorical_shifts_are_counted(engine):
    """Feature distribution shifts must win over missing performance labels."""
    reference = [{"value": index % 2 + 1, "kind": "ab"[index % 2]} for index in range(40)]
    changed = [
        {"id": str(index), "value": 100 * (index % 2 + 1), "kind": "cd"[index % 2]}
        for index in range(40)
    ]
    predictions = [
        {"id": str(index), "prediction": "yes" if index % 2 else "no"} for index in range(40)
    ]
    result = report(engine, reference=reference, current=changed, predictions=predictions)
    assert result["status"] == "drift" and result["drifted_columns"] == 2
    assert metric(result, "drift", "psi_categorical", "kind")["has_issue"]
    assert any(item["has_issue"] for item in result["metrics"] if item["column_name"] == "value")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_empty_and_all_null_features_are_not_healthy(engine):
    """No observations and all missing observations cannot vote healthy."""
    empty = report(engine, current=[], predictions=[])
    assert empty["status"] == "no_data"
    missing = report(
        engine,
        current=[{"id": "one", "value": None, "kind": None}],
        predictions=[{"id": "one", "prediction": "no"}],
    )
    assert missing["status"] == "degraded"
    assert metric(missing, "drift", "unavailable", "value")["status"] == "unavailable"
    assert metric(missing, "quality", "missing_fraction", "value")["value"] == 1


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_schema_and_unsupported_features_remain_visible(engine):
    """Absent and unsupported declared features must not disappear from the verdict."""
    absent = report(engine, current=[{"id": "one", "value": 1}, {"id": "two", "value": 2}])
    assert absent["status"] == "drift" and absent["drifted_columns"] == 1
    assert metric(absent, "drift", "schema_missing", "kind")["has_issue"]
    nested = [{"value": 1, "kind": ["a"]}, {"value": 2, "kind": ["b"]}]
    present = [{"id": "one", "value": 1, "kind": ["a"]}, {"id": "two", "value": 2, "kind": ["b"]}]
    unsupported = report(engine, reference=nested, current=present)
    assert unsupported["status"] == "degraded"
    assert metric(unsupported, "drift", "unavailable", "kind")["status"] == "unavailable"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_shuffled_predictions_join_by_key_and_future_labels_wait(engine):
    """Arrival order and future outcome rows cannot alter observed accuracy."""
    predictions = [{"id": "two", "prediction": "yes"}, {"id": "one", "prediction": "no"}]
    labels = [
        {"id": "one", "outcome": "no", "available_at": AS_OF},
        {"id": "two", "outcome": "no", "available_at": AS_OF + timedelta(days=1)},
    ]
    result = report(engine, predictions=predictions, labels=labels)
    assert result["labeled_rows"] == 1
    assert result["label_coverage"] == 0.5
    assert metric(result, "performance", "accuracy")["status"] == "unavailable"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("source", ["current", "predictions", "labels"])
@pytest.mark.parametrize("bad_key", [None, "one"])
def test_duplicate_and_null_keys_rejected(engine, source, bad_key):
    """Ambiguous record identity must fail before any metric is computed."""
    rows = {
        "current": [
            {"id": "one", "value": 1, "kind": "a"},
            {"id": bad_key, "value": 2, "kind": "b"},
        ],
        "predictions": [{"id": "one", "prediction": "no"}, {"id": bad_key, "prediction": "yes"}],
        "labels": [
            {"id": "one", "outcome": "no", "available_at": AS_OF},
            {"id": bad_key, "outcome": "yes", "available_at": AS_OF},
        ],
    }
    with pytest.raises(ValueError, match="key"):
        report(engine, **{source: rows[source]})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_prediction_must_belong_to_current_population(engine):
    """A saved prediction from another population must not enter performance."""
    with pytest.raises(ValueError, match="prediction.*key"):
        report(engine, predictions=[{"id": "other", "prediction": "no"}])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("available", [datetime(2026, 10, 1), "not a date", None])
def test_invalid_label_availability_rejected(engine, available):
    """Naive or malformed timestamps must not make a label eligible."""
    labels = [{"id": "one", "outcome": "no", "available_at": available}]
    with pytest.raises(ValueError, match="availability"):
        report(engine, labels=labels)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_excluded_predictions_do_not_enter_performance(engine):
    """A skipped scoring row stays in the population count but outside accuracy."""
    predictions = [
        {
            "id": "one",
            "prediction": "no",
            "scoring_status": "predicted",
            "probability_0": 0.9,
            "probability_1": 0.1,
        },
        {
            "id": "two",
            "prediction": None,
            "scoring_status": "excluded",
            "probability_0": None,
            "probability_1": None,
        },
    ]
    labels = [
        {"id": "one", "outcome": "no", "available_at": AS_OF},
        {"id": "two", "outcome": "yes", "available_at": AS_OF},
    ]
    result = report(engine, predictions=predictions, labels=labels)
    assert result["scored_rows"] == 1 and result["labeled_rows"] == 1
    assert metric(result, "performance", "accuracy")["status"] == "unavailable"
    assert any("excluded" in note for note in result["notes"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_missing_classifier_labels_do_not_count_as_observed(engine):
    """Partially populated outcome tables must not turn unobserved labels into failures."""
    labels = [
        {"id": "one", "outcome": "no", "available_at": AS_OF},
        {"id": "two", "outcome": None, "available_at": AS_OF},
    ]
    result = report(engine, labels=labels)
    assert result["labeled_rows"] == 1
    assert result["label_coverage"] == 0.5


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_probability_loss_uses_saved_class_order(engine):
    """Nonalphabetic class order must not invert otherwise correct probability evidence."""
    import math

    predictions = [
        {"id": "one", "prediction": "yes", "probability_0": 0.9, "probability_1": 0.1},
        {"id": "two", "prediction": "no", "probability_0": 0.1, "probability_1": 0.9},
    ]
    labels = [
        {"id": "one", "outcome": "yes", "available_at": AS_OF},
        {"id": "two", "outcome": "no", "available_at": AS_OF},
    ]
    result = report(engine, predictions=predictions, labels=labels, classes=("yes", "no"))
    assert metric(result, "performance", "log_loss")["value"] == pytest.approx(-math.log(0.9))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_regression_uses_only_saved_keyed_pairs(engine):
    """Regression evidence must come from saved predictions, including shuffled keys."""
    predictions = [{"id": "two", "prediction": 3.0}, {"id": "one", "prediction": 1.0}]
    labels = [
        {"id": "one", "outcome": 1.0, "available_at": AS_OF},
        {"id": "two", "outcome": 2.0, "available_at": AS_OF},
    ]
    result = report(engine, predictions=predictions, labels=labels, task="regression", classes=())
    assert metric(result, "performance", "rmse")["value"] == pytest.approx(2**-0.5)
    assert metric(result, "performance", "mae")["value"] == 0.5
    assert metric(result, "performance", "r2")["value"] == -1


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_binary_strings_use_saved_positive_class_and_probabilities(engine):
    """Binary F1 and probability metrics must respect saved class order."""
    predictions = [
        {"id": "one", "prediction": "yes", "probability_0": 0.2, "probability_1": 0.8},
        {"id": "two", "prediction": "no", "probability_0": 0.7, "probability_1": 0.3},
    ]
    labels = [
        {"id": "one", "outcome": "yes", "available_at": AS_OF},
        {"id": "two", "outcome": "yes", "available_at": AS_OF},
    ]
    result = report(engine, predictions=predictions, labels=labels)
    assert metric(result, "performance", "f1")["value"] == pytest.approx(2 / 3)
    json.dumps(result, allow_nan=False)
    assert metric(result, "performance", "log_loss")["status"] == "measured"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_invalid_classes_probabilities_and_thresholds_rejected(engine):
    """Malformed model evidence cannot be scored as if it matched the fitted contract."""
    with pytest.raises(ValueError, match="class"):
        report(engine, classes=("yes", "yes"))
    with pytest.raises(ValueError, match="probabilit"):
        report(
            engine,
            predictions=[
                {"id": "one", "prediction": "no", "probability_0": 0.9},
                {"id": "two", "prediction": "yes", "probability_0": 0.2},
            ],
        )
    with pytest.raises(ValueError, match="threshold"):
        report(engine, thresholds={"psi": float("nan")})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_nonfinite_feature_and_prediction_values_never_serialize_as_nan(engine):
    """NaN and infinity must create finite JSON evidence or explicit unavailability."""
    changed = [
        {"id": "one", "value": float("nan"), "kind": "a"},
        {"id": "two", "value": float("inf"), "kind": "b"},
    ]
    result = report(engine, current=changed)
    assert result["status"] == "degraded"
    assert metric(result, "quality", "nonfinite_fraction", "value")["value"] == 1
    json.dumps(result, allow_nan=False)
    assert metric(result, "drift", "unavailable", "value")["status"] == "unavailable"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "task,prediction", [("regression", float("nan")), ("classification", "unknown")]
)
def test_invalid_saved_prediction_fails_even_without_labels(engine, task, prediction):
    """Missing outcomes cannot mask broken outputs from rows declared predicted."""
    with pytest.raises(ValueError, match="Saved prediction"):
        report(engine, predictions=[{"id": "one", "prediction": prediction}], task=task)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("empty", [False, True])
def test_no_predicted_outputs_cannot_be_healthy(engine, empty):
    """A fully excluded population still has feature evidence but no usable model outputs."""
    predictions = (
        []
        if empty
        else [
            {"id": key, "prediction": None, "scoring_status": "excluded"} for key in ("one", "two")
        ]
    )
    result = report(engine, predictions=predictions)
    assert result["scored_rows"] == 0
    assert result["status"] == "degraded"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_extended_classification_metrics_preserve_saved_positive_class(engine):
    """Core's G-score, MCC and PR-AUC must survive saved-class ordering and key joins."""
    truth = ["yes", "yes", "no", "no"]
    guesses = ["yes", "yes", "yes", "no"]
    probabilities = [0.1, 0.2, 0.6, 0.9]
    result = report(
        engine,
        classes=("yes", "no"),
        current=[{"id": str(i), "value": i, "kind": "a"} for i in range(4)],
        predictions=[
            {
                "id": str(i),
                "prediction": guesses[i],
                "probability_0": 1 - probabilities[i],
                "probability_1": probabilities[i],
            }
            for i in reversed(range(4))
        ],
        labels=[
            {"id": str(i), "outcome": value, "available_at": AS_OF} for i, value in enumerate(truth)
        ],
    )
    for name, expected in {
        "accuracy": 0.75,
        "balanced_accuracy": 0.75,
        "g_score": 0.75,
        "matthews_corrcoef": 3**-0.5,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 2 / 3,
        "pr_auc": 1.0,
        "roc_auc": 1.0,
    }.items():
        assert metric(result, "performance", name)["value"] == pytest.approx(expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_multiclass_probability_metrics_use_full_saved_class_order(engine):
    """All Core multiclass AUC variants use the same saved probability columns."""
    classes = ("z", "a", "m")
    result = report(
        engine,
        classes=classes,
        current=[{"id": str(i), "value": i, "kind": "a"} for i in range(3)],
        predictions=[
            {
                "id": str(i),
                "prediction": label,
                **{f"probability_{j}": 0.8 if i == j else 0.1 for j in range(3)},
            }
            for i, label in enumerate(classes)
        ],
        labels=[
            {"id": str(i), "outcome": label, "available_at": AS_OF}
            for i, label in enumerate(classes)
        ],
    )
    for name in (
        "roc_auc_ovr",
        "roc_auc_ovo",
        "roc_auc_ovr_weighted",
        "roc_auc_ovo_weighted",
        "pr_auc_weighted",
        "g_score",
    ):
        assert metric(result, "performance", name)["value"] == pytest.approx(1.0)
    assert metric(result, "performance", "log_loss")["value"] == pytest.approx(0.2231435513)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_extended_regression_metrics_use_saved_predictions(engine):
    """Monitoring exposes Core's MSE, MAPE and explained variance without rerunning inference."""
    result = report(
        engine,
        task="regression",
        classes=(),
        predictions=[{"id": "one", "prediction": 1.0}, {"id": "two", "prediction": 3.0}],
        labels=[
            {"id": "one", "outcome": 1.0, "available_at": AS_OF},
            {"id": "two", "outcome": 2.0, "available_at": AS_OF},
        ],
    )
    assert metric(result, "performance", "mse")["value"] == 0.5
    assert metric(result, "performance", "mape")["value"] == 0.25
    assert metric(result, "performance", "explained_variance")["value"] == 0.0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("decimal_columns", ["labels", "predictions", "both"])
def test_decimal_regression_outcomes_match_float_metrics(engine, decimal_columns):
    """Spark Decimal outcomes must count toward coverage and match the numeric metric oracle."""
    truth = [1.0, 2.0]
    predictions = [1.5, 2.5]
    if decimal_columns in {"labels", "both"}:
        truth = [Decimal(str(value)) for value in truth]
    if decimal_columns in {"predictions", "both"}:
        predictions = [Decimal(str(value)) for value in predictions]
    result = report(
        engine,
        task="regression",
        classes=(),
        predictions=[
            {"id": key, "prediction": value}
            for key, value in zip(("one", "two"), predictions, strict=True)
        ],
        labels=[
            {"id": key, "outcome": value, "available_at": AS_OF}
            for key, value in zip(("one", "two"), truth, strict=True)
        ],
    )
    assert result["labeled_rows"] == 2
    assert result["label_coverage"] == 1
    assert metric(result, "performance", "mae")["value"] == pytest.approx(0.5)
    assert metric(result, "performance", "mse")["value"] == pytest.approx(0.25)
    assert metric(result, "performance", "rmse")["value"] == pytest.approx(0.5)
    assert metric(result, "performance", "r2")["value"] == pytest.approx(0.0)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("missing", [None, float("nan")])
@pytest.mark.parametrize("current_missing", [5, 10])
@pytest.mark.parametrize("shift", [0, 10])
def test_processed_baseline_missingness_allows_other_feature_drift(
    engine, missing, current_missing, shift
):
    """Missingness already handled by the model must not suppress valid drift decisions."""
    from skyulf.integrations.databricks.jobs.lifecycle.retraining_task import (
        _drift_metrics_decision,
    )

    reference = [{"value": missing if i < 10 else i % 10, "kind": i % 10} for i in range(100)]
    current = [
        {
            "id": str(i),
            "value": missing if i < current_missing else i % 10,
            "kind": i % 10 + shift,
        }
        for i in range(100)
    ]
    result = report(
        engine,
        reference=reference,
        current=current,
        predictions=[{"id": str(i), "prediction": "yes"} for i in reversed(range(100))],
    )
    assert metric(result, "quality", "missing_fraction", "value")["value"] == current_missing / 100
    assert metric(result, "quality", "missing_fraction", "value")["threshold"] == 0.1
    assert not any(item["has_issue"] for item in result["metrics"] if item["category"] == "quality")
    assert result["status"] == ("drift" if shift else "healthy")
    assert _drift_metrics_decision(result["metrics"]) == ("ready" if shift else "no_drift")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("case", ["increased", "new", "excluded", "unpredicted", "infinity"])
def test_missingness_policy_retains_input_quality_guards(engine, case):
    """New or significantly increased missingness and unhandled rows still block training."""
    from skyulf.integrations.databricks.jobs.lifecycle.retraining_task import (
        _drift_metrics_decision,
    )

    reference_missing = 0 if case == "new" else 10
    current_missing = 30 if case == "increased" else 10
    reference = [
        {"value": None if i < reference_missing else float(i % 10), "kind": i % 10}
        for i in range(100)
    ]
    current = [
        {
            "id": str(i),
            "value": None if i < current_missing else float(i % 10),
            "kind": i % 10 + 10,
        }
        for i in range(100)
    ]
    predictions = [
        {"id": str(i), "prediction": "yes", "scoring_status": "predicted"} for i in range(100)
    ]
    if case == "excluded":
        predictions[0]["scoring_status"] = "excluded"
    elif case == "unpredicted":
        predictions.pop(0)
    elif case == "infinity":
        reference[20]["value"] = current[20]["value"] = float("inf")
    result = report(engine, reference=reference, current=current, predictions=predictions)
    assert result["status"] == "drift"
    assert _drift_metrics_decision(result["metrics"]) == "quality_issue"


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("current_missing,expected", [(204, "ready"), (500, "quality_issue")])
def test_missingness_distinguishes_sampling_variation_from_increase(
    engine, current_missing, expected
):
    """A 10% to 10.2% sampling change must not block drift, but 10% to 25% must."""
    from skyulf.integrations.databricks.jobs.lifecycle.retraining_task import (
        _drift_metrics_decision,
    )

    reference = [{"value": None if i < 200 else i % 10, "kind": i % 10} for i in range(2000)]
    current = [
        {"id": str(i), "value": None if i < current_missing else i % 10, "kind": i % 10 + 10}
        for i in range(2000)
    ]
    result = report(
        engine,
        reference=reference,
        current=current,
        predictions=[{"id": str(i), "prediction": "yes"} for i in range(2000)],
    )

    assert result["status"] == "drift"
    assert _drift_metrics_decision(result["metrics"]) == expected


def test_same_missingness_process_rarely_flags_sampling_noise():
    """Independent samples from the same process must not systematically block retraining."""
    import numpy as np

    from skyulf.integrations.databricks.observability.monitoring.local.monitoring_metrics import (
        _feature_quality,
    )

    rng = np.random.default_rng(42)
    flagged = 0
    for _ in range(100):
        baseline, current = rng.binomial(2000, 0.1, size=2)
        reference = pl.DataFrame({"value": [None] * baseline + [1.0] * (2000 - baseline)})
        frame = pl.DataFrame({"value": [None] * current + [1.0] * (2000 - current)})
        flagged += _feature_quality(frame, "value", reference)[0]["has_issue"]

    assert flagged <= 3
