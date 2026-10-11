"""Bundle decision policies must survive persistence and keep selection data separate."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import predict_pipeline
from skyulf.integrations.databricks.scoring.batch.frame_batch import fit_workflow
from skyulf.integrations.databricks.training.thresholds.threshold_training import (
    calibration_partition,
)
from skyulf.integrations.databricks.training.tuning.cv import CVSpec, evaluate_training_cv
from skyulf.integrations.databricks.training.tuning.search import prepare_search_pipeline
from skyulf.modeling._evaluation.thresholds import apply_thresholds


@pytest.mark.parametrize("cutoff", [0.0, 0.5, 0.8, 1.0])
def test_explicit_positive_class_includes_exact_boundary(cutoff):
    """The chosen positive label must win ties even in probability column zero."""
    result = apply_thresholds(
        [[cutoff, 1 - cutoff]], cutoff, classes=["yes", "no"], positive_class="yes"
    )
    assert result.tolist() == ["yes"]


def fixture_frame(classes=2):
    """Provide nontrivial probabilities and string labels in a deterministic population."""
    features, labels = make_classification(
        n_samples=240,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        n_classes=classes,
        n_clusters_per_class=1,
        random_state=17,
    )
    frame = pd.DataFrame(features, columns=list("abcd"))
    frame["target"] = np.array([f"class_{value}" for value in labels])
    return frame


def recipe(policy):
    """Keep threshold controls separate from sklearn constructor parameters."""
    return {
        "preprocessing": [
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": list("abcd")}}
        ],
        "modeling": {"type": "logistic_regression", "params": {"max_iter": 500}},
        "decision_threshold": policy,
    }


def fit_artifact(tmp_path, frame, policy, engine):
    """Exercise the production local artifact path with no remote services."""
    native = pl.from_pandas(frame) if engine == "polars" else frame
    return fit_workflow(
        recipe(policy),
        SplitDataset(train=native, test=native.head(0)),
        target_column="target",
        artifact_path=tmp_path / engine,
        max_rows=1000,
        max_bytes=10_000_000,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_manual_binary_uses_explicit_positive_class(tmp_path, engine):
    """A first-column positive label must not silently become the second class."""
    frame = fixture_frame()
    artifact = fit_artifact(
        tmp_path, frame, {"mode": "manual", "positive_class": "class_0", "value": 0.8}, engine
    )
    output = predict_pipeline(frame.drop(columns="target"), artifact)
    expected = np.where(output.probability_0 >= 0.8, "class_0", "class_1")
    assert artifact.manifest.use_tuned_thresholds
    np.testing.assert_array_equal(output.prediction, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_manual_multiclass_matches_scaled_argmax(tmp_path, engine):
    """Every class value follows labels, independently of declaration ordering."""
    frame = fixture_frame(3)
    values = [
        {"class": "class_2", "value": 0.2},
        {"class": "class_0", "value": 0.9},
        {"class": "class_1", "value": 0.5},
    ]
    artifact = fit_artifact(tmp_path, frame, {"mode": "manual", "thresholds": values}, engine)
    output = predict_pipeline(frame.drop(columns="target"), artifact)
    expected = np.array(["class_0", "class_1", "class_2"])[
        np.argmax(output.filter(like="probability_").to_numpy() / [0.9, 0.5, 0.2], axis=1)
    ]
    assert artifact.manifest.use_tuned_thresholds
    np.testing.assert_array_equal(output.prediction, expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_auto_fits_scaler_and_model_without_calibration_rows(tmp_path, engine):
    """A plain sklearn reconstruction detects calibration leakage into fitting."""
    frame = fixture_frame()
    policy = {
        "mode": "auto",
        "metric": "balanced_accuracy",
        "validation_fraction": 0.2,
        "random_state": 42,
    }
    artifact = fit_artifact(tmp_path, frame, policy, engine)
    train, validation = train_test_split(
        frame, test_size=0.2, random_state=42, stratify=frame.target
    )
    scaler = StandardScaler().fit(train[list("abcd")])
    model = LogisticRegression(max_iter=500).fit(
        scaler.transform(train[list("abcd")]), train.target
    )
    fitted = artifact.pipeline.model_estimator._unwrap_tuned_model()
    np.testing.assert_allclose(fitted.coef_, model.coef_, atol=1e-10, rtol=0)
    proba = model.predict_proba(scaler.transform(validation[list("abcd")]))[:, 1]
    candidates = np.linspace(0, 1, 103)[1:-1]
    best = max(
        candidates,
        key=lambda t: (
            balanced_accuracy_score(validation.target, np.where(proba >= t, "class_1", "class_0")),
            -abs(t - 0.5),
        ),
    )
    assert artifact.manifest.use_tuned_thresholds
    assert artifact.pipeline._tuned_thresholds["class_1"] == pytest.approx(best)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_auto_threshold_objective_stays_unweighted_when_only_calibration_weights_change(engine):
    """Training weights must not silently change the documented unweighted selection objective."""
    from skyulf.integrations.databricks.training.thresholds.threshold_training import (
        fit_threshold_pipeline,
    )

    frame = fixture_frame()
    frame.loc[::7, "target"] = np.where(frame.loc[::7, "target"] == "class_0", "class_1", "class_0")
    policy = {
        "mode": "auto",
        "metric": "balanced_accuracy",
        "validation_fraction": 0.25,
        "random_state": 42,
    }
    fitting, calibration = train_test_split(
        np.arange(len(frame)), test_size=0.25, random_state=42, stratify=frame.target
    )
    weights = np.linspace(0.1, 3, len(frame))
    scaler = StandardScaler().fit(frame.iloc[fitting][list("abcd")])
    reference = LogisticRegression(max_iter=500).fit(
        scaler.transform(frame.iloc[fitting][list("abcd")]),
        frame.iloc[fitting].target,
        sample_weight=weights[fitting],
    )
    labels = frame.iloc[calibration].target.to_numpy()
    proba = reference.predict_proba(scaler.transform(frame.iloc[calibration][list("abcd")]))
    candidates = np.linspace(0, 1, 103)[1:-1]
    expected = max(
        candidates,
        key=lambda cutoff: (
            balanced_accuracy_score(labels, np.where(proba[:, 1] >= cutoff, "class_1", "class_0")),
            -abs(cutoff - 0.5),
        ),
    )
    predicted = np.where(proba[:, 1] >= expected, "class_1", "class_0")
    changed_weights = weights.copy()
    changed_weights[calibration] = np.where(predicted != labels, 1000.0, 1.0)
    expected_score = balanced_accuracy_score(labels, predicted)
    weighted_score = balanced_accuracy_score(
        labels, predicted, sample_weight=changed_weights[calibration]
    )
    assert abs(expected_score - weighted_score) > 0.1
    native = pl.from_pandas(frame) if engine == "polars" else frame
    for selected_weights in (weights, changed_weights):
        pipeline = fit_threshold_pipeline(
            recipe(policy),
            SplitDataset(train=native, test=native.head(0), train_sample_weight=selected_weights),
            "target",
        )
        assert pipeline.model_estimator is not None
        fitted = pipeline.model_estimator._unwrap_tuned_model()
        np.testing.assert_allclose(fitted.coef_, reference.coef_, rtol=0, atol=1e-10)
        assert pipeline._tuned_thresholds is not None
        assert pipeline._tuned_thresholds["class_1"] == pytest.approx(expected)
        evidence = pipeline._decision_threshold_evidence
        assert evidence is not None
        assert evidence["fitting_rows"] == 180 and evidence["calibration_rows"] == 60
        assert evidence["selected_score"] == pytest.approx(expected_score)


@pytest.mark.parametrize(
    "policy",
    [
        {"mode": "typo"},
        {"mode": "manual", "value": float("nan"), "positive_class": "class_1"},
        {"mode": "manual", "value": 1.2, "positive_class": "class_1"},
        {"mode": "auto", "metric": "roc_auc"},
        {"mode": "off", "value": 0.2},
    ],
)
def test_invalid_threshold_policy_is_rejected(tmp_path, policy):
    """Unsupported or ambiguous requests must never silently produce a native model."""
    with pytest.raises(ValueError, match="threshold|Threshold"):
        fit_artifact(tmp_path, fixture_frame(), policy, "pandas")


def test_off_preserves_native_classifier(tmp_path):
    """Disabled decision policies retain the existing native predict behavior."""
    frame = fixture_frame()
    artifact = fit_artifact(tmp_path, frame, {"mode": "off"}, "pandas")
    output = predict_pipeline(frame.drop(columns="target"), artifact)
    assert not artifact.manifest.use_tuned_thresholds
    np.testing.assert_array_equal(
        output.prediction,
        np.array(["class_0", "class_1"])[
            np.argmax(output.filter(like="probability_").to_numpy(), axis=1)
        ],
    )


def test_cv_evaluates_manual_decisions():
    """CV must score the deployed cutoff rather than the classifier's native decisions."""
    from skyulf.integrations.databricks.training.tuning.cv import CVSpec, evaluate_training_cv

    frame = fixture_frame()
    result = evaluate_training_cv(
        frame,
        recipe({"mode": "manual", "value": 0.0, "positive_class": "class_0"}),
        CVSpec(enabled=True, folds=3, method="stratified_k_fold"),
        target_column="target",
    )
    assert result is not None
    assert result["aggregated_metrics"]["balanced_accuracy"]["mean"] == pytest.approx(0.5)
    assert len(result["decision_threshold_folds"]) == 3


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("classes", [2, 3])
@pytest.mark.parametrize("mode", ["off", "manual", "auto"])
def test_reference_coefficients_probabilities_and_holdout_independence(
    tmp_path, engine, classes, mode
):
    """A separately trained sklearn pipeline must reproduce weighted fits and probabilities."""
    frame = fixture_frame(classes)
    weights = np.linspace(0.1, 3, len(frame))
    policy = {"mode": mode}
    if mode == "manual":
        policy["thresholds"] = [
            {"class": f"class_{index}", "value": 0.3 + index * 0.2} for index in range(classes)
        ]
    config = recipe(policy)
    native = pl.from_pandas(frame) if engine == "polars" else frame
    holdout = frame.head(20).copy()
    holdout[list("abcd")] += 10000
    holdout.target = "unseen_holdout_label"
    native_holdout = pl.from_pandas(holdout) if engine == "polars" else holdout
    # Empty test for native fit, deliberately adversarial final holdout for calibration.
    data = SplitDataset(
        train=native,
        test=native_holdout if mode == "auto" else native.head(0),
        train_sample_weight=weights,
    )
    artifact = fit_workflow(
        config,
        data,
        target_column="target",
        artifact_path=tmp_path / "weighted",
        max_rows=1000,
        max_bytes=10_000_000,
    )
    positions = np.arange(len(frame))
    if mode == "auto":
        positions, _ = train_test_split(
            positions, test_size=0.2, random_state=42, stratify=frame.target
        )
    fitting = frame.iloc[positions]
    scaler = StandardScaler().fit(fitting[list("abcd")])
    reference = LogisticRegression(max_iter=500).fit(
        scaler.transform(fitting[list("abcd")]), fitting.target, sample_weight=weights[positions]
    )
    actual = predict_pipeline(frame.drop(columns="target"), artifact)
    expected_probabilities = reference.predict_proba(scaler.transform(frame[list("abcd")]))
    np.testing.assert_allclose(
        actual.filter(like="probability_").to_numpy(), expected_probabilities, rtol=0, atol=1e-10
    )
    if mode == "off":
        expected = reference.predict(scaler.transform(frame[list("abcd")]))
    elif classes == 2 and mode == "auto":
        assert artifact.pipeline._tuned_thresholds is not None
        cutoff = artifact.pipeline._tuned_thresholds["class_1"]
        expected = np.where(expected_probabilities[:, 1] >= cutoff, "class_1", "class_0")
    else:
        assert artifact.pipeline._tuned_thresholds is not None
        thresholds = np.array(
            [artifact.pipeline._tuned_thresholds[label] for label in reference.classes_]
        )
        expected = reference.classes_[np.argmax(expected_probabilities / thresholds, axis=1)]
    np.testing.assert_array_equal(actual.prediction, expected)


@pytest.mark.parametrize(
    "method", ["stratified_k_fold", "group_k_fold", "time_series_split", "nested_cv"]
)
@pytest.mark.parametrize("search", [False, True])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_auto_cv_respects_split_metadata(tmp_path, method, search, engine):
    """Temporal/group metadata must guide calibration without becoming estimator inputs."""
    frame = fixture_frame()
    cv = CVSpec(
        enabled=True,
        folds=3,
        method=method,
        shuffle=method != "time_series_split",
        group_column="group" if method == "group_k_fold" else None,
        gap=2 if method == "time_series_split" else 0,
    )
    event = "event" if cv.temporal else None
    if event:
        frame[event] = pd.date_range("2020-01-01", periods=len(frame), freq="h")
    if cv.group_column:
        frame["group"] = np.arange(len(frame)) // 4
    config = recipe({"mode": "auto"})
    if search:
        config["modeling"] = {
            "type": "hyperparameter_tuner",
            "base_model": config["modeling"],
            "strategy": "grid",
            "search_space": {"C": [0.5, 1.0]},
            "metric": "balanced_accuracy",
        }
    config = prepare_search_pipeline(config, cv, target_column="target", event_column=event)
    native = pl.from_pandas(frame) if engine == "polars" else frame
    result = evaluate_training_cv(native, config, cv, target_column="target", event_column=event)
    assert result is not None
    assert len(result["decision_threshold_folds"]) == 3
    assert all(fold["fitting_rows"] < 160 for fold in result["decision_threshold_folds"])
    assert 0 <= result["aggregated_metrics"]["balanced_accuracy"]["mean"] <= 1


def test_calibration_whole_groups_and_chronological_gap():
    """Calibration cannot share a group or a timestamp with fitting rows."""
    frame = fixture_frame()
    frame["group"] = np.arange(len(frame)) // 4
    frame["event"] = pd.date_range("2020-01-01", periods=len(frame), freq="h")
    policy = {"validation_fraction": 0.2, "random_state": 42}
    train, valid = calibration_partition(frame, "target", policy, {"group_column": "group"})
    assert set(frame.iloc[train].group).isdisjoint(frame.iloc[valid].group)
    train, valid = calibration_partition(
        frame, "target", policy, {"split_strategy": "temporal", "event_column": "event", "gap": 3}
    )
    assert frame.iloc[train].event.max() < frame.iloc[valid].event.min()
    assert len(train) + len(valid) == len(frame) - 3


def test_competition_scores_saved_decision_policy(tmp_path):
    """Candidate ranking must use threshold decisions on the common outer validation rows."""
    from skyulf.integrations.databricks.training.competition.competition_evaluation import (
        evaluate_competition_candidate,
    )

    frame = fixture_frame()
    artifact = fit_artifact(
        tmp_path, frame, {"mode": "manual", "value": 0.0, "positive_class": "class_0"}, "pandas"
    )
    cv = CVSpec(enabled=True, folds=3, method="stratified_k_fold")
    result = evaluate_competition_candidate(
        frame,
        artifact,
        cv,
        target_column="target",
        metric="heldout_balanced_accuracy",
        max_rows=1000,
        max_bytes=10_000_000,
    )
    assert result["mean"] == pytest.approx(0.5)
    assert result["fold_scores"] == pytest.approx([0.5] * 3)
    from skyulf.integrations.databricks.training.competition.competition import choose_winner

    result["candidate"] = "threshold_candidate"
    assert choose_winner([result], {"threshold_candidate"})["winner"] == "threshold_candidate"


@pytest.mark.parametrize("nested,expected", [(False, 8), (True, 14)])
def test_threshold_competition_budget_includes_outer_searches(nested, expected):
    """Repeated threshold-aware fold searches must not bypass the declared search budget."""
    from skyulf.integrations.databricks.training.competition.competition import (
        _pipeline_trial_bound,
    )

    config = recipe({"mode": "auto"})
    config["modeling"] = {
        "type": "hyperparameter_tuner",
        "base_model": config["modeling"],
        "strategy": "grid",
        "search_space": {"C": [0.1, 1.0]},
    }
    cv = CVSpec(enabled=True, folds=3, method="nested_cv" if nested else "stratified_k_fold")
    assert _pipeline_trial_bound(config, cv) == expected


def test_frozen_recipe_and_standalone_export_keep_positive_class(tmp_path):
    """Runtime evidence must preserve frozen config equality and exported decisions."""
    from skyulf.inference.bundle import build_bundle, load_bundle, predict_local, save_bundle

    frame = fixture_frame()
    policy = {"mode": "auto", "positive_class": "class_0"}
    artifact = fit_artifact(tmp_path, frame, policy, "pandas")
    assert artifact.pipeline.config == recipe(policy)
    bundle = build_bundle(
        artifact.pipeline, input_stage="raw", feature_order=tuple("abcd"), use_tuned_thresholds=True
    )
    save_bundle(bundle, tmp_path / "standalone")
    restored = load_bundle(tmp_path / "standalone")
    actual = predict_local(frame.drop(columns="target"), restored)
    expected = predict_pipeline(frame.drop(columns="target"), artifact)
    assert restored.manifest.thresholds.positive_class == "class_0"
    np.testing.assert_array_equal(actual.prediction, expected.prediction)


def test_threshold_cv_chart_contract():
    """Threshold CV must retain the fold metric schema used by optional chart reports."""
    pytest.importorskip("matplotlib")
    from skyulf.integrations.databricks.observability.charts.evaluation_chart_report import (
        _cv_charts,
    )

    report = evaluate_training_cv(
        fixture_frame(),
        recipe({"mode": "manual", "value": 0.8, "positive_class": "class_0"}),
        CVSpec(enabled=True, folds=2),
        target_column="target",
    )
    assert report is not None
    charts = _cv_charts(report, "classification")
    assert "cv_accuracy" in charts


@pytest.mark.parametrize("mode", ["manual", "auto"])
def test_probability_incapable_model_fails(tmp_path, mode):
    """Probability-free SVMs cannot silently fall back to a native decision."""
    frame = fixture_frame()
    policy = (
        {"mode": mode, "positive_class": "class_1", "value": 0.7}
        if mode == "manual"
        else {"mode": mode}
    )
    config = recipe(policy)
    config["modeling"] = {"type": "svc", "params": {"probability": False}}
    with pytest.raises(ValueError, match="predict_proba|probabilit"):
        fit_workflow(
            config,
            SplitDataset(train=frame, test=frame.head(0)),
            target_column="target",
            artifact_path=tmp_path / "invalid",
            max_rows=1000,
            max_bytes=10_000_000,
        )


def test_core_split_positions_matches_sklearn_reference():
    """Shared Core splitting must retain exact stratified and whole-group membership."""
    from sklearn.model_selection import GroupShuffleSplit

    from skyulf.preprocessing.split import DataSplitter

    frame = fixture_frame()
    splitter = DataSplitter(test_size=0.2, random_state=42)
    actual = splitter.split_indices(len(frame), stratify=frame.target)
    expected = train_test_split(
        np.arange(len(frame)), test_size=0.2, random_state=42, stratify=frame.target
    )
    for actual_indices, expected_indices in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_indices, expected_indices)
    groups = np.arange(len(frame)) // 4
    actual = splitter.split_indices(len(frame), groups=groups)
    expected = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(frame, groups=groups)
    )
    for actual_indices, expected_indices in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_indices, expected_indices)


def test_core_threshold_metric_supports_macro_f1():
    """The shared Core resolver must not silently fall back for the Bundle macro objective."""
    from sklearn.metrics import f1_score

    from skyulf.modeling._tuning.refit import resolve_threshold_metric

    metric, name = resolve_threshold_metric("f1_macro", None)
    actual, prediction = [0, 0, 1, 2, 2], [0, 1, 1, 1, 2]
    assert name == "f1_macro"
    assert metric(actual, prediction) == pytest.approx(
        f1_score(actual, prediction, average="macro")
    )


def test_temporal_calibration_rejects_groups_spanning_boundary():
    """Chronological ordering cannot excuse entity leakage across calibration membership."""
    frame = fixture_frame()
    frame["event"] = pd.date_range("2020-01-01", periods=len(frame), freq="h")
    frame["group"] = np.arange(len(frame)) % 4
    with pytest.raises(ValueError, match="disjoint groups"):
        calibration_partition(
            frame,
            "target",
            {"validation_fraction": 0.2, "random_state": 42},
            {"split_strategy": "temporal", "event_column": "event", "group_column": "group"},
        )


@pytest.mark.parametrize(
    "metric",
    [
        "roc_auc_weighted",
        "roc_auc_ovr",
        "roc_auc_ovo",
        "roc_auc_ovr_weighted",
        "roc_auc_ovo_weighted",
        "pr_auc_weighted",
    ],
)
def test_binary_competition_probability_metric_aliases(tmp_path, metric):
    """Admitted probability metric aliases must score binary folds without missing-key errors."""
    from skyulf.integrations.databricks.training.competition.competition_evaluation import (
        evaluate_competition_candidate,
    )

    frame = fixture_frame()
    artifact = fit_artifact(
        tmp_path, frame, {"mode": "manual", "positive_class": "class_0", "value": 0.8}, "pandas"
    )
    cv = CVSpec(enabled=True, folds=2, method="stratified_k_fold")
    result = evaluate_competition_candidate(
        frame,
        artifact,
        cv,
        target_column="target",
        metric=f"heldout_{metric}",
        max_rows=1000,
        max_bytes=10_000_000,
    )
    base = "pr_auc" if metric == "pr_auc_weighted" else "roc_auc"
    expected = evaluate_competition_candidate(
        frame,
        artifact,
        cv,
        target_column="target",
        metric=f"heldout_{base}",
        max_rows=1000,
        max_bytes=10_000_000,
    )
    assert result["fold_scores"] == pytest.approx(expected["fold_scores"])
    from skyulf.inference.pipeline_evaluation import evaluate_holdout

    heldout = evaluate_holdout(artifact, frame, target_column="target")
    assert heldout[f"heldout_{metric}"] == pytest.approx(heldout[f"heldout_{base}"])
