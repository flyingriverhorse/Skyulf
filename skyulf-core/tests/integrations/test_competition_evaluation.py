"""Candidate ranking must compare complete scores on identical training folds."""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.integrations.databricks.competition_evaluation import (
    competition_metric,
    evaluate_competition_candidate,
    validate_competition_preprocessing,
)
from skyulf.integrations.databricks.local_batch import fit_local_workflow
from skyulf.integrations.databricks.local_cv import LocalCVSpec
from skyulf.integrations.databricks.local_search import prepare_search_pipeline


def _frame(engine="pandas", classification=False):
    """Keep enough rows for every outer, inner, and stacking fold."""
    x = np.arange(80, dtype=float)
    frame = pd.DataFrame({"x": x, "target": (x % 2).astype(int) if classification else x * 3 + 2})
    return pl.from_pandas(frame) if engine == "polars" else frame


def _fit(
    tmp_path,
    frame,
    cv,
    model="linear_regression",
    params=None,
    search=False,
    metric="rmse",
    threshold=False,
):
    """Fit a real Core artifact with the same admitted workflow recipe."""
    selected = {"type": model, "params": params or {}}
    modeling = selected
    if search:
        modeling = {
            "type": "hyperparameter_tuner",
            "base_model": selected,
            "metric": metric,
            "strategy": "grid",
            "search_space": {},
        }
        modeling["search_space"] = (
            {"C": [0.2, 1.0]}
            if model == "logistic_regression"
            else {"fit_intercept": [True, False]}
        )
        if model.startswith(("voting_", "stacking_")):
            member = selected["params"]["base_estimators"][0]
            axis = "C" if model.endswith("classifier") else "alpha"
            modeling["search_space"] = {f"{member}__{axis}": [0.5, 1.0]}
        modeling["tune_threshold"] = threshold
    config = prepare_search_pipeline(
        {
            "preprocessing": [
                {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}}
            ],
            "modeling": modeling,
        },
        cv,
        target_column="target",
        event_column="event" if cv.temporal else None,
    )
    fit_frame = frame
    if not search:
        metadata = cv.group_column or ("event" if cv.temporal else None)
        if metadata:
            fit_frame = (
                frame.drop(metadata)
                if isinstance(frame, pl.DataFrame)
                else frame.drop(columns=[metadata])
            )
    return fit_local_workflow(
        config,
        SplitDataset(train=fit_frame, test=fit_frame.head(0)),
        target_column="target",
        artifact_path=tmp_path / "artifact",
        max_rows=1000,
        max_bytes=1000000,
    )


def _evaluate(frame, artifact, cv, metric="heldout_rmse"):
    """Call the evidence boundary with explicit bounded training data."""
    return evaluate_competition_candidate(
        frame,
        artifact,
        cv,
        target_column="target",
        event_column="event" if cv.temporal else None,
        metric=metric,
        max_rows=1000,
        max_bytes=1000000,
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["k_fold", "shuffle_split", "group_k_fold", "time_series_split"])
def test_fixed_and_tuned_share_membership(tmp_path, engine, method):
    """Fixed and selected recipes must use identical rows and positive error units."""
    frame = _frame()
    if method == "group_k_fold":
        frame["entity"] = np.repeat(np.arange(20), 4)
    if method == "time_series_split":
        frame["event"] = pd.date_range("2026-01-01", periods=len(frame), tz="UTC")
    frame = pl.from_pandas(frame) if engine == "polars" else frame
    cv = LocalCVSpec(
        enabled=True,
        folds=3,
        method=method,
        shuffle=method != "time_series_split",
        group_column="entity" if method == "group_k_fold" else None,
    )
    fixed = _fit(tmp_path / "fixed", frame, cv)
    tuned = _fit(tmp_path / "tuned", frame, cv, search=True)
    a, b = _evaluate(frame, fixed, cv), _evaluate(frame, tuned, cv)
    assert a["fold_membership_sha256"] == b["fold_membership_sha256"]
    assert a["direction"] == b["direction"] == "minimize"
    assert a["mean"] == pytest.approx(0, abs=1e-8)
    assert b["mean"] == pytest.approx(0, abs=1e-8)
    assert len(a["fold_scores"]) == len(b["fold_scores"]) == 3
    assert b["evaluation_mode"] == "post_selection_cv"


@pytest.mark.parametrize(
    "metric,task,expected",
    [
        ("heldout_rmse", "regression", "rmse"),
        ("heldout_f1", "classification", "f1"),
        ("heldout_roc_auc_weighted", "classification", "roc_auc_ovr_weighted"),
    ],
)
def test_metric_mapping(metric, task, expected):
    """Workflow metric names must map to the same admitted Core objective."""
    assert competition_metric(metric, task) == expected


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_competition_target_encoding_keeps_same_positive_class(tmp_path, engine):
    """Candidates with the same predictions must receive identical F1 despite target recoding."""
    from sklearn.datasets import make_classification

    values, labels = make_classification(
        n_samples=160, n_features=6, n_informative=4, weights=[0.75], random_state=7
    )
    frame = pd.DataFrame(values, columns=list("abcdef"))
    frame["target"] = np.where(labels == 1, 10, 2)
    if engine == "polars":
        frame = pl.from_pandas(frame)
    cv = LocalCVSpec(enabled=True, folds=3, method="stratified_k_fold")
    reports = []
    for encoded in (False, True):
        steps = (
            [{"name": "labels", "transformer": "LabelEncoder", "params": {"columns": ["target"]}}]
            if encoded
            else []
        )
        artifact = fit_local_workflow(
            {
                "preprocessing": steps,
                "modeling": {"type": "logistic_regression", "params": {"max_iter": 1000}},
            },
            SplitDataset(train=frame, test=frame.head(0)),
            target_column="target",
            artifact_path=tmp_path / str(encoded),
            max_rows=1000,
            max_bytes=1000000,
        )
        reports.append(_evaluate(frame, artifact, cv, "heldout_f1"))

    assert reports[0]["fold_membership_sha256"] == reports[1]["fold_membership_sha256"]
    assert reports[0]["fold_scores"] == pytest.approx(reports[1]["fold_scores"])


@pytest.mark.parametrize(
    "metric,task",
    [
        ("heldout_accuracy", "regression"),
        ("heldout_rmse", "classification"),
        ("heldout_mape", "regression"),
        ("accuracy", "classification"),
    ],
)
def test_unsupported_metric_is_rejected(metric, task):
    """An unsupported objective must fail before candidates train."""
    with pytest.raises(ValueError, match="competition metric"):
        competition_metric(metric, task)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["stratified_k_fold", "stratified_group_k_fold", "nested_cv"])
def test_classification_scores_string_labels_consistently(tmp_path, engine, method):
    """Fixed and searched classifiers must share positive-label and metric semantics."""
    frame = _frame(classification=True)
    frame["target"] = frame["target"].map({0: "negative", 1: "positive"})
    grouped = method == "stratified_group_k_fold"
    if grouped:
        frame["entity"] = np.repeat(np.arange(20), 4)
    native = pl.from_pandas(frame) if engine == "polars" else frame
    cv = LocalCVSpec(
        enabled=True,
        folds=3,
        inner_folds=2,
        method=method,
        group_column="entity" if grouped else None,
    )
    a = _fit(tmp_path / "fixed", native, cv, model="logistic_regression")
    b = _fit(tmp_path / "tuned", native, cv, model="logistic_regression", search=True, metric="f1")
    first, second = _evaluate(native, a, cv, "heldout_f1"), _evaluate(native, b, cv, "heldout_f1")
    assert first["fold_membership_sha256"] == second["fold_membership_sha256"]
    assert first["scoring_metric"] == second["scoring_metric"] == "f1"
    assert first["direction"] == second["direction"] == "maximize"
    assert all(0 <= score <= 1 for score in second["fold_scores"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("nested_type", ["auto", "group_k_fold", "time_series_split"])
def test_nested_fixed_uses_same_requested_metric_and_outer_policy(tmp_path, engine, nested_type):
    """Nested singleton models and real searches must report the same RMSE objective."""
    frame = _frame()
    if nested_type == "group_k_fold":
        frame["entity"] = np.repeat(np.arange(20), 4)
    if nested_type == "time_series_split":
        frame["event"] = pd.date_range("2026-01-01", periods=len(frame), tz="UTC")
    native = pl.from_pandas(frame) if engine == "polars" else frame
    cv = LocalCVSpec(
        enabled=True,
        folds=3,
        inner_folds=2,
        method="nested_cv",
        nested_type=nested_type,
        shuffle=nested_type != "time_series_split",
        group_column="entity" if nested_type == "group_k_fold" else None,
    )
    a = _fit(tmp_path / "fixed", native, cv)
    b = _fit(tmp_path / "tuned", native, cv, search=True)
    first, second = _evaluate(native, a, cv), _evaluate(native, b, cv)
    assert first["fold_membership_sha256"] == second["fold_membership_sha256"]
    assert first["scoring_metric"] == second["scoring_metric"] == "neg_root_mean_squared_error"
    assert first["evaluation_mode"] == second["evaluation_mode"] == "nested_cv"
    assert second["mean"] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["k_fold", "nested_cv"])
@pytest.mark.parametrize(
    "model", ["voting_regressor", "stacking_regressor", "voting_classifier", "stacking_classifier"]
)
def test_ensemble_preserves_selected_members(tmp_path, engine, method, model):
    """Voting and stacking must evaluate their actual selected components on shared folds."""
    classification = model.endswith("classifier")
    native = _frame(engine, classification)
    params = {"base_estimators": ["logistic_regression"] if classification else ["ridge"]}
    if model.startswith("stacking"):
        params.update(cv=2, final_estimator="logistic_regression" if classification else "ridge")
    cv = LocalCVSpec(enabled=True, folds=2, inner_folds=2, method=method)
    artifact = _fit(tmp_path / model, native, cv, model=model, params=params)
    before = deepcopy(artifact.pipeline.config)
    result = _evaluate(
        native, artifact, cv, "heldout_accuracy" if classification else "heldout_rmse"
    )
    assert len(result["fold_scores"]) == 2
    assert all(np.isfinite(result["fold_scores"]))
    assert artifact.pipeline.config == before


@pytest.mark.parametrize("bad_score", [None, True, float("nan"), float("inf")])
def test_nested_rejects_missing_or_nonfinite_outer_scores(tmp_path, bad_score):
    """A final best score must never rescue incomplete outer-fold evidence."""
    frame = _frame()
    cv = LocalCVSpec(enabled=True, folds=2, inner_folds=2, method="nested_cv")
    artifact = _fit(tmp_path, frame, cv, search=True)
    result = artifact.pipeline.model_estimator.model[1]
    result.best_score = 99999.0
    result.nested_cv["folds"][0]["outer_score"] = bad_score
    with pytest.raises(ValueError, match="finite score"):
        _evaluate(frame, artifact, cv)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_report",
        "missing_fold",
        "different_policy",
        "different_metric",
        "different_membership",
    ],
)
def test_nested_rejects_incompatible_reports(tmp_path, mutation):
    """Report validation must fail closed for stale or incompatible nested evidence."""
    frame = _frame()
    cv = LocalCVSpec(enabled=True, folds=2, inner_folds=2, method="nested_cv")
    artifact = _fit(tmp_path, frame, cv, search=True)
    result = artifact.pipeline.model_estimator.model[1]
    if mutation == "missing_report":
        result.nested_cv = None
    elif mutation == "missing_fold":
        result.nested_cv["folds"].pop()
    elif mutation == "different_policy":
        result.nested_cv["split_policy"]["random_state"] = 1234
    elif mutation == "different_metric":
        result.nested_cv["scoring_metric"] = "neg_mean_squared_error"
    else:
        result.nested_cv["folds"][0]["split"]["train_rows"] = 1234
    with pytest.raises(ValueError, match="[Cc]ompetition"):
        _evaluate(frame, artifact, cv)


def test_ordinary_rejects_one_failed_fold(tmp_path, monkeypatch):
    """Ranking must never average only the surviving successful folds."""
    from skyulf.integrations.databricks import competition_evaluation

    frame = _frame()
    cv = LocalCVSpec(enabled=True, folds=3)
    artifact = _fit(tmp_path, frame, cv)
    scores = iter([1.0, float("nan"), 2.0])
    monkeypatch.setattr(
        competition_evaluation, "fit_and_score_candidate_fold", lambda **kwargs: next(scores)
    )
    with pytest.raises(ValueError, match="fold 2 failed"):
        _evaluate(frame, artifact, cv)


def test_shuffle_plan_is_not_kfold(tmp_path):
    """Repeated shuffle folds must hold out twenty percent rather than one K-fold share."""
    frame = _frame()
    cv = LocalCVSpec(enabled=True, folds=3, method="shuffle_split")
    artifact = _fit(tmp_path, frame, cv)
    result = _evaluate(frame, artifact, cv)
    assert [fold["test_rows"] for fold in result["folds"]] == [16, 16, 16]


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"max_rows": 10}, "exceeds"),
        ({"max_bytes": 1}, "exceeds"),
        ({"max_rows": True}, "positive integer"),
        ({"max_bytes": None}, "positive integer"),
        ({"target_column": "missing"}, "separate target"),
        ({"cv": LocalCVSpec()}, "enabled CV"),
    ],
)
def test_evaluation_admission(tmp_path, changes, match):
    """Invalid bounds and missing targets must fail before fold evaluation starts."""
    frame = _frame()
    cv = LocalCVSpec(enabled=True, folds=2)
    artifact = _fit(tmp_path, frame, cv)
    arguments: dict[str, Any] = {
        "cv": cv,
        "target_column": "target",
        "metric": "heldout_rmse",
        "max_rows": 1000,
        "max_bytes": 1000000,
        **changes,
    }
    with pytest.raises(ValueError, match=match):
        evaluate_competition_candidate(frame, artifact, **arguments)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["k_fold", "nested_cv"])
@pytest.mark.parametrize(
    "model", ["voting_regressor", "stacking_regressor", "voting_classifier", "stacking_classifier"]
)
def test_tuned_ensemble_keeps_selected_structure(tmp_path, engine, method, model):
    """Nested member parameter choices must survive the post-selection fixed refit."""
    classification = model.endswith("classifier")
    frame = _frame(engine, classification)
    params = {"base_estimators": ["logistic_regression"] if classification else ["ridge"]}
    if model.startswith("stacking"):
        params.update(cv=2, final_estimator="logistic_regression" if classification else "ridge")
    cv = LocalCVSpec(enabled=True, folds=2, inner_folds=2, method=method)
    artifact = _fit(
        tmp_path,
        frame,
        cv,
        model,
        params,
        search=True,
        metric="accuracy" if classification else "rmse",
    )
    result = _evaluate(
        frame, artifact, cv, "heldout_accuracy" if classification else "heldout_rmse"
    )
    assert len(result["fold_scores"]) == 2
    assert np.isfinite(result["mean"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fold_preprocessing_sees_training_members_only(tmp_path, monkeypatch, engine):
    """Learned preprocessing must be fitted independently without validation rows."""
    from sklearn.model_selection import KFold

    from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter

    frame = _frame(engine)
    cv = LocalCVSpec(enabled=True, folds=3)
    artifact = _fit(tmp_path, frame, cv)
    observed = []
    original = FeatureEngineerFoldAdapter.fit_transform

    def record(self, X, y):
        """Record source row identities before the real preprocessing fit."""
        observed.append(X["x"].to_list())
        return original(self, X, y)

    monkeypatch.setattr(FeatureEngineerFoldAdapter, "fit_transform", record)
    result = _evaluate(frame, artifact, cv)
    expected = [
        train.tolist() for train, _ in KFold(3, shuffle=True, random_state=42).split(np.arange(80))
    ]
    assert observed == expected
    assert result["mean"] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_nested_threshold_reuses_outer_scores_without_refitting(tmp_path, monkeypatch, engine):
    """Final threshold or best-score values must not replace honest nested outer scores."""
    from skyulf.integrations.databricks import competition_evaluation

    frame = _frame(engine, classification=True)
    cv = LocalCVSpec(enabled=True, folds=2, inner_folds=2, method="nested_cv")
    artifact = _fit(
        tmp_path, frame, cv, "logistic_regression", search=True, metric="f1", threshold=True
    )
    fitted = artifact.pipeline.model_estimator.model[1]
    expected = [fold["outer_score"] for fold in fitted.nested_cv["folds"]]
    fitted.best_score = 99999.0
    fitted.decision_thresholds = {0: 0.01, 1: 0.99}

    def forbidden(*args, **kwargs):
        """A stored nested report must not trigger a new full-training evaluation."""
        raise AssertionError("unexpected fold refit")

    monkeypatch.setattr(competition_evaluation, "fit_and_score_candidate_fold", forbidden)
    result = _evaluate(frame, artifact, cv, "heldout_f1")
    assert result["fold_scores"] == expected
    assert result["mean"] == pytest.approx(np.mean(expected))


def test_metric_conflicts_and_multiclass_binary_alias_fail(tmp_path):
    """A binary heldout name must never silently rank multiclass weighted scores."""
    frame = _frame(classification=True)
    cv = LocalCVSpec(enabled=True, folds=2)
    artifact = _fit(tmp_path / "binary", frame, cv, "logistic_regression", search=True, metric="f1")
    with pytest.raises(ValueError, match="search metric"):
        _evaluate(frame, artifact, cv, "heldout_accuracy")
    frame["target"] = np.arange(80) % 3
    multi = _fit(tmp_path / "multi", frame, cv, "logistic_regression")
    with pytest.raises(ValueError, match="Binary competition metric"):
        _evaluate(frame, multi, cv, "heldout_f1")


@pytest.mark.parametrize(
    "transformer,params",
    [
        ("DropMissingRows", {}),
        ("IQR", {}),
        ("LagFeatures", {"drop_na": True}),
        ("RollingAggregate", {"sort_by": "event"}),
        ("Oversampling", {}),
        ("TrainTestSplitter", {}),
    ],
)
def test_row_changing_candidate_preprocessing_is_rejected(transformer, params):
    """A candidate may not improve its score by dropping difficult validation rows."""
    pipeline = {"preprocessing": [{"name": "filter", "transformer": transformer, "params": params}]}
    with pytest.raises(ValueError, match="preserve row membership"):
        validate_competition_preprocessing(pipeline)


def test_evaluator_rejects_validation_row_filtering(tmp_path):
    """Post-fit evaluation must enforce the same membership rules as source preflight."""
    frame = _frame()
    cv = LocalCVSpec(enabled=True, folds=2)
    artifact = _fit(tmp_path, frame, cv)
    artifact.pipeline.config["preprocessing"].append(
        {"name": "filter", "transformer": "IQR", "params": {}}
    )
    with pytest.raises(ValueError, match="shared pre_split"):
        _evaluate(frame, artifact, cv)


def test_aggregate_overflow_cannot_produce_nonfinite_evidence(tmp_path, monkeypatch):
    """Individually finite extreme folds must not serialize an infinite aggregate."""
    from skyulf.integrations.databricks import competition_evaluation

    frame = _frame()
    cv = LocalCVSpec(enabled=True, folds=2)
    artifact = _fit(tmp_path, frame, cv)
    monkeypatch.setattr(
        competition_evaluation, "fit_and_score_candidate_fold", lambda **kwargs: 1e308
    )
    with pytest.raises(ValueError, match="finite score"):
        _evaluate(frame, artifact, cv)
