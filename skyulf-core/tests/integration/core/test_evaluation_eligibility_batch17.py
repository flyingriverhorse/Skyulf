"""Explicit evaluation eligibility retains paired rows and discloses excluded observations."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from skyulf.data.dataset import SplitDataset
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.nested_threshold import score_nested_threshold, select_nested_threshold
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.regression import LinearRegressionCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter
from skyulf.preprocessing.function_steps import filter_step
from skyulf.preprocessing.pipeline import FeatureEngineer


def _steps(outliers=False):
    """Declare the population explicitly and optionally learn train-only outlier bounds."""
    result = [{"name": "eligible", "transformer": "DropMissingRows", "params": {"subset": ["x"]}}]
    if outliers:
        result.append({"name": "bounds", "transformer": "IQR", "params": {"columns": ["x"]}})
    return result


def _pair(values, labels, engine):
    """Use duplicate indexes to prevent accidental label-based target alignment."""
    X = pd.DataFrame({"x": values}, index=[4] * len(values))
    y = pd.Series(labels, index=X.index, name="target")
    return (pl.from_pandas(X), pl.from_pandas(y)) if engine == "polars" else (X, y)


def _nonnegative(frame):
    """Supply an importable explicit eligibility rule for replay and artifact tests."""
    return frame["x"] >= 0


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_function_filter_and_deduplication_select_the_same_paired_rows(engine):
    """Every explicitly configured eligibility operation must run during evaluation replay."""
    steps = [
        filter_step("nonnegative", _nonnegative, columns=["x"]),
        {"name": "unique", "transformer": "Deduplicate", "params": {"subset": ["x"]}},
    ]
    engineer = FeatureEngineer(steps)
    engineer.fit_transform(_pair([-1, 1, 1, 2], [99, 10, 999, 20], engine))
    incoming = _pair([-2, 2, 2, 3], [99, 20, 999, 30], engine)
    X_t, y_t = engineer.transform(incoming)
    assert np.asarray(X_t["x"]).tolist() == [2, 3]
    assert np.asarray(y_t).tolist() == [20, 30]
    assert engineer.last_transform_coverage_["excluded_rows"] == 2
    assert len(engineer.transform(incoming, preserve_rows=True)[0]) == 4


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fit_and_evaluation_replay_share_explicit_eligibility(engine):
    """An explicit missing-row rule must select the same held-out population on every route."""
    train = _pair([0.0, 1.0, 2.0, 3.0], [0, 10, 20, 30], engine)
    valid = _pair([1.0, None, 100.0], [10, 999, 1000], engine)
    engineer = FeatureEngineer(_steps(outliers=True))
    fitted, _ = engineer.fit_transform(SplitDataset(train=train, test=valid, validation=valid))
    replayed = engineer.transform(valid)
    assert len(fitted.test[0]) == len(replayed[0]) == 1
    assert np.asarray(replayed[1]).tolist() == [10]
    expected = {"input_rows": 3, "scored_rows": 1, "excluded_rows": 2}
    assert {key: engineer.last_transform_coverage_[key] for key in expected} == expected
    assert [step["excluded_rows"] for step in engineer.last_transform_coverage_["steps"]] == [1, 1]
    assert engineer.evaluation_coverage_["test"]["input_rows"] == 3
    assert engineer.evaluation_coverage_["validation"]["excluded_rows"] == 2
    assert len(valid[0]) == 3 and np.asarray(valid[1]).tolist() == [10, 999, 1000]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_prediction_preserves_existing_filter_skip_and_outlier_guard(engine):
    """Evaluation eligibility must not shorten an unkeyed prediction response."""
    train = _pair([0.0, 1.0, 2.0, 3.0], [0, 1, 2, 3], engine)
    raw = _pair([1.0, None, 100.0], [1, 2, 3], engine)[0]
    engineer = FeatureEngineer(_steps())
    engineer.fit_transform(train)
    assert len(engineer.transform(raw, preserve_rows=True)) == 3
    filtered = FeatureEngineer(_steps(outliers=True))
    filtered.fit_transform(train)
    with pytest.raises(ValueError, match="changed row count"):
        filtered.transform(raw, preserve_rows=True)


@pytest.mark.parametrize("strategy", ["grid", "random", "optuna", "halving_grid", "halving_random"])
def test_tuning_all_strategies_report_the_scored_fold_population(strategy):
    """Scorers must filter X/y together and retain fold denominators without adding fits."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    x = np.arange(60.0)
    y = pd.Series(2 * x + 1, name="target")
    X = pd.DataFrame({"x": x})
    X.loc[[2, 14, 32, 44], "x"] = np.nan
    config = TuningConfig(
        strategy=strategy,
        metric="neg_mean_squared_error",
        search_space={"fit_intercept": [True]},
        n_trials=1,
        cv_folds=2,
        cv_shuffle=False,
        n_jobs=2 if strategy.startswith("halving") else 1,
        strategy_params={"min_resources": 60, "max_resources": 60},
    )
    _, result = TuningCalculator(LinearRegressionCalculator()).fit(
        X, y, config, preprocessing=FeatureEngineerFoldAdapter(_steps(), "target")
    )
    assert result.best_score == pytest.approx(0, abs=1e-20)
    coverage = result.trials[0]["evaluation_coverage"]
    assert len(coverage) == 2
    assert sum(fold["input_rows"] for fold in coverage) == 60
    assert sum(fold["scored_rows"] for fold in coverage) == 56
    assert sum(fold["excluded_rows"] for fold in coverage) == 4
    assert X["x"].isna().sum() == 4


@pytest.mark.parametrize("encode", [False, True])
def test_nested_threshold_oof_and_outer_score_use_eligible_rows(encode):
    """Threshold selection must disclose filtered OOF coverage and score paired outer rows."""
    X = pd.DataFrame({"x": np.tile([-2.0, -1.0, 1.0, 2.0], 15)})
    y = pd.Series(np.where(X["x"] > 0, "yes", "no"), name="target")
    X.loc[[2, 13, 28, 41], "x"] = np.nan
    tuner = SimpleNamespace(
        problem_type="classification",
        model_calculator=SimpleNamespace(model_class=LogisticRegression, default_params={}),
    )
    config = TuningConfig(metric="f1", random_state=23)
    steps = _steps()
    if encode:
        steps.append(
            {"name": "encode", "transformer": "LabelEncoder", "params": {"columns": ["target"]}}
        )
    adapter = FeatureEngineerFoldAdapter(steps, "target")
    selected = select_nested_threshold(
        tuner, X, y, config, {}, StratifiedKFold(3), preprocessing=adapter
    )
    assert selected["oof_rows"] == 56
    assert selected["evaluation_coverage"]["input_rows"] == 60
    assert selected["evaluation_coverage"]["excluded_rows"] == 4
    outer_X = pd.DataFrame({"x": [-2.0, None, 2.0]})
    outer_y = pd.Series(["no", "no", "yes"])
    coverage = {}
    score = score_nested_threshold(
        tuner,
        X,
        y,
        outer_X,
        outer_y,
        config,
        {},
        selected,
        preprocessing=adapter,
        evaluation_coverage=coverage,
    )
    assert score == 1.0
    assert coverage["input_rows"] == 3 and coverage["scored_rows"] == 2


def test_empty_eligible_validation_fails_with_explicit_population_reason():
    """Removing every validation row must not produce a successful or zero-filled score."""
    adapter = FeatureEngineerFoldAdapter(_steps(), "target")
    adapter.fit_transform(pd.DataFrame({"x": [0.0, 1.0]}), pd.Series([0, 1]))
    from skyulf.data.coverage import transform_evaluation

    with pytest.raises(ValueError, match="No eligible.*evaluation"):
        transform_evaluation(adapter, pd.DataFrame({"x": [None]}), pd.Series([0]))


def test_disabled_optuna_pruning_keeps_coverage_without_extra_refitting(monkeypatch):
    """Disabling pruning must not restore unfiltered scoring or add a hidden best-model fit."""
    pytest.importorskip("optuna")
    from sklearn.linear_model import LinearRegression

    fitted_rows = []
    original = LinearRegression.fit

    def record_fit(model, X, y, **kwargs):
        """Count the existing two folds and final refit through the public calculator."""
        fitted_rows.append(len(X))
        return original(model, X, y, **kwargs)

    monkeypatch.setattr(LinearRegression, "fit", record_fit)
    X = pd.DataFrame({"x": np.arange(20.0)})
    y = pd.Series(2 * X["x"])
    X.loc[[2, 12], "x"] = np.nan
    _, result = TuningCalculator(LinearRegressionCalculator()).fit(
        X,
        y,
        TuningConfig(
            strategy="optuna",
            metric="neg_mean_squared_error",
            n_trials=1,
            cv_folds=2,
            cv_shuffle=False,
            search_space={"fit_intercept": [True]},
            strategy_params={"pruning": False},
        ),
        preprocessing=FeatureEngineerFoldAdapter(_steps(), "target"),
    )
    assert fitted_rows == [9, 9, 18]
    assert sum(row["excluded_rows"] for row in result.trials[0]["evaluation_coverage"]) == 2


@pytest.mark.parametrize("strategy", ["halving_grid", "halving_random"])
@pytest.mark.parametrize("filter_rows", [False, True])
def test_halving_retains_every_round_population_without_multimetric_selection(
    strategy, filter_rows
):
    """Parallel successive rounds retain earlier counts and scalar best-candidate selection."""
    from skyulf.modeling.regression import RidgeRegressionCalculator

    X = pd.DataFrame({"x": np.arange(80.0)})
    y = pd.Series(2 * X["x"] + 1)
    if filter_rows:
        X.loc[[3, 13, 23, 43, 53, 63], "x"] = np.nan
    _, result = TuningCalculator(RidgeRegressionCalculator()).fit(
        X,
        y,
        TuningConfig(
            strategy=strategy,
            metric="neg_mean_squared_error",
            cv_folds=2,
            cv_shuffle=False,
            n_jobs=2,
            n_trials=4,
            search_space={"alpha": [0.1, 1.0, 10.0, 100.0]},
            strategy_params={"min_resources": 20, "max_resources": 80, "factor": 2},
        ),
        preprocessing=FeatureEngineerFoldAdapter(_steps(), "target") if filter_rows else None,
    )
    assert len(result.trials) == 7
    coverage = [row["evaluation_coverage"] for row in result.trials]
    assert [sum(fold["input_rows"] for fold in row) for row in coverage] == [20] * 4 + [40] * 2 + [
        80
    ]
    assert all(
        fold["input_rows"] == fold["scored_rows"] + fold["excluded_rows"]
        for row in coverage
        for fold in row
    )
    assert sum(fold["excluded_rows"] for fold in coverage[-1]) == (6 if filter_rows else 0)
    assert np.isfinite(result.best_score)


@pytest.mark.parametrize("cv_type", ["k_fold", "nested_cv"])
def test_ordinary_cv_discloses_outer_eligible_population(cv_type):
    """Legacy direct CV and nested fixed-model evaluation retain their true fold denominators."""
    from skyulf.modeling.cross_validation import perform_cross_validation
    from skyulf.modeling.regression import LinearRegressionApplier

    X = pd.DataFrame({"x": np.arange(40.0)})
    y = pd.Series(2 * X["x"] + 1)
    X.loc[[3, 23], "x"] = np.nan
    result = perform_cross_validation(
        LinearRegressionCalculator(),
        LinearRegressionApplier(),
        X,
        y,
        {},
        n_folds=2,
        cv_type=cv_type,
        shuffle=False,
        preprocessing=FeatureEngineerFoldAdapter(_steps(), "target"),
    )
    coverage = [row["evaluation_coverage"] for row in result["folds"]]
    assert sum(row["input_rows"] for row in coverage) == 40
    assert sum(row["scored_rows"] for row in coverage) == 38
    assert sum(row["excluded_rows"] for row in coverage) == 2


def test_final_report_can_retain_an_empty_evaluation_population():
    """Final reports may disclose zero eligible rows while CV remains strict by default."""
    from skyulf.data.coverage import transform_evaluation

    adapter = FeatureEngineerFoldAdapter(_steps(), "target")
    adapter.fit_transform(pd.DataFrame({"x": [0.0, 1.0]}), pd.Series([0, 1]))
    X, y, coverage = transform_evaluation(
        adapter,
        pd.DataFrame({"x": [None]}),
        pd.Series([0]),
        allow_empty=True,
    )
    assert len(X) == len(y) == 0
    assert coverage["input_rows"] == coverage["excluded_rows"] == 1
    assert coverage["scored_rows"] == 0


@pytest.mark.parametrize("strategy", ["grid", "random", "optuna", "halving_grid", "halving_random"])
def test_tuning_empty_eligible_fold_explains_actual_failure(strategy):
    """A wholly excluded fold must report eligibility rather than generic NaN advice."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
    X = pd.DataFrame({"x": [*np.arange(10.0), *([np.nan] * 20)]})
    y = pd.Series(np.arange(30.0), name="target")
    config = TuningConfig(
        strategy=strategy,
        metric="neg_mean_squared_error",
        search_space={"fit_intercept": [True]},
        n_trials=1,
        cv_type="time_series_split",
        cv_folds=2,
        n_jobs=2 if strategy.startswith("halving") else 1,
        strategy_params={"min_resources": 30, "max_resources": 30},
    )
    with pytest.raises(ValueError, match="No eligible rows remain for evaluation"):
        TuningCalculator(LinearRegressionCalculator()).fit(
            X, y, config, preprocessing=FeatureEngineerFoldAdapter(_steps(), "target")
        )


class _ExpandingApplier:
    """Exercise a third-party applier that adds synthetic rows at evaluation time."""

    def apply(self, data, artifact):
        """Add a row without mutating the caller's feature frame."""
        return pd.concat([data, data.iloc[[0]]])


def test_evaluation_expansion_error_names_responsible_step():
    """Custom expansion diagnostics must identify the exact offending pipeline step."""
    engineer = FeatureEngineer([])
    engineer.fitted_steps = [
        {
            "name": "duplicate_observations",
            "type": "CustomExpansion",
            "applier": _ExpandingApplier(),
            "artifact": {},
        }
    ]
    with pytest.raises(ValueError, match="duplicate_observations.*2.*3"):
        engineer.transform(pd.DataFrame({"x": [1, 2]}))
