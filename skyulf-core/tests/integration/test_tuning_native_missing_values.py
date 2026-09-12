"""Tuning validates missing features using the installed estimator's capabilities."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor

from skyulf.modeling._tuning import engine as tuning_engine
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import (
    DecisionTreeClassifierCalculator,
    ExtraTreesClassifierCalculator,
    LogisticRegressionCalculator,
    RandomForestClassifierCalculator,
)
from skyulf.modeling.regression import (
    DecisionTreeRegressorCalculator,
    ExtraTreesRegressorCalculator,
    RandomForestRegressorCalculator,
)
from skyulf.modeling.sklearn_wrapper import SklearnCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


def _missing_data() -> tuple[pd.DataFrame, pd.Series]:
    """Create numeric missing features with enough balanced labels for every fold."""
    X = pd.DataFrame({"x": np.arange(60, dtype=float), "z": np.tile([0.0, 1.0], 30)})
    X.loc[::5, "x"] = np.nan
    return X, pd.Series(np.tile([0, 1], 30), name="target")


@pytest.mark.parametrize(
    "calculator_type",
    [
        RandomForestClassifierCalculator,
        ExtraTreesClassifierCalculator,
        DecisionTreeClassifierCalculator,
        RandomForestRegressorCalculator,
        ExtraTreesRegressorCalculator,
        DecisionTreeRegressorCalculator,
    ],
)
@pytest.mark.parametrize("strategy", ["grid", "halving_grid"])
def test_native_missing_features_reach_search_and_best_model(
    calculator_type: type[Any], strategy: Any
) -> None:
    """A model that can fit the supplied missing features must also be tunable."""
    calculator = calculator_type()
    X, y = _missing_data()
    direct = calculator.fit(X, y, {"params": {"max_depth": 2}})
    assert np.isfinite(direct.predict(X.to_numpy())).all()

    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy=strategy,
            metric="accuracy" if calculator.problem_type == "classification" else "r2",
            search_space={"max_depth": [2]},
            cv_folds=2,
        ),
    )

    assert result.best_params == {"max_depth": 2}
    assert np.isfinite(result.best_score)
    assert np.isfinite(model.predict(X.to_numpy())).all()


@pytest.mark.parametrize("settings", [{"criterion": "absolute_error"}, {"monotonic_cst": [1, 0]}])
def test_missing_unsupported_tree_configuration_is_rejected(settings: dict[str, Any]) -> None:
    """NaN support for one tree configuration must not authorize incompatible settings."""
    X, y = _missing_data()
    calculator = SklearnCalculator(DecisionTreeRegressor, settings, "regression")
    with pytest.raises(ValueError, match="NaN"):
        TuningCalculator(calculator).fit(
            X, y, TuningConfig(strategy="grid", metric="r2", search_space={"max_depth": [2]})
        )


@pytest.mark.parametrize("strategy", ["grid", "halving_grid"])
def test_search_can_select_missing_support_over_an_incompatible_default(strategy: Any) -> None:
    """A searched criterion must replace a fixed default before judging its NaN support."""
    X, y = _missing_data()
    calculator = SklearnCalculator(
        DecisionTreeRegressor, {"criterion": "absolute_error"}, "regression"
    )
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy=strategy,
            metric="r2",
            search_space={"criterion": ["squared_error"]},
            cv_folds=2,
        ),
    )
    assert result.best_params == {"criterion": "squared_error"}
    assert np.isfinite(model.predict(X.to_numpy())).all()


def test_capability_probe_preserves_required_searchable_structure() -> None:
    """Preflight must not remove required estimator structure when the search replaces it."""
    X, y = _missing_data()
    X = X.fillna(0)
    estimators = [("tree", DecisionTreeRegressor(criterion="absolute_error", max_depth=2))]
    calculator = SklearnCalculator(VotingRegressor, {"estimators": estimators}, "regression")
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy="grid",
            metric="r2",
            search_space={"estimators": [estimators]},
            cv_folds=2,
        ),
    )
    assert np.isfinite(result.best_score)
    assert np.isfinite(model.predict(X.to_numpy())).all()


def test_legacy_estimator_tags_still_admit_missing_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Supported sklearn releases before get_tags must retain native NaN tuning."""

    class LegacyTaggedTree(DecisionTreeRegressor):
        """Expose the tag API used by the supported sklearn 1.4 and 1.5 releases."""

        def _get_tags(self) -> dict[str, bool]:
            """Report the capability consumed by the older estimator-tag contract."""
            return {"allow_nan": True}

    monkeypatch.setattr(tuning_engine, "get_tags", None)
    X, y = _missing_data()
    calculator = SklearnCalculator(LegacyTaggedTree, {}, "regression")
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(strategy="grid", metric="r2", search_space={"max_depth": [2]}, cv_folds=2),
    )
    assert result.best_params == {"max_depth": 2}
    assert np.isfinite(model.predict(X.to_numpy())).all()


@pytest.mark.parametrize("invalid", ["features_inf", "target_nan", "target_inf"])
def test_native_missing_support_does_not_allow_other_nonfinite_values(invalid: str) -> None:
    """Native feature handling must not weaken target or infinity validation."""
    X, y = _missing_data()
    target_values = y.to_numpy(dtype=float)
    if invalid == "features_inf":
        X.loc[0, "x"] = np.inf
    else:
        target_values[0] = np.nan if invalid == "target_nan" else np.inf
    with pytest.raises(ValueError, match="Infinite|Target variable"):
        TuningCalculator(DecisionTreeClassifierCalculator()).fit(
            X,
            pd.Series(target_values),
            TuningConfig(strategy="grid", search_space={"max_depth": [2]}),
        )


@pytest.mark.parametrize("strategy", ["grid", "halving_grid"])
def test_fold_imputation_allows_missing_features_for_non_native_model(strategy: Any) -> None:
    """Raw missing rows remain valid when preprocessing fills them within each fold."""
    X, y = _missing_data()
    preprocessor = FeatureEngineerFoldAdapter(
        [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["x"], "strategy": "mean"},
            }
        ],
        target_column="target",
    )
    model, result = TuningCalculator(LogisticRegressionCalculator()).fit(
        X,
        y,
        TuningConfig(strategy=strategy, search_space={"C": [1.0]}, cv_folds=2),
        preprocessing=preprocessor,
    )
    transformed, _ = preprocessor.transform(X, y)
    assert np.isfinite(result.best_score)
    assert np.isfinite(model.predict(transformed.to_numpy())).all()
