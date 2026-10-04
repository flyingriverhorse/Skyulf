"""Tuning validates missing features using the installed estimator's capabilities."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
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
@pytest.mark.parametrize("strategy", ["grid", "halving_grid"])
def test_missing_tree_configuration_matches_direct_fit(
    settings: dict[str, Any], strategy: Any
) -> None:
    """Tuning must follow installed tree support, which expanded in sklearn 1.9."""
    X, y = _missing_data()
    calculator = SklearnCalculator(DecisionTreeRegressor, settings, "regression")
    config = TuningConfig(
        strategy=strategy, metric="r2", search_space={"max_depth": [2]}, cv_folds=2
    )
    direct = DecisionTreeRegressor(**settings, max_depth=2)
    try:
        direct.fit(X.to_numpy(), y.to_numpy())
    except ValueError as exc:
        assert "NaN" in str(exc)
        with pytest.raises(ValueError, match="NaN"):
            TuningCalculator(calculator).fit(X, y, config)
        return

    assert np.isfinite(direct.predict(X.to_numpy())).all()
    model, result = TuningCalculator(calculator).fit(X, y, config)
    assert result.best_params == {"max_depth": 2}
    assert np.isfinite(result.best_score)
    assert all(model.get_params()[key] == value for key, value in settings.items())
    assert np.isfinite(model.predict(X.to_numpy())).all()


@pytest.mark.parametrize("estimator_type", [DecisionTreeRegressor, Ridge])
@pytest.mark.parametrize("strategy", ["grid", "halving_grid"])
def test_missing_bagging_features_follow_configured_estimator(
    estimator_type: type[Any], strategy: Any
) -> None:
    """A fixed base estimator must govern bagging's NaN admission before search."""
    X, y = _missing_data()
    settings = {"estimator": estimator_type(), "n_estimators": 2}
    calculator = SklearnCalculator(BaggingRegressor, settings, "regression")
    config = TuningConfig(
        strategy=strategy, metric="r2", search_space={"n_estimators": [2]}, cv_folds=2
    )
    direct = BaggingRegressor(**settings)
    if estimator_type is Ridge:
        with pytest.raises(ValueError, match="NaN"):
            direct.fit(X.to_numpy(), y.to_numpy())
        with pytest.raises(ValueError, match=r"Input features \(X\) contain NaN"):
            TuningCalculator(calculator).fit(X, y, config)
        return

    direct.fit(X.to_numpy(), y.to_numpy())
    assert np.isfinite(direct.predict(X.to_numpy())).all()
    model, result = TuningCalculator(calculator).fit(X, y, config)
    assert result.best_params == {"n_estimators": 2}
    assert np.isfinite(result.best_score)
    assert isinstance(model.estimator_, estimator_type)
    assert np.isfinite(model.predict(X.to_numpy())).all()


@pytest.mark.parametrize("strategy", ["grid", "halving_grid"])
def test_search_can_replace_a_nan_incompatible_base_estimator(strategy: Any) -> None:
    """A searched native base estimator must replace bagging's non-native default."""
    X, y = _missing_data()
    calculator = SklearnCalculator(
        BaggingRegressor, {"estimator": Ridge(), "n_estimators": 2}, "regression"
    )
    with pytest.raises(ValueError, match="NaN"):
        BaggingRegressor(estimator=Ridge(), n_estimators=2).fit(X.to_numpy(), y.to_numpy())

    replacement = DecisionTreeRegressor(max_depth=2)
    model, result = TuningCalculator(calculator).fit(
        X,
        y,
        TuningConfig(
            strategy=strategy,
            metric="r2",
            search_space={"estimator": [replacement]},
            cv_folds=2,
        ),
    )
    assert result.best_params == {"estimator": replacement}
    assert np.isfinite(result.best_score)
    assert isinstance(model.estimator_, DecisionTreeRegressor)
    assert model.estimator_.max_depth == 2
    assert np.isfinite(model.predict(X.to_numpy())).all()


@pytest.mark.parametrize("strategy", ["grid", "halving_grid"])
def test_search_can_replace_a_criterion_with_version_dependent_nan_support(strategy: Any) -> None:
    """The searched criterion must work even on releases rejecting the fixed default."""
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
