"""An unspecified Elastic Net mix must have the same meaning in fits and searches."""

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.params import normalize_logistic_search_config
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import LogisticRegressionCalculator
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    """Use seeded informative and noise features to distinguish Elastic Net from L2."""
    values, labels = make_classification(
        n_samples=120, n_features=6, n_informative=3, n_redundant=0, random_state=17
    )
    return pd.DataFrame(values), pd.Series(labels)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_null_elasticnet_ratio_matches_the_omitted_default(
    classification_data, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The nullable UI default must not silently train a pure L2 model."""
    X, y = classification_data
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    if engine == "polars":
        X, y = pl.from_pandas(X), pl.from_pandas(y)
    settings = {
        "penalty": "elasticnet",
        "solver": "saga",
        "max_iter": 1000,
        "random_state": 42,
        "C": 0.1,
        "tol": 1e-7,
    }
    calculator = LogisticRegressionCalculator()

    explicit_null = calculator.fit(X, y, {"params": {**settings, "l1_ratio": None}})
    omitted = calculator.fit(X, y, {"params": settings})

    assert explicit_null.l1_ratio == 0.5
    np.testing.assert_allclose(explicit_null.coef_, omitted.coef_)
    np.testing.assert_allclose(
        explicit_null.predict_proba(X.to_numpy()), omitted.predict_proba(X.to_numpy())
    )


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
@pytest.mark.parametrize("ratio_config", ["null", "omitted", "string-none", "fixed-null"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_search_and_final_refit_use_the_same_elasticnet_default(
    strategy: Any,
    classification_data,
    ratio_config: str,
    wrapped: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search scores must describe the same penalty mixture used by the returned model."""
    X, y = classification_data
    fitted_ratios = []
    original_fit = LogisticRegression.fit

    def record_fit(self, *args, **kwargs):
        """Observe the actual estimators used by search folds and the final refit."""
        result = original_fit(self, *args, **kwargs)
        fitted_ratios.append(self.l1_ratio)
        return result

    monkeypatch.setattr(LogisticRegression, "fit", record_fit)
    space: dict[str, list[Any]] = {
        "penalty": ["elasticnet"],
        "solver": ["saga"],
        "C": [0.1],
        "max_iter": [1000],
    }
    calculator = LogisticRegressionCalculator()
    if ratio_config == "fixed-null":
        calculator.default_params.update(
            {"penalty": "elasticnet", "l1_ratio": None, "solver": "saga"}
        )
        del space["penalty"]
    elif ratio_config != "omitted":
        space["l1_ratio"] = [None if ratio_config == "null" else "none"]
    config = TuningConfig(
        strategy=strategy,
        n_trials=1,
        cv_folds=2,
        cv_type="stratified_k_fold",
        n_jobs=1,
        search_space=space,
    )
    original_config, original_defaults = deepcopy(config), deepcopy(calculator.default_params)
    preprocessing = (
        FeatureEngineerFoldAdapter(
            [{"name": "scale", "transformer": "StandardScaler", "params": {}}],
            target_column="target",
        )
        if wrapped
        else None
    )
    model, result = TuningCalculator(calculator).fit(X, y, config, preprocessing=preprocessing)

    assert config == original_config
    assert calculator.default_params == original_defaults
    assert np.isfinite(result.best_score)
    assert len(fitted_ratios) >= 3
    assert fitted_ratios == [0.5] * len(fitted_ratios)
    assert model.l1_ratio == 0.5


@pytest.mark.parametrize("ratio", [0.0, 0.25, 1.0])
def test_explicit_elasticnet_ratio_is_preserved(classification_data, ratio: float) -> None:
    """Resolving a null default must not replace an intentional mixture or endpoint."""
    X, y = classification_data
    settings = {"penalty": "elasticnet", "solver": "saga", "l1_ratio": ratio}
    model = LogisticRegressionCalculator().fit(X, y, settings)
    tuned, _ = TuningCalculator(LogisticRegressionCalculator()).fit(
        X,
        y,
        TuningConfig(
            strategy="halving_grid",
            cv_folds=2,
            search_space={key: [value] for key, value in settings.items()},
        ),
    )
    assert model.l1_ratio == ratio
    assert tuned.l1_ratio == ratio


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_pipeline_resolves_the_nullable_elasticnet_setting(
    classification_data, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The serialized modeling configuration must reach the corrected estimator default."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    X, y = classification_data
    data = X.rename(columns=str).assign(target=y)
    if engine == "polars":
        data = pl.from_pandas(data)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [],
            "modeling": {
                "type": "logistic_regression",
                "params": {
                    "penalty": "elasticnet",
                    "solver": "saga",
                    "l1_ratio": None,
                },
            },
        }
    )

    pipeline.fit(data, target_column="target")

    assert pipeline.model_estimator is not None
    model = pipeline.model_estimator.model
    assert isinstance(model, LogisticRegression)
    assert model.l1_ratio == 0.5
    assert len(
        pipeline.predict(data.drop("target") if engine == "polars" else data.drop(columns="target"))
    ) == len(data)


@pytest.mark.parametrize("strategy", ["grid", "random", "halving_grid", "halving_random", "optuna"])
@pytest.mark.parametrize("ratios", [None, [None], [0.25, None]])
def test_mixed_penalty_search_requires_separate_unspecified_elasticnet(
    classification_data, strategy: Any, ratios: Any
) -> None:
    """A conditional default must not silently assign one mixture to unrelated penalties."""
    X, y = classification_data
    space: dict[str, list[Any]] = {"penalty": ["l2", "elasticnet"], "solver": ["saga"]}
    if ratios is not None:
        space["l1_ratio"] = ratios
    with pytest.raises(ValueError, match="search elasticnet separately.*l1_ratio"):
        TuningCalculator(LogisticRegressionCalculator()).fit(
            X, y, TuningConfig(strategy=strategy, cv_folds=2, search_space=space)
        )


@pytest.mark.parametrize("distribution_case", ["numeric", "penalty", "nullable-category"])
def test_optuna_distributions_preserve_their_search_semantics(
    classification_data, distribution_case: str
) -> None:
    """Resolving null defaults must preserve supported distribution objects and caller state."""
    from optuna.distributions import CategoricalDistribution, FloatDistribution

    X, y = classification_data
    space: dict[str, Any] = {"solver": ["saga"], "penalty": ["elasticnet"]}
    if distribution_case == "numeric":
        space["l1_ratio"] = FloatDistribution(0.2, 0.8)
    else:
        space["penalty"] = CategoricalDistribution(["elasticnet"])
        space["l1_ratio"] = (
            CategoricalDistribution([None]) if distribution_case == "nullable-category" else [0.5]
        )
    config = TuningConfig(strategy="optuna", search_space=space, cv_folds=2, n_trials=1)
    original = deepcopy(config)

    model, result = TuningCalculator(LogisticRegressionCalculator()).fit(X, y, config)

    assert config == original
    assert np.isfinite(result.best_score)
    assert model.l1_ratio == result.best_params["l1_ratio"]
    if distribution_case == "numeric":
        assert 0.2 <= model.l1_ratio <= 0.8
    else:
        assert model.l1_ratio == 0.5


def test_nullable_categorical_ratios_stay_categorical() -> None:
    """CMA-ES must not reinterpret discrete ratio choices as a continuous numeric range."""
    from optuna.distributions import CategoricalDistribution

    space: dict[str, Any] = {
        "penalty": ["elasticnet"],
        "l1_ratio": CategoricalDistribution([None, 0.75]),
    }
    config = TuningConfig(strategy="optuna", search_space=space)
    resolved = normalize_logistic_search_config(LogisticRegression, {}, config)

    ratios = resolved.search_space["l1_ratio"]
    assert isinstance(ratios, CategoricalDistribution)
    assert ratios.choices == (0.5, 0.75)
    assert space["l1_ratio"].choices == (None, 0.75)
