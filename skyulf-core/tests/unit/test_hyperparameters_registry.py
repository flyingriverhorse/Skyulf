"""Default search spaces must describe useful, executable model candidates."""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs, make_regression
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.ensemble import VotingRegressorCalculator
from skyulf.modeling.hyperparameters import (
    build_ensemble_search_space,
    get_default_search_space,
    get_hyperparameters,
)


@pytest.mark.parametrize("strategy", ["random", "grid"])
@pytest.mark.parametrize(
    ("model_key", "model_class", "cluster_param"),
    [
        ("kmeans", KMeans, "n_clusters"),
        ("minibatch_kmeans", MiniBatchKMeans, "n_clusters"),
        ("gaussian_mixture", GaussianMixture, "n_components"),
        ("birch", Birch, "n_clusters"),
    ],
)
def test_clustering_default_candidates_fit(model_key, model_class, cluster_param, strategy):
    """Missing spaces or unsupported candidates must not reach the tuning UI."""
    space = get_default_search_space(model_key, strategy)
    assert cluster_param in space
    assert len(space[cluster_param]) > 1
    fields = {field["name"]: field for field in get_hyperparameters(model_key)}
    assert all(fields[name]["tunable"] for name in space)
    candidates = ParameterGrid(space)
    if strategy == "grid":
        assert len(candidates) <= 32

    X, _ = make_blobs(
        n_samples=120, centers=12, cluster_std=0.05, center_box=(-25, 25), random_state=42
    )
    with threadpool_limits(limits=1):
        for candidate in candidates:
            model = model_class(**candidate)
            if "random_state" in model.get_params():
                model.set_params(random_state=42)
            model.fit(X)
            labels = model.predict(X[:10])
            assert labels.shape == (10,)
            assert np.isfinite(labels).all()


@pytest.mark.parametrize("strategy", ["random", "grid"])
def test_voting_regressor_exposes_only_fixed_top_level_fields(strategy):
    """Structural selection and CPU parallelism must not be offered as trial candidates."""
    fields = get_hyperparameters("voting_regressor")
    assert {field["name"] for field in fields} == {"base_estimators", "n_jobs"}
    assert get_default_search_space("voting_regressor", strategy) == {}
    assert all(field["tunable"] is False for field in fields)


@pytest.mark.parametrize("strategy", ["random", "grid"])
def test_voting_regressor_tunes_selected_base_models(strategy):
    """An empty meta-parameter space must preserve opt-in tuning of custom base learners."""
    calculator = VotingRegressorCalculator()
    base_config = {"base_estimators": ["linear_regression", "lasso"], "tune_base_models": True}
    calculator.prepare_tuning_params(base_config)
    assert (
        calculator.build_tuning_search_space({**base_config, "tune_base_models": False}, strategy)
        == {}
    )
    space = calculator.build_tuning_search_space(base_config, strategy)
    assert set(space) == {"linear_regression__fit_intercept", "lasso__alpha"}
    X, y = make_regression(n_samples=40, n_features=3, noise=0.1, random_state=42)
    config = TuningConfig(
        strategy=strategy,
        metric="r2",
        search_space=space,
        cv_folds=2,
        n_trials=2,
    )
    with threadpool_limits(limits=1):
        model, result = TuningCalculator(calculator).fit(
            pd.DataFrame(X), pd.Series(y), config=config.__dict__
        )
    assert [name for name, _ in model.estimators] == ["linear_regression", "lasso"]
    assert result.n_trials >= 2
    assert np.isfinite(result.best_score)
    assert all(model.get_params()[name] == value for name, value in result.best_params.items())


def test_build_ensemble_search_space_skips_unmapped_base_estimator_key():
    """A base_estimators key with no registry mapping should be silently skipped."""
    space = build_ensemble_search_space(
        "voting_classifier",
        base_estimators=["not_a_real_base_estimator_key"],
        problem_type="classification",
    )
    # No nested keys should be added for the unmapped estimator.
    assert not any(k.startswith("not_a_real_base_estimator_key__") for k in space)
    # The ensemble's own meta-param space should still be present.
    assert "voting" in space
