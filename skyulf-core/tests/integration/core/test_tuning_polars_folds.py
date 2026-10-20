"""Every tuner must convert post-preprocessing frames at the estimator boundary."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("lightgbm")

from skyulf.engines.sklearn_bridge import SklearnBridge
from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import LGBMClassifierCalculator
from skyulf.modeling.regression import LGBMRegressorCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter


class PolarsImputer:
    """Run a real fold-local imputer and return its native Polars output."""

    def __init__(self):
        """Store an unfitted recipe so searcher copies own their fitted statistics."""
        self.adapter = FeatureEngineerFoldAdapter(
            [{"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x", "z"]}}],
            target_column="target",
        )

    @staticmethod
    def _native(X, y):
        """Preserve fold values when a searcher has reconstructed pandas inputs."""
        features = X if isinstance(X, pl.DataFrame) else pl.from_pandas(pd.DataFrame(X))
        target = None if y is None else pl.Series("target", np.asarray(y).reshape(-1))
        return features, target

    def fit_transform(self, X, y):
        """Fit imputation on this fold only without returning the global fitted state."""
        return self.adapter.fit_transform(*self._native(X, y))

    def transform(self, X, y):
        """Apply only the current fold's statistics to the paired held-out rows."""
        return self.adapter.transform(*self._native(X, y))


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize(
    "strategy,pruner",
    [
        ("grid", "none"),
        ("random", "none"),
        ("halving_grid", "none"),
        ("halving_random", "none"),
        ("optuna", "none"),
        ("optuna", "median"),
    ],
)
def test_lightgbm_search_accepts_native_fold_outputs(
    strategy: Any, pruner: str, classification: bool
):
    """Grid, searcher and native-pruning paths must agree on their numeric model inputs."""
    if strategy == "optuna":
        pytest.importorskip("optuna")
        pytest.importorskip("optuna_integration")
    rng = np.random.default_rng(19)
    x = rng.normal(size=96)
    frame = pl.DataFrame({"x": x, "z": rng.normal(size=96)})
    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise(pl.col("x")).alias("x")
    )
    target = pl.Series("target", (x > 0).astype(int) if classification else 2 * x)
    adapter = PolarsImputer()
    calculator = LGBMClassifierCalculator() if classification else LGBMRegressorCalculator()
    model, result = TuningCalculator(calculator).fit(
        frame,
        target,
        preprocessing=adapter,
        config=TuningConfig(
            strategy=strategy,
            metric="accuracy" if classification else "r2",
            n_trials=1,
            cv_folds=2,
            n_jobs=1,
            search_space={"n_estimators": [4], "n_jobs": [1], "min_child_samples": [2]},
            strategy_params={"pruner": pruner},
        ),
    )
    assert result.n_trials >= 1
    assert np.isfinite(result.best_score)
    features, labels = adapter.transform(frame, target)
    values, _ = SklearnBridge.to_sklearn((features, labels), validate_features=True)
    predictions = model.predict(values)
    assert len(predictions) == len(frame)
    assert np.isfinite(predictions).all()
