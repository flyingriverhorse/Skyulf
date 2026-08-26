"""Benchmark: leakage-free CV + holdout tuning (v0.8.4) — old vs new scores.

SECTION 1 — leakage probe: pure-noise target + memorising WOE encoder.
  Any score above ~0.65 (disc) is memorisation of held-out rows.
  OLD: preprocessing fitted once on the full split before CV/holdout scoring.
  NEW: chain refits inside every fold (CV) / on train rows only via a single
       PredefinedSplit fold (holdout), scoring untouched held-out rows.

SECTION 2 — real-signal imbalanced data + SMOTE oversampling.
  Shows the honest scores stay meaningful: the drop vs OLD is the optimistic
  bias leaving, not the signal.

Reference numbers (seeded, reproducible): the OLD column reproduces the
literal v0.7.9 behaviour — the same probe run against a checkout of tag
v0.7.9 scored 0.8669 on the noise dataset. The NEW column is the per-fold
refit; chance level on the noise probe is 0.50.

Run from the repo root:
    .venv/Scripts/python.exe skyulf-core/benchmarks/bench_holdout_refit_leakage.py
"""

from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from skyulf.modeling._tuning.engine import TuningCalculator
from skyulf.modeling._tuning.schemas import TuningConfig
from skyulf.modeling.classification import (
    LogisticRegressionCalculator,
    XGBClassifierCalculator,
)
from skyulf.preprocessing.encoding import WOEEncoderApplier, WOEEncoderCalculator
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter

_Strategy = Literal["grid", "random", "optuna", "halving_grid", "halving_random"]

WOE_STEPS = [
    {
        "name": "woe_city",
        "transformer": "WOEEncoder",
        "params": {"columns": ["city"], "regularization": 0.5},
    }
]
SMOTE_STEPS = [
    {
        "name": "smote",
        "transformer": "Oversampling",
        "params": {"method": "smote", "target_column": "target"},
    }
]
LR_SPACE = {"C": [0.5, 1.0]}
XGB_SPACE = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1, 0.3],
    "n_estimators": [40, 80],
}


def disc(auc: float) -> float:
    return max(auc, 1.0 - auc)


def tune_once(calculator, X, y, config, **kwargs):
    _m, result = calculator.fit(X, y, config=config, **kwargs)
    return float(result.best_score)


def noise_woe_frames():
    rng = np.random.default_rng(42)
    n, n_categories = 400, 200
    X = pd.DataFrame({"city": [f"c{v}" for v in rng.integers(0, n_categories, size=n)]})
    y = pd.Series(rng.integers(0, 2, size=n), name="target")
    return X.iloc[:320], y.iloc[:320], X.iloc[320:], y.iloc[320:]


def section_1() -> None:
    print("=" * 78)
    print("SECTION 1 — pure-noise target + memorising WOE (leakage probe)")
    print("disc(ROC-AUC) above ~0.65 = memorisation of held-out rows; chance = 0.50")
    print("=" * 78)
    X_tr, y_tr, X_val, y_val = noise_woe_frames()

    # OLD behaviour: preprocessing fitted once on the FULL frame before any
    # split (the app's pre-split FE run), then CV/holdout score the
    # transformed rows — held-out labels sit inside the WOE table.
    X_all = pd.concat([X_tr, X_val], ignore_index=True)
    y_all = pd.concat([y_tr, y_val], ignore_index=True)
    # functools.wraps copies the inner (X, y, config) signature, but the
    # @fit_method/@apply_method wrappers take (df, config) — cast keeps the
    # call shape honest for static analyzers.
    leaky_params = cast(Any, WOEEncoderCalculator()).fit((X_all, y_all), WOE_STEPS[0]["params"])
    X_leaky, _ = cast(Any, WOEEncoderApplier()).apply((X_all, y_all), dict(leaky_params))

    cv_config = TuningConfig(
        strategy="grid", metric="roc_auc", search_space=dict(LR_SPACE), cv_folds=5
    )
    old_cv = tune_once(
        TuningCalculator(LogisticRegressionCalculator()),
        X_leaky.iloc[:320],
        y_all.iloc[:320],
        cv_config,
    )
    # NEW CV: chain refits inside every fold.
    new_cv = tune_once(
        TuningCalculator(LogisticRegressionCalculator()),
        X_tr,
        y_tr,
        cv_config,
        preprocessing=FeatureEngineerFoldAdapter(WOE_STEPS, "target"),
    )

    print(f"{'mode':<30}{'OLD (leaky)':>14}{'NEW (refit)':>14}")
    print(f"{'CV tuning (grid)':<30}{old_cv:>14.4f}{new_cv:>14.4f}")

    def holdout_config(strategy: _Strategy) -> TuningConfig:
        return TuningConfig(
            strategy=strategy, metric="roc_auc", n_trials=4, search_space=dict(LR_SPACE)
        )

    old_holdout = tune_once(
        TuningCalculator(LogisticRegressionCalculator()),
        X_leaky.iloc[:320],
        y_all.iloc[:320],
        holdout_config("grid"),
        validation_data=(X_leaky.iloc[320:], y_all.iloc[320:]),
    )
    print(f"{'holdout tuning (grid)':<30}{old_holdout:>14.4f}{'':>14}")
    strategies: list[_Strategy] = [
        "grid",
        "random",
        "halving_grid",
        "halving_random",
        "optuna",
    ]
    for strategy in strategies:
        new = tune_once(
            TuningCalculator(LogisticRegressionCalculator()),
            X_tr,
            y_tr,
            holdout_config(strategy),
            validation_data=(X_val, y_val),
            preprocessing=FeatureEngineerFoldAdapter(WOE_STEPS, "target"),
            validation_frames=(X_val, y_val),
        )
        print(f"{'holdout ' + strategy + ' (NEW refit)':<30}{'':>14}{new:>14.4f}")
    print()


def imbalanced_split():
    X_arr, y_arr = make_classification(
        n_samples=1500,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        weights=[0.9, 0.1],
        flip_y=0.03,
        random_state=42,
    )
    X = pd.DataFrame(X_arr, columns=pd.Index([f"f{i}" for i in range(8)]))
    y = pd.Series(y_arr, name="target")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    return X_tr.reset_index(drop=True), y_tr.reset_index(drop=True), X_te, y_te


def section_2() -> None:
    print("=" * 78)
    print("SECTION 2 — real-signal imbalanced data + SMOTE (honest scores survive)")
    print("=" * 78)
    X_tr, y_tr, _X_te, _y_te = imbalanced_split()
    print(f"train: {len(X_tr)} rows, minority {y_tr.mean() * 100:.1f}%")
    print(f"{'strategy':<18}{'OLD (leaky)':>14}{'NEW (per-fold)':>16}{'delta':>10}")
    searches: list[tuple[_Strategy, int]] = [
        ("grid", 12),
        ("random", 12),
        ("halving_random", 15),
        ("optuna", 15),
    ]
    for strategy, n_trials in searches:
        config = TuningConfig(
            strategy=strategy,
            metric="roc_auc",
            n_trials=n_trials,
            search_space=dict(XGB_SPACE),
            cv_folds=3,
            cv_type="stratified_k_fold",
        )
        leaky_adapter = FeatureEngineerFoldAdapter(SMOTE_STEPS, "target")
        X_aug, y_aug = leaky_adapter.fit_transform(X_tr, y_tr)
        old = tune_once(TuningCalculator(XGBClassifierCalculator()), X_aug, y_aug, config)
        new = tune_once(
            TuningCalculator(XGBClassifierCalculator()),
            X_tr,
            y_tr,
            config,
            preprocessing=FeatureEngineerFoldAdapter(SMOTE_STEPS, "target"),
        )
        print(f"{strategy:<18}{old:>14.4f}{new:>16.4f}{old - new:>10.4f}")

    # NEW holdout with a validation split (previously refused with ValueError).
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, random_state=7, stratify=y_tr
    )
    holdout_config = TuningConfig(
        strategy="grid",
        metric="roc_auc",
        search_space=dict(XGB_SPACE),
    )
    new_holdout = tune_once(
        TuningCalculator(XGBClassifierCalculator()),
        X_fit.reset_index(drop=True),
        y_fit.reset_index(drop=True),
        holdout_config,
        validation_data=(X_val.reset_index(drop=True), y_val.reset_index(drop=True)),
        preprocessing=FeatureEngineerFoldAdapter(SMOTE_STEPS, "target"),
        validation_frames=(X_val.reset_index(drop=True), y_val.reset_index(drop=True)),
    )
    print(f"\nNEW holdout tuning with validation split (previously ValueError): {new_holdout:.4f}")


if __name__ == "__main__":
    section_1()
    section_2()
