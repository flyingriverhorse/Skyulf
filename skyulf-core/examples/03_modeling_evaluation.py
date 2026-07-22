"""Model families, metrics, tuning, and optional SHAP explanations.

Run from the repository root:
    python skyulf-core/examples/03_modeling_evaluation.py

This is intentionally compact: use it as an executable API map, then choose
the family that suits the problem.  The same Calculator/Applier pairs are what
the pipeline engine resolves from its model registry.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from skyulf.modeling._evaluation import (
    calculate_classification_metrics,
    calculate_clustering_metrics,
    calculate_regression_metrics,
)
from skyulf.modeling._explainability.shap_explanation import compute_shap_explanation
from skyulf.modeling._tuning import TuningCalculator
from skyulf.modeling.classification import LogisticRegressionApplier, LogisticRegressionCalculator
from skyulf.modeling.clustering import (
    BirchApplier,
    BirchCalculator,
    GaussianMixtureApplier,
    GaussianMixtureCalculator,
    KMeansApplier,
    KMeansCalculator,
    MiniBatchKMeansApplier,
    MiniBatchKMeansCalculator,
)
from skyulf.modeling.ensemble import (
    StackingClassifierApplier,
    StackingClassifierCalculator,
    VotingClassifierApplier,
    VotingClassifierCalculator,
)
from skyulf.modeling.regression import RidgeRegressionApplier, RidgeRegressionCalculator


def fit_classifier(
    calculator: Any, applier: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame
) -> Any:
    """Fit one classifier through Skyulf's public calculator interface."""
    model = calculator.fit(X_train, y_train, {"params": {"random_state": 42, "max_iter": 500}})
    return model, applier.predict(X_test, model)


def classification_and_ensembles() -> tuple[Any, pd.DataFrame, pd.Series]:
    """Evaluate logistic regression plus voting and stacking ensembles."""
    dataset = load_breast_cancer(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42, stratify=dataset.target
    )
    scaler = StandardScaler().fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    logistic, predictions = fit_classifier(
        LogisticRegressionCalculator(), LogisticRegressionApplier(), X_train, y_train, X_test
    )
    metrics = calculate_classification_metrics(logistic, X_test, y_test)
    print(
        f"Logistic regression accuracy: {metrics['accuracy']:.4f}; ROC AUC: {metrics['roc_auc']:.4f}"
    )

    ensemble_config = {
        "params": {
            "base_estimators": ["logistic_regression", "random_forest"],
            "voting": "soft",
            "cv": 3,
        }
    }
    for name, calculator, _applier in [
        ("Voting", VotingClassifierCalculator(), VotingClassifierApplier()),
        ("Stacking", StackingClassifierCalculator(), StackingClassifierApplier()),
    ]:
        model = calculator.fit(X_train, y_train, ensemble_config)
        score = calculate_classification_metrics(model, X_test, y_test)["accuracy"]
        print(f"{name} ensemble accuracy: {score:.4f}")

    return logistic, X_test, y_test


def regression() -> None:
    """Fit a regression model and calculate the standardized regression metrics."""
    dataset = load_diabetes(as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=42
    )
    calculator = RidgeRegressionCalculator()
    model = calculator.fit(X_train, y_train, {"params": {"alpha": 1.0}})
    metrics = calculate_regression_metrics(model, X_test, y_test)
    print(f"Ridge regression RMSE: {metrics['rmse']:.3f}; R²: {metrics['r2']:.3f}")


def clustering() -> None:
    """Fit all deployable clustering/segmentation models and show silhouette scores."""
    dataset = load_iris(as_frame=True)
    X = dataset.data
    models = [
        ("KMeans", KMeansCalculator(), KMeansApplier(), {"n_clusters": 3, "random_state": 42}),
        (
            "MiniBatchKMeans",
            MiniBatchKMeansCalculator(),
            MiniBatchKMeansApplier(),
            {"n_clusters": 3, "random_state": 42},
        ),
        (
            "GaussianMixture",
            GaussianMixtureCalculator(),
            GaussianMixtureApplier(),
            {"n_components": 3, "random_state": 42},
        ),
        ("Birch", BirchCalculator(), BirchApplier(), {"n_clusters": 3}),
    ]
    for name, calculator, applier, params in models:
        model = calculator.fit(X, None, {"params": params})
        labels = applier.predict(X, model)
        metrics = calculate_clustering_metrics(X, labels)
        print(
            f"{name} clusters={int(metrics['n_clusters'])}; silhouette={metrics['silhouette_score']:.3f}"
        )


def tuning(X: pd.DataFrame, y: pd.Series) -> None:
    """Run small Grid, Random, and Optuna searches through the tuning engine."""
    calculator = LogisticRegressionCalculator()
    tuner = TuningCalculator(calculator)
    common = {
        "metric": "accuracy",
        "cv_type": "stratified_k_fold",
        "cv_folds": 2,
        "cv_random_state": 42,
        "random_state": 42,
        "search_space": {"C": [0.1, 1.0], "solver": ["liblinear"]},
    }
    for strategy, trials in [("grid", 4), ("random", 2), ("optuna", 2)]:
        model, result = tuner.fit(X, y, {**common, "strategy": strategy, "n_trials": trials})
        assert model is not None
        print(
            f"{strategy} tuning: best CV accuracy={result.best_score:.4f}; "
            f"params={result.best_params}"
        )


def explain(model: Any, X: pd.DataFrame) -> None:
    """Compute a bounded SHAP explanation when the optional dependency is installed."""
    explanation = compute_shap_explanation(model, X, max_samples=40, max_display_samples=2)
    if explanation is None:
        print("SHAP explanation unavailable (install skyulf-core[explainability]).")
        return
    top_feature = max(
        explanation["mean_abs_importance"], key=explanation["mean_abs_importance"].get
    )
    print(f"SHAP explanation produced; top feature: {top_feature}")


def main() -> None:
    """Run each modeling family once on small, deterministic datasets."""
    warnings.filterwarnings("ignore", message="OptunaSearchCV is experimental")
    logistic, X_test, y_test = classification_and_ensembles()
    regression()
    clustering()
    tuning(X_test, y_test)
    explain(logistic, X_test)
    print("Modeling, tuning, evaluation, and explainability examples completed.")


if __name__ == "__main__":
    main()
