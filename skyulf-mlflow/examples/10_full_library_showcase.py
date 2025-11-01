"""Complete end-to-end showcase for the Skyulf-MLFlow library.

This script exercises the core modules shipped with the library:
- Data ingestion via DataSaver/DataLoader
- Preprocessing with imputers and scalers
- Feature engineering with FeatureMath, SmartBinning, and OneHotEncoder
- Feature selection using FeatureSelector
- Pipeline orchestration via make_pipeline
- Model training and evaluation with RandomForestClassifier and MetricsCalculator
- Model registry persistence through ModelRegistry

The example uses the scikit-learn wine dataset and augments it with
categorical and datetime signals to demonstrate the breadth of the toolkit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from skyulf_mlflow_library.data_ingestion import DataLoader, DataSaver
from skyulf_mlflow_library.features import FeatureMath, OneHotEncoder, SmartBinning
from skyulf_mlflow_library.features.selection import FeatureSelector
from skyulf_mlflow_library.modeling import MetricsCalculator, ModelRegistry
from skyulf_mlflow_library.modeling.classifiers import RandomForestClassifier
from skyulf_mlflow_library.pipeline import Pipeline, make_pipeline
from skyulf_mlflow_library.preprocessing import SimpleImputer, StandardScaler

RNG = np.random.default_rng(42)


def _prepare_raw_dataset(destination: Path) -> Tuple[pd.DataFrame, Path]:
    """Create and persist an enriched wine dataset for the demo."""
    wine = load_wine(as_frame=True)
    frame = wine.frame.copy()

    # Rename a couple of awkward column names for readability.
    frame = frame.rename(
        columns={
            "od280/od315_of_diluted_wines": "od280_od315",
            "nonflavanoid_phenols": "nonflavanoid",
        }
    )

    # Map classes to human readable labels and add categorical context.
    quality_labels = np.array(["table", "reserve", "collector"])
    frame["quality_target"] = wine.target
    frame["quality_label"] = quality_labels[wine.target]
    frame["region"] = np.take(
        np.array(["north_valley", "coastal", "highlands"]),
        RNG.integers(0, 3, size=len(frame)),
    )

    # Add a harvest_date column to highlight datetime feature extraction.
    base_date = np.datetime64("2014-01-01")
    frame["harvest_date"] = pd.to_datetime(
        base_date + RNG.integers(0, 365 * 4, size=len(frame)).astype("timedelta64[D]")
    )

    # Introduce a few missing values for the imputer to resolve.
    for col in ["malic_acid", "magnesium", "color_intensity"]:
        missing_idx = RNG.choice(frame.index, size=12, replace=False)
        frame.loc[missing_idx, col] = np.nan

    # Persist the dataset so DataLoader can demonstrate file-based ingestion.
    destination.parent.mkdir(parents=True, exist_ok=True)
    saver = DataSaver(destination, index=False)
    saver.save(frame)
    return frame, destination


def _build_feature_pipeline(numeric_columns: Tuple[str, ...]) -> Pipeline:
    """Attach the primary transformers into a reusable pipeline."""
    feature_math_ops = [
        {
            "type": "ratio",
            "numerator": ["magnesium"],
            "denominator": ["malic_acid"],
            "output": "magnesium_to_acid",
            "round": 4,
        },
        {
            "type": "stat",
            "method": "mean",
            "columns": ["total_phenols", "flavanoids"],
            "output": "phenols_mean",
        },
        {
            "type": "arithmetic",
            "method": "add",
            "columns": ["color_intensity", "od280_od315"],
            "output": "intensity_plus_od",
        },
        {
            "type": "datetime",
            "columns": ["harvest_date"],
            "features": ["year", "month", "is_weekend"],
            "prefix": "harvest_",
        },
    ]

    return make_pipeline(
        SimpleImputer(strategy="median", columns=list(numeric_columns)),
        FeatureMath(feature_math_ops),
        SmartBinning(
            strategy="equal_frequency",
            columns=["alcohol"],
            n_bins=4,
            labels="ordinal",
            suffix="_band",
        ),
        OneHotEncoder(columns=["region"]),
        FeatureSelector(
            method="select_k_best",
            problem_type="classification",
            k=14,
            drop_unselected=True,
        ),
        StandardScaler(columns=None),
    )


def _train_and_evaluate(
    pipeline: object,
    df: pd.DataFrame,
    target_column: str,
) -> Dict[str, float]:
    """Fit the pipeline and classifier, returning evaluation metrics."""
    features = df.drop(columns=[target_column, "quality_label"])
    target = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=7, stratify=target
    )

    X_train_prepared = pipeline.fit_transform(X_train, y_train)
    X_test_prepared = pipeline.transform(X_test)

    # Drop the raw datetime column after feature extraction to keep the model inputs numeric.
    if "harvest_date" in X_train_prepared.columns:
        X_train_prepared = X_train_prepared.drop(columns=["harvest_date"])
    if "harvest_date" in X_test_prepared.columns:
        X_test_prepared = X_test_prepared.drop(columns=["harvest_date"])

    classifier = RandomForestClassifier(random_state=21, n_estimators=250)
    classifier.fit(X_train_prepared, y_train)
    predictions = classifier.predict(X_test_prepared)
    probabilities = classifier.predict_proba(X_test_prepared)

    metrics_calc = MetricsCalculator("classification")
    metrics = metrics_calc.calculate(y_test, predictions, y_prob=probabilities)

    # Persist the trained model and metrics for completeness.
    registry = ModelRegistry(Path(__file__).resolve().parent / "model_registry")
    registry.save_model(
        model=classifier,
        name="wine_quality_classifier",
        problem_type="classification",
        metrics=metrics,
        tags=["example", "demo"],
    )

    metrics_path = Path(__file__).resolve().parent / "data" / "wine_metrics.csv"
    saver = DataSaver(metrics_path, index=False)
    saver.save(pd.DataFrame([metrics]))

    print("Top-level evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key:>12}: {value:.4f}")

    print("\nRegistered models:")
    print(registry.list_models())

    return metrics


def main() -> None:
    """Stage data, run the pipeline, and report results."""
    base_path = Path(__file__).resolve().parent
    data_path = base_path / "data" / "wine_dataset.csv"

    raw_frame, dataset_path = _prepare_raw_dataset(data_path)

    loader = DataLoader(dataset_path)
    loaded_frame = loader.load()
    loaded_frame["harvest_date"] = pd.to_datetime(loaded_frame["harvest_date"])

    numeric_cols = tuple(
        col
        for col in loaded_frame.select_dtypes(include=[np.number]).columns
        if col not in {"quality_target"}
    )

    pipeline = _build_feature_pipeline(numeric_cols)
    metrics = _train_and_evaluate(pipeline, loaded_frame, target_column="quality_target")

    print("\nWorkflow complete. Metrics saved alongside the dataset.")
    print(metrics)


if __name__ == "__main__":
    main()
