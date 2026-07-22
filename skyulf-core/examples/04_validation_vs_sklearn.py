"""Validate equivalent Skyulf and raw scikit-learn pipelines on the same split.

Run from the repository root:
    python skyulf-core/examples/04_validation_vs_sklearn.py

This comparison is deliberately split *before* either implementation learns
imputation, categorical vocabulary, or scaling parameters.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skyulf import SkyulfPipeline
from skyulf.data.dataset import SplitDataset


def dense_one_hot_encoder() -> OneHotEncoder:
    """Build an sklearn encoder compatible with installed sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def main() -> None:
    """Fit both implementations and assert that their predictions match."""
    raw = load_breast_cancer(as_frame=True)
    frame = raw.frame.rename(columns={"target": "label"}).copy()
    frame["radius_band"] = pd.cut(
        frame["mean radius"], bins=[0, 12, 15, 100], labels=["small", "medium", "large"]
    ).astype("object")
    frame.loc[np.random.default_rng(42).choice(frame.index, 25, replace=False), "mean texture"] = (
        np.nan
    )

    target = "label"
    categorical = ["radius_band"]
    numeric = [column for column in frame.columns if column not in [target, *categorical]]
    X_train, X_test, y_train, y_test = train_test_split(
        frame[numeric + categorical],
        frame[target],
        test_size=0.25,
        random_state=42,
        stratify=frame[target],
    )

    sklearn_pipeline = Pipeline(
        [
            (
                "preprocess",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())]),
                            numeric,
                        ),
                        (
                            "categorical",
                            Pipeline(
                                [
                                    ("impute", SimpleImputer(strategy="most_frequent")),
                                    ("encode", dense_one_hot_encoder()),
                                ]
                            ),
                            categorical,
                        ),
                    ]
                ),
            ),
            ("model", LogisticRegression(max_iter=1_000, random_state=42)),
        ]
    )
    sklearn_pipeline.fit(X_train, y_train)
    sklearn_predictions = sklearn_pipeline.predict(X_test)

    dataset = SplitDataset(
        train=X_train.assign(label=y_train).reset_index(drop=True),
        test=X_test.assign(label=y_test).reset_index(drop=True),
    )
    skyulf_pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "impute_numeric",
                    "transformer": "SimpleImputer",
                    "params": {"columns": numeric, "strategy": "mean"},
                },
                {
                    "name": "impute_categorical",
                    "transformer": "SimpleImputer",
                    "params": {"columns": categorical, "strategy": "most_frequent"},
                },
                {
                    "name": "encode",
                    "transformer": "OneHotEncoder",
                    "params": {
                        "columns": categorical,
                        "drop_original": True,
                        "handle_unknown": "ignore",
                    },
                },
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"columns": numeric},
                },
            ],
            "modeling": {
                "type": "logistic_regression",
                "params": {"max_iter": 1_000, "random_state": 42},
            },
        }
    )
    skyulf_pipeline.fit(dataset, target_column=target)
    skyulf_predictions = skyulf_pipeline.predict(X_test)

    sklearn_accuracy = accuracy_score(y_test, sklearn_predictions)
    skyulf_accuracy = accuracy_score(y_test, skyulf_predictions)
    np.testing.assert_array_equal(sklearn_predictions, skyulf_predictions.to_numpy())
    print(f"sklearn accuracy: {sklearn_accuracy:.4f}")
    print(f"Skyulf accuracy:  {skyulf_accuracy:.4f}")
    print("Predictions match exactly.")


if __name__ == "__main__":
    main()
