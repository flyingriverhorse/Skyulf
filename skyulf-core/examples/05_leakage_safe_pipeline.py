"""Prove that fitted preprocessing learns only from the training partition.

Run from the repository root:
    python skyulf-core/examples/05_leakage_safe_pipeline.py

The platform's DAG validator hard-blocks a data-dependent preprocessing node
upstream of ``TrainTestSplitter`` or ``Split``.  The data-dependent list is:
imputers; Standard/MinMax/Robust/MaxAbs scalers; learned encoders; IQR,
ZScore, Winsorize, and EllipticEnvelope; statistical feature selection;
General/EqualWidth/EqualFrequency/KBins binning; PowerTransformer; and
Count/TF-IDF vectorizers.  Safe rule-based steps include casting, cleaning,
manual bins, date/geo/lag/rolling/math/interactions, tokenizer, hashing
vectorizer, and sentence embedding.

Skyulf Core is standalone, so this example demonstrates the same contract with
an explicit ``SplitDataset``: poison held-out rows and verify fitted artifacts
remain unchanged.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skyulf import SkyulfPipeline
from skyulf.data.dataset import SplitDataset


def make_dataset() -> pd.DataFrame:
    """Create data whose train statistics differ observably from poisoned test data."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "age": rng.normal(40, 12, 200),
            "fare": rng.lognormal(3.5, 0.5, 200),
            "sex": rng.choice(["female", "male"], 200),
            "survived": rng.integers(0, 2, 200),
        }
    ).assign(age=lambda frame: frame["age"].mask(frame.index < 10))


def dataset_from_split(train: pd.DataFrame, test: pd.DataFrame) -> SplitDataset:
    """Construct the exact input object expected by a train-only fit."""
    return SplitDataset(train=train.reset_index(drop=True), test=test.reset_index(drop=True))


def config() -> dict:
    """Return the safe ordering: split first, then every learned transform."""
    return {
        "preprocessing": [
            {
                "name": "impute_age",
                "transformer": "SimpleImputer",
                "params": {"columns": ["age"], "strategy": "mean"},
            },
            {
                "name": "encode_sex",
                "transformer": "OneHotEncoder",
                "params": {"columns": ["sex"], "drop_original": True, "handle_unknown": "ignore"},
            },
            {
                "name": "scale_fare",
                "transformer": "StandardScaler",
                "params": {"columns": ["fare"]},
            },
        ],
        "modeling": {
            "type": "logistic_regression",
            "params": {"max_iter": 500, "random_state": 42},
        },
    }


def main() -> None:
    """Fit clean and poisoned held-out data, then compare learned artifacts."""
    frame = make_dataset()
    train, test = train_test_split(
        frame, test_size=0.3, random_state=42, stratify=frame["survived"]
    )

    clean = SkyulfPipeline(config())
    clean.fit(dataset_from_split(train, test), target_column="survived")

    poisoned_test = test.copy()
    poisoned_test["age"] = 10_000.0
    poisoned_test["fare"] = 1_000_000.0
    poisoned_test["sex"] = "unseen_category"
    poisoned = SkyulfPipeline(config())
    poisoned.fit(dataset_from_split(train, poisoned_test), target_column="survived")

    clean_imputer = clean.feature_engineer.fitted_steps[0]["artifact"]["fill_values"]["age"]
    poisoned_imputer = poisoned.feature_engineer.fitted_steps[0]["artifact"]["fill_values"]["age"]
    clean_scaler = clean.feature_engineer.fitted_steps[2]["artifact"]
    poisoned_scaler = poisoned.feature_engineer.fitted_steps[2]["artifact"]

    np.testing.assert_allclose(clean_imputer, train["age"].mean())
    np.testing.assert_allclose(clean_imputer, poisoned_imputer)
    np.testing.assert_allclose(clean_scaler["mean"], poisoned_scaler["mean"])
    np.testing.assert_allclose(clean_scaler["scale"], poisoned_scaler["scale"])

    print(f"Train-only imputer mean: {clean_imputer:.4f}")
    print(f"Full-frame mean (would leak): {frame['age'].mean():.4f}")
    print("Poisoned held-out values did not alter imputation or scaling artifacts.")
    print("Leakage-safe pipeline proof completed.")


if __name__ == "__main__":
    main()
