"""A reproducible, offline Spaceship Titanic competition-style workflow.

Run from the repository root:
    python skyulf-core/examples/08_kaggle_spaceship_titanic.py

The bundled Kaggle train/test files let this run without credentials.  The
competition test labels are unavailable here, so this script never claims a
leaderboard score or submits anything.  It uses a stratified CV search on an
outer training partition, then reports the score on an outer held-out
partition that was never used for tuning or preprocessing fitting.

Leaderboard context (accessed 2026-07-22): the Kaggle competition page is the
primary source for the task and its public leaderboard:
https://www.kaggle.com/competitions/spaceship-titanic.  Live leaderboard
research was unavailable in this execution environment.  Public notebooks and
post-competition discussions commonly report clean local CV around 0.80--0.82
for strong tabular models; high public-board positions are distorted by the
small public test subset and leakage/overfitting attempts.  Treat ~0.80+ clean
CV as a defensible goal, not evidence of a literal top-50 placement.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from skyulf import SkyulfPipeline
from skyulf.data.dataset import SplitDataset

DATA_DIR = Path(__file__).parent / "data" / "spaceship_titanic"
SPEND_COLUMNS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
NUMERIC_COLUMNS = ["Age", "CabinNum", "GroupSize", "TotalSpend", "AnySpend", *SPEND_COLUMNS]
CATEGORICAL_COLUMNS = [
    "HomePlanet",
    "CryoSleep",
    "CabinDeck",
    "CabinSide",
    "Destination",
    "VIP",
    "AgeBand",
]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic, partition-local features without fitting statistics."""
    result = frame.copy()
    passenger_parts = result["PassengerId"].str.split("_", expand=True)
    result["PassengerGroup"] = passenger_parts[0]
    result["GroupMember"] = pd.to_numeric(passenger_parts[1], errors="coerce")

    cabin_parts = result["Cabin"].fillna("").str.split("/", expand=True)
    result["CabinDeck"] = cabin_parts[0].replace("", float("nan"))
    result["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    result["CabinSide"] = cabin_parts[2].replace("", float("nan"))

    # Group size is intentionally computed inside each partition: no held-out
    # row contributes to a train feature, even though it uses no target labels.
    result["GroupSize"] = result.groupby("PassengerGroup")["PassengerGroup"].transform("size")
    spend = result[SPEND_COLUMNS]
    result["TotalSpend"] = spend.fillna(0).sum(axis=1)
    result["AnySpend"] = (result["TotalSpend"] > 0).astype(int)
    result["AgeBand"] = pd.cut(
        result["Age"], bins=[0, 12, 18, 30, 45, 65, 120], labels=False, include_lowest=True
    )

    keep = [*NUMERIC_COLUMNS, *CATEGORICAL_COLUMNS]
    if "Transported" in result:
        keep.append("Transported")
    return result[keep]


def pipeline_config() -> dict:
    """Build an outer-train-only preprocessing pipeline plus randomized CV tuning."""
    return {
        "preprocessing": [
            {
                "name": "impute_numeric",
                "transformer": "SimpleImputer",
                "params": {"columns": NUMERIC_COLUMNS, "strategy": "median"},
            },
            {
                "name": "impute_categorical",
                "transformer": "SimpleImputer",
                "params": {"columns": CATEGORICAL_COLUMNS, "strategy": "most_frequent"},
            },
            {
                "name": "encode_categories",
                "transformer": "OneHotEncoder",
                "params": {
                    "columns": CATEGORICAL_COLUMNS,
                    "drop_original": True,
                    "handle_unknown": "ignore",
                },
            },
        ],
        "modeling": {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "random_forest_classifier"},
            "strategy": "random",
            "metric": "accuracy",
            "n_trials": 4,
            "search_space": {
                "n_estimators": [150, 250],
                "max_depth": [8, 14, None],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
            },
            "cv_enabled": True,
            "cv_type": "stratified_k_fold",
            "cv_folds": 3,
            "cv_random_state": 42,
            "random_state": 42,
            "n_jobs": 1,
        },
    }


def main() -> None:
    """Tune on outer-train rows, score a never-seen holdout, and predict Kaggle test rows."""
    warnings.filterwarnings("ignore", message="Downcasting object dtype arrays on .fillna")
    raw_train = pd.read_csv(DATA_DIR / "train.csv")
    raw_test = pd.read_csv(DATA_DIR / "test.csv")
    outer_train_raw, outer_test_raw = train_test_split(
        raw_train, test_size=0.2, random_state=42, stratify=raw_train["Transported"]
    )
    outer_train = make_features(outer_train_raw)
    outer_test = make_features(outer_test_raw)

    pipeline = SkyulfPipeline(pipeline_config())
    pipeline.fit(
        SplitDataset(
            train=outer_train.reset_index(drop=True),
            test=outer_test.reset_index(drop=True),
        ),
        target_column="Transported",
    )
    holdout_predictions = pipeline.predict(outer_test.drop(columns="Transported"))
    holdout_accuracy = accuracy_score(outer_test["Transported"], holdout_predictions)

    fitted_model, tuning_result = pipeline.model_estimator.model
    assert fitted_model is not None
    kaggle_features = make_features(raw_test)
    kaggle_predictions = pipeline.predict(kaggle_features)
    submission_preview = pd.DataFrame(
        {"PassengerId": raw_test["PassengerId"], "Transported": kaggle_predictions.astype(bool)}
    )

    print(f"Rows: train={len(raw_train)}, Kaggle holdout={len(raw_test)}")
    print(f"Best 3-fold CV accuracy on outer train: {tuning_result.best_score:.4f}")
    print(f"Outer held-out accuracy: {holdout_accuracy:.4f}")
    print(f"Best parameters: {tuning_result.best_params}")
    print("Submission preview:")
    print(submission_preview.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
