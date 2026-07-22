"""Runnable preprocessing recipes and the ordering rules behind them.

Run from the repository root:
    python skyulf-core/examples/02_preprocessing_recipes.py

Data-dependent steps (imputation, learned encoders/scalers, IQR and feature
selection, and learned bins) must receive a ``SplitDataset`` or follow a
``TrainTestSplitter``.  Deterministic cleaning and feature construction can
run before that boundary.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.pipeline import FeatureEngineer


def split_frame(frame: pd.DataFrame, target: str) -> SplitDataset:
    """Create a reproducible train/test SplitDataset without fitting any transform."""
    train, test = train_test_split(frame, test_size=0.25, random_state=42, stratify=frame[target])
    return SplitDataset(train=train.reset_index(drop=True), test=test.reset_index(drop=True))


def run_learned_recipe() -> None:
    """Fit learned tabular preprocessing only on training rows."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "income": rng.normal(60_000, 15_000, 160),
            "age": rng.integers(18, 80, 160).astype(float),
            "city": rng.choice(["A", "B", "C"], 160),
            "constant": 1.0,
            "target": rng.integers(0, 2, 160),
        }
    )
    frame.loc[:7, "income"] = np.nan
    frame.loc[8, "age"] = 500.0

    steps = [
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"columns": ["income", "age"], "strategy": "median"},
        },
        {
            "name": "encode",
            "transformer": "OneHotEncoder",
            "params": {"columns": ["city"], "drop_original": True, "handle_unknown": "ignore"},
        },
        {"name": "clip_outliers", "transformer": "IQR", "params": {"columns": ["age"]}},
        {
            "name": "scale",
            "transformer": "StandardScaler",
            "params": {"columns": ["income", "age"]},
        },
        {"name": "drop_constant", "transformer": "VarianceThreshold", "params": {"threshold": 0.0}},
        {
            "name": "quantile_bins",
            "transformer": "GeneralBinning",
            "params": {"columns": ["income"], "n_bins": 4, "strategy": "quantile"},
        },
    ]
    transformed, _ = FeatureEngineer(steps).fit_transform(split_frame(frame, "target"))
    print(
        "Learned tabular recipe:",
        f"train={transformed.train.shape}, test={transformed.test.shape}",
    )


def run_deterministic_recipe() -> None:
    """Demonstrate cleaning, dates, geo, rolling, lags, and interactions."""
    frame = pd.DataFrame(
        {
            "when": pd.date_range("2025-01-01", periods=8, freq="D"),
            "group": ["a"] * 4 + ["b"] * 4,
            "value": [1, 3, 2, 4, 2, 5, 4, 6],
            "amount": ["10.0", "12.5", "9.5", "11.0", "8.0", "13.0", "15.0", "7.0"],
            "note": ["  HELLO!!  "] * 8,
            "lat1": [51.5] * 8,
            "lon1": [-0.1] * 8,
            "lat2": [48.86] * 8,
            "lon2": [2.35] * 8,
        }
    )
    steps = [
        {
            "name": "cast_amount",
            "transformer": "Casting",
            "params": {"column_types": {"amount": "float"}, "coerce_on_error": True},
        },
        {
            "name": "clean_note",
            "transformer": "TextCleaning",
            "params": {
                "columns": ["note"],
                "operations": [
                    {"operation": "trim", "mode": "both"},
                    {"operation": "case", "mode": "lower"},
                    {"operation": "remove_special", "mode": "keep_alphanumeric"},
                ],
            },
        },
        {
            "name": "date_parts",
            "transformer": "DateFeatures",
            "params": {"columns": ["when"], "features": ["year", "month", "weekday"]},
        },
        {
            "name": "distance",
            "transformer": "GeoDistance",
            "params": {
                "lat1_col": "lat1",
                "lon1_col": "lon1",
                "lat2_col": "lat2",
                "lon2_col": "lon2",
                "method": "haversine",
                "unit": "km",
                "output_column": "london_to_paris_km",
            },
        },
        {
            "name": "ratio",
            "transformer": "FeatureMath",
            "params": {
                "operations": [
                    {
                        "operation_type": "ratio",
                        "input_columns": ["value"],
                        "secondary_columns": ["amount"],
                        "output_column": "value_per_amount",
                    }
                ]
            },
        },
        {
            "name": "interactions",
            "transformer": "FeatureInteraction",
            "params": {"columns": ["value", "amount"], "degree": 2, "interaction_only": True},
        },
        {
            "name": "rolling",
            "transformer": "RollingAggregate",
            "params": {
                "columns": ["value"],
                "window": 2,
                "aggregations": ["mean"],
                "group_by": ["group"],
                "sort_by": "when",
            },
        },
        {
            "name": "lag",
            "transformer": "LagFeatures",
            "params": {
                "columns": ["value"],
                "lags": [1],
                "group_by": ["group"],
                "sort_by": "when",
            },
        },
        {
            "name": "fixed_bins",
            "transformer": "CustomBinning",
            "params": {"columns": ["amount"], "bins": [0, 10, 20], "drop_original": False},
        },
    ]
    transformed, _ = FeatureEngineer(steps).fit_transform(frame)
    print("Deterministic recipe columns:", sorted(transformed.columns))


def main() -> None:
    """Run the learned and deterministic preprocessing recipes."""
    run_learned_recipe()
    run_deterministic_recipe()
    print("Preprocessing recipes completed.")


if __name__ == "__main__":
    main()
