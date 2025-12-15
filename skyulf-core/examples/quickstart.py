"""
Skyulf SDK Quickstart Example.

This script demonstrates how to:
1. Define a pipeline configuration.
2. Train a model using SkyulfPipeline.
3. Save the trained pipeline.
4. Load it back and make predictions.
"""

import numpy as np
import pandas as pd
from skyulf import SkyulfPipeline


def create_dummy_data():
    """Create a dummy dataset for demonstration."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n),
            "income": np.random.normal(50000, 15000, n),
            "city": np.random.choice(["New York", "London", "Paris"], n),
            "is_customer": np.random.choice([0, 1], n),
        }
    )
    # Add some missing values
    df.loc[0:10, "income"] = np.nan
    return df


def main():
    print("1. Creating dummy data...")
    data = create_dummy_data()
    print(f"   Data shape: {data.shape}")

    # 2. Define Pipeline Configuration
    # This config defines the steps for preprocessing and the model to use.
    config = {
        "preprocessing": [
            # Step 0: Split Data
            {
                "name": "split_data",
                "transformer": "TrainTestSplitter",
                "params": {"test_size": 0.2, "target_column": "is_customer"},
            },
            # Step 1: Impute missing income with mean
            {
                "name": "impute_income",
                "transformer": "SimpleImputer",
                "params": {"columns": ["income"], "strategy": "mean"},
            },
            # Step 2: One-Hot Encode 'city'
            {
                "name": "encode_city",
                "transformer": "OneHotEncoder",
                "params": {"columns": ["city"]},
            },
            # Step 3: Scale numerical features
            {
                "name": "scale_features",
                "transformer": "StandardScaler",
                "params": {"columns": ["age", "income"]},
            },
        ],
        "modeling": {
            "type": "random_forest_classifier",
            "params": {"n_estimators": 50, "max_depth": 5},
        },
    }

    print("\n2. Initializing Pipeline...")
    pipeline = SkyulfPipeline(config)

    print("\n3. Training Pipeline...")
    # fit() runs preprocessing and trains the model
    metrics = pipeline.fit(data, target_column="is_customer")

    print("   Training Complete!")
    print("   Metrics:", metrics.keys())
    if "modeling" in metrics:
        # Access Pydantic model attributes
        report = metrics["modeling"]
        if "splits" in report and "test" in report["splits"]:
            acc = report["splits"]["test"].metrics.get("accuracy")
            print(f"   Test Accuracy: {acc:.4f}")

    # 4. Save Pipeline
    print("\n4. Saving Pipeline to 'my_model.pkl'...")
    pipeline.save("my_model.pkl")

    # 5. Load Pipeline
    print("\n5. Loading Pipeline...")
    loaded_pipeline = SkyulfPipeline.load("my_model.pkl")

    # 6. Make Predictions
    print("\n6. Making Predictions on new data...")
    new_data = pd.DataFrame(
        {
            "age": [25, 40],
            "income": [60000, np.nan],  # Missing value will be handled by imputer
            "city": ["London", "Paris"],
        }
    )

    predictions = loaded_pipeline.predict(new_data)
    print("   Predictions:", predictions.tolist())

    # Cleanup
    import os

    if os.path.exists("my_model.pkl"):
        os.remove("my_model.pkl")
        print("\n   (Cleaned up 'my_model.pkl')")


if __name__ == "__main__":
    main()
