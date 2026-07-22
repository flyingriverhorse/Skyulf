"""Train, persist, reload, and predict with a leakage-safe Polars pipeline.

Run from the repository root:
    python skyulf-core/examples/01_quickstart.py

The only tabular dependency used directly here is Polars.  Skyulf's
``SklearnBridge`` sends a ``pl.DataFrame`` straight to NumPy for sklearn; it
does not require users to import pandas.  ``to_arrow()`` remains available for
Arrow-oriented integrations.
"""

from pathlib import Path

import numpy as np
import polars as pl

from skyulf import SkyulfPipeline
from skyulf.engines.polars_engine import PolarsEngine
from skyulf.engines.sklearn_bridge import SklearnBridge


def make_customers(rows: int = 240) -> pl.DataFrame:
    """Create a deterministic binary-classification dataset with one missing field."""
    rng = np.random.default_rng(42)
    age = rng.integers(18, 75, rows).astype(float)
    income = rng.normal(55_000, 12_000, rows)
    city = rng.choice(["London", "Nairobi", "Singapore"], rows)
    target = ((age > 42) | (city == "Singapore")).astype(int)
    return pl.DataFrame(
        {"age": age, "income": income, "city": city, "purchased": target}
    ).with_columns(
        pl.when(pl.int_range(pl.len()) < 10).then(None).otherwise(pl.col("income")).alias("income")
    )


def main() -> None:
    """Fit the pipeline, round-trip its artifact, and make two predictions."""
    data = make_customers()
    features = data.drop("purchased")

    # The train/test split is the leakage boundary: every fitted transform follows it.
    config = {
        "preprocessing": [
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {
                    "target_column": "purchased",
                    "test_size": 0.25,
                    "random_state": 42,
                    "stratify": True,
                },
            },
            {
                "name": "impute_income",
                "transformer": "SimpleImputer",
                "params": {"columns": ["income"], "strategy": "median"},
            },
            {
                "name": "encode_city",
                "transformer": "OneHotEncoder",
                "params": {
                    "columns": ["city"],
                    "drop_original": True,
                    "handle_unknown": "ignore",
                },
            },
            {
                "name": "scale_numeric",
                "transformer": "StandardScaler",
                "params": {"columns": ["age", "income"]},
            },
        ],
        "modeling": {
            "type": "logistic_regression",
            "params": {"max_iter": 500, "random_state": 42},
        },
    }

    pipeline = SkyulfPipeline(config)
    metrics = pipeline.fit(data, target_column="purchased")
    assert pipeline.is_fitted()

    artifact = Path(__file__).with_name("quickstart_model.pkl")
    try:
        pipeline.save(str(artifact))
        restored = SkyulfPipeline.load(str(artifact))
        predictions = restored.predict(
            pl.DataFrame(
                {
                    "age": [25.0, 61.0],
                    "income": [48_000.0, None],
                    "city": ["London", "Singapore"],
                }
            )
        )
    finally:
        artifact.unlink(missing_ok=True)

    # This is the no-hidden-pandas boundary: Polars -> NumPy for sklearn.
    X_numpy, _ = SklearnBridge.to_sklearn(features)
    arrow_table = PolarsEngine.wrap(features).to_arrow()
    assert X_numpy.shape == features.shape
    assert arrow_table.num_rows == features.height

    print(pipeline.describe())
    print(f"Fitted: {pipeline.is_fitted()} | fingerprint: {pipeline.fingerprint()[:12]}...")
    print(f"Model metric payload: {metrics['modeling']}")
    print(f"Polars -> NumPy shape: {X_numpy.shape}; Polars -> Arrow rows: {arrow_table.num_rows}")
    print(f"Predictions after save/load: {predictions.tolist()}")


if __name__ == "__main__":
    main()
