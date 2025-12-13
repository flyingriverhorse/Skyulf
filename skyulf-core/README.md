# Skyulf Core

**Skyulf Core** (`skyulf-core`) is the standalone machine learning library that powers the Skyulf MLOps platform. It provides a robust, type-safe, and modular set of tools for:

- **Data Preprocessing**: A comprehensive suite of transformers for cleaning, scaling, encoding, and feature engineering.
- **Modeling**: Unified interfaces for classification and regression models, wrapping Scikit-Learn and other libraries.
- **Pipeline Management**: Tools to build, serialize, and execute complex ML pipelines.
- **Tuning**: Advanced hyperparameter tuning capabilities with support for Grid Search, Random Search, and Optuna.
- **Evaluation**: Standardized metrics and evaluation schemas for model performance tracking.

## Installation

```bash
pip install skyulf-core
```

## Usage Example

```python
import pandas as pd

from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.pipeline import SkyulfPipeline

# --- 1) Preprocessing only ---
df = pd.DataFrame({
    "x1": [1, 2, 2, None],
    "x2": [10, 20, 20, 40],
    "target": [0, 1, 1, 0],
})

steps = [
    {"name": "dedupe", "transformer": "Deduplicate", "params": {}},
    {"name": "impute", "transformer": "SimpleImputer", "params": {"strategy": "mean"}},
]

fe = FeatureEngineer(steps)
df_transformed, fe_metrics = fe.fit_transform(df)

# --- 2) End-to-end pipeline (preprocessing + modeling) ---
config = {
    "preprocessing": steps,
    "modeling": {"type": "random_forest_classifier", "node_id": "rf"},
}

pipe = SkyulfPipeline(config)
metrics = pipe.fit(df_transformed, target_column="target")
preds = pipe.predict(df_transformed.drop(columns=["target"]))

print("metrics keys:", metrics.keys())
print("preds shape:", preds.shape)
```

## Features

- **Type-Safe**: Built with modern Python type hints and Pydantic models.
- **Modular**: Use only the components you need.
- **Serializable**: All components are designed to be easily serialized for storage and deployment.
- **Extensible**: Easy to extend with your own custom transformers and models.

## License

This project is licensed under the terms of the Apache 2.0 license.
