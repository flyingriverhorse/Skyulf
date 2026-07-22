# Skyulf Core

**Skyulf Core** (`skyulf-core`) is a standalone, installable Python ML library
for teams who want sklearn's dependable estimators with a cohesive,
Polars-friendly pipeline layer around them. It unifies preprocessing,
classification, regression, clustering, text models, hyperparameter tuning,
evaluation, and optional SHAP explanations behind one composable API.

Use it when you want to move from a notebook to a repeatable model artifact
without assembling a different interface for every transformer and estimator.
Skyulf builds on and is validated against scikit-learn rather than replacing
it: sklearn remains the modeling foundation while Skyulf provides pipeline
configuration, artifacts, metrics, and safe execution conventions.

<!-- Quick badges + links -->
[![Docs](https://img.shields.io/website?down_color=red&down_message=offline&up_message=online&url=https://flyingriverhorse.github.io/Skyulf)](https://flyingriverhorse.github.io/Skyulf) [![PyPI](https://img.shields.io/pypi/v/skyulf-core.svg)](https://pypi.org/project/skyulf-core) [![License](https://img.shields.io/github/license/flyingriverhorse/Skyulf)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/skyulf-core.svg)](https://pypi.org/project/skyulf-core) [![issues](https://img.shields.io/github/issues/flyingriverhorse/Skyulf.svg)](https://github.com/flyingriverhorse/Skyulf/issues) [![contributors](https://img.shields.io/github/contributors/flyingriverhorse/Skyulf.svg)](https://github.com/flyingriverhorse/Skyulf/graphs/contributors)

**Website & Documentation**

- Project site / docs: https://www.skyulf.com
- Repository: https://github.com/flyingriverhorse/Skyulf
- PyPI package: https://pypi.org/project/skyulf-core

## Installation

Skyulf Core currently packages for Python 3.12+.

```bash
pip install skyulf-core

# EDA-focused install (core EDA + optional advanced EDA + visualization)
pip install skyulf-core[eda,viz]

# For visualization support (Rich dashboard + Matplotlib plots)
pip install skyulf-core[viz]

# For advanced EDA add-ons (sentiment + causal discovery)
pip install skyulf-core[eda]

# For hyperparameter tuning engines
pip install skyulf-core[tuning]

# For SHAP explainability
pip install skyulf-core[explainability]

# For dense SentenceEmbedder support
pip install skyulf-core[nlp]

# For imbalance-aware preprocessing (e.g., SMOTE)
pip install skyulf-core[preprocessing-imbalanced]

# For XGBoost modeling nodes
pip install skyulf-core[modeling-xgboost]

# For LightGBM modeling nodes
pip install skyulf-core[modeling-lightgbm]

# For geospatial feature engineering (H3 indexing, spatial stats)
pip install skyulf-core[geo]

# For text sentiment features
pip install skyulf-core[text]
```

## Quick start

```python
import polars as pl
from skyulf import SkyulfPipeline

customers = pl.read_csv("customers.csv")  # contains a `purchased` target
pipeline = SkyulfPipeline(
    {
        "preprocessing": [
            {
                "name": "split",
                "transformer": "TrainTestSplitter",
                "params": {"target_column": "purchased", "test_size": 0.2, "random_state": 42},
            },
            {
                "name": "impute_income",
                "transformer": "SimpleImputer",
                "params": {"columns": ["income"], "strategy": "median"},
            },
            {
                "name": "encode_city",
                "transformer": "OneHotEncoder",
                "params": {"columns": ["city"], "drop_original": True, "handle_unknown": "ignore"},
            },
        ],
        "modeling": {
            "type": "logistic_regression",
            "params": {"max_iter": 500, "random_state": 42},
        },
    }
)

pipeline.fit(customers, target_column="purchased")
pipeline.save("customer_model.pkl")
predictions = SkyulfPipeline.load("customer_model.pkl").predict(
    pl.read_csv("new_customers.csv")
)
```

See [`examples/01_quickstart.py`](examples/01_quickstart.py) for a complete,
executed save/load round trip.

## Data leakage safety

**Split before any data-dependent preprocessing.** Fitting an imputer, scaler,
learned encoder, feature selector, learned binning step, outlier detector, or
Count/TF-IDF vectorizer on rows that include validation/test data contaminates
evaluation. Put `TrainTestSplitter` first in a pipeline, or explicitly create
a `SplitDataset` first and then fit `FeatureEngineer`/`SkyulfPipeline`.

The larger Skyulf platform validates DAGs and hard-blocks data-dependent nodes
upstream of a train/test split. Skyulf Core is also useful on its own, so the
same ordering is an essential API-level contract. Read and run
[`examples/05_leakage_safe_pipeline.py`](examples/05_leakage_safe_pipeline.py)
for an executable poisoned-holdout proof.

## Polars-native, no hidden pandas

Users can pass `polars.DataFrame` inputs directly. `PolarsEngine` and
`SklearnBridge` convert Polars straight to NumPy at the sklearn boundary—there
is no user-facing pandas round trip. Arrow integrations use the explicit
`to_arrow()` path (and therefore `pyarrow`). Run
[`examples/01_quickstart.py`](examples/01_quickstart.py) and
[`examples/07_eda_and_polars.py`](examples/07_eda_and_polars.py) to see both
paths without importing pandas in user code.

## Examples

All scripts run from the repository root with
`python skyulf-core/examples/<file>.py`.

1. [`01_quickstart.py`](examples/01_quickstart.py) — Polars-native pipeline configuration, fit, save, load, predict, NumPy bridge, and Arrow.
2. [`02_preprocessing_recipes.py`](examples/02_preprocessing_recipes.py) — Imputation, encoding, scaling, outliers, selection, binning, casting, cleaning, date/geo/rolling/lag/math/interaction features.
3. [`03_modeling_evaluation.py`](examples/03_modeling_evaluation.py) — Classification, regression, voting/stacking ensembles, four clustering models, Grid/Random/Optuna tuning, metrics, and SHAP.
4. [`04_validation_vs_sklearn.py`](examples/04_validation_vs_sklearn.py) — Exact prediction-equivalence check against an equivalent raw sklearn pipeline.
5. [`05_leakage_safe_pipeline.py`](examples/05_leakage_safe_pipeline.py) — Split-first guidance and train-only fitted-artifact proof under poisoned holdout data.
6. [`06_text_nlp_vectorization.py`](examples/06_text_nlp_vectorization.py) — Tokenizer, Count/TF-IDF/Hashing vectorizers, Naive Bayes, and SentenceEmbedder guidance.
7. [`07_eda_and_polars.py`](examples/07_eda_and_polars.py) — Automated `EDAAnalyzer` profiling and the explicit Polars-to-Arrow path.
8. [`08_kaggle_spaceship_titanic.py`](examples/08_kaggle_spaceship_titanic.py) — Offline Kaggle-style, leakage-safe Spaceship Titanic feature engineering, CV tuning, and honest held-out scoring.

## Automated EDA

Skyulf Core includes automated exploratory data analysis for Polars frames.

```python
import polars as pl
from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.visualizer import EDAVisualizer

df = pl.read_csv("data.csv")
profile = EDAAnalyzer(df).analyze(
    target_col="target",
    date_col="timestamp",  # Optional
    lat_col="latitude",  # Optional
    lon_col="longitude",  # Optional
)

EDAVisualizer(profile, df).summary()  # Requires skyulf-core[viz]
```

## Features

- **Unified pipelines**: Serializable preprocessing and model artifacts with
  readable descriptions, fingerprints, model cards, and prediction APIs.
- **Leakage-aware execution**: Split-first guidance and platform validation
  guard against fitting learned preprocessing on held-out rows.
- **Preprocessing and feature engineering**: Cleaning, casting, imputation,
  encoders, scalers, outliers, selection, binning, dates, geospatial features,
  time-series lags/rolling windows, and interactions.
- **Modeling**: sklearn-backed classification, regression, segmentation, and
  Voting/Stacking ensembles, plus text-specific Naive Bayes models.
- **Tuning and evaluation**: Grid Search, Random Search, Optuna, standardized
  classification/regression/clustering metrics, and optional SHAP.
- **Automated EDA**: Data quality, distributions, outliers, temporal,
  geospatial, and target analysis for Polars data.
- **Polars and Arrow**: Native Polars support with direct NumPy bridging for
  sklearn and an explicit Arrow export path.

## License

This project is licensed under the terms of the Apache 2.0 license.
