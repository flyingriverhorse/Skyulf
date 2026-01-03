# Skyulf Core

**Skyulf Core** (`skyulf-core`) is the standalone machine learning library that powers the Skyulf MLOps platform. It provides a robust, type-safe, and modular set of tools for:

- **Data Preprocessing**: A comprehensive suite of transformers for cleaning, scaling, encoding, and feature engineering.
- **Modeling**: Unified interfaces for classification and regression models, wrapping Scikit-Learn and other libraries.
- **Pipeline Management**: Tools to build, serialize, and execute complex ML pipelines.
- **Tuning**: Advanced hyperparameter tuning capabilities with support for Grid Search, Random Search, and Optuna.
- **Evaluation**: Standardized metrics and evaluation schemas for model performance tracking.

<!-- Quick badges + links -->
[![Docs](https://img.shields.io/website?down_color=red&down_message=offline&up_message=online&url=https://flyingriverhorse.github.io/Skyulf)](https://flyingriverhorse.github.io/Skyulf) [![PyPI](https://img.shields.io/pypi/v/skyulf-core.svg)](https://pypi.org/project/skyulf-core) [![License](https://img.shields.io/github/license/flyingriverhorse/Skyulf)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/skyulf-core.svg)](https://pypi.org/project/skyulf-core) [![issues](https://img.shields.io/github/issues/flyingriverhorse/Skyulf.svg)](https://github.com/flyingriverhorse/Skyulf/issues) [![contributors](https://img.shields.io/github/contributors/flyingriverhorse/Skyulf.svg)](https://github.com/flyingriverhorse/Skyulf/graphs/contributors)

**Website & Documentation**

Visit the full documentation and project site for guides, API reference, and examples:

- Project site / docs: https://www.skyulf.com
- Repository: https://github.com/flyingriverhorse/Skyulf
- PyPI package: https://pypi.org/project/skyulf-core

## Installation

```bash
pip install skyulf-core

# For visualization support (Rich dashboard + Matplotlib plots)
pip install skyulf-core[viz]
```

## Quick Start: Automated EDA

Skyulf Core includes a powerful automated Exploratory Data Analysis (EDA) module.

```python
import polars as pl
from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.visualizer import EDAVisualizer

# 1. Load Data
df = pl.read_csv("data.csv")

# 2. Analyze
analyzer = EDAAnalyzer(df)
# Optional: Manually specify special columns if auto-detection fails
profile = analyzer.analyze(
    target_col="target",
    date_col="timestamp",  # Optional
    lat_col="latitude",    # Optional
    lon_col="longitude"    # Optional
)

# 3. Visualize
viz = EDAVisualizer(profile, df)
viz.summary()  # Prints rich terminal dashboard
viz.plot()     # Opens interactive plots
```

## Features

- **Automated EDA**: One-line profiling with Data Quality, Outliers, Time Series, and Geospatial analysis.
- **Type-Safe**: Built with modern Python type hints and Pydantic models.
- **Modular**: Use only the components you need.
- **Serializable**: All components are designed to be easily serialized for storage and deployment.
- **Extensible**: Easy to extend with your own custom transformers and models.

## License

This project is licensed under the terms of the Apache 2.0 license.
