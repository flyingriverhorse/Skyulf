# Installation

## Requirements

- **Python >=3.12**
- pip 21 or newer

## Install from PyPI

```bash
pip install skyulf-core
```

## Optional extras

Install only what you need:

| Extra | What it adds | Command |
|---|---|---|
| `viz` | Matplotlib, Rich (visualisation helpers) | `pip install skyulf-core[viz]` |
| `eda` | VADER sentiment, causal-learn | `pip install skyulf-core[eda]` |
| `text` | VADER sentiment (text features) | `pip install skyulf-core[text]` |
| `nlp` | Sentence-transformers (dense embeddings) | `pip install skyulf-core[nlp]` |
| `geo` | GeoPandas, H3, spatial stats (native deps) | `pip install skyulf-core[geo]` |
| `tuning` | Optuna + Optuna-integration | `pip install skyulf-core[tuning]` |
| `modeling-xgboost` | XGBoost estimators | `pip install skyulf-core[modeling-xgboost]` |
| `modeling-lightgbm` | LightGBM estimators | `pip install skyulf-core[modeling-lightgbm]` |
| `preprocessing-imbalanced` | imbalanced-learn (SMOTE, etc.) | `pip install skyulf-core[preprocessing-imbalanced]` |
| `explainability` | SHAP explainability | `pip install skyulf-core[explainability]` |
| `dev` | pytest, twine, build | `pip install skyulf-core[dev]` |

Install all non-geospatial optional runtime features:

```bash
pip install skyulf-core[all]
```

Install geospatial functionality separately because it has native dependencies:

```bash
pip install skyulf-core[geo]
```

## Editable install (contributor workflow)

If you cloned the repository and want live changes reflected:

```bash
git clone https://github.com/flyingriverhorse/Skyulf.git
cd Skyulf
pip install -e ./skyulf-core
```

## Runtime dependencies (auto-installed)

These are pulled in automatically by `pip install skyulf-core`:

- **pandas** >= 2.0
- **numpy** >= 1.24
- **scikit-learn** >= 1.4
- **polars** >= 1.36
- **pyarrow** >= 21.0
- **pydantic** >= 2.0
- **scipy** >= 1.10
- **statsmodels** >= 0.14

## Self-hosted platform

[Run the separate self-hosted platform](../guides/platform_setup.md)

## Import check

Verify the installation:

```python
from skyulf import SkyulfPipeline
from skyulf.data.dataset import SplitDataset

print("skyulf-core installed successfully")
```
