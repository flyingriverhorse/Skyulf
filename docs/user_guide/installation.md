# Installation

## Requirements

- **Python 3.9+** (3.10 or 3.11 or newer recommended)
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
| `tuning` | Optuna + Optuna-integration | `pip install skyulf-core[tuning]` |
| `modeling-xgboost` | XGBoost estimators | `pip install skyulf-core[modeling-xgboost]` |
| `preprocessing-imbalanced` | imbalanced-learn (SMOTE, etc.) | `pip install skyulf-core[preprocessing-imbalanced]` |
| `dev` | pytest, twine, build | `pip install skyulf-core[dev]` |

Install everything at once:

```bash
pip install skyulf-core[viz,eda,tuning,modeling-xgboost,preprocessing-imbalanced]
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

## Full platform (Docker)

To run the complete Skyulf platform (backend + frontend + Celery workers):

```bash
docker-compose up --build
```

See the [Quick Start guide](../guides/getting_started.md) for details.

## Import check

Verify the installation:

```python
from skyulf.pipeline import SkyulfPipeline
from skyulf.data.dataset import SplitDataset

print("skyulf-core installed successfully")
```
