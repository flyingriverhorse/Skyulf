# Installation

## Editable install (repo checkout)

From the repository root:

```bash
pip install -e ./skyulf-core
```

## Runtime dependencies

`skyulf-core` primarily relies on:

- Polars
- Pandas
- NumPy
- Scikit-Learn

Some preprocessing nodes use optional dependencies (e.g., `rapidfuzz` for string similarity in feature generation).

## Import check

```python
from skyulf.pipeline import SkyulfPipeline
from skyulf.data.dataset import SplitDataset
```
