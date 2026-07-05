"""Load small, checked-in "real-shaped" sample datasets for integration tests.

These CSVs are hand-built (not scraped/downloaded) but mimic real-world data
quirks — missing values, mixed dtypes, categorical columns, lat/lon pairs —
so preprocessing/modeling nodes can be exercised end-to-end against something
closer to production data than a purely synthetic ``np.random`` fixture.

Usage:

```python
from tests.utils.dataset_loader import load_sample_dataset

def test_scaler_on_real_shaped_data():
    df = load_sample_dataset("customers")
    ...
```
"""

from pathlib import Path
from typing import Literal, overload

import pandas as pd
import polars as pl

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@overload
def load_sample_dataset(name: str, engine: Literal["pandas"] = "pandas") -> pd.DataFrame: ...
@overload
def load_sample_dataset(name: str, engine: Literal["polars"]) -> pl.DataFrame: ...


def load_sample_dataset(
    name: str, engine: Literal["pandas", "polars"] = "pandas"
) -> pd.DataFrame | pl.DataFrame:
    """Load a checked-in sample CSV from ``tests/data/`` as a DataFrame.

    Args:
        name: File stem under ``tests/data/``, e.g. ``"customers"`` for
            ``tests/data/customers.csv``.
        engine: Which DataFrame library to return the data as.

    Returns:
        The parsed CSV as a pandas or polars DataFrame, per ``engine``.

    Raises:
        FileNotFoundError: If no matching ``.csv`` file exists.
        ValueError: If ``engine`` is not ``"pandas"`` or ``"polars"``.
    """
    csv_path = DATA_DIR / f"{name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No sample dataset found at {csv_path}")

    if engine == "pandas":
        return pd.read_csv(csv_path)
    if engine == "polars":
        return pl.read_csv(csv_path)
    raise ValueError(f"Unsupported engine {engine!r}; expected 'pandas' or 'polars'")
