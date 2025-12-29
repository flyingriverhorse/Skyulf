# Adding a New Compute Engine (Dask, Spark, etc.)

Skyulf is designed to be engine-agnostic. While it currently supports **Pandas** and **Polars**, the architecture allows for adding distributed computing engines like **Dask** or **Spark** (via PySpark) without rewriting the entire codebase.

## Architecture Overview

The core abstraction is the `EngineRegistry` and the `BaseEngine` interface.

1.  **Engine Registry:** `skyulf.engines.registry.EngineRegistry` manages available engines.
2.  **Base Engine:** `skyulf.engines.base.BaseEngine` defines the contract that all engines must fulfill.
3.  **Nodes:** `Calculator` and `Applier` classes use the engine to perform operations.

## Steps to Add a New Engine

### 1. Implement the Engine Class

Create a new file in `skyulf-core/skyulf/engines/` (e.g., `dask_engine.py`).

```python
from typing import Any, List, Union
import dask.dataframe as dd
from skyulf.engines.base import BaseEngine

class DaskEngine(BaseEngine):
    @property
    def name(self) -> str:
        return "dask"

    def is_dataframe(self, data: Any) -> bool:
        return isinstance(data, dd.DataFrame)

    def is_series(self, data: Any) -> bool:
        return isinstance(data, dd.Series)

    def get_columns(self, df: Any) -> List[str]:
        return list(df.columns)

    def shape(self, df: Any) -> tuple:
        # Dask shape is lazy, might need compute() or delayed
        return (len(df), len(df.columns))

    # ... implement other required methods
```

### 2. Register the Engine

In `skyulf-core/skyulf/engines/__init__.py` or `registry.py`, register your new engine.

```python
from .dask_engine import DaskEngine
from .registry import EngineRegistry

EngineRegistry.register(DaskEngine())
```

### 3. Update Nodes (Optional but Recommended)

Most existing nodes use `SklearnBridge` or generic Python logic.
*   **SklearnBridge:** If you implement `to_numpy()` or `to_pandas()` in your engine (or if the dataframe supports it), many nodes will work out-of-the-box by converting data to local memory (Pandas/Numpy).
*   **Native Implementation:** For true distributed performance, you should update critical nodes (like `Scaling`, `Join`, `Aggregation`) to use native Dask/Spark API.

Example of a node supporting multiple engines:

```python
class StandardScalerCalculator(BaseCalculator):
    def fit(self, df, config):
        engine = get_engine(df)
        
        if engine.name == "dask":
            # Use Dask native logic
            mean = df.mean().compute()
            std = df.std().compute()
            return {"mean": mean, "std": std}
        
        elif engine.name == "polars":
            # Use Polars logic
            ...
        
        else:
            # Fallback to Pandas/Sklearn
            ...
```

### 4. Update Data Ingestion

Ensure `DataService` or your ingestion layer can load data into the new format (e.g., `dd.read_csv`).

## Summary

You do **not** need to rewrite everything.
1.  **Infrastructure:** Add the Engine implementation (One-time setup).
2.  **Compatibility:** Existing nodes will work via fallback (conversion to Pandas), assuming the data fits in memory.
3.  **Optimization:** Gradually optimize specific nodes to use the new engine's native capabilities for large-scale data.
