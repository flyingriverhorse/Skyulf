# Skyulf Engine Mechanics

This guide explains how Skyulf handles different dataframes (Polars and Pandas) under the hood, ensuring both high performance and compatibility with the Python ML ecosystem (Scikit-Learn).

## 1. Specifically, what is `SkyulfDataFrame`?

You might see `SkyulfDataFrame` in our type hints. It is **not** a new class. It is a **Type Alias** (Protocol) that means:
*"This variable can be either a `polars.DataFrame` or a `pandas.DataFrame`."*

```python
# Conceptually:
SkyulfDataFrame = Union[pl.DataFrame, pd.DataFrame]
```

This ensures our code knows how to handle both formats without forcing you to convert everything manually.

## 2. Engine Detection: How do we know which one it is?

We don't ask you to specify the engine. We detect it automatically using the `get_engine(df)` utility.

### The Logic (`skyulf.engines`)
When a DataFrame enters a Calculator or Applier:
1.  We check `isinstance(df, polars.DataFrame)`.
    *   If **True**: We tag it as `Engine.POLARS`.
2.  If not, we check `isinstance(df, pandas.DataFrame)`.
    *   If **True**: We tag it as `Engine.PANDAS`.
3.  Otherwise, we raise an error.

```python
from skyulf.engines import get_engine

def fit(self, df: SkyulfDataFrame, ...):
    engine = get_engine(df)
    
    if engine.name == "polars":
        # Run optimized Polars logic
    else:
        # Run standard Pandas logic
```

## 3. The Hybrid Architecture

We want the **speed of Polars** but the **ecosystem of Scikit-Learn**. To achieve this, we use a hybrid strategy.

### Path A: Pure Polars (The Fast Path)
For operations that Polars supports natively, we stay in Polars. This is zero-copy and extremely fast.
*   **Examples:** `StandardScaler`, `MinMaxScaler`, `SimpleImputing` (mean/median), `LogTransformation`, `Encoding`.
*   **Mechanism:** We construct Polars **expressions** (e.g., `pl.col(c).mean()`) and apply them lazy or eager.

### Path B: The "Hybrid Bridge" (Compatibility Path)
For complex algorithms that only exist in Scikit-Learn (e.g., `IsolationForest`, `RFE`, `PolynomialFeatures`), we temporarily bridge to Pandas/Numpy.

**The Workflow:**
1.  **Input:** Receive `pl.DataFrame`.
2.  **Bridge:** Convert to `pd.DataFrame` (using Arrow for speed).
3.  **Compute:** Run Scikit-Learn (e.g., `sklearn_model.fit(df_pandas)`).
4.  **Return:**
    *   If the result is small (parameters), we just store them.
    *   If the result is data (transformed rows), we convert the result back to `pl.DataFrame`.

#### Why Arrow? (The "Zero-Copy" Magic)
A common fear with hybrid systems is: *"Won't converting data back and forth double my memory usage and kill performance?"*

This is where **Apache Arrow** comes in.
*   **Polars** is built on top of the Arrow memory format.
*   **Pandas (2.0+)** supports Arrow backends, and even older Pandas can ingest Arrow very efficiently.

When we call `df_polars.to_pandas()`, it doesn't serialize data to Python objects (slow). It hands over the **pointer to the Arrow memory buffer**. This is often **Zero-Copy** (or near zero-copy), meaning the data stays in the same place in RAM, and Pandas just "views" it. This makes the hybrid bridge incredibly lightweight compared to traditional CSV/method conversion.

#### The `SklearnBridge` Utility
To keep our code clean, we use a utility called `SklearnBridge` in `skyulf.engines.sklearn_bridge`.

Instead of writing `if polars: to_numpy()` everywhere, `SklearnBridge` handles the standardization:

1.  **Standardizes Input:** Accepts Polars DF, Pandas DF, or Tuples `(X, y)`.
2.  **Handles Targets (The "Flattening" Problem):**
    *   **The Issue:** DataFrames treat a single column as a 2D structure (a list of lists, like `[[1], [2], [3]]`, shape `(N, 1)`).
    *   **The Need:** Scikit-Learn models often expect the target `y` to be a simple 1D array (like `[1, 2, 3]`, shape `(N,)`).
    *   **The Fix:** The Bridge automatically detects this and **"flattens"** (ravels) the array so Scikit-Learn doesn't throw a shape mismatch error.
3.  **Safe Output:** Ensures the result is always a Numpy array ready for `.fit()`.

```python
# Inside your complex Calculator
from skyulf.engines.sklearn_bridge import SklearnBridge

def fit(self, df, ...):
    # Doesn't matter if df is Pandas or Polars
    X_matrix, y_vector = SklearnBridge.to_sklearn(df)
    
    model = IsolationForest()
    model.fit(X_matrix, y_vector)
```
This isolates the "compatibility boilerplate" away from your ML logic.

This allows us to support *every* ML feature without rewriting Scikit-Learn from scratch in Polars.

## 4. Calculator vs. Applier in the Hybrid World

### Calculator (`fit`)
*   **Polars Engine:** Calculates stats (min, max, mean) using fast Polars aggregations. Returns a simple Python dictionary (e.g., `{'mean': 5.2}`).
*   **Pandas Engine:** Calculates stats using `.mean()`. Returns the same dictionary structure.
*   **Result:** The `params` dictionary is **engine-agnostic**. It doesn't care where it came from.

### Applier (`apply`)
*   Receives the **agnostic params**.
*   Checks the *current* dataframe's engine.
*   **If Polars:** Uses `pl.col("A") - params["mean"]`.
*   **If Pandas:** Uses `df["A"] - params["mean"]`.

This means you can **train on Pandas** and **predict on Polars** (or vice versa)!

## Summary
| Feature | Polars Path | Pandas Path |
| :--- | :--- | :--- |
| **Speed** | 🚀 High | 🐢 Standard |
| **Memory** | Efficient (Rust) | High Overhead |
| **Compatibility** | Native ops (Expr) | Full Scikit-Learn support |
| **Usage** | Default for ETL/Serving | Default for Training complex models |

Skyulf handles the switching for you. You just pass the data.
