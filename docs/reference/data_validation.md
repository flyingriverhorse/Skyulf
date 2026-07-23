# Data Validation Expectations

`skyulf.profiling.expect` is a lightweight, dependency-free data-validation
helper — a tiny subset of what Great Expectations offers, but with **zero extra
dependencies**. Each `expect_*` function checks a single condition and raises
[`ExpectationError`][skyulf.profiling.expect.ExpectationError] with a precise
message when the condition is violated.

It is **engine-agnostic**: Pandas frames are used directly; Polars (or any frame
exposing `to_pandas()`) is converted first.

## When to use it

These are **manual assertions** — they are *not* wired into profiling or CI
automatically. You call them yourself in two main places:

1. **In tests / CI** — guard a dataset contract so a bad upstream change fails
   the build.
2. **In a pipeline** — assert preconditions before an expensive step, so you get
   a clear error instead of a deep traceback later.

## Available expectations

| Function | Checks |
| --- | --- |
| `expect_columns_exist(df, columns)` | Every name in `columns` is present. |
| `expect_no_nulls(df, columns=None)` | Given columns (default: all) have no nulls. |
| `expect_value_range(df, column, *, minimum, maximum, inclusive=True)` | All values fall within `[minimum, maximum]`. |
| `expect_unique(df, columns)` | The combination of `columns` has no duplicate rows. |

## Example: a dataset contract in CI

```python
import pandas as pd
from skyulf import (
    ExpectationError,
    expect_columns_exist,
    expect_no_nulls,
    expect_unique,
    expect_value_range,
)


def validate_customers(df: pd.DataFrame) -> None:
    """Raises ExpectationError if the customers frame breaks its contract."""
    expect_columns_exist(df, ["customer_id", "age", "signup_date"])
    expect_unique(df, ["customer_id"])
    expect_no_nulls(df, ["customer_id", "signup_date"])
    expect_value_range(df, "age", minimum=0, maximum=120)
```

Wire it into a test so CI enforces it:

```python
def test_customers_contract():
    df = pd.read_parquet("data/customers.parquet")
    validate_customers(df)  # raises ExpectationError on violation → test fails
```

## Example: a pipeline guard

```python
from skyulf import expect_no_nulls

def run(df):
    # Fail fast with a clear message before an expensive fit.
    expect_no_nulls(df, ["target"])
    ...
```

## API reference

::: skyulf.profiling.expect
