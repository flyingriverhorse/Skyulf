# Data Validation Expectations

`skyulf.profiling.expect` is a lightweight, dependency-free data-validation
helper — a tiny subset of what Great Expectations offers, but with **zero extra
dependencies**. Each `expect_*` function checks a single condition and raises
[`ExpectationError`][skyulf.profiling.expect.ExpectationError] with a precise
message when the condition is violated.

It is **engine-agnostic**: Pandas frames are used directly; raw and wrapped
Polars frames stay native for simple predicates. Boolean range checks and
other frames exposing `to_pandas()` use Pandas conversion.

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
| `expect_no_nulls(df, columns=None, *, allow_empty=False)` | Given columns (default: all) have no nulls. |
| `expect_value_range(df, column, *, minimum=None, maximum=None, inclusive=True, allow_empty=False)` | All non-null values fall within the optional bounds. |
| `expect_unique(df, columns, *, allow_empty=False)` | The combination of `columns` has no duplicate rows. |

## Empty datasets

The null, range and uniqueness checks require at least one row by default.
An ingestion that unexpectedly returns zero rows therefore raises
`ExpectationError`, even if the expected columns are present.

If empty partitions are valid in your workflow, explicitly opt in on each check:

```python
import pandas as pd
from skyulf import expect_no_nulls, expect_unique, expect_value_range

empty = pd.DataFrame({"age": pd.Series(dtype="float64")})
expect_no_nulls(empty, ["age"], allow_empty=True)
expect_value_range(empty, "age", minimum=0, maximum=120, allow_empty=True)
expect_unique(empty, ["age"], allow_empty=True)
```

Previously these checks accepted empty frames automatically; existing callers
that rely on that behavior must add `allow_empty=True`. Requested columns must
still exist. `expect_columns_exist` only checks the schema and accepts zero rows.
Range checks continue to ignore null values, including an all-null column in a
frame that has rows; combine them with `expect_no_nulls` when nulls are invalid.

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
