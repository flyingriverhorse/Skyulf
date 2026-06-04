# Deprecation Policy & Helpers

`skyulf.core.deprecation` provides a small, consistent way to retire public
symbols without breaking callers overnight.

## Policy

A public symbol marked deprecated in minor **`X.Y`** keeps working through
**`X.(Y+1)`** and may be removed in **`X.(Y+2)`**. Deprecating a symbol emits a
`DeprecationWarning` — silent by default, but visible under `python -W default`
and in test suites (pytest shows them in its warnings summary).

## `@deprecated` — for callables

Wrap any function or method. Calling it emits a standardised warning while the
original behaviour and return value are preserved.

```python
from skyulf.core.deprecation import deprecated


@deprecated(since="0.5", removed_in="0.7", replacement="new_transform")
def old_transform(df):
    return new_transform(df)
```

Calling `old_transform(df)` emits:

```
DeprecationWarning: 'old_transform' is deprecated since v0.5 and is scheduled
for removal in v0.7. Use 'new_transform' instead.
```

## `warn_deprecated` — for everything else

Use it directly when a decorator cannot wrap the thing being deprecated — for
example a config key, an enum value, or a positional argument.

```python
from skyulf.core.deprecation import warn_deprecated


def fit(self, df, config):
    if "old_key" in config:
        warn_deprecated(
            "config key 'old_key'",
            since="0.5",
            removed_in="0.7",
            replacement="new_key",
        )
        config["new_key"] = config.pop("old_key")
    ...
```

## Verifying deprecations in tests

```python
import pytest


def test_old_transform_warns():
    with pytest.warns(DeprecationWarning):
        old_transform(df)
```

## API reference

::: skyulf.core.deprecation
