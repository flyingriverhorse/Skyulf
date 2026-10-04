"""Final candidate fits retain empty test partitions without changing feature schemas."""

import pandas as pd
import pytest

from skyulf.preprocessing.encoding import OneHotEncoderApplier, OneHotEncoderCalculator


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("drop_original", [True, False])
def test_onehot_empty_partition_keeps_fitted_columns(engine, drop_original):
    """A fit-only split must not fail or consume holdout rows to satisfy sklearn transform."""
    frame = pd.DataFrame({"category": ["a", "b", "a"], "amount": [1.0, 2.0, 3.0]})
    if engine == "polars":
        import polars as pl

        frame = pl.from_pandas(frame)
    params = OneHotEncoderCalculator().fit(
        frame, {"columns": ["category"], "drop_original": drop_original}
    )
    applier = OneHotEncoderApplier()
    populated = applier.apply(frame, params)
    empty = applier.apply(frame.head(0), params)
    assert len(empty) == 0
    assert list(empty.columns) == list(populated.columns)
    assert list(empty.dtypes) == list(populated.dtypes)
