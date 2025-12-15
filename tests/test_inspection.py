import numpy as np
import pandas as pd
import pytest
from skyulf.preprocessing.inspection import (
    DatasetProfileApplier,
    DatasetProfileCalculator,
    DataSnapshotApplier,
    DataSnapshotCalculator,
)


def test_dataset_profile():
    df = pd.DataFrame({"a": [1, 2, 3, np.nan], "b": ["x", "y", "z", "w"]})
    calc = DatasetProfileCalculator()
    applier = DatasetProfileApplier()

    artifacts = calc.fit(df, {})
    result = applier.apply(df, artifacts)

    assert result.equals(df)  # Should be identity
    assert "profile" in artifacts
    assert artifacts["profile"]["rows"] == 4
    assert artifacts["profile"]["missing"]["a"] == 1


def test_data_snapshot():
    df = pd.DataFrame({"a": range(10)})
    calc = DataSnapshotCalculator()
    applier = DataSnapshotApplier()

    config = {"n_rows": 2}
    artifacts = calc.fit(df, config)
    result = applier.apply(df, artifacts)

    assert result.equals(df)
    assert len(artifacts["snapshot"]) == 2
    assert artifacts["snapshot"][0]["a"] == 0
