import numpy as np
import pandas as pd

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
    assert isinstance(artifacts, dict)
    result = applier.apply(df, dict(artifacts))

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
    assert isinstance(artifacts, dict)
    result = applier.apply(df, dict(artifacts))

    assert result.equals(df)
    assert len(artifacts["snapshot"]) == 2
    snapshot = artifacts["snapshot"]
    assert isinstance(snapshot, list)
    assert snapshot[0]["a"] == 0
