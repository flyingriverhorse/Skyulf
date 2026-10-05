"""Training reference preparation verifies original values, not only row membership."""

import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_source_evidence import (
    source_evidence,
    validate_source_evidence,
)


def test_same_keys_with_changed_training_values_cannot_become_a_baseline():
    """A recreated table with matching row keys must not replace original training evidence."""
    frame = pd.DataFrame({"id": [1, 2], "x": [3.0, 4.0], "y": [5.0, 6.0]})
    receipt = source_evidence(frame, ("id", "x", "y"), "dataset")
    validate_source_evidence(receipt, frame.iloc[::-1], ("id", "x", "y"), "dataset")
    changed = frame.copy()
    changed.loc[0, "x"] = 99
    with pytest.raises(ValueError, match="original training"):
        validate_source_evidence(receipt, changed, ("id", "x", "y"), "dataset")
    assert receipt["rows"] == 2
