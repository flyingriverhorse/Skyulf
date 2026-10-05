"""Classification diagnostics retain real class labels and bounded aggregate counts."""

from unittest.mock import MagicMock

import pytest

from skyulf.integrations.databricks.spark_monitoring_metrics import _classification


def test_confusion_cells_preserve_saved_class_order_and_population_counts():
    """Dashboard labels must describe actual outcomes, not encoded class positions."""
    pairs = MagicMock()
    pairs.groupBy.return_value.count.return_value.limit.return_value.collect.return_value = [
        ("no", "yes", 2),
        ("yes", "yes", 1_100_000),
        ("no", "no", 3),
    ]
    values, matrix = _classification(pairs, ("yes", "no"))
    assert values["accuracy"] == pytest.approx(1_100_003 / 1_100_005)
    assert matrix == {
        "status": "measured",
        "cells": [
            {
                "actual_label": "yes",
                "predicted_label": "yes",
                "actual_index": 0,
                "predicted_index": 0,
                "count": 1_100_000,
            },
            {
                "actual_label": "yes",
                "predicted_label": "no",
                "actual_index": 0,
                "predicted_index": 1,
                "count": 0,
            },
            {
                "actual_label": "no",
                "predicted_label": "yes",
                "actual_index": 1,
                "predicted_index": 0,
                "count": 2,
            },
            {
                "actual_label": "no",
                "predicted_label": "no",
                "actual_index": 1,
                "predicted_index": 1,
                "count": 3,
            },
        ],
    }
