import sys
import warnings
from pathlib import Path

import pandas as pd

from core.feature_engineering.nodes.feature_eng.binning import (
    _apply_binning_discretization,
    _normalize_binning_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_normalize_binning_config_defaults():
    config = {
        "strategy": "unknown",
        "columns": ["col1", "col1", " col2 "],
        "equal_width_bins": 10,
        "missing_strategy": "label",
        "missing_label": "N/A",
    }

    normalized = _normalize_binning_config(config)

    assert normalized.strategy == "equal_width"
    assert normalized.columns == ["col1", "col2"]
    assert normalized.equal_width_bins == 10
    assert normalized.missing_strategy == "label"
    assert normalized.missing_label == "N/A"


def test_apply_binning_discretization_equal_width():
    frame = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    node = {
        "id": "node_1",
        "data": {
            "config": {
                "strategy": "equal_width",
                "columns": ["value"],
                "equal_width_bins": 2,
            }
        },
    }

    result_frame, summary, signal = _apply_binning_discretization(frame, node)

    assert summary == "Binning: value→value_binned (equal-width, 2 bins)"
    assert "value_binned" in result_frame.columns
    assert result_frame["value_binned"].notna().all()
    assert signal.applied_columns
    applied = signal.applied_columns[0]
    assert applied.strategy == "equal_width"
    assert applied.actual_bins == 2


def test_apply_binning_discretization_kbins_degenerate_bins_are_skipped():
    values = [0.0] * 20 + [1e-9] * 5 + list(range(5))
    frame = pd.DataFrame({"value": values})
    node = {
        "id": "node_kbins",
        "data": {
            "config": {
                "strategy": "kbins",
                "columns": ["value"],
                "kbins_n_bins": 15,
            }
        },
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error")
        result_frame, summary, signal = _apply_binning_discretization(frame, node)

    assert not caught
    assert "value_binned" not in result_frame.columns
    assert "degenerate bins" in summary
    assert any("degenerate bins" in reason for reason in signal.skipped_columns)
