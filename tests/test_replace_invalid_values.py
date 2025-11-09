import pandas as pd

from core.feature_engineering.nodes.data_consistency.replace_invalid_values import (
    apply_replace_invalid_values,
)


def test_replace_invalid_values_custom_bounds():
    frame = pd.DataFrame(
        {
            "metric": [10, -5, 15, None],
            "other": [1, 2, 3, 4],
        }
    )

    node = {
        "id": "invalid-node",
        "data": {
            "config": {
                "columns": ["metric"],
                "mode": "custom_range",
                "min_value": 0,
                "max_value": 12,
            }
        },
    }

    transformed, summary, signal = apply_replace_invalid_values(frame, node)

    assert pd.isna(transformed.loc[1, "metric"])
    assert summary.startswith("Replace invalid values: processed 1 column")
    assert signal.total_replacements == 2
