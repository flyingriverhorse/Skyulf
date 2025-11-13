import pandas as pd

from core.feature_engineering.preprocessing.cleaning.standardize_dates import (
    apply_standardize_date_formats,
)


def test_standardize_dates_iso_format():
    frame = pd.DataFrame(
        {
            "date_col": ["01/02/2024", "02/03/2024", None],
            "other": [1, 2, 3],
        }
    )

    node = {
        "id": "date-node",
        "data": {
            "config": {
                "columns": ["date_col"],
                "mode": "iso_date",
            }
        },
    }

    transformed, summary, signal = apply_standardize_date_formats(frame, node)

    assert transformed.loc[0, "date_col"] == "2024-01-02"
    assert transformed.loc[1, "date_col"] == "2024-02-03"
    assert summary.startswith("Standardize dates: processed 1 column")
    assert signal.total_converted_values >= 1
