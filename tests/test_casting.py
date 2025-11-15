import pandas as pd

from core.feature_engineering.preprocessing.casting import _apply_cast_column_types


def test_apply_cast_column_types_casts_to_float_and_tracks_signal():
    frame = pd.DataFrame({"value": ["1.5", "2.0", "bad"]})
    node = {
        "data": {
            "config": {
                "columns": ["value"],
                "target_dtype": "float64",
                "coerce_on_error": True,
            }
        }
    }

    result_frame, summary, signal = _apply_cast_column_types(frame, node)

    assert "Cast columns: attempted 1 column(s) to float64" in summary
    assert result_frame["value"].dtype == "float64"
    assert signal.applied_columns == ["value"]
    assert signal.coerced_values == 1
    assert pd.isna(result_frame.loc[2, "value"])
