import pandas as pd

from core.feature_engineering.nodes.feature_eng.feature_math import (
    FeatureMathConfig,
    FeatureMathOperation,
    _apply_datetime_operation,
)


def test_apply_datetime_operation_creates_expected_features():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([
                "2023-01-15 08:30:00",
                "2023-06-20 14:45:00",
                None,
            ])
        }
    )

    operation = FeatureMathOperation(
        operation_id="dt1",
        operation_type="datetime_extract",
        input_columns=["timestamp"],
        datetime_features=["year", "month", "day_name"],
        fillna="missing",
    )

    config = FeatureMathConfig()

    created, message = _apply_datetime_operation(frame, operation, config)

    assert set(created) == {"timestamp_year", "timestamp_month", "timestamp_day_name"}
    assert "year" in message and "month" in message and "day_name" in message
    assert frame["timestamp_day_name"].iloc[2] == "missing"


def test_apply_datetime_operation_respects_timezone_and_output_prefix():
    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime([
                "2023-01-01 00:00:00",
                "2023-01-01 02:00:00",
            ])
        }
    )

    operation = FeatureMathOperation(
        operation_id="dt2",
        operation_type="datetime_extract",
        input_columns=["ts"],
        datetime_features=["hour"],
        timezone="Europe/Berlin",
        output_prefix="feat_",
    )

    config = FeatureMathConfig(default_timezone="UTC")

    created, _ = _apply_datetime_operation(frame, operation, config)

    assert created == ["feat_hour"]
    assert frame[created[0]].dtype.kind in {"i", "u"}
