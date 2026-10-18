"""Optional custom callbacks selected by config/scoring.yml."""


def long_trip_fare(inputs, predictions, params):
    """Weight the fare estimate by the long-trip probability from the second model."""
    expected = predictions["fare__prediction"] * predictions["long_trip__probability_1"]
    return expected.astype("float64").to_frame("expected_long_trip_fare")
