from __future__ import annotations

from typing import List

import pytest
from sklearn.preprocessing import LabelEncoder

from backend.ml_pipeline.services.prediction_utils import decode_int_like


def _make_label_encoder(classes: List[str]) -> LabelEncoder:
    encoder = LabelEncoder()
    encoder.fit(classes)
    return encoder


def test_decode_int_like_does_not_decode_non_integer_floats() -> None:
    encoder = _make_label_encoder(["cat", "dog", "mouse"])

    values = [0.2, 1.7, 2.9]
    decoded = decode_int_like(values, encoder)

    assert decoded == values


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0, 1, 2], ["cat", "dog", "mouse"]),
        ([0.0, 1.0, 2.0], ["cat", "dog", "mouse"]),
        ([True, False, True], ["dog", "cat", "dog"]),
    ],
)
def test_decode_int_like_decodes_integer_like_values(values, expected) -> None:
    encoder = _make_label_encoder(["cat", "dog", "mouse"])

    decoded = decode_int_like(values, encoder)

    assert decoded == expected
