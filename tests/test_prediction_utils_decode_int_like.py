import pytest
from sklearn.preprocessing import LabelEncoder

from backend.ml_pipeline._services.prediction_utils import decode_int_like


def _make_label_encoder(classes: list[str]) -> LabelEncoder:
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


def test_decode_int_like_decodes_stringified_integers() -> None:
    """y_proba's "classes" list comes from DataFrame column names (e.g.
    ``model.classes_``), which sklearn/pandas often stringifies (e.g.
    "0"/"1"/"2") even though they represent the same encoded integer
    indices as y_true/y_pred. Regression test for a bug where these were
    silently left undecoded because ``str`` arrays don't have an
    integer/float dtype.kind, while y_true/y_pred (real int arrays) decoded
    fine — causing the frontend's decoded-label lookups to mismatch.
    """
    encoder = _make_label_encoder(["cat", "dog", "mouse"])

    values = ["0", "1", "2"]
    decoded = decode_int_like(values, encoder)

    assert decoded == ["cat", "dog", "mouse"]


def test_decode_int_like_does_not_decode_non_numeric_strings() -> None:
    encoder = _make_label_encoder(["cat", "dog", "mouse"])

    values = ["cat", "dog", "mouse"]
    decoded = decode_int_like(values, encoder)

    assert decoded == values
