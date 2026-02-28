"""Tests for prediction label decoding.

Verifies that extract_target_label_encoder() finds the correct encoder
under both ``__target__`` (post-split encoding) and by target column name
(pre-split encoding) so that integer predictions decode back to labels.
"""

import pytest
from sklearn.preprocessing import LabelEncoder

from backend.ml_pipeline.services.prediction_utils import (
    decode_int_like,
    extract_target_label_encoder,
)


def _make_feature_engineer(fitted_steps):
    """Fake FeatureEngineer with the given fitted_steps attribute."""

    class FakeFE:
        pass

    fe = FakeFE()
    fe.fitted_steps = fitted_steps  # type: ignore[attr-defined]
    return fe


@pytest.fixture
def species_label_encoder():
    """A fitted LabelEncoder for Species: setosa=0, versicolor=1, virginica=2."""
    le = LabelEncoder()
    le.fit(["setosa", "versicolor", "virginica"])
    return le


# ---------- extract_target_label_encoder ---------- #


class TestExtractTargetLabelEncoder:
    """Tests for the encoder extraction helper."""

    def test_finds_dunder_target(self, species_label_encoder):
        """Standard path: encoder stored under __target__ key."""
        fe = _make_feature_engineer(
            [
                {
                    "name": "step",
                    "type": "LabelEncoder",
                    "applier": None,
                    "artifact": {"encoders": {"__target__": species_label_encoder}},
                }
            ]
        )
        enc = extract_target_label_encoder(fe)
        assert enc is species_label_encoder

    def test_finds_by_target_column_name(self, species_label_encoder):
        """Fallback path: encoder keyed by column name (pre-split encoding)."""
        fe = _make_feature_engineer(
            [
                {
                    "name": "step",
                    "type": "LabelEncoder",
                    "applier": None,
                    "artifact": {"encoders": {"Species": species_label_encoder}},
                }
            ]
        )
        # Without target_column, should NOT find it
        assert extract_target_label_encoder(fe) is None

        # With target_column, should find it
        enc = extract_target_label_encoder(fe, target_column="Species")
        assert enc is species_label_encoder

    def test_dunder_target_takes_priority(self, species_label_encoder):
        """__target__ key has priority over column name."""
        other_le = LabelEncoder()
        other_le.fit(["a", "b"])

        fe = _make_feature_engineer(
            [
                {
                    "name": "step",
                    "type": "LabelEncoder",
                    "applier": None,
                    "artifact": {
                        "encoders": {
                            "__target__": species_label_encoder,
                            "Species": other_le,
                        }
                    },
                }
            ]
        )
        enc = extract_target_label_encoder(fe, target_column="Species")
        assert enc is species_label_encoder  # __target__ wins

    def test_returns_none_when_no_label_encoder(self):
        fe = _make_feature_engineer(
            [{"name": "step", "type": "StandardScaler", "applier": None, "artifact": {}}]
        )
        assert extract_target_label_encoder(fe) is None

    def test_returns_none_for_empty_steps(self):
        fe = _make_feature_engineer([])
        assert extract_target_label_encoder(fe) is None


# ---------- decode_int_like ---------- #


class TestDecodeIntLike:
    """Tests for the decoding helper."""

    def test_decodes_integer_list(self, species_label_encoder):
        result = decode_int_like([0, 1, 2, 0], species_label_encoder)
        assert result == ["setosa", "versicolor", "virginica", "setosa"]

    def test_decodes_float_like_integers(self, species_label_encoder):
        """Values like 0.0, 1.0 should still be decoded."""
        result = decode_int_like([0.0, 1.0, 2.0], species_label_encoder)
        assert result == ["setosa", "versicolor", "virginica"]

    def test_skips_non_integer_floats(self, species_label_encoder):
        """True floats (e.g. regression predictions) should NOT be decoded."""
        original = [0.5, 1.7, 2.3]
        result = decode_int_like(original, species_label_encoder)
        assert result == original

    def test_handles_empty_list(self, species_label_encoder):
        result = decode_int_like([], species_label_encoder)
        assert result == []
