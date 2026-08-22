"""Tests for the ``SKYULF_ENGINE`` dataframe-engine setting.

Phase 1a of the backend Polars migration: one explicit, observable engine
switch. Product decision: default is **polars** (deviating from the
migration plan's "pandas until Phase 5" sequencing).
"""

import pytest

from backend.config.base import Settings


def test_skyulf_engine_defaults_to_polars(monkeypatch):
    """The platform ships Polars-first: unset env means polars."""
    monkeypatch.delenv("SKYULF_ENGINE", raising=False)
    assert Settings().SKYULF_ENGINE == "polars"


def test_skyulf_engine_accepts_pandas_override(monkeypatch):
    """Pandas stays a first-class, explicitly selectable option."""
    monkeypatch.setenv("SKYULF_ENGINE", "pandas")
    assert Settings().SKYULF_ENGINE == "pandas"


def test_skyulf_engine_normalizes_case_and_whitespace(monkeypatch):
    """Env values arrive messy; normalize instead of rejecting."""
    monkeypatch.setenv("SKYULF_ENGINE", " Polars ")
    assert Settings().SKYULF_ENGINE == "polars"


def test_skyulf_engine_rejects_unknown_engine(monkeypatch):
    """An unknown engine name fails closed at startup, not at read time."""
    monkeypatch.setenv("SKYULF_ENGINE", "spark")
    with pytest.raises(ValueError):
        Settings()


def test_skyulf_engine_validator_passes_non_strings_through():
    """Non-string values bypass normalization untouched so pydantic's own
    default/required handling still applies (e.g. an unset env entry)."""
    assert Settings.normalize_skyulf_engine(None) is None
    assert Settings.normalize_skyulf_engine(42) == 42
