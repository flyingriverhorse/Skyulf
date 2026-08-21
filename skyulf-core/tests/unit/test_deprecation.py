"""Tests for the deprecation helpers (`core.deprecation`)."""

import warnings

import pytest

from skyulf.core.deprecation import deprecated, warn_deprecated


def test_deprecated_decorator_warns_and_preserves_behaviour():
    @deprecated(since="0.2", removed_in="0.4", replacement="new_add")
    def old_add(a, b):
        return a + b

    with pytest.warns(DeprecationWarning) as record:
        result = old_add(2, 3)

    assert result == 5
    message = str(record[0].message)
    assert "old_add" in message
    assert "v0.2" in message
    assert "v0.4" in message
    assert "new_add" in message


def test_deprecated_preserves_metadata():
    @deprecated(since="0.2")
    def documented():
        """Original docstring."""

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Original docstring."


def test_warn_deprecated_emits_warning():
    with pytest.warns(DeprecationWarning, match="old_key"):
        warn_deprecated("old_key", since="0.3", replacement="new_key")


def test_warning_is_silent_by_default():
    @deprecated()
    def noisy():
        return 1

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert noisy() == 1
