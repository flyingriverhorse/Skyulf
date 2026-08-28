"""Tests for skyulf.profiling._analyzer._utils module-level helpers and dependency probes."""

import importlib
import importlib.util

import polars as pl
import pytest

import skyulf.profiling._analyzer._utils as utils_mod


def test_collect_narrows_lazyframe_to_dataframe() -> None:
    """``_collect`` should eagerly evaluate a LazyFrame and return a concrete DataFrame."""
    lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    result = utils_mod._collect(lf)
    assert isinstance(result, pl.DataFrame)
    assert result["a"].to_list() == [1, 2, 3]


def test_optional_dependency_flags_are_true_when_packages_installed() -> None:
    """In this environment all optional deps (sklearn/scipy/statsmodels/vader) are installed."""
    pytest.importorskip("vaderSentiment")
    assert utils_mod.SKLEARN_AVAILABLE is True
    assert utils_mod.SCIPY_AVAILABLE is True
    assert utils_mod.STATSMODELS_AVAILABLE is True
    assert utils_mod.VADER_AVAILABLE is True


def test_optional_dependency_flags_flip_to_false_when_package_missing(monkeypatch) -> None:
    """Simulate missing optional deps and confirm each flag degrades to False.

    ``_utils`` probes sklearn/scipy/statsmodels/vaderSentiment with
    ``importlib.util.find_spec`` at import time (F-28: probes must not execute
    module code). We force find_spec to report those packages absent and reload
    the module, then restore it to its normal (dependencies-available) state.
    """
    real_find_spec = importlib.util.find_spec
    blocked = {"sklearn", "scipy", "statsmodels", "vaderSentiment"}

    def fake_find_spec(name, *args, **kwargs):
        if name in blocked:
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    try:
        importlib.reload(utils_mod)
        assert utils_mod.SKLEARN_AVAILABLE is False
        assert utils_mod.SCIPY_AVAILABLE is False
        assert utils_mod.STATSMODELS_AVAILABLE is False
        assert utils_mod.VADER_AVAILABLE is False
    finally:
        monkeypatch.undo()
        # Restore real availability flags for any other test relying on this module.
        importlib.reload(utils_mod)
        assert utils_mod.SKLEARN_AVAILABLE is True
