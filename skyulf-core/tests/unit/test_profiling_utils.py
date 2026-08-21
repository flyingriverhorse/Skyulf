"""Tests for skyulf.profiling._analyzer._utils module-level helpers and dependency probes."""

import builtins
import importlib

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


def test_optional_dependency_flags_flip_to_false_on_import_error(monkeypatch) -> None:
    """Simulate missing optional deps at import time and confirm each flag degrades to False.

    ``_utils`` probes sklearn/scipy/statsmodels/vaderSentiment with module-level
    try/except ImportError blocks. We force those imports to fail and reload the
    module to exercise the ``except ImportError: FOO_AVAILABLE = False`` branches,
    then restore the module to its normal (dependencies-available) state.
    """
    real_import = builtins.__import__
    blocked_prefixes = ("sklearn", "scipy", "statsmodels", "vaderSentiment")

    def fake_import(name, *args, **kwargs):
        if name.startswith(blocked_prefixes):
            raise ImportError(f"mocked missing dependency: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
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
