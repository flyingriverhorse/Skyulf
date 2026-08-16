"""Tests targeting gap lines in skyulf.modeling.regression (optional-import guards)."""

import importlib
import sys


def test_regression_xgboost_import_failure_sets_flag_false(monkeypatch):
    """Simulating an unimportable xgboost must leave XGBOOST_AVAILABLE False after reload."""
    import skyulf.modeling.regression as reg_mod

    monkeypatch.setitem(sys.modules, "xgboost", None)
    try:
        importlib.reload(reg_mod)
        assert reg_mod.XGBOOST_AVAILABLE is False
    finally:
        monkeypatch.delitem(sys.modules, "xgboost", raising=False)
        importlib.reload(reg_mod)
        assert reg_mod.XGBOOST_AVAILABLE is True


def test_regression_lightgbm_import_failure_sets_flag_false(monkeypatch):
    """Simulating an unimportable lightgbm must leave LIGHTGBM_AVAILABLE False after reload."""
    import skyulf.modeling.regression as reg_mod

    monkeypatch.setitem(sys.modules, "lightgbm", None)
    try:
        importlib.reload(reg_mod)
        assert reg_mod.LIGHTGBM_AVAILABLE is False
    finally:
        monkeypatch.delitem(sys.modules, "lightgbm", raising=False)
        importlib.reload(reg_mod)
        assert reg_mod.LIGHTGBM_AVAILABLE is True


def test_regression_silent_lgbm_logger_info_and_warning_are_no_ops():
    """_SilentLgbmLogger.info/.warning must be callable no-ops (silences native lgbm logs)."""
    from skyulf.modeling.regression import _SilentLgbmLogger

    logger_instance = _SilentLgbmLogger()
    assert logger_instance.info("some native message") is None
    assert logger_instance.warning("some native warning") is None
