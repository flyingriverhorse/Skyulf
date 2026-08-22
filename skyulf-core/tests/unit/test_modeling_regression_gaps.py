"""Tests targeting gap lines in skyulf.modeling.regression (optional-import guards)."""

from tests.utils.reload_guard import reload_module_preserving_registry


def test_regression_xgboost_import_failure_sets_flag_false(monkeypatch):
    """Simulating an unimportable xgboost must leave XGBOOST_AVAILABLE False after reload."""
    import skyulf.modeling.regression as reg_mod

    with reload_module_preserving_registry(reg_mod, monkeypatch, "xgboost") as mod:
        assert mod.XGBOOST_AVAILABLE is False
    assert reg_mod.XGBOOST_AVAILABLE is True


def test_regression_lightgbm_import_failure_sets_flag_false(monkeypatch):
    """Simulating an unimportable lightgbm must leave LIGHTGBM_AVAILABLE False after reload."""
    import skyulf.modeling.regression as reg_mod

    with reload_module_preserving_registry(reg_mod, monkeypatch, "lightgbm") as mod:
        assert mod.LIGHTGBM_AVAILABLE is False
    assert reg_mod.LIGHTGBM_AVAILABLE is True


def test_regression_silent_lgbm_logger_info_and_warning_are_no_ops():
    """_SilentLgbmLogger.info/.warning must be callable no-ops (silences native lgbm logs)."""
    from skyulf.modeling.regression import _SilentLgbmLogger

    logger_instance = _SilentLgbmLogger()
    assert logger_instance.info("some native message") is None
    assert logger_instance.warning("some native warning") is None
