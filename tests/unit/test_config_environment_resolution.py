"""OC-130 — ``FASTAPI_ENV`` resolution fails closed.

An unrecognized value used to select ``DevelopmentSettings`` silently, booting
the most permissive profile: wildcard CORS combined with credentials,
``DEBUG=True``, no ``SECRET_KEY`` check and no security headers. The variable was
also read with a bare ``os.getenv``, and because it is not a ``Settings`` field
pydantic-settings never exported a ``.env`` value into ``os.environ`` — so the
documented configuration file was a dead channel for this one setting.
"""

import pytest
from pydantic import ValidationError

from backend.config import KNOWN_ENVIRONMENTS
from backend.config import environments as env_profiles
from backend.config.base import Settings, resolve_environment
from backend.config.factory import _ENV_SETTINGS_MAP

UNKNOWN_VALUES = ["prod", "prd", "staging", "production!", "", " ", "dev"]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """Run in an empty CWD with every settings variable scrubbed.

    Both halves matter. The CWD change keeps the repo's real ``.env`` out, and
    the scrub keeps a developer shell that exports ``SECRET_KEY`` from quietly
    satisfying the production guard these tests assert on — with it set, the
    guard tests passed vacuously.
    """
    monkeypatch.chdir(tmp_path)
    for name in ("FASTAPI_ENV", *Settings.model_fields):
        monkeypatch.delenv(name, raising=False)


def test_unset_defaults_to_development():
    assert resolve_environment() == "development"


@pytest.mark.parametrize("name", KNOWN_ENVIRONMENTS)
def test_each_known_environment_resolves_to_itself(monkeypatch, name):
    monkeypatch.setenv("FASTAPI_ENV", name)
    assert resolve_environment() == name


@pytest.mark.parametrize("value", ["PRODUCTION", "Production", "production "])
def test_value_is_normalized_not_rejected(monkeypatch, value):
    """Case and surrounding whitespace are normalized.

    ``"production "`` is the trailing-space case a YAML/CI variable picks up; it
    used to fall through to ``DevelopmentSettings``.
    """
    monkeypatch.setenv("FASTAPI_ENV", value)
    assert resolve_environment() == "production"


@pytest.mark.parametrize("value", UNKNOWN_VALUES)
def test_unknown_value_raises_instead_of_falling_back(monkeypatch, value):
    monkeypatch.setenv("FASTAPI_ENV", value)
    with pytest.raises(ValueError, match="Unknown FASTAPI_ENV") as excinfo:
        resolve_environment()
    # The message must name the accepted values, or the operator cannot fix it.
    for name in KNOWN_ENVIRONMENTS:
        assert name in str(excinfo.value)


def test_dotenv_only_value_is_honored(tmp_path):
    """A ``.env``-only ``FASTAPI_ENV`` reaches the resolver.

    This channel was dead before: ``os.getenv`` cannot see a value that
    pydantic-settings loaded from the dotenv into the model.
    """
    (tmp_path / ".env").write_text("FASTAPI_ENV=production\n", encoding="utf-8")
    assert resolve_environment() == "production"


def test_dotenv_unknown_value_also_fails_closed(tmp_path):
    (tmp_path / ".env").write_text("FASTAPI_ENV=prod\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown FASTAPI_ENV"):
        resolve_environment()


def test_process_env_wins_over_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("FASTAPI_ENV=testing\n", encoding="utf-8")
    monkeypatch.setenv("FASTAPI_ENV", "production")
    assert resolve_environment() == "production"


def test_settings_map_covers_every_known_environment():
    """A key added to ``KNOWN_ENVIRONMENTS`` but missing from the map must not
    silently select a fallback profile."""
    assert set(_ENV_SETTINGS_MAP) == set(KNOWN_ENVIRONMENTS)
    assert _ENV_SETTINGS_MAP["development"] is env_profiles.DevelopmentSettings
    assert _ENV_SETTINGS_MAP["production"] is env_profiles.ProductionSettings
    assert _ENV_SETTINGS_MAP["testing"] is env_profiles.TestingSettings


def test_secret_key_guard_shares_the_resolver(monkeypatch):
    """The production ``SECRET_KEY`` check must agree with class selection.

    Both readers used to do their own exact-match ``os.getenv``, so a value the
    factory accepted the guard could still skip. ``Settings()`` itself has no
    ``model_post_init``, so constructing it does not touch the filesystem or the
    root logger.
    """
    monkeypatch.setenv("FASTAPI_ENV", "production")
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings()


def test_secret_key_guard_survives_trailing_space(monkeypatch):
    """``"production "`` skipped the ``SECRET_KEY`` guard entirely before."""
    monkeypatch.setenv("FASTAPI_ENV", "production ")
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings()


def test_unknown_value_reaches_the_guard_too(monkeypatch):
    """A typo must not quietly downgrade to a dev ``Settings`` that boots."""
    monkeypatch.setenv("FASTAPI_ENV", "prod")
    with pytest.raises(ValidationError, match="Unknown FASTAPI_ENV"):
        Settings()


def test_non_production_environment_boots_without_a_secret_key(monkeypatch):
    """Control: the guard is production-only.

    Bare ``Settings`` is used rather than ``DevelopmentSettings`` because the
    subclass ``model_post_init`` calls ``setup_logging()``, which strips every
    handler off the root logger for the rest of the pytest session.
    """
    monkeypatch.setenv("FASTAPI_ENV", "development")
    assert Settings().DEBUG is not None


def test_production_boots_once_a_secret_key_is_supplied(monkeypatch):
    """Positive control: the guard fires on absence, not on production itself."""
    monkeypatch.setenv("FASTAPI_ENV", "production")
    monkeypatch.setenv("SECRET_KEY", "x" * 32)
    assert Settings().SECRET_KEY == "x" * 32
