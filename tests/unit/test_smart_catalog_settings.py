"""OC-183: resolve the default S3 bucket through Settings, including dotenv files."""

import os
from unittest.mock import MagicMock

import pytest

import backend.data.catalog as catalog_module
from backend.config.base import Settings


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch, tmp_path):
    """Keep the developer's real dotenv and exported bucket names out of the tests."""
    monkeypatch.chdir(tmp_path)
    for name in (*Settings.model_fields, "S3_BUCKET_NAME"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("name", ["AWS_BUCKET_NAME", "S3_BUCKET_NAME"])
@pytest.mark.parametrize("source", ["dotenv", "environment"])
def test_bucket_setting_initializes_smart_catalog(name, source, monkeypatch, tmp_path):
    """Both documented and legacy names must work without exporting dotenv values."""
    if source == "dotenv":
        (tmp_path / ".env").write_text(f"{name}=configured-bucket\n", encoding="utf-8")
    else:
        monkeypatch.setenv(name, "configured-bucket")
    settings = Settings()
    monkeypatch.setattr(catalog_module, "get_settings", lambda: settings)
    constructor = MagicMock()
    monkeypatch.setattr(catalog_module, "S3Catalog", constructor)
    catalog = catalog_module.SmartCatalog(session=MagicMock(), fs_catalog=MagicMock())
    assert settings.AWS_BUCKET_NAME == "configured-bucket"
    constructor.assert_called_once_with(bucket_name="configured-bucket")
    assert catalog.s3_catalog is constructor.return_value
    if source == "dotenv":
        assert name not in os.environ


@pytest.mark.parametrize("source", ["dotenv", "environment"])
def test_canonical_bucket_name_wins_within_one_source(source, monkeypatch, tmp_path):
    """Conflicting names must resolve predictably to the canonical AWS setting."""
    if source == "dotenv":
        (tmp_path / ".env").write_text(
            "AWS_BUCKET_NAME=canonical\nS3_BUCKET_NAME=legacy\n", encoding="utf-8"
        )
    else:
        monkeypatch.setenv("AWS_BUCKET_NAME", "canonical")
        monkeypatch.setenv("S3_BUCKET_NAME", "legacy")
    assert Settings().AWS_BUCKET_NAME == "canonical"


@pytest.mark.parametrize("env_name", ["AWS_BUCKET_NAME", "S3_BUCKET_NAME"])
def test_environment_bucket_overrides_dotenv_alias(env_name, monkeypatch, tmp_path):
    """A deployment environment override must beat the file even across aliases."""
    file_name = "S3_BUCKET_NAME" if env_name == "AWS_BUCKET_NAME" else "AWS_BUCKET_NAME"
    (tmp_path / ".env").write_text(f"{file_name}=file-bucket\n", encoding="utf-8")
    monkeypatch.setenv(env_name, "environment-bucket")
    assert Settings().AWS_BUCKET_NAME == "environment-bucket"


def test_no_bucket_keeps_filesystem_catalog(monkeypatch):
    """Local-only installations must not initialize the optional S3 dependency."""
    monkeypatch.setattr(catalog_module, "get_settings", Settings)
    constructor = MagicMock()
    monkeypatch.setattr(catalog_module, "S3Catalog", constructor)
    local_catalog = MagicMock()
    catalog = catalog_module.SmartCatalog(session=MagicMock(), fs_catalog=local_catalog)
    constructor.assert_not_called()
    assert catalog.s3_catalog is None
    assert catalog._get_catalog_for_path("local.csv") is local_catalog


def test_injected_s3_catalog_takes_precedence(monkeypatch):
    """Explicit catalog dependencies must not be replaced by global settings."""
    monkeypatch.setattr(catalog_module, "get_settings", MagicMock(side_effect=AssertionError))
    provided_catalog = MagicMock()
    catalog = catalog_module.SmartCatalog(
        session=MagicMock(), fs_catalog=MagicMock(), s3_catalog=provided_catalog
    )
    assert catalog.s3_catalog is provided_catalog


def test_missing_s3fs_keeps_optional_integration_disabled(monkeypatch):
    """Configuring a bucket must preserve graceful startup without the optional SDK."""
    settings = Settings(AWS_BUCKET_NAME="configured-bucket")
    monkeypatch.setattr(catalog_module, "get_settings", lambda: settings)
    constructor = MagicMock(side_effect=ImportError("s3fs missing"))
    monkeypatch.setattr(catalog_module, "S3Catalog", constructor)
    catalog = catalog_module.SmartCatalog(session=MagicMock(), fs_catalog=MagicMock())
    constructor.assert_called_once_with(bucket_name="configured-bucket")
    assert catalog.s3_catalog is None
