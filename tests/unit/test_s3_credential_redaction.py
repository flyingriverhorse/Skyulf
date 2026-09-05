"""OC-150 — S3 credential redaction keys off value shape, not setting names.

The two byte-identical ``_sanitize_error`` copies triggered only on four literal
key names, matched case-sensitively, and replaced the *entire* message when one
hit. Both directions were wrong: real AWS formats contain none of those four
strings — ``AWSAccessKeyId=`` does not contain ``key=`` — so S3's own 403 body,
any presigned URL and an s3fs options dict repr were logged verbatim, while a
benign object-key message lost all of its text.

A presigned URL is a bearer credential: whoever can read the log can replay it
against the object until it expires, with no AWS account required.
"""

import logging
import sys
import types

import pytest

from backend.data_ingestion.connectors.s3 import S3Connector
from backend.ml_pipeline.artifacts.s3 import S3ArtifactStore
from backend.utils.logging_utils import redact_credentials

ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
TEMP_KEY_ID = "ASIAIOSFODNN7EXAMPLE"
SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
SIGV2_SIGNATURE = "aBcDeFgHiJkLmNoPqRsTuVwXyZ0"
SIGV4_SIGNATURE = "9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a"
SESSION_TOKEN = "FQoGZXIvYXdzEBYaDH0123"

ALL_SECRETS = [
    ACCESS_KEY_ID,
    TEMP_KEY_ID,
    SECRET_ACCESS_KEY,
    SIGV2_SIGNATURE,
    SIGV4_SIGNATURE,
    SESSION_TOKEN,
]

PRESIGNED_SIGV2 = (
    f"https://bucket.s3.amazonaws.com/obj.csv?AWSAccessKeyId={ACCESS_KEY_ID}"
    f"&Expires=1780000000&Signature={SIGV2_SIGNATURE}%3D"
)
PRESIGNED_SIGV4 = (
    "https://bucket.s3.amazonaws.com/obj.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    f"&X-Amz-Credential={ACCESS_KEY_ID}%2F20260905%2Fus-east-1%2Fs3%2Faws4_request"
    f"&X-Amz-Date=20260905T000000Z&X-Amz-Signature={SIGV4_SIGNATURE}"
)

LEAK_CASES = {
    "s3_403_xml_body": (
        "<Error><Code>SignatureDoesNotMatch</Code>"
        "<Message>The request signature we calculated does not match</Message>"
        f"<SignatureProvided>{SIGV2_SIGNATURE}=</SignatureProvided>"
        "<StringToSign>GET\n\n\n1780000000\n/bucket/obj</StringToSign>"
        f"<AWSAccessKeyId>{ACCESS_KEY_ID}</AWSAccessKeyId></Error>"
    ),
    "presigned_url_sigv2": PRESIGNED_SIGV2,
    "presigned_url_sigv4": PRESIGNED_SIGV4,
    "sts_temp_credentials": (f"AccessDenied for {TEMP_KEY_ID} session_token={SESSION_TOKEN}"),
    "s3fs_options_dict_repr": (
        f"S3FileSystem failed with {{'key': '{ACCESS_KEY_ID}', "
        f"'secret': '{SECRET_ACCESS_KEY}', 'region': 'us-east-1'}}"
    ),
    "options_json_repr": (
        f'{{"aws_access_key_id": "{ACCESS_KEY_ID}", '
        f'"aws_secret_access_key": "{SECRET_ACCESS_KEY}"}}'
    ),
    "bare_key_id_in_prose": f"Not authorized to perform sts:AssumeRole on {ACCESS_KEY_ID}",
}

BENIGN_CASES = {
    "timeout": "Generic connection timeout after 30s",
    "missing_bucket": "Unable to locate credentials for bucket=my-bucket in us-east-1",
    "key_error": "KeyError: 'column_x' while reading s3://bucket/path/file.parquet",
    "endpoint_url": "Could not connect to endpoint_url=https://s3.us-east-1.amazonaws.com",
}


@pytest.mark.parametrize("message", list(LEAK_CASES.values()), ids=list(LEAK_CASES))
def test_no_credential_survives_redaction(message):
    out = redact_credentials(message)
    leaked = [secret for secret in ALL_SECRETS if secret in out]
    assert not leaked, f"leaked {leaked} into: {out}"


@pytest.mark.parametrize("message", list(BENIGN_CASES.values()), ids=list(BENIGN_CASES))
def test_benign_message_pass_through_unchanged(message):
    assert redact_credentials(message) == message


def test_surrounding_diagnostic_survives():
    """Redaction scrubs in place; it must not discard the whole message.

    All-or-nothing redaction destroys the diagnostic, which is why an engineer
    is tempted to disable the control entirely.
    """
    out = redact_credentials(f"SignatureDoesNotMatch for {ACCESS_KEY_ID} on bucket=my-bucket")
    assert out == "SignatureDoesNotMatch for [REDACTED] on bucket=my-bucket"


def test_object_key_message_keeps_its_text():
    """A benign object key is scrubbed but the message stays readable.

    ``key`` is the access-key-id option name in both S3 modules, so its value
    cannot be trusted; the object path is logged separately via ``self.path``.
    The old code replaced this entire message with "redacted sensitive S3 error".
    """
    out = redact_credentials("NoSuchKey: The specified key does not exist. key=reports/2026/q3.csv")
    assert "NoSuchKey" in out
    assert "does not exist" in out
    assert "reports/2026/q3.csv" not in out


def test_redaction_is_idempotent():
    once = redact_credentials(PRESIGNED_SIGV4)
    assert redact_credentials(once) == once


def test_accepts_non_string_input():
    assert ACCESS_KEY_ID not in redact_credentials(RuntimeError(f"bad key {ACCESS_KEY_ID}"))


def test_both_modules_share_one_redactor():
    """A security control maintained in two copies drifts; only one gets fixed."""
    assert not hasattr(S3Connector, "_sanitize_error")
    assert not hasattr(S3ArtifactStore, "_sanitize_error")


def test_constructor_log_redacts_a_presigned_path(caplog):
    """``path`` is caller-supplied and may itself be a presigned URL."""
    with caplog.at_level(logging.INFO, logger="backend.data_ingestion.connectors.s3"):
        S3Connector(PRESIGNED_SIGV2)
    assert "Initialized S3Connector" in caplog.text
    assert ACCESS_KEY_ID not in caplog.text
    assert SIGV2_SIGNATURE not in caplog.text


@pytest.mark.asyncio
async def test_connect_failure_logs_redacted(monkeypatch, caplog):
    connector = S3Connector(PRESIGNED_SIGV2)

    async def _boom():
        raise RuntimeError(f"403 Forbidden for {PRESIGNED_SIGV2} secret={SECRET_ACCESS_KEY}")

    monkeypatch.setattr(connector, "get_schema", _boom)

    with (
        caplog.at_level(logging.ERROR, logger="backend.data_ingestion.connectors.s3"),
        pytest.raises(ConnectionError),
    ):
        await connector.connect()

    assert "S3 connection check failed" in caplog.text
    for secret in (ACCESS_KEY_ID, SIGV2_SIGNATURE, SECRET_ACCESS_KEY):
        assert secret not in caplog.text


def test_artifact_store_init_failure_logs_redacted(monkeypatch, caplog):
    fake_s3fs = types.ModuleType("s3fs")

    def _boom(**kwargs):
        raise RuntimeError(f"InvalidAccessKeyId: {ACCESS_KEY_ID} secret={SECRET_ACCESS_KEY}")

    fake_s3fs.S3FileSystem = _boom
    monkeypatch.setitem(sys.modules, "s3fs", fake_s3fs)

    with (
        caplog.at_level(logging.ERROR, logger="backend.ml_pipeline.artifacts.s3"),
        pytest.raises(RuntimeError, match="Failed to initialize S3 artifact storage"),
    ):
        S3ArtifactStore("my-bucket")

    assert "Failed to initialize S3 filesystem client" in caplog.text
    # The diagnostic survives; only the credentials are gone.
    assert "InvalidAccessKeyId" in caplog.text
    assert ACCESS_KEY_ID not in caplog.text
    assert SECRET_ACCESS_KEY not in caplog.text
