"""OC-169: uncaught request errors must not bypass credential redaction."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

import backend.main as main_module
from backend.data_ingestion.connectors.s3 import S3Connector

_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"
_SECRET = "oc169-dummy-secret"
_SIGNATURE = "oc169-dummy-signature"
_PRESIGNED = (
    f"https://bucket.s3.amazonaws.com/obj.csv?AWSAccessKeyId={_ACCESS_KEY}"
    f"&Expires=1780000000&Signature={_SIGNATURE}"
)
_LOGGERS = {"backend.middleware.error_handler", "backend.middleware.logging", "backend.main"}


def _app(endpoint):
    """Use the production middleware order without startup or database work."""
    app = FastAPI()
    app.add_api_route("/probe", endpoint, methods=["GET"])
    main_module._add_exception_handlers(app)
    main_module._add_middleware(
        app, SimpleNamespace(ALLOWED_HOSTS=[], CORS_ORIGINS=["https://canvas.test"])
    )
    return app


def _assert_logs_redacted(caplog):
    """Check plain output, structured extras and exception-formatter inputs."""
    records = [record for record in caplog.records if record.name in _LOGGERS]
    assert records
    formatter = logging.Formatter("%(levelname)s %(message)s")
    for record in records:
        surfaces = formatter.format(record) + json.dumps(vars(record), default=str)
        for secret in (_ACCESS_KEY, _SECRET, _SIGNATURE, "oc169-source-only"):
            assert secret not in surfaces
        assert record.exc_info is None
    return records


def _raise_failure(mode):
    """Exercise causes, implicit context, notes and source lines in real tracebacks."""
    if mode == "source":
        raise RuntimeError("source diagnostic")  # password=oc169-source-only
    if mode == "notes":
        error = RuntimeError("noted diagnostic")
        error.add_note(f"secret={_SECRET}")
        raise error
    if mode == "group":
        raise ExceptionGroup("parallel diagnostic", [ValueError(f"secret={_SECRET}")])
    try:
        raise ValueError(f"AccessDenied for {_PRESIGNED} secret={_SECRET}")
    except ValueError as cause:
        if mode == "cause":
            raise ConnectionError("outer diagnostic") from cause
        if mode == "context":
            raise ConnectionError("outer diagnostic")  # noqa: B904 - exercise implicit chaining
        raise


@pytest.mark.parametrize("mode", ["direct", "cause", "context", "group", "source", "notes"])
def test_uncaught_failure_redacts_every_log_surface(mode, caplog):
    """The full middleware stack must scrub the message and every traceback representation."""

    async def fail():
        """Raise the selected error from a real HTTP endpoint."""
        _raise_failure(mode)

    app = _app(fail)
    with caplog.at_level(logging.DEBUG, logger="backend"):
        response = TestClient(app).get("/probe", headers={"Origin": "https://canvas.test"})

    assert response.status_code == 500
    assert response.json() == {
        "success": False,
        "message": "Internal server error",
        "error": "An unexpected error occurred. Please try again later.",
        "request_id": response.headers["X-Request-ID"],
    }
    assert response.headers["Access-Control-Allow-Origin"] == "https://canvas.test"
    records = _assert_logs_redacted(caplog)
    error_record = next(record for record in records if record.name.endswith("error_handler"))
    assert error_record.request_id == response.headers["X-Request-ID"]
    assert "_raise_failure" in error_record.traceback
    assert "[REDACTED]" in error_record.traceback
    assert error_record.traceback in error_record.getMessage()


def test_s3_connection_failure_stays_redacted_through_http_stack(monkeypatch, caplog):
    """The real connector's chained exception must remain scrubbed by both middleware layers."""
    connector = S3Connector(_PRESIGNED)

    async def failed_schema():
        """Represent the external S3 failure without making a network call."""
        raise RuntimeError(f"SignatureDoesNotMatch secret={_SECRET}")

    async def fail():
        """Run the connector's actual exception wrapping and logging."""
        await connector.connect()

    monkeypatch.setattr(connector, "get_schema", failed_schema)
    app = _app(fail)
    with caplog.at_level(logging.DEBUG, logger="backend"):
        response = TestClient(app).get("/probe")
    assert response.status_code == 500
    records = _assert_logs_redacted(caplog)
    error_record = next(record for record in records if record.name.endswith("error_handler"))
    assert "SignatureDoesNotMatch" in error_record.traceback
    assert "ConnectionError" in error_record.traceback


@pytest.mark.parametrize("status_code", [200, 422, 500])
def test_request_metadata_is_redacted_without_changing_request(status_code, caplog):
    """URL/header secrets must not leak on success, handled errors or unhandled errors."""
    seen = {}

    async def endpoint(request: Request):
        """Observe the original request before completing or raising."""
        seen["signature"] = request.query_params["Signature"]
        seen["user_agent"] = request.headers["user-agent"]
        if status_code == 500:
            raise RuntimeError("metadata diagnostic")
        if status_code == 422:
            raise HTTPException(status_code=422, detail="invalid column")
        return {"ok": True}

    app = _app(endpoint)
    with caplog.at_level(logging.DEBUG, logger="backend"):
        response = TestClient(app).get(
            f"/probe?Signature={_SIGNATURE}&page=2",
            headers={"user-agent": f"client secret={_SECRET}"},
        )
    assert response.status_code == status_code
    assert seen == {"signature": _SIGNATURE, "user_agent": f"client secret={_SECRET}"}
    records = _assert_logs_redacted(caplog)
    assert all("page=2" in record.url for record in records)
    if status_code != 500:
        assert float(response.headers["X-Process-Time"]) >= 0
    assert response.headers["X-Request-ID"]


@pytest.mark.parametrize("mode", ["direct", "cause"])
async def test_general_exception_handler_redacts_log_and_persisted_diagnostics(
    mode, monkeypatch, caplog
):
    """The fallback handler must use the supplied exception and scrub its persistence payload."""
    app = FastAPI()
    main_module._add_exception_handlers(app)
    record_error = AsyncMock()
    monkeypatch.setattr(main_module, "_record_error", record_error)
    try:
        _raise_failure(mode)
    except (ValueError, ConnectionError) as error:
        captured_error = error
    request = Request(
        {"type": "http", "method": "GET", "path": f"/probe/{_ACCESS_KEY}", "headers": []}
    )
    with caplog.at_level(logging.ERROR, logger="backend"):
        response = await app.exception_handlers[Exception](request, captured_error)
    assert response.status_code == 500
    records = _assert_logs_redacted(caplog)
    record_error.assert_awaited_once()
    details = record_error.call_args.kwargs
    assert details["route"] == "/probe/[REDACTED]"
    assert details["error_type"] == type(captured_error).__name__
    assert ("outer diagnostic" if mode == "cause" else "AccessDenied") in details["message"]
    assert "AccessDenied" in details["traceback"]
    assert "[REDACTED]" in details["traceback"]
    assert details["traceback"] in records[0].getMessage()
    assert all(secret not in json.dumps(details) for secret in (_ACCESS_KEY, _SECRET, _SIGNATURE))


def test_handled_http_error_keeps_status_and_request_headers():
    """A normal client error must not be turned into an unhandled server error."""

    async def invalid():
        """Return an ordinary application validation error."""
        raise HTTPException(status_code=422, detail="invalid column")

    response = TestClient(_app(invalid)).get("/probe")
    assert response.status_code == 422
    assert response.headers["X-Request-ID"]
    assert float(response.headers["X-Process-Time"]) >= 0
