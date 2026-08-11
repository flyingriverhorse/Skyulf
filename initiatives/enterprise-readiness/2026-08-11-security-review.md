# Enterprise Readiness — Dedicated Security Review

**Date:** 2026-08-11
**Method:** Read-only static analysis via the `security-review` specialist
agent, scoped to high-confidence, exploitable findings only (not general code
quality — see [technical-debt-deep-dive.md](2026-08-11-technical-debt-deep-dive.md)
for that). Covered: injection (SQL/command/path/deserialization), SSRF,
auth/session/CORS, secrets handling, file-upload risks (zip-slip), dependency
versions, and frontend XSS.

## Executive summary

No SQL injection, command injection, unsafe deserialization of
externally-supplied data, frontend XSS, or committed secrets/`.env` files
were confirmed. **Two real, Medium-severity SSRF vulnerabilities were
found and verified** — both stem from the same root cause: a datasource's
user-supplied S3 configuration is sanitized in one code path
(`S3Connector`) but **not** in two others (EDA analysis, pipeline
resolution/execution), letting an attacker who can create/edit a
datasource redirect the server's outbound S3 requests to an arbitrary
host (SSRF against internal infrastructure, cloud metadata endpoints,
etc.).

## Findings

### SEC-01 — SSRF via EDA endpoint's unsanitized `endpoint_url`

**Severity: Medium | Confidence: 9/10**

`backend/eda/router.py:283-317` reads a datasource's top-level
`config_creds.get("endpoint_url")` and forwards it straight into the
Pandas/Polars S3 storage options used to load the file for
decomposition/EDA analysis (`backend/eda/router.py:320-355`,
`backend/services/data_service.py:86-110`):

```python
"endpoint_url": config_creds.get("endpoint_url"),
...
storage_options = {
    "key": creds.get("aws_access_key_id") or creds.get("key"),
    "secret": creds.get("aws_secret_access_key") or creds.get("secret"),
    "token": creds.get("aws_session_token") or creds.get("token"),
    "endpoint_url": creds.get("endpoint_url"),
}
```

`DataSourceCreate.config` is an unrestricted, caller-supplied dictionary
(`backend/data_ingestion/schemas/ingestion.py:27-31`) persisted verbatim by
`/api/ingestion/database`. `S3Connector` (the "correct" path elsewhere in
the codebase) explicitly strips request-provided endpoint URLs to prevent
exactly this attack — but the EDA decomposition/analysis path bypasses
`S3Connector` and its guard entirely.

**Exploit scenario:** an attacker registers an S3-type datasource with
`config.endpoint_url` (or nested `client_kwargs.endpoint_url`) pointed at
an internal host (e.g. `http://169.254.169.254` for cloud metadata, or an
internal admin service), then triggers EDA/decomposition on that
datasource. The server-side S3 client issues requests to the
attacker-chosen host using the server's network position — classic SSRF,
with the added risk of credential/metadata exfiltration if pointed at a
cloud metadata endpoint.

**Fix:** apply one centralized S3-storage-options sanitizer immediately
before every S3 client construction (there should be exactly one code path
that builds `storage_options`, reused by connector, EDA, and pipeline
resolution). It must strip both top-level and nested endpoint-routing keys
from anything datasource/user-supplied and only ever set the endpoint from
server (operator) configuration.

### SEC-02 — SSRF via nested `client_kwargs.endpoint_url` in pipeline resolution

**Severity: Medium | Confidence: 9/10**

`backend/data/catalog.py:226-242` strips only the top-level
`endpoint_url`/`aws_endpoint_url` keys, leaving a nested
`client_kwargs.endpoint_url` untouched:

```python
opts.pop("endpoint_url", None)
opts.pop("aws_endpoint_url", None)
configured_endpoint = get_settings().AWS_ENDPOINT_URL
if configured_endpoint:
    if "client_kwargs" not in opts:
        opts["client_kwargs"] = {}
    opts["client_kwargs"]["endpoint_url"] = configured_endpoint
```

But this sanitizer is only reached on some code paths. Pipeline
resolution forwards a datasource's persisted `storage_options` unchanged
when present:

```python
if str(path).startswith("s3://") and ds.config and "storage_options" in ds.config:
    return cast(dict[str, Any], ds.config["storage_options"])
```
(`backend/ml_pipeline/resolution.py:55-57`)

Those raw options are used to construct `S3Catalog`
(`backend/data/catalog.py:518-529`), which passes `client_kwargs` straight
to `s3fs.S3FileSystem` (`:175-201`) — and `s3fs`/`boto3` will honor a
supplied `client_kwargs.endpoint_url` to route all S3 traffic to that host.

**Exploit scenario:** an attacker sets
`{"storage_options": {"client_kwargs": {"endpoint_url": "http://internal-host"}}}`
on a datasource's config, then runs a pipeline preview or execution that
reads from it. The resulting S3 client sends requests to the
attacker-chosen host.

**Fix:** never trust a datasource-provided `storage_options` blob
verbatim for transport-routing keys. Reconstruct `client_kwargs` /
`endpoint_url` from trusted server configuration only, on every path that
builds an S3 filesystem/client — including pipeline resolution, not just
the connector and EDA sanitizer paths found in SEC-01.

## What was checked and found clean (high confidence)

- **SQL injection**: no raw string-built SQL with unsanitized user input found; ORM usage appears parameterized throughout the sampled routes.
- **Command injection**: no `subprocess`/`os.system` calls with unsanitized user input found.
- **Unsafe deserialization**: no externally-reachable `pickle.load`/`joblib.load`/`torch.load` path accepting untrusted uploaded artifacts was confirmed exploitable in this pass (note: this is a *narrower* claim than the earlier `weights_only` finding in the DL-plan rubber-duck review — see `deep-learning/2026-08-11-findings.md` — which flagged the *planned* DL artifact-loading design, not existing code, as a future risk; both should be tracked).
- **Zip-slip / archive extraction**: no exploitable path traversal in archive extraction confirmed in this pass.
- **Frontend XSS**: no unescaped `dangerouslySetInnerHTML`/`innerHTML` assignment of user-controlled content found.
- **Committed secrets**: no `.env` file or hardcoded credential found in the current tree or reachable git history in this pass.
- **CORS/JWT**: no wildcard-origin-plus-credentials misconfiguration or JWT algorithm-confusion/missing-signature-verification issue confirmed.

These are scoped to what was reachable via static reading in this pass —
they reduce risk confidence but do not constitute a certified clean bill of
health (e.g., a full dependency CVE scan against a live vulnerability
database was not performed; the existing CI `security.yml` OSV scan and
CodeQL, confirmed present by the testing/CI audit, are the actual ongoing
control for this).

## Prioritized top findings

1. **SEC-01 — EDA endpoint_url SSRF** (Medium, high confidence) — fix first, smallest scope.
2. **SEC-02 — Pipeline resolution nested client_kwargs SSRF** (Medium, high confidence) — same root cause, fix together with SEC-01 via one shared sanitizer.
3. No further confirmed high-confidence findings this pass.

## Recommended remediation approach

Both findings share one fix: extract a single `sanitize_s3_storage_options()`
helper (or equivalent) that is the *only* function permitted to construct
S3 client/filesystem options anywhere in the backend, used identically by
`S3Connector`, the EDA analysis path, and pipeline resolution/execution.
Add a regression test asserting that a datasource-supplied `endpoint_url`
or `client_kwargs.endpoint_url` is never present in the options actually
passed to `s3fs`/`boto3`, regardless of entry point.
