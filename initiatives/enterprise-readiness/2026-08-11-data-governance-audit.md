# Enterprise Readiness — Data Governance, Privacy & Compliance Audit

**Date:** 2026-08-11  
**Scope:** Customer-data governance controls in the FastAPI backend, `skyulf-core`
profiling library, and React canvas. This complements, rather than repeats, the
authentication/tenancy findings in
[2026-08-11-backend-blockers.md](2026-08-11-backend-blockers.md).

## Executive assessment

Skyulf has useful building blocks (limited PII *detection*, dataset deletion
that attempts physical erasure, redaction of sensitive source-config fields in
API responses, and optional Sentry configured not to send default PII). It does
not yet provide the controls an enterprise buyer would expect for governed
customer data: comprehensive immutable audit evidence, enforced retention and
DSAR workflows, encryption at rest, residency policy enforcement, or
tenant-scoped data access logging. This is **not ready** to give affirmative
SOC 2 or GDPR control statements.

Severity below is the priority as an **enterprise procurement/sales blocker**,
not a statement of exploitability.

## Findings

### DG-01 — PII identification is narrow, advisory-only, and has no masking workflow

**Priority: High | Effort: Medium**

The profiler does identify likely email addresses and phone numbers from the
first 20 non-null values of Text/Categorical columns
(`skyulf-core/skyulf/profiling/_analyzer/text.py:106-123`,
`skyulf-core/skyulf/profiling/_analyzer/column.py:188-217,246-266`). It emits
a `PII` error alert, rather than applying a policy or changing the data.

This is not a general sensitive-data control:

- It does not identify names, SSNs/national identifiers, payment-card numbers,
  addresses, dates of birth, health data, or custom regulated fields.
- It has no confidence/evidence record, column classification/approval state,
  data catalogue label, scan history, or detection at ingestion time.
- There is no masking, tokenization, pseudonymization, access restriction, or
  export suppression for a detected column. The only nearby protection is an
  explicit caller-provided exclusion from profiling samples
  (`skyulf-core/skyulf/profiling/analyzer.py:512-516`), not automatic PII
  handling.

**Remediation:** implement configurable detectors (name + value patterns,
including Luhn/SSN checks), persisted column sensitivity classifications and
review/override workflow. Apply a policy at preview, profiling, training, and
export boundaries; offer irreversible masking/tokenization and record every
override.

### DG-02 — Dataset deletion attempts physical erasure, but there is no retention, DSAR, or verified purge program

**Priority: Critical | Effort: Large**

This is the documented deliberate tradeoff already noted in the backend audit:
local file deletion is attempted before the `DataSource` row is removed
(`backend/data_ingestion/service.py:83-120`). If `unlink()` fails, the DB row
is still deleted and an orphan is logged for manual cleanup
(`backend/data_ingestion/service.py:108-116`). That is reasonable
user-facing behavior, but does not furnish provable erasure; there is no
reconciliation sweep.

There is no customer-data retention schedule or automated N-day deletion.
`ENABLE_RETENTION` is only an unused setting flag
(`backend/config/mixins/files.py:41-44`). The scheduled retention task deletes
only `ErrorEvent` records after 30 days
(`backend/monitoring/tasks.py:14-38`;
`backend/config/mixins/celery.py:12-13`). No organization-wide data-subject
access/export/delete workflow exists. In particular, there is no
organization/tenant ownership model on which to scope such a request, as
documented separately in `2026-08-11-backend-blockers.md`.

**Remediation:** define data classes and retention periods; implement scheduled,
tenant-scoped lifecycle deletion across source files, S3 objects, EDA reports,
pipeline versions, artifacts, caches, backups, and logs. Make deletion
idempotent, track completion/exception evidence, reconcile orphaned objects,
and provide authenticated DSAR export/delete workflows with legal-hold support.

### DG-03 — “Audit Log” is pipeline-save history, not a compliance audit trail

**Priority: Critical | Effort: Large**

The UI accurately describes itself as an audit trail for *canvas pipeline
saves* (`frontend/ml-canvas/src/pages/AuditLogPage.tsx:1-15,340-346`). Its
backend is a derived diff of append-only `PipelineVersion` snapshots, explicitly
avoiding a dedicated audit table (`backend/ml_pipeline/_internal/_routers/pipelines_io.py:255-263`).
Each snapshot records only the pipeline graph, timestamp, optional user ID, and
save metadata (`backend/ml_pipeline/_services/pipeline_versions_service.py:45-78`).

It does **not** establish a durable, immutable trail for:

- dataset uploads, reads, samples, downloads/exports, deletion, or failed
  access;
- model training, registry changes, deployment/promotion/rollback, inference,
  or artifact retrieval;
- organization membership, roles, permissions, API keys, or configuration
  changes; and
- administrator/auditor viewing or exporting audit records.

HTTP middleware logs method, full URL, IP, user agent, status, and duration
(`backend/middleware/logging.py:41-87`), but no authenticated actor, resource
classification, authorization decision, durable append-only store, integrity
protection, or retention policy. It is operational request logging, not a
compliance audit system.

**Remediation:** create an append-only audit-event store with event ID,
timestamp, actor/service principal, organization/workspace, request/correlation
ID, action, resource/type/ID, purpose/outcome, IP, and before/after metadata
that never contains raw customer data or secrets. Emit from a central
authorization/data-access layer and all deployment/admin mutations; make it
queryable/exportable to SIEM with retention, access controls, and
tamper-evidence/WORM storage.

### DG-04 — Dataset read and export activity is not access-audited

**Priority: Critical | Effort: Medium**

The dataset sample endpoint returns data directly
(`backend/data_ingestion/router.py:70-81`), and the export endpoint reads a
sample and returns CSV or Parquet as an attachment
(`backend/data_ingestion/router.py:95-134`). Neither emits a data-access audit
event. The frontend exposes this export operation
(`frontend/ml-canvas/src/core/api/datasets.ts:210-230`).

The maximum export is 50,000 rows per request
(`backend/data_ingestion/router.py:97-105`), so it is a partial per-dataset
download rather than an organization portability feature. It also means a
customer cannot later determine who viewed or downloaded which records.

**Remediation:** require a scoped principal and authorize each preview/sample,
query, download/export, and artifact retrieval. Log successful and denied
events with dataset ID/version, row/column scope, destination/export format and
request ID; add rate/volume monitoring and export approvals where required.

### DG-05 — No enforceable data-residency policy or organization-level portability export

**Priority: High | Effort: Large**

Local uploads default to `uploads/data`
(`backend/config/mixins/files.py:13-18`). Artifact storage can be configured to
use an S3 bucket and AWS region (`backend/config/mixins/aws.py:7-15`;
`backend/ml_pipeline/artifacts/factory.py:157-174`), but the application has no
tenant-level region selection, allowed-region policy, residency metadata,
cross-region-transfer control, or proof/reporting mechanism. A global operator
can choose a bucket/region; that is infrastructure configuration, not a GDPR
data-residency control.

The only identified data portability path is the limited, individual-dataset
CSV/Parquet export in DG-04. There is no one-request user/organization export
covering datasets, raw files, metadata, models/artifacts, pipelines, and audit
history.

**Remediation:** define supported deployment/residency regions; bind every
tenant and storage key to a residency region, deny incompatible connectors and
transfers, and expose residency/subprocessor documentation. Add an asynchronous,
authorized organization export with a manifest, checksums, expiry, encryption,
and audit events.

### DG-06 — Customer data and connection credentials have no application-level encryption at rest; TLS is not enforced by the app

**Priority: Critical | Effort: Large**

Uploaded data is streamed in plaintext to the local filesystem
(`backend/data_ingestion/service.py:430-459`) and defaults to a local SQLite
database (`backend/config/mixins/database.py:11-12`). There is no file,
field, or database encryption implementation in the repository. Although the
`DataSource.credentials` comment says “encrypted in production,” it is a JSON
column (`backend/database/models.py:104-109`), with no encryption routine.
Response serialization does remove selected sensitive config keys
(`backend/data_ingestion/schemas/ingestion.py:6-24,65-70`), which prevents one
API disclosure path but does not encrypt stored values. S3 artifact writes do
not request SSE/KMS encryption (`backend/ml_pipeline/artifacts/s3.py:118-128`);
any bucket-default encryption is external, optional infrastructure policy.

For transit, PostgreSQL SSL mode/root certificate are optional configuration
fields (`backend/config/mixins/database.py:43-51,75-84`), but HTTPS redirect,
certificate configuration, and TLS-required middleware are absent. The
application’s default OpenAPI server is explicitly `http://`
(`backend/main.py:135-166`). Production headers define HSTS
(`backend/config/environments.py:40-50`), but no code applies
`SECURITY_HEADERS`; therefore that declaration does not enforce TLS.

**Remediation:** use managed encrypted storage with per-tenant keys/KMS,
encrypted database volumes/managed DB encryption, and envelope/field encryption
for connection credentials. Enforce TLS at an application-aware ingress and
between service dependencies; require verified DB TLS and S3 SSE-KMS policy,
rotate keys, and document key ownership/restore/backups.

### DG-07 — Third-party transfer inventory is incomplete; S3 and optional Sentry are real outbound paths

**Priority: High | Effort: Medium**

S3 ingestion reads customer data from `s3://` sources
(`backend/data_ingestion/connectors/s3.py:130-179`), and optional S3 artifact
storage sends model artifacts to the configured bucket
(`backend/ml_pipeline/artifacts/s3.py:118-128`). These are third-party data
processors/transfers when AWS or an S3-compatible provider is customer-external;
there is no processor inventory, DPA/subprocessor disclosure, residency guard,
or per-tenant connector egress approval.

Sentry is enabled when `SENTRY_DSN` is set. It uses a 10% trace sample rate and
sets `send_default_pii=False` (`backend/main.py:59-74`; worker equivalent:
`celery_worker.py:41-53`). This is a helpful default, but it is neither a
formal data-processing assessment nor a scrubber for custom exception/context
data. No frontend analytics SDK is declared in `frontend/ml-canvas/package.json`
or invoked in source reviewed. LLM provider keys/URLs are configuration-only in
the reviewed code (`backend/config/mixins/llm.py:6-35`); no runtime provider
call was found, so they should not be represented as an active data transfer
without further feature review.

**Remediation:** maintain a data-flow/subprocessor register and connector
allow-list by tenant/region, document Sentry configuration and scrubbing, make
telemetry opt-in/contractually configurable, and audit connector reads/writes.

## Framework readiness

### SOC 2 Type II

**Single biggest blocker: absence of comprehensive, actor-attributed,
tamper-evident audit evidence for customer-data access and administrative
changes (DG-03/DG-04).** A Type II examination needs sustained operating
evidence, not merely a pipeline-save history and local request logs. The
separately documented missing authentication/tenant isolation compounds this:
even a new event table would not have a trustworthy principal or organization
scope until those controls exist.

Additional material gaps are retention/deletion evidence (DG-02), encryption
and key-management controls (DG-06), vendor/telemetry governance (DG-07), and
formal policies/control ownership. A readiness project should establish and
operate the control set long enough to collect evidence before claiming Type II
readiness.

### GDPR

**Single biggest blocker: no tenant-/subject-scoped lifecycle capable of
locating, exporting, and demonstrably deleting personal data across all copies
(DG-02).** The platform lacks organization scoping and a DSAR mechanism; it
cannot reliably satisfy access, erasure, or accountability requests. Narrow PII
alerts are not data mapping/classification, and there is no enforceable
residency/transfer governance (DG-01/DG-05/DG-07).

This is a product-control assessment, not legal advice. GDPR readiness also
requires the controller/processor role analysis, lawful basis, DPA,
subprocessor terms, privacy notices, retention schedule, records of
processing, breach process, and deployment-specific technical measures.

## Prioritized top five enterprise compliance-review blockers

1. **Critical — Build tenant-scoped identity, authorization, and an
   append-only audit system for every data read/export and administrative/model
   action.** Without actor and organization context, SOC 2 evidence is not
   credible.
2. **Critical — Implement policy-driven retention, DSAR export/erasure, and
   verified deletion/reconciliation across files, DB rows, artifacts, caches,
   and backups.**
3. **Critical — Deliver encryption at rest and enforced TLS in a supported
   production reference architecture, including encrypted connection
   credentials and key management.**
4. **High — Add governed sensitive-data discovery/classification and masking
   controls before profiling, previewing, training, and exporting datasets.**
5. **High — Establish data-residency, subprocessor/telemetry, and
   organization-portability controls: regional tenant storage, egress
   allow-lists, disclosure/contract evidence, and complete export manifests.**
