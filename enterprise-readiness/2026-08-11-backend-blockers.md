# Enterprise Readiness — Backend Blockers

**Date:** 2026-08-11
**Status:** Investigation complete (subagent audit, spot-checked against real code)
**Scope:** Backend/infrastructure blockers that would prevent enterprise
(multi-user organization, governance/security/scale guarantees) adoption of
Skyulf. Companion to
[2026-08-11-node-flexibility.md](2026-08-11-node-flexibility.md), which
covers consumer-facing node/canvas flexibility.

## How this was produced

A background agent audited the real codebase against 10 enterprise
readiness areas (auth, multi-tenancy, database/scale, secrets, audit
logging, observability, API stability, deployment/HA, data-at-rest
encryption, licensing/quotas). The most severe claims were independently
spot-checked against the actual files before being recorded here — all
checked out, including one area where the code **already contains a
self-documented `KNOWN-GAP` comment** acknowledging the issue.

## 1. Authentication & Authorization — **BLOCKER, Critical, Large effort**

- `DataSource.has_permission()` unconditionally returns `True` —
  `backend/database/models.py:156-160`, literally commented `# Placeholder`.
  (Corrected via rubber-duck review: this method and the adjacent
  `is_admin` placeholder belong to `DataSource`, not `User` — `User` has no
  `has_permission` method at all, and separately has a real, persisted
  `is_admin` boolean column at `models.py:57` that this placeholder does
  not affect.)
- Data-ingestion endpoints operate with **no real user context**: list
  endpoints pass `user_id=None` (`backend/data_ingestion/router.py:44,55`)
  and upload/create endpoints hardcode `user_id = 1` with an explicit
  in-code comment: `# KNOWN-GAP: Auth not implemented yet — hardcoded
  user_id=1.` (`router.py:148-151, 170-173`).
- The root `pyproject.toml` declares an `auth` optional-dependency group
  (`python-jose`, `passlib`, `bcrypt`), but **it's never installed**:
  `requirements-fastapi.txt` (the file actually used to build the running
  app) contains none of those packages, and `skyulf-core/setup.py` has no
  matching extra either — the auth group is declared but effectively
  orphaned.
- `backend/config/mixins/security.py:43-54` — fallback/default security
  config is dev-only and unwired to any enforcement path.

**Recommendation:** Build `backend/auth/` with real JWT/OIDC verification
middleware, a `CurrentPrincipal` FastAPI dependency required on every
customer-data route, real password hashing (wire up the already-declared
`passlib`/`bcrypt` deps), and role/permission checks replacing the
`DataSource.has_permission()` placeholder. For enterprise specifically, add
OIDC/SAML federation (Okta/Azure AD/Google Workspace) since large orgs
require SSO, not username/password.

## 2. Multi-Tenancy & Data Isolation — **BLOCKER, Critical, Large effort**

- No `Organization`/`Tenant`/`Workspace` model exists anywhere in
  `backend/database/models.py` — confirmed by listing every model class
  (`User`, `DataSource`, `FeatureEngineeringPipeline`, `PipelineVersion`,
  `MLJob`, `Deployment`, etc.): none carry a tenant/workspace foreign key.
- Combined with §1, this means **every dataset, pipeline, and job is
  visible and mutable by anyone who can reach the API** — there is no
  logical partition between customers today. This is the single largest
  gap for enterprise SaaS adoption, more severe than missing auth alone,
  because even adding auth without adding tenancy would only add a login
  screen in front of one shared global workspace.
- Job APIs retrieve/cancel/promote by bare global ID with no ownership
  predicate (`backend/ml_pipeline/_internal/_routers/jobs.py:105-124,
  173-191`).

**Recommendation:** Introduce `Organization`/`Membership`/`Workspace`
models; add a non-nullable `workspace_id` to every customer-owned table and
every artifact storage key; enforce it via a mandatory scoped-repository
dependency (not ad-hoc per-route checks, which are easy to miss). Consider
PostgreSQL Row-Level Security as defense-in-depth once §3 makes PostgreSQL
mandatory.

## 3. Database & Scale — **BLOCKER, Critical, Large effort**

- SQLite is the default database (`backend/config/mixins/database.py:11-21,
  40-42`; `docker-compose.yml:25` forces it), using `StaticPool`
  (`backend/database/engine.py:19,55-69`) — confirmed: SQLite is not safe
  for concurrent multi-writer production use, and this matches what the
  (unmerged) Ray migration docs on branch `080` already independently flag
  as a blocker for their own distributed-compute plan.
- PostgreSQL async pooling exists (`engine.py:71-80`), but the sync
  code path rewrites the URL to `psycopg2` (`engine.py:91-98`) while
  normal installs only bring in `asyncpg` — `psycopg2-binary` is only in the
  optional `all` extra (`pyproject.toml:90-93`), so a "PostgreSQL-configured"
  deployment can still fail on the sync path unless that extra is
  separately installed.
- **No Alembic** (or equivalent migration tool). Startup runs
  `create_all()` plus hand-written, exception-swallowing `ALTER` statements
  (`engine.py:169-244`) — this is unsafe for coordinated, zero-downtime
  production rollouts and has no rollback story.

**Recommendation:** Make PostgreSQL the only supported production database
(keep SQLite for local dev only, clearly labeled); add `psycopg2`/`psycopg`
as a core (not optional) dependency if the sync path stays; replace the
startup DDL block with Alembic migrations run as a singleton pre-deploy job,
not on every app boot.

## 4. Secrets Management — **BLOCKER, High severity, Medium effort**

- Settings load plaintext from `.env` (`backend/config/base.py:51-55`); DB
  password and AWS static keys are plain settings fields
  (`backend/config/mixins/database.py:43-51`,
  `backend/config/mixins/aws.py:7-13`).
- `SECRET_KEY` auto-generates randomly if unset, and production mode only
  checks that *something* was explicitly set, not its strength/rotation
  (`backend/config/mixins/security.py:12-18`, `backend/config/base.py:
  173-189`); `.env.example` ships a placeholder secret.
- S3 credentials are passed directly into client constructors
  (`backend/ml_pipeline/artifacts/factory.py:157-168`) — no Vault/AWS
  Secrets Manager/workload-identity (IAM role) integration.

**Recommendation:** Add a secrets-provider abstraction (env var today,
pluggable to Vault/AWS Secrets Manager/K8s Secrets tomorrow); support IAM
instance-role/IRSA credentials for S3 instead of static keys in production;
add JWT signing-key rotation/versioning once real auth (§1) exists.

## 5. Audit Logging & Compliance — **BLOCKER, High severity, Large effort**

- The only audit-adjacent feature is a derived pipeline-version diff, not an
  immutable, cross-resource audit log
  (`backend/ml_pipeline/_internal/_routers/pipelines_io.py:255-262,
  308-335`).
- Request logging middleware records IP/URL but not an authenticated actor,
  and isn't tamper-resistant/durable (`backend/middleware/logging.py:
  41-87`).
- Dataset deletion is a deliberate, documented tradeoff rather than a bug:
  the code intentionally removes the file first, and if that fails it still
  deletes the DB row while emitting an operator-visible `ERROR` log
  flagging the orphaned file for manual cleanup (`backend/data_ingestion/
  service.py:83-120`, docstring at 87-93, error log at 109-116) —
  reasonable today, but there's still no automated reconciliation job, so
  it depends on someone watching logs. Only error events get a 30-day
  cleanup job (`backend/monitoring/tasks.py:14-38`); no equivalent
  retention/deletion policy exists for user data, and no reconciliation
  sweep exists for the orphaned-file case either.

**Recommendation:** Add an append-only audit-event table (actor, org,
request id, resource, action, outcome, timestamp) written on every
data-touching action; build data-retention and deletion workflows with
verifiable storage erasure (add an automated reconciliation sweep for the
orphaned-file case above); add DSAR (data subject access request)
export/delete support for GDPR/CCPA.

## 6. Observability — **Significant gap, Medium severity, Medium effort**

- **What exists:** optional Sentry with 10% trace sampling and PII disabled
  (`backend/main.py:59-74`, `celery_worker.py:41-53`); health/dependency
  check endpoints (`backend/health/routes.py:45-102`).
- **What's missing:** logging is unstructured, rotating local files
  (`backend/utils/logging_utils.py:70-84, 122-168`) — no JSON/structured
  logs to stdout, no correlation/tenant IDs threaded through. No
  Prometheus metrics, no OpenTelemetry tracing across API→Celery boundary.
  The readiness probe runs a trivial sklearn fit but doesn't actually verify
  DB/Redis connectivity (`health/routes.py:105-130`) — a probe that can
  report "ready" while the database is unreachable.

**Recommendation:** Structured JSON logging to stdout (12-factor style) with
correlation/tenant IDs; Prometheus metrics for HTTP, job queue, DB pool, and
storage; OpenTelemetry tracing propagated from API through Celery tasks;
fix the readiness probe to actually check DB/Redis.

## 7. API Stability, Rate Limiting, Security Headers — **Significant gap, Medium severity, Medium effort**

- No `/v1`-style API versioning — routes are mounted directly under mixed
  `/api`, `/data`, and root prefixes (`backend/main.py:372-391`), making any
  future breaking change disruptive to integrators.
- Rate limiting exists (`slowapi`) but keys only on remote IP
  (`backend/middleware/rate_limiter.py:4,17` — `get_remote_address`), not
  per-user/per-org, and is in-memory/single-process (won't work correctly
  behind multiple API replicas without a shared backing store).
- CORS/trusted-host middleware exists (`main.py:351-369`), but
  production security headers (HSTS, CSP, etc.) are only *configured*, never
  actually *installed* as middleware (`backend/config/environments.py:
  40-51, 75-89`).

**Recommendation:** Add explicit API versioning and a documented
deprecation policy; move rate limiting to a Redis-backed store keyed by
authenticated principal/org once §1/§2 exist; actually mount a
security-headers middleware in production, and make client-IP resolution
reverse-proxy-aware (`X-Forwarded-For` trust boundary).

## 8. Deployment & High Availability — **BLOCKER, High severity, Large effort**

- The production `Dockerfile` starts Uvicorn with **`--reload`**
  (`Dockerfile:33`) — a development flag that should never run in
  production (extra file-watching overhead, auto-restart on any file
  touch, not appropriate for a served container).
- `docker-compose.yml` is explicitly a development setup: bind-mounts the
  repo, uses SQLite, runs a single Celery worker with `--pool=solo`, and
  has no service healthchecks (`docker-compose.yml:2-48`).
- **No Kubernetes manifests or Helm chart exist.** The realtime event bus
  has a local single-process implementation
  (`backend/realtime/local_bus.py:1-12`) alongside a Redis pub/sub
  implementation that does support multi-process/multi-replica delivery
  (`backend/realtime/manager.py:64-76, 107-146`) — the multi-replica-ready
  piece exists but isn't the default and has no deployment story around it.

**Recommendation:** Ship a production entrypoint without `--reload`; provide
a reference production Compose/Helm chart wired to PostgreSQL/Redis/S3 with
liveness/readiness probes, resource requests/limits, HPA for the API and
worker autoscaling, and a documented backup/restore/DR runbook.

## 9. Data Residency & Encryption at Rest — **BLOCKER, High severity, Medium effort**

- Default uploads/artifacts live on local disk
  (`backend/config/mixins/files.py:13-39`,
  `backend/ml_pipeline/artifacts/factory.py:102-108`) — no encryption at
  rest for local storage.
- S3 support passes region/endpoint/static credentials but doesn't
  configure SSE-S3/SSE-KMS, customer-managed keys, or bucket policy
  (`backend/ml_pipeline/artifacts/factory.py:157-174`,
  `backend/ml_pipeline/artifacts/s3.py:118-128`).

**Recommendation:** Require encrypted, managed object storage in
production; support SSE-KMS with customer-managed keys and per-tenant
bucket prefixes/IAM boundaries; document regional data-placement controls
for customers with data-residency requirements (common enterprise ask,
especially EU customers).

## 10. Licensing, Billing & Quota Enforcement — **Significant commercial gap, Medium severity, Medium effort**

- The project is AGPL (backend/frontend) + Apache (core) with a documented
  commercial exception/enterprise-support tier
  (`COMMERCIAL-LICENSE.md:3-16, 25-29`) — this materially changes how
  "enterprise features" should be gated: they likely belong behind a
  server-side entitlement check tied to the commercial license, not just a
  feature flag.
- No usage-metering, quota, or per-organization entitlement model exists.
  Current limits are global operational caps (e.g. max parallel branches,
  `backend/config/mixins/celery.py:42-59`), not customer-specific
  entitlements.

**Recommendation:** Once §2 (multi-tenancy) exists, add an entitlement
service keyed by organization — storage caps, pipeline/job count limits,
concurrent-job/CPU-hour quotas — enforced at submission and ingestion time,
with usage events emitted for billing.

## Top 3 Blockers to Fix First

1. **Identity + organization-scoped authorization (§1 + §2 together).**
   Today, anonymous requests can read, modify, export, and cancel any
   customer's data and jobs — this alone makes multi-customer deployment
   unsafe regardless of any other feature work. These two areas must be
   designed and delivered together (auth without tenancy just adds a login
   screen in front of one shared global workspace).
2. **Production data/state plane (§3 + §9).** Mandatory PostgreSQL with
   real migrations, and encrypted managed object storage — the current
   SQLite/local-disk defaults are not a production-safe foundation for any
   of the other enterprise work to build on.
3. **Production operating model (§6 + §7 + §8).** HA deployment assets
   (Helm/K8s, no `--reload`), secrets-manager integration, structured
   logging/metrics/tracing — the operational maturity a production
   enterprise SRE team will expect before onboarding.

## Relationship to other in-flight plans

- The (unmerged, docs-only) Ray migration on branch `080` already
  identifies the SQLite/local-artifact-storage limitation independently,
  from a distributed-compute angle rather than a multi-tenancy angle — §3
  and §9 above are consistent with, and should be delivered alongside, that
  migration rather than duplicated.
- The [deep-learning integration plan](../deep-learning/README.md) assumes
  today's single-global-namespace job model; once multi-tenancy (§2) lands,
  DL job submission/artifact paths must also be updated to be
  workspace-scoped — noted here so it isn't missed when both efforts are
  in flight concurrently.
