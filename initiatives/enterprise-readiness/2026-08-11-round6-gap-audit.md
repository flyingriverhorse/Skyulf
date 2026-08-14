# Enterprise Readiness — Round 6 Meta Gap-Check

**Date:** 2026-08-11
**Purpose:** A meta-audit of the 5 prior investigation rounds (18+
subagents) plus the two feasibility studies (code-escape-hatch,
training-visualization) and the two orthogonal initiatives
(deep-learning, ray-migration). This is **not** a re-investigation of
anything already covered — it targets blind spots the checklist in the
task explicitly called out, verifies each one for real against the live
code, and reports only genuine, previously-undocumented findings.

## Method

Read every document listed in the task in full, then grepped/viewed the
live codebase (`backend/`, `skyulf-core/`, `frontend/ml-canvas/`, repo
root) to confirm or refute each of the 10 checklist areas before writing
anything down. Three areas turned out to be already adequately covered
(see the closing section) and were not re-documented here. Six produced
genuine, verifiable findings below.

---

## Finding 1 — Model Registry & Deployment: only one model can ever be "live" platform-wide, with no per-pipeline/environment scoping

**Severity:** High (architectural, not cosmetic — blocks a core enterprise
use case)
**Effort to fix:** Large (schema + service + API + UI)

### Location / Evidence

- `backend/database/models.py:371-396` — the `Deployment` table has **no**
  `pipeline_id`, `workspace_id`, or `environment` column at all. It has
  `job_id`, `model_type`, `artifact_uri`, `is_active` (a bare boolean),
  `deployed_by`, `previous_deployment_id`.
- `backend/ml_pipeline/deployment/service.py:120-127` — deploying a new
  model unconditionally deactivates **every** other deployment in the
  entire table, with no scoping filter:
  ```python
  previous_deployment = await DeploymentService.get_active_deployment(session)
  await session.execute(
      update(Deployment).where(Deployment.is_active).values(is_active=False)
  )
  ```
- `backend/ml_pipeline/deployment/service.py:150-152` — `get_active_deployment`
  reads the single globally active row:
  ```python
  stmt = select(Deployment).where(Deployment.is_active).order_by(Deployment.created_at.desc())
  result = await session.execute(stmt)
  return result.scalars().first()
  ```
- `backend/ml_pipeline/deployment/api.py:45-56` (`GET /deployment/active`)
  and `:85-124` (`POST /deployment/predict`) both call this same
  singular getter — there is exactly one prediction endpoint for the
  whole platform, backed by exactly one active model at a time.
- Confirmed this was **not** the focus of the existing UX writeup: the
  redesign doc (`2026-08-11-redesign-existing-pages.md:210-238`) documents
  UI problems ("Deployment status is binary active/inactive with no
  health/traffic/environment/rollback... posture") but frames it as a
  page-design gap, not the underlying single-row data-model limitation
  that makes multi-model, multi-environment, or multi-pipeline serving
  structurally impossible today without a schema change.

### Actual vs. Expected

**Actual:** an org training 10 different pipelines (e.g., churn model,
fraud model, forecasting model) can have at most one of them actively
served for real-time prediction at any time — deploying model B silently
un-deploys model A, with no environment concept (dev/staging/prod), no
canary/shadow/A-B traffic split, and no per-model-type or per-pipeline
"active" slot.

**Expected for an enterprise SaaS ML platform:** each pipeline/use-case
should have its own independent deployment slot (and ideally
per-environment slots), so deploying a new fraud model doesn't take the
churn model offline.

### Recommendation

Add `pipeline_id` (and eventually `workspace_id`, `environment`) to
`Deployment`; scope `is_active`/`get_active_deployment`/`/predict` uniqueness
to `(pipeline_id, environment)` instead of the whole table. Do this as
part of Phase 0 multi-tenancy work (workspace_id lands then anyway) rather
than as a separate migration later. Treat this as a schema-level
prerequisite before Phase 5's Model Registry/Deployments redesign —
otherwise the redesigned UI would be documenting a UX for a capability
the backend doesn't actually support (multiple simultaneous
environments/rollback targets).

---

## Finding 2 — Zero licensing/billing/entitlement enforcement code exists anywhere (confirmed at the implementation level, not just strategically)

**Severity:** Medium-High (commercial-viability gap, not a security bug)
**Effort to fix:** Medium once Phase 0 tenancy lands

### Location / Evidence

Prior round-1 doc (`backend-blockers.md:222-238`) already flagged this
*strategically* ("No usage-metering, quota, or per-organization
entitlement model exists"). This round verified there is **exactly zero**
supporting code anywhere, confirming the gap is total, not partial:

```
grep -rln "license_key|LICENSE_KEY|validate_license|seat_limit|SEAT_LIMIT" backend/ frontend/  → no matches
grep -rln "class Plan|PlanTier|SubscriptionTier|pricing_tier" backend/               → no matches
grep -rln "class Usage|UsageEvent|usage_events|track_usage" backend/                → no matches
```

- `COMMERCIAL-LICENSE.md:1-29` describes the AGPLv3/commercial split at
  the **legal document** level only — there is no technical mechanism
  (license key, feature flag, seat cap) anywhere in the codebase that
  distinguishes an AGPLv3 self-hosted deployment from a commercially
  licensed one. Anyone can run the identical code either way; enforcement
  is 100% honor-system/legal.
- No `TrainingJob`/`Deployment`/`DataSource` row anywhere carries a
  `duration`, `compute_seconds`, or `cost` field — `started_at` exists
  (`backend/database/models.py:269`) but there is no `finished_at`-minus-
  `started_at` derived cost metric surfaced or stored anywhere (see
  Finding 4).

### Actual vs. Expected

**Actual:** the differentiation-strategy.md document discusses pricing
*positioning* (Bet-level business strategy) but this round confirms there
is no feasibility gap analysis of what it would take to actually
implement metering/enforcement — because there is truly nothing to build
on top of; it is a from-scratch data model + middleware + admin UI, not a
"wire up existing plumbing" job like several Phase 8 items.

**Expected:** at minimum, a `usage_events` table (job runs, storage bytes,
API calls) keyed to the future `workspace_id`, plus a `plan`/`entitlement`
row per organization, checked at submission time — this is table-stakes
for any tiered/paid SaaS offering and was not scoped as an implementation
task anywhere in the existing 5 rounds (only as a one-line Phase 1 item:
"Usage-metering/entitlement service tied to the commercial license
tier" in `master-fix-list.md:45`, with no supporting design).

### Recommendation

Add a dedicated design pass (like the code-escape-hatch and
training-visualization feasibility studies got) scoping: (a) what events
to meter (job-seconds, storage bytes, API calls, seats), (b) where to
enforce (submission-time check vs. post-hoc invoice), (c) minimal schema.
Sequence after Phase 0 (needs `workspace_id` to attach usage to).

---

## Finding 3 — No cost/FinOps visibility: a tenant cannot see what any pipeline run cost, and no cost data is computed or stored anywhere

**Severity:** Medium-High (directly promised by the checklist question,
zero existing investigation)
**Effort to fix:** Medium

### Location / Evidence

- `backend/database/models.py:269` — `started_at` exists on jobs, but a
  grep across the whole backend for cost-related fields returns nothing:
  ```
  grep -rln "cost\b|cpu_time|gpu_hour|compute_cost|billing_event|usage_event" backend/ --include=*.py → no matches (excluding this doc)
  ```
- The Jobs/Experiments pages (per `redesign-existing-pages.md`) show
  status/duration-adjacent data but no cost column is mentioned anywhere
  in any of the 5 rounds' page-redesign docs — `smooth-experience-fixes.md`,
  `redesign-existing-pages.md`, and `differentiation-strategy.md` all
  omit it.
- `initiatives/ray-migration/*.md` (all 8 files, ~7,300 lines total) —
  grepped for "cost" and found **zero** mentions anywhere. Ray migration
  is precisely the point at which GPU/CPU-second accounting becomes
  natural to add (Ray tasks/actors report resource usage), yet no design
  doc considers it.
- `initiatives/deep-learning/*.md` — GPU node scheduling is discussed at
  length (queueing, hardware selection) but never in terms of "what does
  this GPU-hour cost the tenant."

### Actual vs. Expected

**Actual:** no compute-cost estimation, attribution, or display exists in
any form — not even a naive `duration_seconds × $/hour` estimate.

**Expected:** enterprise buyers evaluating AutoML/no-code platforms
consistently cite cost predictability as a top concern (this mirrors the
"pricing opacity" complaint already surfaced externally in
`user-complaints-research.md`, but that doc is about external pricing
perception, not an internal cost-attribution feature — a related but
distinct gap that was not connected).

### Recommendation

Bare minimum: compute and surface `(finished_at - started_at) × per-second
worker rate` per job as an estimated cost figure on the Jobs/Experiments
pages; store it on `TrainingJob` for historical reporting. This is cheap,
reuses timestamps that already exist, and should be scoped alongside
Finding 2's usage-metering work (same underlying event stream) and
explicitly considered as part of the Ray migration's resource-accounting
design (Ray already tracks per-task resource usage, making per-tenant
cost rollups substantially easier post-migration than today).

---

## Finding 4 — `.env.example` omits entire configuration classes that the app actually reads, making self-hosted setup misleading

**Severity:** Medium (first-run/self-host friction, not a security bug)
**Effort to fix:** Small

### Location / Evidence

- `.env.example:1-100` (full file) documents DB, LLM keys, Redis, Celery,
  tuning parallelism, uploads, `SECRET_KEY`/`DEBUG`/`ALLOWED_HOSTS`/CORS,
  logging, and commented-out JWT vars — but contains **zero** mention of
  AWS/S3/object storage configuration.
- `backend/config/mixins/aws.py:1-14` defines a full `AWSMixin` read by
  the running app: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `AWS_SESSION_TOKEN`, `AWS_DEFAULT_REGION`, `AWS_ENDPOINT_URL`,
  `AWS_BUCKET_NAME`, `S3_ARTIFACT_BUCKET`, `UPLOAD_TO_S3_FOR_LOCAL_FILES`,
  `SAVE_S3_ARTIFACTS_LOCALLY` — none of these 9 settings appear in
  `.env.example` at all.
- This directly touches a Phase 0 blocker already flagged
  (`backend-blockers.md §9`, "encrypted, managed object storage") — but
  none of the prior docs checked whether the *documented onboarding path*
  (`.env.example`) even mentions the object-storage variables a real
  self-hoster switching off local-disk storage would need to set. It
  doesn't — an admin following the example file has no discoverable path
  to enabling S3 at all without reading `aws.py` source directly.

### Recommendation

Add a `# ====== Object Storage (S3/MinIO) ======` section to
`.env.example` mirroring the PostgreSQL section's pattern (documented,
commented-out, with inline guidance) — cheap, isolated, and directly
unblocks the "how would an enterprise actually self-host this" question
the task asked about. Fold into whatever Phase 1 doc ends up producing
the reference Compose/Helm chart (`backend-blockers.md §8`), since that
chart will need these same variables wired through.

---

## Finding 5 — Notebook export has never been checked for standalone execution correctness — only for security/faithfulness of structure

**Severity:** Medium (correctness risk sitting directly behind a Phase 9/
15a differentiation bet)
**Effort to fix:** Medium (test infrastructure), ongoing (fixing what it finds)

### Location / Evidence

- `code-escape-hatch/2026-08-11-feasibility-and-security.md:40-48` audits
  the notebook exporter exclusively for **security/faithfulness**
  ("not... a decompilation of each node's... implementation", "not
  possible to faithfully export arbitrary edits") — it never asks whether
  the code the exporter emits today actually runs end-to-end outside the
  app.
- `tests/test_notebook_training_step_type.py:1-60` and
  `tests/reproduce_branched_notebook.py:1-40` — both existing notebook
  tests only call internal Python builder functions (`_classify`,
  `modeling_cells`, `build_full_branched`) and assert on **structural**
  properties (which node lands in which bucket, that a smoke build
  "succeeds" in-process). Neither test ever serializes the notebook to
  `.ipynb` and executes it with a real kernel.
- Repo-wide search confirms no notebook-execution test tooling is even
  installed: `nbclient`/`papermill`/`ipykernel` execution harnesses are
  absent from `requirements-dev.txt`/`pyproject.toml` (only `jupyter` — an
  interactive/authoring dependency, not a headless-execution one — appears
  in `requirements-dev.txt` and `pyproject.toml`).
- The exported "full" mode notebook (`_notebook_builders.py:353-364`)
  emits `import skyulf` and calls into `NodeRegistry.get_calculator(...)`
  directly — meaning correctness of the exported notebook is coupled to
  the end user having `skyulf-core` installed with the exact right
  optional extras (`skyulf-core[preprocessing-imbalanced,modeling-xgboost]`
  etc., per `README.md:127-133`) for whatever nodes their specific
  pipeline used. Nothing in the exporter emits a `pip install
  skyulf-core[...]` cell listing the extras actually required by the
  exported graph — a user exporting a pipeline using, say, an imbalanced
  resampler or XGBoost would get an `ImportError` on a bare `pip install
  skyulf-core` with no guidance in the notebook itself about which extra
  was missing.

### Actual vs. Expected

**Actual:** "does the exported notebook actually run, unmodified, in a
fresh environment" has never been tested — only "is the generated code
structurally well-formed and non-exploitable."

**Expected:** given notebook export is explicitly called out as answering
the #1 external user complaint (vendor lock-in, per
`user-complaints-research.md`) and is slated for expansion in Phase 9 and
Phase 15a Phase A, its actual reliability as an artifact users will run
outside Skyulf needs the same rigor — a CI job that generates a notebook
for a handful of representative pipeline shapes (each modeling family,
each resampler, tuned vs. fixed) and executes it headlessly (e.g. via
`nbclient`/`jupyter nbconvert --execute`) against a matching
`skyulf-core[...]` extras install, asserting it runs to completion.

### Recommendation

Add `nbclient` (or `papermill`) as a dev dependency; add one CI-run
integration test per major node family that generates + executes the
notebook end-to-end against a small fixture dataset. Also have the
exporter emit an explicit `pip install skyulf-core[<detected extras>]`
cell derived from the node types actually present in the graph, instead
of a generic install instruction — this is a small, contained fix once
node→extra mapping is known (it's already implicit in `pyproject.toml`'s
`optional-dependencies` table). Sequence before Phase 9's "two-way
notebook export/import loop" bet, since that bet builds directly on top
of export reliability.

---

## Finding 6 — No backup/disaster-recovery/multi-region story exists anywhere — only passing one-line mentions, never investigated

**Severity:** Medium-High (procurement blocker for regulated/enterprise
buyers, but distinct from the already-covered data-governance/retention
work)
**Effort to fix:** Large (requires the Phase 0 PostgreSQL+object-storage
migration as a prerequisite before it's even meaningful)

### Location / Evidence

- Grepping all 20 enterprise-readiness docs for backup/DR language finds
  only incidental one-line mentions, never a dedicated analysis:
  - `backend-blockers.md:203` — "...worker autoscaling, and a documented
    backup/restore/DR runbook" (a single clause inside a larger
    recommendation sentence, not investigated further).
  - `data-governance-audit.md:77,189,257` — three mentions, all in the
    context of *retention/deletion* scope ("Make deletion... cover
    pipeline versions, artifacts, caches, **backups**, and logs"), i.e.
    backups are referenced only as something that must also honor
    deletion requests — not as their own investigated capability.
  - `scale-load-audit.md:76` — flags local-disk artifact storage as a
    multi-instance/failover risk, but stops at "make object storage
    mandatory," never addressing point-in-time recovery, RPO/RTO targets,
    or cross-region replication.
- No repo-level backup tooling exists at all:
  `find . -iname "*backup*"` (excluding vendored `.venv`/dependencies)
  returns nothing — no backup scripts, cron jobs, or documented `pg_dump`/
  snapshot procedure anywhere in the repo.
- `docker-compose.yml:1-52` (full file) has no volume-snapshot or backup
  service defined; SQLite is a bind-mounted file with no backup
  automation at all (consistent with `scale-load-audit.md:31`'s point
  that the dev Compose targets a local `mlops_database.db` file).
- No mention anywhere in any of the 5 rounds of multi-region deployment,
  RPO/RTO targets, or a documented restore drill — this was never a
  question any of the 18 prior subagents were asked to investigate, and
  none volunteered it.

### Actual vs. Expected

**Actual:** zero backup/DR capability, zero backup/DR *documentation*
beyond a single unelaborated clause, zero multi-region consideration
anywhere in the codebase or the 20+ investigation docs.

**Expected:** enterprise security questionnaires (SOC2, vendor risk
assessments) near-universally ask for documented RPO/RTO, backup
frequency/retention, and a tested restore procedure — this is typically a
harder procurement blocker in practice than several items already ranked
higher (e.g., PII masking) because it's a checkbox question sales teams
hit immediately, not a nuanced compliance discussion.

### Recommendation

Add this as an explicit line item once Phase 0's PostgreSQL/object-storage
migration lands (it's not meaningful to design backup/DR for the current
SQLite/local-disk default — that's already flagged as unfit for
production for unrelated reasons). Minimum viable: automated
`pg_dump`/WAL-archiving schedule + retention policy, object-storage
versioning/replication, and a written, periodically-tested restore
runbook with target RPO/RTO numbers. Treat as part of Phase 1 (Production
Operating Model) rather than a new phase — it belongs next to the
already-listed "reference Compose/Helm chart" and "secrets-manager
integration" items in `master-fix-list.md:36-45`.

---

## What was checked and found ALREADY adequately covered (not duplicated here)

Per the task's 10-item checklist:

1. **Deployment & packaging** — largely covered: `backend-blockers.md §8`
   and `scale-load-audit.md` already confirm no K8s/Helm exists, the
   `Dockerfile` runs `--reload` in production, and Compose is dev-only.
   This round independently re-confirmed those same facts (no
   `helm`/`k8s` directories anywhere in the repo) but found nothing
   beyond what's documented — **except** the `.env.example` completeness
   angle specifically requested, which was a genuine gap (Finding 4).
2. **Billing/licensing/monetization mechanics** — the *business-strategy*
   framing was covered in `differentiation-strategy.md`, and the
   *existence* of the gap was flagged in `backend-blockers.md §10`, but
   neither prior doc verified the code-level absence this concretely or
   treated it as a from-scratch design problem — **new** (Finding 2).
3. **Model registry & versioning depth** — the *UX* problems were
   documented (`redesign-existing-pages.md §6`) and *performance* issues
   were documented (`scale-load-audit.md`), but the underlying
   single-global-active-deployment **data model** limitation was not —
   **new** (Finding 1).
4. **Data connectors breadth** — already thoroughly covered in
   `node-flexibility.md` (existing `BaseConnector` abstraction, file/S3
   only today, SQL connectors recommended) — confirmed still accurate
   (`backend/data_ingestion/connectors/` contains only `base.py`,
   `file.py`, `s3.py`) — **not duplicated**.
5. **Plugin/extension SDK** — already thoroughly covered in
   `node-flexibility.md §1` (two-tier metadata/code-plugin design already
   proposed) — **not duplicated**.
6. **Backup/DR/multi-region** — only ever mentioned in passing, never
   investigated — **new** (Finding 6).
7. **Cost tracking/FinOps** — essentially uninvestigated (one line in the
   round-3 README summary, "cost visibility," never expanded into its own
   finding anywhere) — **new** (Finding 3).
8. **Onboarding/first-run experience** — thoroughly covered in
   `smooth-experience-fixes.md` (no sample dataset, no onboarding
   tour/tutorial confirmed via grep, template-binding friction) —
   confirmed still accurate — **not duplicated**.
9. **Notebook/export code quality (correctness, not security)** — the
   feasibility study covered security/faithfulness only; standalone
   execution reliability was never checked — **new** (Finding 5).
10. **Other blind spot** — `.env.example` completeness (Finding 4) was the
    other genuine new item found while re-reading the deployment story.

## Genuinely new findings from this round (summary)

| # | Finding | Severity | Effort |
|---|---|---|---|
| 1 | Single global active-deployment slot — no per-pipeline/environment model serving | High | Large |
| 2 | Zero licensing/billing/entitlement code exists (confirmed total absence) | Medium-High | Medium (post-tenancy) |
| 3 | No cost/FinOps tracking anywhere, not even naive duration-based estimates | Medium-High | Medium |
| 4 | `.env.example` omits all AWS/S3 object-storage variables the app actually reads | Medium | Small |
| 5 | Notebook export never tested for standalone execution correctness | Medium | Medium |
| 6 | No backup/DR/multi-region story, tooling, or documentation exists | Medium-High | Large |
