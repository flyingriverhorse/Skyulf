# Celery to Ray Migration Implementation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Coordinate the independent, testable plans that replace Celery with Ray without losing Skyulf job behavior.

**Architecture:** FastAPI remains the control plane, PostgreSQL remains the job-state source of truth, Ray Jobs/Core become the compute plane, and S3-compatible storage becomes the production data plane. The migration keeps local and Celery adapters until Ray passes parity and reliability gates.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy asyncio/sync, PostgreSQL, Ray Jobs/Core, joblib, Redis pub/sub, S3-compatible storage, pytest, Docker Compose, optional KubeRay.

## Global Constraints

- Preserve existing public pipeline, job, cancellation, and WebSocket API
  paths and response shapes. The approved attempt model intentionally changes
  manual retry identity: retry returns the same logical job ID and appends a
  physical attempt.
- Production Ray mode requires PostgreSQL and shared S3-compatible storage.
- Production never silently falls back from Ray to in-process execution.
- Ray Dashboard and Jobs API remain private network endpoints.
- API and Ray workers use the same versioned application image.
- Do not adopt Ray Data, Ray Train, Ray Tune, or Ray Serve in the initial migration unless a later plan explicitly says so.
- Every implementation task follows TDD and ends with a focused commit containing `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.
- After Python changes run `ruff check .`, `ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`, and `ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`.
- After dependency changes run the repository's dependency/security checks, including Codacy Trivy when available.

---

## Plan Order

| Order | Plan | Deliverable | Depends On |
|---|---|---|---|
| 1 | `01-execution-backend-foundation-plan.md` | Backend-neutral submission contract, config split, event transport split | None |
| 2 | `02-job-attempt-lifecycle-plan.md` | Durable execution attempts, cancellation states, retry lineage | Plan 1 |
| 3 | `03-ray-jobs-pipeline-runtime-plan.md` | Ray Jobs adapter and pipeline entrypoint | Plans 1-2 |
| 4 | `04-distributed-compute-plan.md` | Ray branch executor, joblib tuning, resource and artifact safety | Plan 3 |
| 5 | `05-operations-deployment-plan.md` | Reconciliation, health, scheduler, Compose, observability | Plans 2-4 |
| 6 | `06-cutover-celery-removal-plan.md` | Parity rollout, default switch, Celery drain and removal | Plans 1-5 |

## Independent Review Gates

Each plan produces working software that can be reviewed and rejected without
invalidating later architectural decisions:

1. **Foundation gate:** Existing local and Celery behavior passes behind the
   new interface before any Ray dependency is used.
2. **Lifecycle gate:** Attempts, cancel-requested state, and retries work with
   the existing Celery adapter.
3. **Ray runtime gate:** A whole pipeline can be submitted, queried, stopped,
   and completed on a single-node Ray cluster.
4. **Compute gate:** Ray produces result parity and a measured benefit for
   branch or tuning workloads.
5. **Operations gate:** Restart, orphan, storage, and scheduler behavior is
   observable and recoverable.
6. **Cutover gate:** Ray becomes default only after production-like acceptance
   tests; Celery removal is the final reversible-boundary change.

## Execution Rule

Do not begin a later plan while an earlier gate has unresolved correctness,
security, data-integrity, or performance failures. Performance alone may stop
the migration: if Ray does not provide a measurable benefit for selected
workloads, keep the backend abstraction and do not remove Celery.
