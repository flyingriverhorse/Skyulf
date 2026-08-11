# Enterprise Readiness — Round 4: Security, Scale, Governance, Testing & Real User Complaints

**Date:** 2026-08-11

This round closes out the investigation phase before implementation
planning begins. It answers a direct question: **is there more to check
before we start writing plans?** — yes, five gaps existed in rounds 1–3:
a dedicated security pass, production-scale/load readiness, data
governance/compliance, testing/CI rigor, and — critically — real
external evidence of what users of comparable products actually complain
about, so prioritization isn't based on internal guesses alone.

Five subagents ran in parallel:

| Agent | Type | Doc |
|---|---|---|
| `security-review-skyulf` | `security-review` (specialist) | [2026-08-11-security-review.md](2026-08-11-security-review.md) |
| `audit-scale-load` | `general-purpose` | [2026-08-11-scale-load-audit.md](2026-08-11-scale-load-audit.md) |
| `audit-data-governance` | `general-purpose` | [2026-08-11-data-governance-audit.md](2026-08-11-data-governance-audit.md) |
| `research-user-complaints` | `research` | [2026-08-11-user-complaints-research.md](2026-08-11-user-complaints-research.md) |
| `audit-testing-ci` | `general-purpose` | [2026-08-11-testing-ci-audit.md](2026-08-11-testing-ci-audit.md) |

All five completed with file:line-cited findings. No cross-agent
contradictions were found this round (unlike rounds 1–2, which each
caught one real factual error via rubber-duck cross-check) — the two
security-adjacent agents (security-review and the SSRF findings) and the
scale-load agent's rate-limiter finding are consistent with each other
and with prior rounds' `IP-only rate limiter` finding in
[backend-blockers.md](2026-08-11-backend-blockers.md).

## Why "real user complaints" matters here

You asked specifically to weight this by what users actually complain
about, not just what we think is theoretically wrong — that's the right
instinct, since internal audits are good at finding *correctness* gaps but
blind to *what actually drives churn*. The research agent could only
reach TrustRadius, Hacker News, and one long-form review (G2/Capterra/
Reddit blocked it — noted transparently in its doc, not glossed over), but
the patterns that emerged were strong and, importantly, **converge with
findings already in this repo's own docs**:

- **"No usable code export / vendor lock-in"** was the single
  best-evidenced complaint (3+ independent sources, one detailed founder
  testimonial from someone who built a competing tool specifically to fix
  this). This directly validates **Differentiation Bet #3** (code-graduation
  loop) in [differentiation-strategy.md](2026-08-11-differentiation-strategy.md)
  — it's not just a nice differentiator, it's the top reason people
  actively leave/avoid this category of tool.
- **"Black-box automation, can't see what AutoML did"** (H2O.ai, AWS
  SageMaker Canvas, DataRobot reviews) validates **Bet #1** (enforced,
  visible guardrails/transparency) — same root complaint, independent
  confirmation.
- **"No per-step/per-node data preview — the schema guessing game"** is a
  new, highly specific finding not previously in any doc this session,
  and it is *directly analogous to Skyulf's own node-based canvas
  architecture* — a competitor tool (Flowfile) was built by an ex-Alteryx
  user specifically to solve this. This is now the **single most
  actionable, evidence-backed UX addition** to consider: click any node,
  see the data at that point, without running the whole pipeline.
- **Pricing opacity / expensive tiers** — real, but more of a
  business/packaging decision outside this session's technical scope;
  noted for awareness.
- **Learning-curve/control mismatch** (power users want an escape hatch,
  casual users want more guidance) reinforces the same "code escape
  hatch per node" idea as the lock-in finding above — two independent
  angles landing on the same fix.

**New action item this round:** add "per-node data preview / inspect
data at any pipeline step without a full run" to the differentiation
bets — see the updated master fix list below.

## Security review summary

Two real, Medium-severity, high-confidence SSRF findings — both from the
same root cause (a datasource's user-controlled S3 `endpoint_url` /
`client_kwargs.endpoint_url` bypasses the one sanitizer that exists,
via the EDA and pipeline-resolution code paths). No SQLi, command
injection, XSS, unsafe deserialization of externally-supplied data, or
committed secrets were confirmed. This is a genuinely good baseline —
but the two SSRF paths should be fixed promptly since they're cheap
(one shared sanitizer function, reused everywhere) and let an attacker
who can create a datasource reach internal infrastructure or cloud
metadata endpoints. See [security-review.md](2026-08-11-security-review.md).

## Scale/load summary

The clearest production-incident risks: (1) full-dataset in-memory
processing with no streaming, against a 10GB upload cap — will OOM
workers; (2) **no per-tenant/per-user resource quotas** at all — one
user's jobs can exhaust shared workers/memory/disk/queue capacity, with
only an IP-keyed rate limiter (200/min default) as the sole throttle;
(3) undefined/default-`solo` Celery worker concurrency in production; (4)
SQLite + local disk won't survive multi-instance deployment (already
known, but now quantified with the multi-instance failure mode); (5)
non-virtualized result tables will choke the DOM at 10,000+ rows. See
[scale-load-audit.md](2026-08-11-scale-load-audit.md).

## Data governance summary

Not SOC 2/GDPR ready today. PII detection exists but is narrow
(email/phone only) and advisory-only — no masking/tokenization workflow.
Dataset deletion attempts real file erasure (a deliberate, already-known
tradeoff) but there's no retention policy, DSAR (data-subject-access-
request) workflow, encryption at rest, or tenant-scoped access-read
logging. The single biggest procurement blocker: no comprehensive,
immutable audit trail comparable to what a SOC 2 Type II review expects.
See [data-governance-audit.md](2026-08-11-data-governance-audit.md).

## Testing/CI summary

Good foundations exist (Ruff/Ty/ESLint/tsc gates, Vitest, Playwright,
OSV dependency scanning, CodeQL, per-project coverage floors) — this is
better than a typical early-stage project. But five real gaps threaten
release quality: (1) **no production-like end-to-end test** — the one
full-inference test is explicitly skipped on CI as "machine-specific";
(2) **no real canvas drag/connect E2E** — the existing Playwright spec
seeds graph state via a test hook because React Flow drag simulation is
unreliable in headless Chromium, so the core interaction (build a
pipeline by dragging/connecting nodes) has zero real-gesture CI coverage;
(3) no coverage gate/ratchet on backend or frontend; (4) auth/authz and
job-service orchestration (`job_service.py`, `pipeline_versions_service.py`)
lack direct tests; (5) **the planned DL/Ray work has no safety net at the
exact integration boundary it will land on** — Ray scheduling,
distributed artifact writes, worker retry/cancellation are entirely
untested today, which independently confirms and sharpens the
architecture-risk finding from round 3's `audit-core-architecture-depth`.
See [testing-ci-audit.md](2026-08-11-testing-ci-audit.md).

## Net new items added to the master fix list

See [2026-08-11-master-fix-list.md](2026-08-11-master-fix-list.md)'s new
**Phase 10 (Security & Scale Hardening)** and **Phase 11 (Testing/CI
Foundations)**, and the new differentiation-bet addition (per-node data
preview) folded into Phase 9.
