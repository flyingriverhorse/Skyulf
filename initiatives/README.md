# Skyulf Initiatives

This folder holds the planning/investigation documentation for Skyulf's
major forward-looking initiatives. Each subfolder is a self-contained
investigation-and-plan with its own README as the entry point.

| Folder | What it covers | Status |
|---|---|---|
| [growth/](growth/README.md) | **The active plan.** Funnel-ordered: bring people in → keep them → earn enterprise. Three stages only (trust floor → measurement → first-run activation), every item backed by a re-run repro or a measured number. Supersedes `roadmap/`. | **Active — start here** |
| [roadmap/](roadmap/2026-08-11-dual-track-versioned-roadmap.md) | Dual-track R1–R19 versioned release ladder. Superseded by `growth/`; version ledger known inaccurate. Task breakdowns, positioning and pricing sections still useful as reference. | Superseded, not maintained |
| [dual-engine-correctness/](dual-engine-correctness/README.md) | Whether the Polars/Pandas dual-engine design is correct end-to-end (engine → nodes → modeling → persistence → inference → monitoring) and whether the leakage claim holds. Five Opus-5 agents plus an adversarial rubber-duck pass, every finding reproduced with an executable probe. 49 findings (4 CRITICAL, 18 HIGH, 17 MED, 10 LOW); Polars is architecturally sound and the numpy handoff is bit-identical, but 6 correctness bugs block a parity claim, 3 CRITICAL inference bugs affect both engines, and the **experiments subsystem** — nearly missed entirely — is the worst-affected layer with 12 LIVE bugs that are mostly metric-semantics/UI-state, not engine, defects. All pre-existing — coverage gaps, not regressions. Also records a **product gap**: Skyulf is presented as Polars-backed, but the backend runs 100% pandas — a phased migration plan is included, gated on fixing the Polars correctness bugs first. Contains a triage section (§0.5): ~24 findings are live for canvas users today, ~17 affect only `skyulf-core` SDK users on Polars, ~8 are latent. Now also holds the **F-15 leakage-free per-fold refit** record: shipped always-on (0.8.0) and extended in **v0.8.4** to holdout tuning with a validation split, with measured noise-at-chance / signal-stays-high scores and two follow-up product realities (both non-defects, recommendations logged). | Investigation complete, 3 plans ready (fix tiers, leakage enforcement, backend Polars migration), **no fixes applied** |
| [ray-migration/](ray-migration/README.md) | Migrating pipeline job execution from Celery to Ray for distributed compute. Docs-only, unmerged design + phased implementation plans. | Design complete, not yet implemented |
| [deep-learning/](deep-learning/README.md) | Adding deep learning (tabular/text/image/time-series) node types to the canvas, config-driven (not a layer-builder), routed through Ray for GPU scheduling. Corrected after rubber-duck validation found 2 blocking design errors in the original plan. | Design complete, validated, not yet implemented |
| [enterprise-readiness/](enterprise-readiness/README.md) | The largest initiative: 8 rounds of subagent-driven investigation (28+ subagents total) covering backend/auth/tenancy blockers, node flexibility, technical debt, page redesigns, new enterprise pages, competitive differentiation strategy, smooth-experience UX fixes, security, scale/load, data governance, testing/CI depth, concrete bug hunting, i18n/mobile/cross-browser, user observability, API contract drift, a final meta gap-check plus dedicated `skyulf-core` DX/coverage/docs research, real external user-complaints research, web-research-backed `skyulf-core` differentiation/quick-win-tech recommendations, and a scientific-literature scan (arXiv/conference papers) across preprocessing, AutoML/tuning, DL training diagnostics, and MLOps/drift. Consolidated into a single phased master fix list (Phase 0-18). | Investigation complete, see [master fix list](enterprise-readiness/2026-08-11-master-fix-list.md) for what's next |
| [code-escape-hatch/](code-escape-hatch/2026-08-11-feasibility-and-security.md) | Feasibility/security study for showing and editing per-node generated code, with faithful export. Verdict: read-only code view is safe now; constrained editing is safe after normal auth; arbitrary Python execution requires tenancy foundations plus a dedicated hardened executor — explicitly not safe on the current shared workers. Folded into the master fix list as Phase 15a. | Feasibility study complete, phased (A/B/C), Phase C blocked pending Phase 0 |
| [training-visualization/](training-visualization/2026-08-11-feasibility-and-plan.md) | Feasibility/plan for live training visualization graphs (classic ML + deep learning). Verdict: ship post-fit diagnostics first by reusing existing chart components; add genuinely live per-epoch curves only once the DL direct-fit path lands. Folded into the master fix list as Phase 15b. | Feasibility study complete, phased plan ready |
| [frontend-consumer-design/](frontend-consumer-design/README.md) | Consumer-perspective UX/design review of the whole `frontend/ml-canvas` app via parallel full-file reviews: (1) full-app review — 27 findings across guided journey, canvas run experience, results clarity, IA, design-system foundations, with a three-wave execution order; (2) [canvas node-journey deep dive](frontend-consumer-design/2026-08-26-canvas-node-journey.md) — connections, node configuration, dataset→training difficulty table, settings-panel placement analysis, 23 changes (N1–N23); (3) [Experiments/Inference IA design](frontend-consumer-design/2026-08-26-experiments-inference-ia.md) — both views proven globally scoped and loosely coupled; recommended hybrid design (standalone routes + canvas tabs as scoped deep links) with dedup program and 3-phase plan; (4) [beyond-pages opportunities](frontend-consumer-design/2026-08-26-beyond-pages-opportunities.md) — missing consumer features (onboarding tour, pipeline lint, undo visibility), unaudited surfaces (mobile, perf, a11y, testing, realtime), codebase/approach items; (5) [charts gaps and additions](frontend-consumer-design/2026-08-26-charts-gaps-and-additions.md) — chart inventory plus five targeted additions (registry sparklines, dashboard outcomes, drift verdicts, node-body training progress, experiments trend) with effort and backend needs. Every finding carries file:line evidence. Overlaps noted with growth/ A2.1–A2.6. | Investigation complete, five reports ready, **no fixes applied** |
| [backend-and-core-review/](backend-and-core-review/README.md) | Fresh four-scope code review of `backend/` + `skyulf-core/`: ~55 findings — 3 CRITICAL (no authentication anywhere; cancelled jobs resurrect on Celery redelivery; batch cancel kills sibling jobs), 12 HIGH (plaintext S3 credentials, CORS wildcard+credentials, unmounted rate limiter, SQLite StaticPool/WAL misconfig, preview-error swallowing — root cause of the frontend's "Check console", zero live job progress, trials buffer broken in Celery mode, ingestion cancel cosmetic, deployment activate race, 5 engine-parity bugs, unversioned pickle save/load), ~30 MED robustness issues, and 10 verified missing capabilities (per-epoch events, run-time estimates, structured failure reasons, drift histograms — already half-built in core, dataset profile endpoint). Four-wave fix plan; cross-referenced with dual-engine-correctness, enterprise-readiness Phase 0, and the frontend reports. Companion: [skyulf-core second-pass deep review](backend-and-core-review/2026-08-26-skyulf-core-second-pass.md) — ~41 NEW core-only findings, incl. CRITICAL README-quickstart crash, time-series X/y misalignment, `group_agg` target leakage, tuning metric-direction inversion, 9 more engine divergences, Celery-worker concurrency hazards, and DX/packaging gaps. | Investigation complete, prioritized fix lists ready, **no fixes applied** |

## Effort comparison

**Read the caveat first.** None of these initiatives carry an effort
estimate of their own. The only folder that ever gave numbers is
`roadmap/`, which is superseded and whose version ledger is known to be
inaccurate. The "estimated effort" column below is therefore a
**judgement call, not a quotation** — derived from countable scope (task
counts, checkbox steps, finding counts) and calibrated against the
`roadmap/` figures. Treat it as an order-of-magnitude comparison for
sequencing decisions, not as a commitment.

| Initiative | Countable scope | Status | Estimated effort* | New infrastructure | Dependency |
|---|---|---|---|---|---|
| [growth/](growth/README.md) | 3 stages, **14 scheduled items** (+8 Stage 3 candidates) | **Active plan** | Stated in the doc: **6–8 weeks at 2–3 days/week, paired** (≈4–5 weeks full-time equivalent) | None | Already absorbs dual-engine Tier 1 |
| [dual-engine-correctness/](dual-engine-correctness/README.md) | **49 findings**, 5 tiers + 2 additional plans | Investigation done, **0 fixes applied** | ~**10–12 weeks** (Tier 1 alone ~1–1.5 weeks) | None | None — can start immediately |
| [code-escape-hatch/](code-escape-hatch/2026-08-11-feasibility-and-security.md) | 3 phases (A/B/C) | Feasibility done | Phase A ~**1–2 weeks**, Phase B ~**3–4 weeks**, Phase C **blocked** | Phase C: hardened isolated executor | Phase B/C → auth + multi-tenancy |
| [deep-learning/](deep-learning/README.md) | **6 phases** (0–5), 4 modalities, new core subpackage + frontend nodes | Design done, **no code** | ~**12–20 weeks** | PyTorch, GPU | Phase 5 → ray-migration plans 01 + 04 |
| [ray-migration/](ray-migration/README.md) | **6 plans, 43 tasks, 291 checkbox steps** (~8,000 lines) | Design approved, **no code** | ~**16–24 weeks** | **Ray cluster + PostgreSQL + S3/MinIO + Redis + new Compose stack** | Strict ordering gate: 01→02→…→06 |

*\*Estimates are the author's judgement; they appear in no source document.*

Naive total: roughly **45–60 weeks** of single-person full-time work.

### But most of it is deliberately parked

That is not an outside opinion — the `growth/` plan already triaged every
other initiative for near-term actionability, and recorded the verdicts:

| Initiative | `growth/` plan's verdict |
|---|---|
| dual-engine-correctness | **"Promoted."** Tier 1 (F-01/F-02/F-03) is part of Stage 0. *"The only prior doc that met this folder's evidence bar unaided."* Separately, on F-02: *"the most severe defect found anywhere in this repo."* |
| code-escape-hatch | **"One slice promoted"** — read-only code view (Stage 3), plus a labelling correction to A2.5 |
| training-visualization | **"One slice promoted"** — post-fit diagnostics (Stage 3) |
| enterprise-readiness | **"Mined, not executed."** Raw material only; its `master-fix-list.md` phases are *not* a schedule |
| deep-learning | **"Nothing actionable."** Two findings recorded in the salvage ledger for if/when it starts |
| ray-migration | **"Nothing."** Six gated plans, no independently cheap phase. *"Its own rule blocks it: no measurable benefit has been measured."* |

So the genuinely near-term body of work is **growth + dual-engine
≈ 12–16 weeks**; the remaining ~35–45 weeks is parked behind either
user-demand evidence or a measured benefit case.

### Three things worth knowing before sequencing

1. **Ray's real cost is not the 24 weeks.** It makes PostgreSQL, S3 and a
   Ray cluster *mandatory* for production — a permanent operational
   burden, not a one-off spend. The plan's own exit criterion is the
   right one: *"if Ray shows no benefit, stop after Task 4 and keep the
   backend abstraction with Celery intact."* Do not start it before the
   benefit is measured.
2. **The highest-value single item is a test, not a fix.** The `growth/`
   plan says of T5 (the registry-wide contract test): *"Three bug fixes
   are worth a week. A test that makes the whole class impossible is
   worth considerably more, and it is the single highest-value artifact
   in this plan."* The evidence backs it: two independent audits both
   examined `lag.py`, each found a **different** bug, and neither found
   the other's. All 49 dual-engine findings pass the existing suites
   cleanly. Without closing that coverage debt the same class of bug
   simply regenerates.
3. **The ordering gates are real and cannot be parallelised away.** The
   backend Polars migration must follow dual-engine Tier 1+2 (migrating
   first would activate every Polars bug for every user at once);
   deep-learning Phase 5 follows Ray; escape-hatch Phase B/C follows
   auth/tenancy.

**Recommended entry point:** dual-engine **Tier 1 — 7 fixes, ~1–1.5
weeks**. It clears every CRITICAL, it is already inside `growth/` Stage 0,
and it depends on nothing.

## Implementation plans

Bite-sized, TDD-structured implementation plans (written via the
`writing-plans` skill, one plan per architecturally-independent subsystem)
live under `docs/superpowers/plans/`:

| Plan | Scope |
|---|---|
| [`docs/superpowers/plans/2026-08-11-phase12-confirmed-bugs.md`](../docs/superpowers/plans/2026-08-11-phase12-confirmed-bugs.md) | Phase 12 of the master fix list — all 9 concrete bugs from `enterprise-readiness/2026-08-11-bug-hunt.md`, each as an independent task with a failing-first test, exact fix, and commit. |

Further phases (0–11, 13–15) each need their own dedicated plan before
execution — do not squeeze multiple independent phases into one plan
document (see the Scope Check rule in `writing-plans`).

## Reading order if you're new to this branch

1. Start with **[growth/README.md](growth/README.md)** — the active plan and
   the operating rules that govern what may enter it. Everything else in
   this folder is research feeding into it.
2. Then **enterprise-readiness/2026-08-11-master-fix-list.md** — the
   consolidated findings. Read it as an evidence index, **not** as a
   schedule; its phase numbers are groupings, not an execution order.
3. **deep-learning/README.md** and **ray-migration/README.md** are parked.
   Both are downstream of having users; revisit when the growth plan's
   Stage 3 data justifies them.

## Why this folder exists

These documents are living planning artifacts for branch `080`, not
end-user or contributor documentation — that lives in `docs/`. Keeping
all in-flight initiative research under one `initiatives/` folder (rather
than scattered at the repo root) keeps the root clean and makes it obvious
these are related, cross-referencing planning efforts rather than
independent one-off notes.
