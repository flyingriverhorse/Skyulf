# Skyulf Initiatives

This folder holds the planning/investigation documentation for Skyulf's
major forward-looking initiatives. Each subfolder is a self-contained
investigation-and-plan with its own README as the entry point.

| Folder | What it covers | Status |
|---|---|---|
| [growth/](growth/README.md) | **The active plan.** Funnel-ordered: bring people in → keep them → earn enterprise. Three stages only (trust floor → measurement → first-run activation), every item backed by a re-run repro or a measured number. Supersedes `roadmap/`. | **Active — start here** |
| [roadmap/](roadmap/2026-08-11-dual-track-versioned-roadmap.md) | Dual-track R1–R19 versioned release ladder. Superseded by `growth/`; version ledger known inaccurate. Task breakdowns, positioning and pricing sections still useful as reference. | Superseded, not maintained |
| [dual-engine-correctness/](dual-engine-correctness/README.md) | Whether the Polars/Pandas dual-engine design is correct end-to-end (engine → nodes → modeling → persistence → inference → monitoring) and whether the leakage claim holds. Five Opus-5 agents plus an adversarial rubber-duck pass, every finding reproduced with an executable probe. 49 findings (4 CRITICAL, 18 HIGH, 17 MED, 10 LOW); Polars is architecturally sound and the numpy handoff is bit-identical, but 6 correctness bugs block a parity claim, 3 CRITICAL inference bugs affect both engines, and the **experiments subsystem** — nearly missed entirely — is the worst-affected layer with 12 LIVE bugs that are mostly metric-semantics/UI-state, not engine, defects. All pre-existing — coverage gaps, not regressions. Also records a **product gap**: Skyulf is presented as Polars-backed, but the backend runs 100% pandas — a phased migration plan is included, gated on fixing the Polars correctness bugs first. Contains a triage section (§0.5): ~24 findings are live for canvas users today, ~17 affect only `skyulf-core` SDK users on Polars, ~8 are latent. | Investigation complete, 3 plans ready (fix tiers, leakage enforcement, backend Polars migration), **no fixes applied** |
| [ray-migration/](ray-migration/README.md) | Migrating pipeline job execution from Celery to Ray for distributed compute. Docs-only, unmerged design + phased implementation plans. | Design complete, not yet implemented |
| [deep-learning/](deep-learning/README.md) | Adding deep learning (tabular/text/image/time-series) node types to the canvas, config-driven (not a layer-builder), routed through Ray for GPU scheduling. Corrected after rubber-duck validation found 2 blocking design errors in the original plan. | Design complete, validated, not yet implemented |
| [enterprise-readiness/](enterprise-readiness/README.md) | The largest initiative: 8 rounds of subagent-driven investigation (28+ subagents total) covering backend/auth/tenancy blockers, node flexibility, technical debt, page redesigns, new enterprise pages, competitive differentiation strategy, smooth-experience UX fixes, security, scale/load, data governance, testing/CI depth, concrete bug hunting, i18n/mobile/cross-browser, user observability, API contract drift, a final meta gap-check plus dedicated `skyulf-core` DX/coverage/docs research, real external user-complaints research, web-research-backed `skyulf-core` differentiation/quick-win-tech recommendations, and a scientific-literature scan (arXiv/conference papers) across preprocessing, AutoML/tuning, DL training diagnostics, and MLOps/drift. Consolidated into a single phased master fix list (Phase 0-18). | Investigation complete, see [master fix list](enterprise-readiness/2026-08-11-master-fix-list.md) for what's next |
| [code-escape-hatch/](code-escape-hatch/2026-08-11-feasibility-and-security.md) | Feasibility/security study for showing and editing per-node generated code, with faithful export. Verdict: read-only code view is safe now; constrained editing is safe after normal auth; arbitrary Python execution requires tenancy foundations plus a dedicated hardened executor — explicitly not safe on the current shared workers. Folded into the master fix list as Phase 15a. | Feasibility study complete, phased (A/B/C), Phase C blocked pending Phase 0 |
| [training-visualization/](training-visualization/2026-08-11-feasibility-and-plan.md) | Feasibility/plan for live training visualization graphs (classic ML + deep learning). Verdict: ship post-fit diagnostics first by reusing existing chart components; add genuinely live per-epoch curves only once the DL direct-fit path lands. Folded into the master fix list as Phase 15b. | Feasibility study complete, phased plan ready |

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
