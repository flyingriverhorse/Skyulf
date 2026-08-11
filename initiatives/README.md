# Skyulf Initiatives

This folder holds the planning/investigation documentation for Skyulf's
major forward-looking initiatives. Each subfolder is a self-contained
investigation-and-plan with its own README as the entry point.

| Folder | What it covers | Status |
|---|---|---|
| [ray-migration/](ray-migration/README.md) | Migrating pipeline job execution from Celery to Ray for distributed compute. Docs-only, unmerged design + phased implementation plans. | Design complete, not yet implemented |
| [deep-learning/](deep-learning/README.md) | Adding deep learning (tabular/text/image/time-series) node types to the canvas, config-driven (not a layer-builder), routed through Ray for GPU scheduling. Corrected after rubber-duck validation found 2 blocking design errors in the original plan. | Design complete, validated, not yet implemented |
| [enterprise-readiness/](enterprise-readiness/README.md) | The largest initiative: 6 rounds of subagent-driven investigation (22+ subagents total) covering backend/auth/tenancy blockers, node flexibility, technical debt, page redesigns, new enterprise pages, competitive differentiation strategy, smooth-experience UX fixes, security, scale/load, data governance, testing/CI depth, concrete bug hunting, i18n/mobile/cross-browser, user observability, API contract drift, and a final meta gap-check plus dedicated `skyulf-core` DX/coverage/docs research — plus real external user-complaints research. Consolidated into a single phased master fix list (Phase 0-16). | Investigation complete, see [master fix list](enterprise-readiness/2026-08-11-master-fix-list.md) for what's next |
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

1. Start with **enterprise-readiness/2026-08-11-master-fix-list.md** — the
   single consolidated, prioritized, phased plan. It cross-references
   every other document, including the other two initiatives.
2. **deep-learning/README.md** and **ray-migration/README.md** are
   orthogonal but interact with the master fix list's `skyulf-core`
   foundational items (see the master fix list's Phase 9 and Cross-References
   sections) — both should be sequenced with those in mind, not built in
   isolation.

## Why this folder exists

These documents are living planning artifacts for branch `080`, not
end-user or contributor documentation — that lives in `docs/`. Keeping
all in-flight initiative research under one `initiatives/` folder (rather
than scattered at the repo root) keeps the root clean and makes it obvious
these are related, cross-referencing planning efforts rather than
independent one-off notes.
