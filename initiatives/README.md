# Skyulf Initiatives

This folder holds the planning/investigation documentation for Skyulf's
major forward-looking initiatives. Each subfolder is a self-contained
investigation-and-plan with its own README as the entry point.

| Folder | What it covers | Status |
|---|---|---|
| [ray-migration/](ray-migration/README.md) | Migrating pipeline job execution from Celery to Ray for distributed compute. Docs-only, unmerged design + phased implementation plans. | Design complete, not yet implemented |
| [deep-learning/](deep-learning/README.md) | Adding deep learning (tabular/text/image/time-series) node types to the canvas, config-driven (not a layer-builder), routed through Ray for GPU scheduling. Corrected after rubber-duck validation found 2 blocking design errors in the original plan. | Design complete, validated, not yet implemented |
| [enterprise-readiness/](enterprise-readiness/README.md) | The largest initiative: 4 rounds of subagent-driven investigation (18 subagents total) covering backend/auth/tenancy blockers, node flexibility, technical debt, page redesigns, new enterprise pages, competitive differentiation strategy, smooth-experience UX fixes, security, scale/load, data governance, and testing/CI depth — plus real external user-complaints research. Consolidated into a single phased master fix list (Phase 0-11). | Investigation complete, see [master fix list](enterprise-readiness/2026-08-11-master-fix-list.md) for what's next |

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
