# Enterprise Readiness Investigation

**Date:** 2026-08-11 (two rounds — see "Investigation rounds" below)
**Question this answers:** *Is there any blocker that needs backend code
changes for Skyulf to become an enterprise app; what can be improved
(including existing nodes) to give consumers more flexibility; what other
issues exist across the codebase more generally; and what should the app's
key pages look like redesigned for a "everyone wants to use it"
enterprise-grade product?*

## Start here

👉 **[2026-08-11-master-fix-list.md](2026-08-11-master-fix-list.md)** —
the single consolidated, prioritized, phased action list synthesized from
every document below. Read this first; the other docs are the detailed
evidence and design behind each item.

## Short answer

**Yes — backend code changes are required, and they are the largest and
most urgent part of the work.** Three backend gaps are true blockers, not
polish: (1) there is effectively no authentication/authorization (multiple
routes hardcode `user_id = 1` with an explicit `KNOWN-GAP` comment already
in the code, and `DataSource.has_permission()` unconditionally returns
`True` — not `User`, see the correction note in
technical-debt-deep-dive.md), (2) there is no multi-tenant/organization
data model at all, and (3) SQLite/local-disk are the defaults with no
migration tool. These three are structurally inseparable — you cannot bolt
real security onto the current single-shared-namespace design without
rearchitecting the data model.

Beyond that foundational blocker, a deeper technical-debt sweep found real
resilience/correctness bugs unrelated to enterprise concerns (a job
cancellation race that lets a cancelled job train anyway, a
pipeline-version-numbering race, decompression-bomb-vulnerable upload
parsing, no schema versioning for saved pipelines), and the frontend has a
serious, concrete accessibility gap (the core "build a pipeline" flow is
provably impossible via keyboard alone). Two dedicated design passes
propose a full enterprise-grade redesign of every major existing page plus
the new pages (login/SSO, org/workspace settings, RBAC, audit log, billing/
usage, API keys) needed once multi-tenancy lands.

## Documents in this folder

| File | Contents |
|---|---|
| **[2026-08-11-master-fix-list.md](2026-08-11-master-fix-list.md)** | **Start here.** Consolidated, phased, prioritized list synthesized from all documents below. |
| [2026-08-11-backend-blockers.md](2026-08-11-backend-blockers.md) | Round 1. 10 areas: auth, multi-tenancy, DB/scale, secrets, audit/compliance, observability, API/rate-limiting, deployment/HA, encryption at rest, licensing/quotas. |
| [2026-08-11-node-flexibility.md](2026-08-11-node-flexibility.md) | Round 1. 7 areas: node extensibility/plugins, pipeline reuse/templates, existing-node rigidity (confirmed concrete gap: `ManualBounds`), collaboration/governance, debuggability/caching, data connectivity, export/canvas scalability. |
| [2026-08-11-technical-debt-deep-dive.md](2026-08-11-technical-debt-deep-dive.md) | Round 2. Deeper backend/core + frontend audit beyond round 1's scope: error handling/resilience, testing quality, maintainability hotspots, dependency hygiene, concurrency/races, input validation, pipeline schema versioning, performance, plus frontend state management, perf, **accessibility** (most severe finding), error feedback, testing, code duplication, mobile, design-system consistency. Includes a full rubber-duck validation of every claim in the round-1 docs (one factual correction made) plus new independently-confirmed findings. |
| [2026-08-11-redesign-existing-pages.md](2026-08-11-redesign-existing-pages.md) | Round 2. Enterprise-grade UX redesign brief for the Pipeline Canvas, Experiments, Jobs, Datasets, Drift Monitoring, and Model Registry/Deployments pages, plus cross-page design-system recommendations. |
| [2026-08-11-new-enterprise-pages.md](2026-08-11-new-enterprise-pages.md) | Round 2. UX design for pages that don't exist yet: Login/SSO, Org & Workspace Settings, RBAC, Audit Log (**correction: this page already exists**, see doc), Usage/Billing/Quota, API Keys — plus a unified app-shell design tying old and new pages together. |

## Investigation rounds

**Round 1** (backend-blockers.md, node-flexibility.md): two `general-purpose`
subagents each audited half the enterprise-readiness scope directly
against the codebase. The most severe claims were independently
re-verified before writing — confirmed exactly, including the `user_id=1`
`KNOWN-GAP` comment and the `ManualBounds` frontend gap.

**Round 2** (technical-debt-deep-dive.md, redesign-existing-pages.md,
new-enterprise-pages.md): five subagents ran in parallel —
two `general-purpose` deep-audit agents (backend/core, frontend) explicitly
scoped to find *new* issues beyond round 1; a `rubber-duck` agent
independently re-verified every claim in *both* round-1 docs and ran its
own additional sweep (cross-checking 3 more node types the ManualBounds
way, plus a forward-looking cross-tenancy deserialization risk); and two
`general-purpose` redesign agents (existing pages, new admin pages)
produced text-based design briefs. **The rubber-duck caught one real
error** in round 1 (a `has_permission()`/`is_admin` placeholder wrongly
attributed to `User` instead of `DataSource`) — corrected in
backend-blockers.md — and reframed one overstated claim (the
orphaned-dataset-file "bug" is actually a deliberate, log-visible
tradeoff). The redesign agents' page inventories were also cross-checked
against the independent frontend audit's actual route list, which caught
that `AuditLogPage.tsx` already exists — noted as a correction in
new-enterprise-pages.md rather than silently fixed, so the reasoning is
visible.

## Suggested sequencing

See [2026-08-11-master-fix-list.md](2026-08-11-master-fix-list.md) for the
full phased plan (Phase 0 through Phase 7). Short version: identity +
multi-tenancy + production data/storage plane first (nothing else is safe
to build on top of it); accessibility fixes as their own priority tier, not
folded into general polish; shared frontend infrastructure (`DataTable`,
`StatusBadge`, design tokens) before any individual page redesign, so pages
aren't redone twice; new enterprise pages can be built as mocked UI in
parallel with backend work but shouldn't be considered done until wired to
real endpoints.

**Relationship to the deep-learning plan:** [../deep-learning/](../deep-learning/README.md)
is largely orthogonal but has two real sequencing dependencies surfaced in
this round: the tuning engine's size/complexity (technical-debt-deep-dive.md
§A3) and the lack of pipeline schema versioning (§A7) should ideally be
addressed before or alongside the DL node additions, since DL adds exactly
the kind of new node types/schema changes that break old saved pipelines
without a migration path.
