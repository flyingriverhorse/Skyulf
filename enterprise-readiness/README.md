# Enterprise Readiness Investigation

**Date:** 2026-08-11
**Question this answers:** *Is there any blocker that needs backend code
changes for Skyulf to become an enterprise app, and separately, what can be
improved (including existing nodes) to give consumers more flexibility?*

## Short answer

**Yes — backend code changes are required, and they are the larger and
more urgent half of the work.** Three backend gaps are true blockers, not
polish: (1) there is effectively no authentication/authorization (multiple
routes hardcode `user_id = 1` with an explicit `KNOWN-GAP` comment already
in the code, and `User.has_permission()` unconditionally returns `True`),
(2) there is no multi-tenant/organization data model at all, and (3)
SQLite/local-disk are the defaults with no migration tool, which is not a
safe foundation for either of the above. These three are structurally
inseparable — you cannot bolt real security onto the current single-shared-
namespace design without rearchitecting the data model. See
[2026-08-11-backend-blockers.md](2026-08-11-backend-blockers.md) for the
full list of 10 areas.

Separately, and independent of the security work, there's a large set of
**node/canvas flexibility improvements** that make the product more capable
for consumers whether or not multi-tenancy lands — these range from a
one-line frontend fix (a fully working backend outlier node, `ManualBounds`,
has zero UI exposure) to larger investments like a plugin system and
persistent node-level caching. See
[2026-08-11-node-flexibility.md](2026-08-11-node-flexibility.md).

## Documents in this folder

| File | Contents |
|---|---|
| [2026-08-11-backend-blockers.md](2026-08-11-backend-blockers.md) | 10 areas audited: auth, multi-tenancy, DB/scale, secrets, audit/compliance, observability, API/rate-limiting, deployment/HA, encryption at rest, licensing/quotas. Each has current-state evidence (file:line), severity, effort, and recommendation. |
| [2026-08-11-node-flexibility.md](2026-08-11-node-flexibility.md) | 7 areas audited: node extensibility/plugins, pipeline reuse/templates, existing-node rigidity (with a confirmed concrete gap), collaboration/governance, debuggability/caching, data connectivity, export/canvas scalability. |

## Methodology

Two independent `general-purpose` background subagents each audited half the
scope directly against the real codebase (not from memory/assumption),
citing exact file:line evidence for every claim. Before writing these
docs, the most severe/load-bearing claims from both reports were
independently re-verified by direct `grep`/`view` against the actual files
— all checked out exactly as reported, including finding that the
`user_id = 1` hardcoding already carries an explicit `KNOWN-GAP` comment in
the code itself, and that the `ManualBounds` outlier node is fully
registered and working on the backend with literally zero frontend
exposure. This follows the same validate-before-write pattern used for the
[deep-learning integration plan](../deep-learning/README.md) in this
session, where a rubber-duck review caught two blocking design errors
before they were written down as fact.

## Suggested sequencing relative to other in-flight work

1. **Backend-blockers §1+§2 (identity + multi-tenancy) first.** Nothing
   else in this investigation, or in the [deep-learning plan](
   ../deep-learning/README.md), is safe to build multi-customer-facing
   features on top of until this lands — it's the true foundation.
2. **Backend-blockers §3+§9 (Postgres/Alembic mandatory, encrypted
   storage) alongside #1.** These are also flagged independently by the
   (unmerged) Ray migration docs on branch `080` from a distributed-compute
   angle — deliver once, not twice.
3. **Node-flexibility's fast, low-risk wins (§3's `ManualBounds` fix, §5's
   node-level caching) can proceed in parallel at any time** — they don't
   depend on the backend security work and improve the product for every
   user immediately.
4. **Node-flexibility's larger items (plugin system, templates/subflows,
   collaboration/governance) should sequence after #1**, since
   collaboration and governance features need a real identity/org model to
   attach permissions and audit trails to.
5. **Deep learning integration** (see `../deep-learning/`) is largely
   orthogonal to this investigation and can proceed independently, but its
   job/artifact model will need a follow-up pass once multi-tenancy lands
   (noted in the backend-blockers doc).
