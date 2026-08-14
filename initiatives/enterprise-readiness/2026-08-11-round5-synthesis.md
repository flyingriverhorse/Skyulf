# Enterprise Readiness — Round 5: Concrete Bugs, i18n/Mobile, User Observability, API Contract Drift

**Date:** 2026-08-11

This round answers the direct follow-up question: *"is there anything left
that we can investigate — code related, UI related, user related, bug
related, anything?"* after 4 prior rounds (14 subagents). Four more
subagents ran in parallel, each scoped to a genuinely uninvestigated angle:

| Agent | Scope | Doc |
|---|---|---|
| `bug-hunt-core-paths` | Concrete, reproducible bugs (not architecture opinions) in calculators, pipeline execution, frontend state, concurrency | [2026-08-11-bug-hunt.md](2026-08-11-bug-hunt.md) |
| `audit-i18n-mobile-crossbrowser` | Internationalization, mobile/tablet usability, cross-browser risk — untouched in rounds 1-4 | [2026-08-11-i18n-mobile-crossbrowser-audit.md](2026-08-11-i18n-mobile-crossbrowser-audit.md) |
| `audit-user-observability` | Can an *end user* (not ops) self-diagnose a failure inside the product UI | [2026-08-11-user-observability-audit.md](2026-08-11-user-observability-audit.md) |
| `audit-api-contract-drift` | Backend/frontend API contract drift beyond the already-documented node-param duplication pattern | [2026-08-11-api-contract-drift-audit.md](2026-08-11-api-contract-drift-audit.md) |

All four were explicitly instructed not to re-report findings already in
rounds 1-4's docs, and cross-checked against those docs — no duplicate
findings were reported back.

## Bug hunt: 9 concrete, reproducible bugs found

This is the most actionable output of this round — real, verified bugs,
each with a file:line citation, exact reproduction steps, and actual-vs-
expected behavior (not opinions about code quality). Top 4 by likelihood ×
impact:

1. **Cross-process duplicate pipeline-job creation** (High) — the
   existing idempotency guard doesn't hold across processes/workers,
   so concurrent duplicate submissions can both proceed.
2. **Lag Features node sorts/drops `X` but returns the original,
   unsorted/undropped `y`** (High) — a silent train/label misalignment
   bug in a shipped feature-engineering node. This is a correctness bug
   in a core node, not an edge case — any pipeline using Lag Features
   with unsorted input data will silently train against
   misaligned labels.
3. **Rolling Aggregate node has the identical `X`/`y` misalignment bug**
   (High) — same root cause as #2, different node; strongly suggests a
   shared underlying helper/pattern should be fixed once and reused,
   not patched node-by-node.
4. **Out-of-order job-list API responses can revert newer job state in
   the UI** (Medium) — a classic stale-response race condition (no
   request sequencing/abort on the polling client).

Plus 5 more Medium-severity findings: a cyclic-graph validation gap (the
canvas accepts cycles and only fails at execution time, with a confusing
error), a misleading upload-size error message (client rejects at 500MB
while the server default is 10GB), two nodes (`Feature Selection`,
`General Binning`) whose advertised/documented default method silently
no-ops instead of executing, and a mixed-timezone datetime-extraction
silent no-op.

**Why #2/#3 matter most:** these are not "could be better" findings —
they are confirmed logic errors in shipped, presumably tested feature
engineering nodes that would silently corrupt model training on any
dataset that isn't already sorted the way the node assumes. These should
be treated as release-blocking bugs, not backlog items, independent of
any broader enterprise-readiness phasing.

## i18n / mobile / cross-browser: no work has started here

No i18n framework exists at all (fully hardcoded English strings, no
locale-aware date/number formatting via `Intl`, no RTL consideration).
Canvas mobile/tablet support has no explicit policy — drag-and-drop node
placement's touch-event support was not confirmible as designed-for. No
declared browser support matrix; E2E tests run against Chromium only
(no Firefox/WebKit Playwright projects). Metric/p-value formatting is
inconsistent across the app (a real significant value like `0.000004` can
render as `0.0000` in one view while a p-value renders in scientific
notation in another) — this last one is a correctness-adjacent UX bug,
not just a nice-to-have.

None of this blocks a US/English-only B2B launch, but every item here
becomes a hard blocker the moment there's a non-English enterprise
customer, a Middle-East expansion, or a customer wanting to review the
product on a tablet during a sales demo.

## User observability: the product already has strong pieces, but they're disconnected

The audit found genuine existing capabilities to *preserve*, not just
gaps: raw job logs, a Pipeline Logs/Error page, preview-node failure
cards, notification history, config diffing, and feature
importance/SHAP diagnostics. The actual gap is that these are
**fragmented** — there's no single canonical "why did this specific run
fail/behave this way" timeline that ties a job to its per-node
execution ledger, data-quality warnings at each step, and a comparison
against a prior run. The progress/ETA event schema already exists in the
database/event model but isn't consistently wired end-to-end to the UI.

This is the clearest connective-tissue gap identified this round: much of
what's needed already exists as separate pieces (job logs, node preview
cards, diff views) — the fix is largely UI/wiring work to unify them into
one per-run story, not new backend capability.

## API contract drift: confirmed, concrete instances beyond node params

The backend generates an OpenAPI spec via FastAPI, but **nothing consumes
it for frontend codegen** — the frontend hand-writes TypeScript
interfaces that must be manually kept in sync, and drift was confirmed
in practice: `JobInfo`'s `created_at` nullability, an omitted `preview`
field, and `output_artifact_id` vs. `output` naming in node execution
results. EDA job-status strings are uppercase and force-cast (`as
JobStatus`) rather than validated, so an unhandled EDA status value would
silently fall through to a default/unknown UI state. WebSocket message
frames are validated on the backend (Pydantic) but **not** on the
frontend despite Zod already being an installed dependency — a renamed
field would silently break live job-progress updates with no visible
error, just a fallback to 30-second polling. No API versioning scheme
exists (`/api/v1` or equivalent) at all.

## Net new items added to the master fix list

See [2026-08-11-master-fix-list.md](2026-08-11-master-fix-list.md)'s new
**Phase 12 (Confirmed Bugs — fix independent of any other phase)**,
**Phase 13 (API Contract Hardening)**, and an addition to **Phase 3
(Accessibility)** broadened to cover i18n/mobile/cross-browser as a
related but distinct "reach" tier, plus a User Observability item folded
into Phase 9's differentiation work (it directly reinforces Bet #1,
transparency/black-box).
