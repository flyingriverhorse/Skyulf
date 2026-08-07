# Frontend UX Roadmap Rerun Design

## Purpose

Rerun the complete frontend UX audit against the current Skyulf frontend while
preserving the existing roadmap as the canonical deliverable. The rerun will
refresh measurements and observations, identify changes since the original
audit, and keep prior evidence available for comparison.

## Scope

The rerun covers the same four journeys and shared foundations as
`.superpowers/plans/2026-08-06-frontend-ux-roadmap.md`:

- Shared navigation, feedback states, forms, accessibility, responsive
  behavior, terminology, hierarchy, and perceived performance.
- Canvas.
- Data and EDA.
- Experiments and Inference.
- Operations.

No product source code, backend behavior, visual styling, or dependencies will
be changed during the audit. Failures discovered during baseline checks or
walkthroughs will be documented as evidence rather than fixed.

## Canonical Output

Update `docs/ux/frontend-ux-roadmap.md` in place. Add a dated rerun section that
explains the comparison method and summarizes the audit delta. Existing
findings remain available unless current evidence proves that they are
resolved, superseded, or no longer applicable.

Each reviewed finding receives one rerun status:

- **New:** Not present in the previous roadmap.
- **Changed:** The user impact, evidence, scope, priority, or proposed behavior
  materially changed.
- **Confirmed:** Current evidence still supports the existing finding without a
  material change.
- **Resolved:** Current evidence demonstrates that the prior user problem no
  longer occurs.

The existing `Observed`, `Measured`, and `Inferred` evidence labels remain
separate from rerun status.

## Evidence Collection

Repeat the complete engineering baseline:

- ESLint.
- TypeScript type checking.
- Production build.
- Unit tests.
- Chromium end-to-end tests.
- Bundle-size checks.

Repeat live walkthroughs for every required route at widths `1440`, `1024`,
`768`, and `390`. Exercise the workflows, failure states, keyboard behavior,
focus management, accessibility checks, responsive layouts, and recovery paths
specified by the original plan.

Record command results and direct UI observations from the rerun. Historical
results remain clearly identified as previous evidence and are not presented as
current measurements.

## Roadmap Reconciliation

After refreshing all journey evidence:

1. Reconcile every existing finding against current evidence.
2. Add new findings with the next available journey-specific ID.
3. Preserve resolved findings in a compact historical section instead of
   silently deleting them.
4. Recalculate normalized ranking and dependencies.
5. Rebuild the Now/Next/Later milestones.
6. Refresh component-boundary recommendations only when current user-facing
   evidence supports them.
7. Refresh the validation matrix for every current Now and Next item.

## Execution

Follow the eight tasks in the original implementation plan in order. Use a
fresh implementer and independent task review for each task. Each task may
commit only `docs/ux/frontend-ux-roadmap.md` and audit-only artifacts explicitly
required for evidence collection.

The existing unrelated modification to
`.github/workflows/dependency-review.yml` must remain outside all audit commits.

## Completion Criteria

The rerun is complete when:

- Every original audit task has fresh evidence.
- All four journeys have equal current coverage.
- Every existing finding has a rerun status.
- New and resolved findings are explicitly represented.
- Rankings, milestones, dependencies, component recommendations, and the
  validation matrix agree with the refreshed evidence.
- The final document contains no placeholders or ambiguous future language.
- The full frontend quality baseline has been rerun and recorded.
