# Frontend UX Roadmap Design

## Purpose

Produce an evidence-based UX improvement roadmap for the complete Skyulf
frontend. The roadmap will cover every major user journey equally while
prioritizing shared improvements that benefit multiple pages.

This phase is an audit and design exercise. It will not include visual
restyling, backend feature work, or implementation changes.

## Current Context

The frontend contains roughly 53,000 lines of TypeScript and TSX across the
pipeline canvas, data management, EDA, experiments, inference, jobs, model
registry, deployments, drift monitoring, error logs, and audit logs.

The application already includes useful shared state components, accessibility
tests, end-to-end tests, route-level code splitting, error boundaries, and
common UI primitives. It also has several page and configuration components
between 500 and 1,650 lines. The audit must therefore examine both user-visible
behavior and structural risks that can cause inconsistent UX.

## Goals

1. Identify the highest-impact UX problems across all major workflows.
2. Separate observable UX problems from maintainability risks that cause UX
   inconsistency or regressions.
3. Define shared foundations before proposing repeated page-specific fixes.
4. Produce a phased Now/Next/Later roadmap with measurable acceptance criteria.
5. Limit architectural recommendations to changes that improve user-facing
   quality, consistency, accessibility, responsiveness, or perceived
   performance.

## Audit Structure

### Shared Foundations

Review cross-cutting behavior in these areas:

- Navigation, orientation, breadcrumbs, and workflow context.
- Loading, empty, error, warning, success, and disabled states.
- Form structure, field help, validation timing, defaults, and destructive
  actions.
- Accessibility, keyboard operation, focus management, labels, contrast, and
  reduced-motion behavior.
- Responsive layouts at desktop, tablet, and mobile widths.
- Terminology, visual hierarchy, density, and component consistency.
- Perceived performance during route loading, canvas interaction, chart
  rendering, and data-heavy views.

### User Journeys

Assess each journey against the shared foundations:

1. **Canvas:** build, configure, validate, save, run, and diagnose pipelines.
2. **Data and EDA:** connect data, inspect datasets, explore profiles, and
   understand analysis results.
3. **Experiments and Inference:** compare runs, interpret metrics and
   explainability output, and perform inference.
4. **Operations:** monitor jobs, manage the model registry and deployments,
   inspect drift, investigate errors, and review audit history.

## Evidence Collection

The audit will use four evidence sources:

1. **Code inspection:** shared components, route structure, state management,
   API handling, large component boundaries, duplicated interaction patterns,
   and existing test coverage.
2. **Live walkthroughs:** complete representative workflows at desktop, tablet,
   and mobile widths.
3. **Accessibility checks:** keyboard-only operation, focus order, dialogs,
   menus, forms, status announcements, and automated accessibility checks.
4. **Engineering signals:** build and bundle output, test coverage patterns,
   loading boundaries, render hotspots, and failure handling.

Findings must state whether they were directly observed in the interface or
inferred from code as a UX regression risk.

## Deliverables

The final roadmap will contain:

1. An executive summary with the 5-10 highest-impact opportunities.
2. A shared-foundations backlog.
3. Separate backlogs for Canvas, Data/EDA, Experiments/Inference, and
   Operations.
4. A phased Now/Next/Later implementation roadmap with dependencies.
5. A component-boundary plan for oversized files only where decomposition
   directly improves UX reliability or consistency.
6. A validation matrix covering interaction tests, accessibility, responsive
   layouts, and key end-to-end paths.

Each finding will include:

- User problem and evidence.
- Affected journeys and surfaces.
- Impact and expected frequency.
- Recommended behavior.
- Acceptance criteria.
- Validation method.
- Effort estimate and regression risk.
- Dependencies and proposed milestone.

## Prioritization

Rank findings in this order:

1. User impact and severity.
2. Frequency within normal workflows.
3. Number of journeys improved.
4. Accessibility or data-loss risk.
5. Implementation effort.
6. Regression risk and dependencies.

The first milestone should favor cross-cutting improvements that benefit
multiple pages, including consistent asynchronous states, clearer
navigation/context, standardized forms and validation, accessible overlays, and
reusable responsive patterns.

## Scope Boundaries

The roadmap will not include:

- Speculative visual redesign without a demonstrated user problem.
- Backend features unrelated to current frontend usability.
- Broad refactors with no user-facing benefit.
- New design-system infrastructure unless existing inconsistency cannot be
  resolved safely through current shared components.

Backend/frontend contract issues are in scope only when they create confusing,
unavailable, invalid, or silently ineffective UI behavior. Performance issues
are in scope when users experience them through loading delays, interaction
latency, slow charts, or canvas responsiveness.

## Validation Expectations

A roadmap item is implementation-ready only when it defines:

- The current user problem.
- The proposed behavior.
- Measurable acceptance criteria.
- Required automated or manual validation.
- Responsive and accessibility expectations.
- Dependencies and affected components.

The future implementation plan should prefer targeted component and journey
tests, then run the existing frontend lint, TypeScript, build, unit, end-to-end,
accessibility, and bundle-size checks appropriate to each change.

