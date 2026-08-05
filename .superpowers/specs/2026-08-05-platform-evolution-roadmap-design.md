# Skyulf Platform Evolution Roadmap Design

## Purpose

Create one evidence-backed roadmap for improving Skyulf as an evolving ML
platform. The roadmap must help prioritize work that benefits individual
practitioners, open-source contributors, small ML teams, and companies without
turning unverified ideas into committed implementation plans.

## Scope

The roadmap will assess the current repository across these connected areas:

1. Core correctness, resource behavior, artifacts, and standalone API quality.
2. Backend/API, asynchronous execution, deployments, monitoring, and
   operational reliability.
3. Frontend canvas workflows, onboarding, configuration feedback, evaluation,
   inference, accessibility, and trust signals.
4. Cross-layer node contracts, registry ownership, configuration validation,
   and schema/preflight behavior.
5. Release engineering, documentation, examples, optional dependencies,
   package quality, and contributor experience.
6. Community adoption, ecosystem integrations, positioning, and
   enterprise-readiness opportunities.

## Evidence Standard

Each finding in the roadmap must be classified as one of:

- **Still present**: verified in current code, tests, docs, or workflows.
- **Partially addressed**: a mitigation exists but leaves a material gap.
- **Resolved**: the original risk is no longer present.
- **Opportunity**: a product or adoption improvement supported by the current
  architecture or market expectations, but not a confirmed defect.

Verified repository findings must cite current file paths and line ranges.
External adoption observations must cite public sources and must not claim
unverified market demand, customers, or security guarantees.

## Roadmap Shape

The final file lives at:

`temp/skyulf-platform-evolution-roadmap-2026-08-05.md`

It will contain:

1. An executive thesis and strengths to preserve.
2. A rebaseline of completed safety and observability work.
3. Findings grouped by platform domain.
4. A ranked portfolio organized into near-term trust/adoption work, platform
   foundations, and later differentiators.
5. For every proposed initiative: affected users, evidence, dependencies,
   smallest next product/design decision, and a measurable success signal.
6. Explicit non-recommendations to avoid premature platform rewrites or
   duplicated cross-layer contracts.

The roadmap is a decision artifact, not an implementation plan. Each selected
initiative must receive its own design and implementation plan later.

## Release Notes Boundary

`changelog/0.7.x.md` will receive a new `## v0.7.4` section that documents
only completed safety and observability changes:

- leakage-safe TargetEncoder pipeline training through deterministic
  cross-fitting;
- collision-free preprocessing metrics with backward-compatible telemetry
  aliases and updated consumer readers;
- correct `tracemalloc` ownership and lifecycle cleanup;
- deterministic, memory-bounded clustering silhouette sampling with reported
  sample size.

The root `CHANGELOG.md` 0.7.x summary will receive a concise matching
description. The roadmap itself will not be presented as released product
functionality.

## Constraints

- Preserve the Calculator -> Applier architecture and Core's standalone value.
- Do not imply that every future feature requires the full backend/frontend
  platform.
- Do not recommend an unbounded compute or storage path without a resource
  contract.
- Do not duplicate Core node semantics in backend or frontend recommendations;
  favor a versioned Core-owned contract where cross-layer coordination is
  required.
- Do not make unsupported security, enterprise, or adoption claims.

## Validation

Before completing the documentation changes:

1. Check the roadmap against this scope for unsupported claims, omissions, and
   duplicate initiatives.
2. Confirm the v0.7.4 text matches only committed completed work.
3. Run `git diff --check`.
4. Run the existing documentation build after editing tracked changelog files.
