# Platform Evolution Roadmap Documentation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish an evidence-backed whole-platform evolution roadmap in
`temp/` and document the completed safety and observability work as v0.7.4.

**Architecture:** Keep the roadmap as a decision artifact, not an implementation
backlog. It consolidates source-verified Core, backend, frontend, release, and
adoption evidence into phased initiatives with explicit dependencies and next
decisions. Release notes describe only already committed behavior.

**Tech Stack:** Markdown, MkDocs Material, Git, existing Core/backend/frontend
audit evidence.

## Global Constraints

- Keep `temp/skyulf-platform-evolution-roadmap-2026-08-05.md` untracked and
  evidence-backed; do not describe roadmap opportunities as shipped features.
- Cite current repository paths and line ranges for verified technical findings.
- Cite public URLs for external adoption expectations and avoid market-size,
  customer, security, or enterprise-readiness claims that lack evidence.
- Preserve the Calculator -> Applier architecture and standalone value of
  `skyulf-core`.
- Do not turn the roadmap into a feature implementation plan; each selected
  initiative needs its own design and plan.
- Create `## v0.7.4` before `## v0.7.3` in `changelog/0.7.x.md`.
- Limit v0.7.4 to completed safety and observability work in commits
  `748929e5..483ee994`.
- Update `CHANGELOG.md`'s 0.7.x table description with a concise matching
  v0.7.4 phrase.
- Do not modify product code, dependencies, generated assets, or package
  versions.

---

## File Structure

| Path | Responsibility |
|---|---|
| `temp/skyulf-platform-evolution-roadmap-2026-08-05.md` | Untracked decision roadmap: evidence, strengths, verified gaps, phased portfolio, dependencies, and adoption signals. |
| `changelog/0.7.x.md` | Tracked v0.7.4 release narrative for completed safety and observability behavior. |
| `CHANGELOG.md` | Concise 0.7.x table summary mentioning v0.7.4. |

## Task 1: Consolidate the Whole-Platform Evidence Roadmap

**Files:**

- Create: `temp/skyulf-platform-evolution-roadmap-2026-08-05.md`
- Read: `temp/skyulf-core-deep-assessment-2026-07-29.md`
- Read: `.superpowers/specs/2026-08-05-platform-evolution-roadmap-design.md`
- Read: current Core, backend, frontend, workflow, and documentation sources
  cited by the completed audits.

**Interfaces:**

- Consumes: the design specification, completed safety work, current source
  evidence, and public adoption references.
- Produces: a decision document that future design work can consume without
  treating it as an implementation specification.

- [ ] **Step 1: Write the executive thesis and strengths to preserve**

  Start the roadmap with this scope statement:

  ```markdown
  # Skyulf Platform Evolution Roadmap

  **Assessment date:** 2026-08-05
  **Scope:** `skyulf-core`, backend/API/jobs/artifacts/deployments,
  `frontend/ml-canvas`, operations, release engineering, and adoption.
  **Purpose:** prioritize verified reliability and adoption work without
  committing unvalidated opportunities to implementation.
  ```

  State that Skyulf's durable strengths are the explicit Calculator -> Applier
  learned-state model, Pandas/Polars support, visual-to-code continuity,
  split-safe execution paths, model/version history, notebook export, and
  local/S3 artifact flexibility. Cite one current source path for each
  technical strength.

- [ ] **Step 2: Rebaseline completed safety and observability work**

  Add a `## Completed in v0.7.4` section with these four verified outcomes:

  ```markdown
  - TargetEncoder training rows use deterministic, leakage-safe cross-fitting;
    held-out and inference rows use the fitted encoder transform.
  - Preprocessing now exposes a collision-free summary-and-steps metrics
    contract while retaining the four flat telemetry aliases used by platform
    consumers.
  - Transformer memory tracing restores ownership correctly after success and
    errors and does not attribute caller-owned historical peaks to a node.
  - Clustering silhouette scoring uses a deterministic, representative,
    memory-bounded sample capped at 10,000 rows and reports the actual score
    sample size.
  ```

  Cite the relevant Core implementation, tests, and cross-layer consumer
  updates. Do not repeat historical report prose.

- [ ] **Step 3: Document verified findings by platform domain**

  Create these headings in order and give every finding one of the statuses
  defined by the design specification:

  ```markdown
  ## Core correctness, resource controls, and artifacts
  ## Cross-layer node contract and configuration integrity
  ## Backend execution, operations, and governance
  ## Frontend product experience and trust
  ## Release engineering and contributor experience
  ## Community, ecosystem, and company adoption
  ```

  For each finding, use this compact table shape:

  ```markdown
  | Status | Evidence | Impact | Next decision |
  |---|---|---|---|
  | Still present | `path:line-range` | Specific user or operator consequence | One bounded design decision |
  ```

  Cover at minimum:

  - tuning budget inconsistencies, unbounded evaluation payloads, profiling
    sampling disclosure, SHAP status/budget, CV failure policy, and drift
    reference retention;
  - duplicated node/leakage contracts, permissive per-node params, registry
    robustness, schema/preflight diagnostics, mutable pipeline config,
    noncanonical fingerprints, raw-Pickle trust, raw-Polars metric gaps, and
    degraded-success evaluation behavior;
  - identity/RBAC/tenant boundaries, branch cancellation semantics, worker
    resource limits/progress, artifact provenance, deployment scope,
    operational metrics/traces, secret lifecycle, and production deployment
    posture;
  - guided first run, schema-preview failures, manual converter drift,
    actionable job failures, promotion/rollback evidence, accessibility, and
    budget visibility;
  - strict docs/link/notebook checks, network-isolated SentenceEmbedder tests,
    optional-extra smoke coverage, package extras verification, contributor
    guide correctness, and trusted-artifact documentation;
  - positioning, Core-first onboarding, template/notebook round trips,
    contributor pathways, support/lifecycle claims, workspace/RBAC,
    reproducibility manifests, governed promotion, scheduled monitoring, and
    interoperability.

- [ ] **Step 4: Rank the portfolio without inventing implementation**

  Add a `## Recommended portfolio` section with three phases:

  ```markdown
  ### Phase 0 — truthful activation and safety contracts
  ### Phase 1 — team trust and reproducibility
  ### Phase 2 — operational adoption and ecosystem reach
  ```

  For each initiative, list:

  ```markdown
  - **Initiative:** short name
  - **Why now:** affected users and verified evidence
  - **Dependencies:** prerequisite contracts or decisions
  - **Next decision:** the smallest product/design choice needed before a plan
  - **Success signal:** an observable adoption, reliability, or operability metric
  ```

  Put identity/workspace boundaries, a shared node contract, resource budgets,
  run manifests, and docs/test reliability ahead of enterprise integrations,
  Kubernetes packaging, a template marketplace, or broad AutoML claims.

- [ ] **Step 5: Add non-recommendations and evidence limits**

  Finish with:

  ```markdown
  ## Deliberate non-recommendations
  ## Evidence limits
  ```

  Explicitly reject a wholesale Core rewrite, forcing all Core users into the
  web platform, metadata-driven replacement of bespoke UI, and claiming
  enterprise readiness before identity, tenancy, governance, and operational
  contracts exist. Note that external ecosystem sources establish expectations,
  not proof of demand for a copied feature set.

- [ ] **Step 6: Verify roadmap integrity**

  Run:

  ```bash
  git check-ignore -q temp/skyulf-platform-evolution-roadmap-2026-08-05.md
  rg -n 'TBD|TODO|implement later|fill in details' \
    temp/skyulf-platform-evolution-roadmap-2026-08-05.md
  ```

  Expected: the roadmap is ignored and the search returns no placeholder
  content. Manually confirm every release claim is separate from opportunity
  language.

## Task 2: Add the v0.7.4 Safety and Observability Release Notes

**Files:**

- Modify: `changelog/0.7.x.md`
- Modify: `CHANGELOG.md`

**Interfaces:**

- Consumes: completed safety work in `748929e5..483ee994`.
- Produces: an accurate release narrative that users can read independently of
  the roadmap.

- [ ] **Step 1: Add the detailed v0.7.4 entry**

  Insert before `## v0.7.3`:

  ```markdown
  ## v0.7.4

  **Leakage-safe TargetEncoder pipeline training.** `FeatureEngineer` and
  `SkyulfPipeline` now use sklearn TargetEncoder's cross-fitted train-time
  representation for training rows, while test, validation, and inference
  rows use the fitted encoder transform. The default policy is deterministic
  five-fold cross-fitting (`shuffle=True`, seed 42) when the split supports
  it; small splits reduce folds without leaking target values, and impossible
  one-row or singleton-class splits now raise clear errors. Direct
  Calculator/Applier use keeps its explicit fit/apply behavior.

  **Trustworthy preprocessing metrics across the platform.**
  `FeatureEngineer.fit_transform()` now preserves one record per preprocessing
  step under `metrics["steps"]`, plus a deliberate pipeline
  `metrics["summary"]`. The existing flat `fit_time`, `peak_memory_bytes`,
  `rows_in`, and `rows_out` keys remain summary compatibility aliases.
  Backend aggregation and frontend preprocessing feedback now read step
  details safely, retain legacy payload support, and avoid presenting
  ambiguous multi-step feature-selection data as a false zero-change result.

  **Correct transformer tracing lifecycle.** `StatefulTransformer` now owns
  only tracing it starts, always cleans it up after success or failure, and
  reports caller-owned memory only as new peak growth after entry. Failed
  transforms also clear stale output-row telemetry.

  **Bounded, representative clustering silhouette scoring.** Clustering
  evaluation now scores silhouette on a deterministic, cluster-representative
  sample capped at 10,000 rows (seed 42), reports the actual
  `silhouette_sample_size`, and leaves Calinski-Harabasz and Davies-Bouldin
  full-input. The sampler uses bounded intermediate memory, validates
  impossible caps clearly, and keeps every feasible predicted cluster in the
  sampled score.
  ```

- [ ] **Step 2: Update the root release-series summary**

  In the **0.7.x** row of `CHANGELOG.md`, append this single clause after the
  existing v0.7.3 threshold-tuning description:

  ```markdown
  ; hardens `skyulf-core` pipeline safety and observability with leakage-safe
  TargetEncoder cross-fitting, per-step preprocessing metrics, correct
  transformer tracing cleanup, and bounded representative clustering
  silhouette scoring (v0.7.4)
  ```

  Preserve the existing series wording; do not create a new table row or
  change a package version.

- [ ] **Step 3: Review the release-note boundary**

  Run:

  ```bash
  git diff -- changelog/0.7.x.md CHANGELOG.md
  ```

  Expected: only completed behavior appears. Remove any future roadmap,
  enterprise, or adoption language from the release notes.

## Task 3: Validate and Commit the Documentation Artifacts

**Files:**

- Inspect: `temp/skyulf-platform-evolution-roadmap-2026-08-05.md`
- Inspect: `changelog/0.7.x.md`
- Inspect: `CHANGELOG.md`

**Interfaces:**

- Consumes: Tasks 1 and 2.
- Produces: an ignored roadmap file and one tracked, reviewable release-note
  commit.

- [ ] **Step 1: Build documentation**

  Run:

  ```bash
  source .venv/bin/activate && mkdocs build
  ```

  Expected: build exits 0. Record pre-existing advisory or navigation messages
  separately from a failure.

- [ ] **Step 2: Check tracked and untracked boundaries**

  Run:

  ```bash
  git diff --check
  git status --short
  git check-ignore -v temp/skyulf-platform-evolution-roadmap-2026-08-05.md
  ```

  Expected: no whitespace errors; the roadmap is ignored; only the changelog
  files and intentional planning records are tracked changes.

- [ ] **Step 3: Commit only tracked release documentation**

  Run:

  ```bash
  git add changelog/0.7.x.md CHANGELOG.md
  git commit -m "docs: add v0.7.4 safety release notes" \
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
  ```

  Do not add `temp/skyulf-platform-evolution-roadmap-2026-08-05.md` to Git.
