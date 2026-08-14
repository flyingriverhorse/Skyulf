# Enterprise Readiness Investigation

> **How this folder is used now (2026-08-11):** as **research to mine, not a
> schedule to execute.** The active plan is
> [`../growth/2026-08-11-growth-plan.md`](../growth/2026-08-11-growth-plan.md).
> `master-fix-list.md`'s Phase 0–18 numbering is a grouping of findings, not
> an execution order. Items are promoted into the growth plan only after
> being re-verified against the code — re-verification has already found
> both under-stated severity (see `2026-08-11-bug-hunt.md`) and at least one
> finding that is a positioning gap rather than a build task (code export
> already ships). The work in here is good; treat its *conclusions* as
> evidence and its *sequencing* as unvalidated.

**Date:** 2026-08-11 (eight rounds — see "Investigation rounds" below)
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
| [2026-08-11-smooth-experience-fixes.md](2026-08-11-smooth-experience-fixes.md) | Round 3. Concrete, verified first-run-friction and janky-UX findings: no reachable sample dataset, generic error messages, missing trust signals (WS live indicator), inconsistent success toasts, undo-history eviction, unused `Skeleton` component. Top 3 prioritized fixes. |
| [2026-08-11-differentiation-strategy.md](2026-08-11-differentiation-strategy.md) | Round 3. Evidence-backed competitive positioning: what makes DataRobot/H2O/Databricks/KNIME/RapidMiner/Dataiku/Modal/Baseten win or lose users (cited), 5 ranked differentiation bets for Skyulf, and a concrete list of what's missing specifically in `skyulf-core` today (AutoML layer, enforced leakage guardrails, forecasting models, partitionable calculator contract, artifact versioning). |
| [2026-08-11-security-review.md](2026-08-11-security-review.md) | Round 4. Dedicated security-specialist pass: 2 confirmed Medium-severity SSRF findings (datasource-controlled S3 endpoint routing bypasses the existing connector guard via EDA and pipeline-resolution paths); no SQLi/command-injection/XSS/unsafe-deserialization/committed-secrets confirmed. |
| [2026-08-11-scale-load-audit.md](2026-08-11-scale-load-audit.md) | Round 4. Production scale/load readiness: no per-tenant resource quotas (only an IP-keyed rate limiter), full-dataset in-memory processing risks OOM, undefined Celery worker concurrency, SQLite/local-disk won't survive multi-instance deployment, non-virtualized 10,000+ row result tables. |
| [2026-08-11-data-governance-audit.md](2026-08-11-data-governance-audit.md) | Round 4. PII/compliance/SOC2/GDPR readiness: narrow, advisory-only PII detection with no masking workflow; no retention/DSAR program; no encryption at rest; biggest procurement blocker is the lack of a comprehensive, immutable audit trail. |
| [2026-08-11-testing-ci-audit.md](2026-08-11-testing-ci-audit.md) | Round 4. Test/CI depth audit: solid gates exist (Ruff/Ty/ESLint/tsc, coverage floors, OSV+CodeQL scanning) but the one full-inference test is skipped on CI, canvas drag-and-drop has zero real-gesture E2E coverage, and the planned DL/Ray work has no safety net at its exact integration boundary. |
| [2026-08-11-user-complaints-research.md](2026-08-11-user-complaints-research.md) | Round 4. Cited, sourced research (TrustRadius/HN/Ars Technica) on what real users of comparable AutoML/no-code tools complain about: vendor lock-in/no code export (strongest signal), pricing opacity, black-box automation, and — a novel, highly actionable finding — no per-node data preview ("the schema guessing game"), directly analogous to Skyulf's own canvas architecture. |
| [2026-08-11-round4-synthesis.md](2026-08-11-round4-synthesis.md) | Round 4 overview: how the 5 round-4 docs cross-validate each other and rounds 1-3, and what net-new items were added to the master fix list (Phase 10, 11, and a Phase 9 addition). |
| [2026-08-11-bug-hunt.md](2026-08-11-bug-hunt.md) | Round 5. 9 confirmed, reproducible bugs (not opinions) with exact repro steps: a cross-process duplicate-job race, and — most severe — two feature-engineering nodes (Lag Features, Rolling Aggregate) that silently misalign `X`/`y` on unsorted input, a real train/label-corruption risk shipping today. |
| [2026-08-11-i18n-mobile-crossbrowser-audit.md](2026-08-11-i18n-mobile-crossbrowser-audit.md) | Round 5. No i18n framework exists (fully hardcoded English, no RTL); canvas mobile/tablet support has no explicit policy; browser testing is Chromium-only; metric/p-value formatting is inconsistent across the app. |
| [2026-08-11-user-observability-audit.md](2026-08-11-user-observability-audit.md) | Round 5. Strong diagnostic pieces already exist (job logs, preview-node failure cards, SHAP diagnostics) but are fragmented across disconnected UI surfaces instead of one canonical per-run timeline. |
| [2026-08-11-api-contract-drift-audit.md](2026-08-11-api-contract-drift-audit.md) | Round 5. OpenAPI spec exists but isn't consumed for frontend codegen; confirmed `JobInfo` field drift, force-cast EDA status strings, and WebSocket frames validated on the backend but not the frontend despite Zod already being installed. |
| [2026-08-11-round5-synthesis.md](2026-08-11-round5-synthesis.md) | Round 5 overview: cross-validation against rounds 1-4 and the net-new master-fix-list additions (Phase 12 confirmed bugs, Phase 13 API contract hardening, Phase 14 i18n/mobile/cross-browser, plus a Phase 9 addition). |
| [2026-08-11-round6-gap-audit.md](2026-08-11-round6-gap-audit.md) | Round 6. Final meta gap-check against rounds 1-5 plus both feasibility studies: model registry only supports one globally-"live" deployment (no per-pipeline/environment scoping); zero licensing/billing/entitlement enforcement code exists anywhere; no cost/FinOps visibility; `.env.example` omits all AWS/S3 vars the app actually reads; notebook export was never checked for standalone execution correctness; no backup/DR/multi-region story anywhere. |
| [2026-08-11-core-dx-improvements.md](2026-08-11-core-dx-improvements.md) | Round 6. `skyulf-core` developer-experience audit (hands-on usage, not just reading): calculators silently no-op on bad config instead of raising; no sklearn `BaseEstimator`/`TransformerMixin` protocol (confirmed `Pipeline` incompatibility); bare untyped `dict` configs; standalone (non-canvas) usage genuinely works today but is undocumented. |
| [2026-08-11-core-coverage-gaps.md](2026-08-11-core-coverage-gaps.md) | Round 6. Full `NodeRegistry` inventory plus algorithm-coverage diff vs. sklearn/feature-engine/category_encoders/imbalanced-learn: no forecasting models (ARIMA/Prophet), no CatBoost, no cyclical calendar encoding, no VIF/multicollinearity node, no group-aware CV — imbalanced-learn and NLP coverage confirmed already strong. |
| [2026-08-11-core-docs-onboarding.md](2026-08-11-core-docs-onboarding.md) | Round 6. `skyulf-core`'s own README/docstrings/packaging-metadata: a strong 438-line README and 9 example notebooks already exist; the "add a new node" guide is good but lives outside `skyulf-core/`; docstring coverage is inconsistent on node source files; `pyproject.toml` is nearly empty with real metadata stuck in `setup.py`. |
| [2026-08-11-core-quickwin-tech-research.md](2026-08-11-core-quickwin-tech-research.md) | Round 7. Web research into concrete quick-win tech additions: MLflow-skinny one-line fit-hook (42.5M downloads/month), CatBoost nodes (direct copy of the existing LightGBM lazy-dep pattern), Pandera-backed schema-capture/drift node, Narwhals as an additive third engine backend, StatsForecast for classical forecasting, DuckDB scoped to ingestion convenience only. Ranked top-5 cheapest/highest-value list included. |
| [2026-08-11-core-differentiation-research.md](2026-08-11-core-differentiation-research.md) | Round 7. External web evidence for genuine `skyulf-core` whitespace: the existing leakage-safe calculator/applier split and JSON-artifact model are structural advantages the wider ecosystem is independently converging toward wanting, but aren't marketed anywhere; also identifies Narwhals-as-engine-abstraction as the single most strategically significant, uniquely-combinable differentiator. |
| [2026-08-11-papers-preprocessing-research.md](2026-08-11-papers-preprocessing-research.md) | Round 8. Scientific-literature scan for preprocessing/feature-engineering: optimal/monotonic binning (`optbinning`), mutual-information/distance-correlation feature selection, PyOD-backed outlier detection — all citing real papers with maintained OSS packages to wrap. |
| [2026-08-11-papers-automl-tuning-research.md](2026-08-11-papers-automl-tuning-research.md) | Round 8. Scientific-literature scan for AutoML/tuning: multi-objective tuning via Optuna, ASHA scheduler for the Ray migration, cross-run warm-starting from Skyulf's own historical tuning data, OBOE-style model recommendation, TabPFN zero-tuning baseline node. |
| [2026-08-11-papers-dl-training-research.md](2026-08-11-papers-dl-training-research.md) | Round 8. Scientific-literature scan beyond basic loss curves for the DL module: Captum integrated-gradients explainability node, LR range finder pre-flight check, `cleanlab` confident-learning label-quality report (reusable across classical-ML and DL canvases). |
| [2026-08-11-papers-mlops-drift-research.md](2026-08-11-papers-mlops-drift-research.md) | Round 8. Scientific-literature scan for drift/monitoring/reproducibility: multivariate drift detection via `alibi-detect`, multi-seed reproducibility scoring grounded in the NeurIPS reproducibility checklist, and TFDV production evidence validating the already-planned schema-capture priority. |

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

**Round 3** (smooth-experience-fixes.md, differentiation-strategy.md): six
more subagents — two `general-purpose` agents auditing `skyulf-core`
itself (feature completeness vs. the ecosystem, and internal architecture
depth); one agent simulating a brand-new user's first 10 minutes; one
`rubber-duck` agent hunting specifically for janky/untrustworthy-feeling
issues not already documented; one `research` agent gathering
cited, verifiable evidence on what differentiates leading AutoML/no-code
ML platforms (and what their own users complain about); and one
`general-purpose` agent auditing post-training MLOps lifecycle gaps
(monitoring, serving, feature stores, cost visibility, code-export). Two
independent agents (the first-run-UX one and the smooth-experience
rubber-duck) converged on the identical root-cause finding from different
angles — no sample dataset is reachable anywhere in the UI despite example
data already existing in the repo — which is treated as high-confidence
and prioritized accordingly.

**Round 4** (security-review.md, scale-load-audit.md,
data-governance-audit.md, testing-ci-audit.md,
user-complaints-research.md): five subagents ran in parallel to close the
remaining investigation gaps identified after round 3 — a dedicated
`security-review` specialist (SSRF, injection, secrets, deserialization,
XSS), a `general-purpose` scale/load audit (DB queries, memory, worker
concurrency, quotas, frontend rendering at scale), a `general-purpose`
data-governance audit (PII, retention, SOC2/GDPR readiness), a `research`
agent gathering cited external evidence of real user complaints about
comparable AutoML/no-code platforms (explicitly requested, since internal
audits are blind to what actually drives user churn), and a
`general-purpose` testing/CI depth audit. All five completed with
file:line-cited, high-confidence findings; unlike rounds 1-2, no
cross-agent factual errors were found this round, though the
user-complaints research **independently corroborated** two existing
differentiation bets (black-box automation, code lock-in/export) and
surfaced one genuinely new, highly actionable finding — no per-node data
preview exists in the canvas, which a real competitor tool was built
specifically to solve. See
[2026-08-11-round4-synthesis.md](2026-08-11-round4-synthesis.md) for the
full cross-validation writeup.

**Round 5** (bug-hunt.md, i18n-mobile-crossbrowser-audit.md,
user-observability-audit.md, api-contract-drift-audit.md): four more
subagents ran in parallel in direct response to "is there anything left
to investigate — code, UI, user, or bug related?" Each was explicitly
scoped to ground not covered by the prior 14 subagents, and instructed to
cross-check against those docs rather than duplicate them — no overlap
was reported back. The bug-hunt agent moved past architecture/design
opinions entirely and found 9 concrete, reproducible bugs with exact
repro steps, the most severe being a train/label misalignment bug shared
by two shipped feature-engineering nodes (Lag Features, Rolling
Aggregate) that silently corrupts model training on unsorted input — this
is treated as release-blocking, independent of any other phase. The other
three agents closed out i18n/mobile/cross-browser (previously entirely
uninvestigated), end-user-facing debuggability (distinct from the
ops-facing observability already covered in backend-blockers.md), and API
contract drift beyond the already-documented node-parameter duplication
pattern. See [2026-08-11-round5-synthesis.md](2026-08-11-round5-synthesis.md)
for the full writeup.

**Round 6** (round6-gap-audit.md, core-dx-improvements.md,
core-coverage-gaps.md, core-docs-onboarding.md): four final subagents ran
in parallel — one `explore` agent doing a pure meta gap-check (reading
every prior round's docs plus both feasibility studies in full, then
verifying against the live codebase only the areas not yet covered:
deployment/packaging, licensing/billing, cost/FinOps, model registry
depth, backup/DR, notebook-export correctness) and three `explore` agents
specifically researching `skyulf-core` improvements as its own product
surface (requested directly, distinct from the canvas/platform lens every
prior round used) — hands-on DX/ergonomics, algorithm/feature coverage vs.
the wider ML ecosystem, and docs/onboarding. The gap-audit explicitly
cross-checked its 10-item candidate list against all prior docs first and
confirmed 4 of 10 areas (deployment/packaging, data connectors, plugin
SDK, onboarding) were already adequately covered, reporting only the 6
genuinely new findings. All three `skyulf-core` agents independently
confirmed the library already has real strengths (a working standalone
usage path with no canvas/registry ceremony required, a strong 438-line
README, 9 example notebooks, comprehensive imbalanced-learn/NLP coverage)
alongside concrete, cheap-to-fix gaps (bad-config silent no-ops, no
sklearn Pipeline interop, missing CatBoost/forecasting models, docstring
coverage holes on the actual node files). Folded into the master fix list
as Phase 16 (16a-16d).

**Round 7** (core-quickwin-tech-research.md, core-differentiation-research.md):
two `research` agents ran in parallel answering the user's direct question
of how `skyulf-core` can differentiate itself from (not clone) the wider
ecosystem — one focused on concrete, cheap technology additions with strong
2024-2025 adoption evidence (live PyPI download-count queries, official docs),
the other on genuine external whitespace grounded in real HN/practitioner
pain points. Both independently converged on the same two headline findings:
(1) the library's existing leakage-safe calculator/applier split and
JSON-artifact model are already a structural advantage that just isn't
marketed anywhere, and (2) adopting Narwhals as an additive internal
expression-layer/engine-backend is the single most strategically significant
move available — it's the only path that combines genuine dataframe-engine
agnosticism with skyulf-core's leakage-safe fit/apply contract, a
combination no competitor (Narwhals, skrub, feature-engine, sklearn) offers
today. Concrete quick wins ranked cheapest-to-highest-value: MLflow-skinny
fit-hook, CatBoost nodes, Pandera schema-drift node, Narwhals-as-engine,
StatsForecast forecasting nodes. Folded into the master fix list as Phase 17
(17a-17b).

**Round 8** (papers-preprocessing-research.md, papers-automl-tuning-research.md,
papers-dl-training-research.md, papers-mlops-drift-research.md): four
`research` agents scanned arXiv/conference proceedings — both recent
(2023-2025) and older-but-overlooked papers — in direct response to the
user's ask for scientific articles that "no one cared but actually could be
useful," split across preprocessing, AutoML/tuning, DL training
diagnostics, and MLOps/drift. Every finding cites a real paper URL and,
where one exists, a maintained OSS package to wrap (`optbinning`, `pyod`,
`alibi-detect`, `captum`, `cleanlab`, `tabpfn`) rather than requiring
published algorithms to be reimplemented from scratch. Standout findings:
optimal/monotonic binning as the single best effort-to-benefit preprocessing
addition; multi-objective tuning and an ASHA scheduler that plugs directly
into the already-planned Ray migration; a Captum-backed explainability node
and a `cleanlab` label-quality report reusable across both the classical-ML
and DL canvases; and TFDV's own production evidence confirming that the
already-planned Pandera-style schema capture (Phase 17a) should stay
prioritized ahead of fancier statistical drift metrics. Folded into the
master fix list as Phase 18 (18a-18d).

## Suggested sequencing

See [2026-08-11-master-fix-list.md](2026-08-11-master-fix-list.md) for the
full phased plan (Phase 0 through Phase 9). Short version: identity +
multi-tenancy + production data/storage plane first (nothing else is safe
to build on top of it); accessibility fixes as their own priority tier, not
folded into general polish; shared frontend infrastructure (`DataTable`,
`StatusBadge`, design tokens) before any individual page redesign, so pages
aren't redone twice; a set of unusually cheap, high-leverage quick wins
(Phase 8 — sample dataset, WS live indicator, missing toasts) that reuse
existing plumbing and should be done early/in parallel; and the
competitive-differentiation bets (Phase 9) that make Skyulf different, not
just at parity, with two `skyulf-core` foundational items (partitionable
calculator contract, versioned artifacts) that should land before more
node types (including deep learning) get added.

**Relationship to the deep-learning plan:** [../deep-learning/](../deep-learning/README.md)
is largely orthogonal but has real sequencing dependencies surfaced across
rounds 2 and 3: the tuning engine's size/complexity and the lack of
pipeline schema versioning (round 2), plus the lack of a
partitionable/stateless calculator contract and versioned artifact schema
in `skyulf-core` itself (round 3) — should ideally be addressed before or
alongside the DL node additions, since DL adds exactly the kind of new
node types/schema changes that make all four gaps more costly to fix
later, and the round-3 architecture audit found these same gaps directly
threaten the planned Ray migration working smoothly.
