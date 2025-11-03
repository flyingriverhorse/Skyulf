# NGI Zero Concept Note — Skyulf: Privacy-Preserving, Self‑Hosted MLOps Building Blocks

Date: 2025‑11‑03
Applicant: Individual (open source maintainer)
License: Apache‑2.0
Repo: https://github.com/flyingriverhorse/Skyulf-mlflow

## Problem and public interest

EU SMEs, municipalities, and civic‑tech teams need to analyze sensitive data but cannot rely on black‑box cloud AI due to legal, privacy, and sovereignty constraints. Today’s ML tooling is either cloud‑centric or heavyweight; self‑hosted options lack privacy‑by‑design features and interoperable packaging that makes models portable across systems.

## Objective

Deliver a set of open building blocks to run privacy‑preserving ML on‑premises:
- Differential‑privacy training and data‑minimization utilities, with reports on privacy budgets and risk.
- A pluggable, secure Federated Learning (FL) adapter (aggregator + client) that keeps data local.
- Interoperable packaging: ONNX export/import and MLflow‑compatible model bundles for reproducible deployment.
- Auditability and reproducibility: provenance (W3C PROV), signed artifacts, and Software Bill of Materials (CycloneDX).

These integrate into the existing FastAPI app while remaining reusable as standalone components.

## Outcomes and impact

- Privacy & sovereignty: Data remains on‑prem; DP limits information leakage; FL enables cross‑site collaboration without data sharing.
- Interoperability & resilience: ONNX/MLflow compatibility lowers switching costs; simple API and CLI promote adoption.
- Public benefit: Templates and tutorials target municipalities and SMEs; results fully FOSS and documented.

## Work plan (6 months, €30k)

Milestone 1 (M1–M2): Differential privacy core and risk reporting (MVP)
- DP‑SGD training wrapper for scikit‑learn‑compatible estimators and a simple trainer API.
- Privacy ledger (ε/δ budgets) and dataset risk heuristics; data minimization helpers (drop/blur/clip).
- Tests, examples, quickstart docs; initial STRIDE threat model and secure‑by‑default configs.

Milestone 2 (M3–M4): Federated Learning adapter (MVP)
- Minimal self‑hosted aggregator + client leveraging Flower as an optional backend to reduce complexity.
- Signed model update flow and pluggable aggregation (FedAvg baseline); docker‑compose reference deployment.
- End‑to‑end demo on a public synthetic dataset; reproducible scripts.

Milestone 3 (M5): Interoperable packaging (MVP)
- ONNX export/import for trained models; minimal MLflow‑compatible bundle (no external server required).
- Lightweight provenance log (run metadata + hashes) and optional CycloneDX SBOM for artifacts.
- CLI utilities to validate bundles and print provenance.

Milestone 4 (M6): Hardening and documentation
- CI checks, security baselines, and admin/developer docs with tutorials.
- Consolidated demo showing on‑prem DP training and a simple FL run; publish tagged release.

## Deliverables per milestone

- Code (Apache‑2.0), unit/integration tests, docs, and tagged releases.
- OpenAPI 3.1 spec updates; demo notebooks and minimal docker‑compose deployment.
- Security documentation: threat model, security headers, and hardening guides.

## Team and feasibility

The repository already provides a FastAPI application with async DB, Celery background tasks, feature engineering, and training utilities. The 6‑month scope prioritizes MVPs for DP, FL, and packaging, each shippable and reusable. CI (pytest + mypy + flake8) will run on every PR; each milestone ships a tagged release.

## Budget (high level)

- Personnel: ~5–5.5 person‑months (architecture, dev, tests, docs, community) — €27k
- Dissemination/infra/misc: documentation, demos, minimal pilot support — €3k

## Sustainability

The features are upstreamed to a permissive FOSS repo; they are packaged as reusable Python modules to encourage downstream reuse. We will maintain long‑term via community issues/PRs and document a lightweight governance model.

## Risks and mitigations

- Heavy dependencies or platform friction → keep core modules light, documented, and with minimal deps.
- Privacy guarantees misunderstood → clear docs, examples, and explicit limitations; optional formal review.
- Adoption barriers → ONNX/MLflow compatibility and easy APIs; docker‑compose examples and tutorials.

## Success criteria

- DP and FL MVPs shipped with tests, docs, and examples; reproducible demo showing on‑prem DP training + a basic FL round.
- Interoperable packaging: ONNX export/import and a minimal MLflow‑compatible bundle validated in two runtimes.
- At least one tutorial or small pilot by a civic/SME team using the demo stack.
