# Skyulf Roadmap (next 1–2 quarters)

This is a living document with small, focused milestones that compound.

## Q1 2026
- Privacy-by-default feature operators
  - PII masking/redaction, hashing, bucketing for common column types
  - Retention & consent enforcement integrated with ingestion and export
- Interoperability
  - ONNX export for trained scikit-learn pipelines (where feasible)
  - MLflow-compatible packaging utilities
- Local-first LLM helper
  - Default to local (Ollama) with explicit opt-in for cloud providers
  - Frontend disclosure of context and redaction/sampling limits

## Q2 2026
- Auditability & supply chain integrity
  - SBOM (CycloneDX) generation for releases and model bundles
  - Reproducibility manifests for datasets/transforms/models
- Admin UX and governance
  - Privacy/consent/retention settings pages
  - Export/erase requests with logs

## Always accepting contributions
- UI polish and accessibility
- Docs/tutorials + translations (EN/TR)
- Tests and type hints across modules

If you want to help, open a Discussion or pick a “good first issue”.