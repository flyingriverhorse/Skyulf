# Deep Learning Integration

This folder contains the investigation findings and implementation plan for
adding deep learning (DL) model support to Skyulf, alongside the existing
scikit-learn/XGBoost/LightGBM model family.

## Documents

- [Findings](2026-08-11-findings.md) — codebase investigation: what exists
  today, what's reusable, what's missing, and the constraints that shape the
  design.
- [Architecture design](2026-08-11-architecture-design.md) — the approved
  target architecture: new `deep_learning` subpackage, data modality
  separation, training execution model, artifact format, frontend nodes, and
  how this relates to the (not-yet-merged) Ray migration on branch `080`.
- [Implementation roadmap](2026-08-11-implementation-roadmap.md) — phased
  delivery plan (Phase 0 shared infra → tabular → text → time-series → image
  → GPU/Ray wiring), with file-level scope, dependencies, and gates per phase.

## Decision Summary

DL models are a **parallel model family**, not a replacement: they register
with the existing `NodeRegistry`/`@node_meta` mechanism and appear as new
node types in the same ml-canvas pipeline (config-driven presets, not a
layer-by-layer architecture builder). PyTorch is the DL framework. Each data
modality (tabular, text, time-series, image) gets an isolated ingestion path
so formats never tangle. Training execution is written against a
backend-neutral interface so it runs on today's Celery worker (CPU) and
transparently gains GPU scheduling once the Ray migration (branch `080`)
lands, reusing Ray's `ResourceSpec`/`entrypoint_num_gpus` mechanism instead of
inventing a second GPU queue.

Delivery order: **Phase 0 (shared infra) → Phase 1 (tabular DL) → Phase 2
(text DL) → Phase 3 (time-series DL) → Phase 4 (image DL) → Phase 5 (GPU
scheduling via Ray)**, each phase independently shippable and reviewable.
