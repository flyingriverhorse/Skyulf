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
- [Frontend design](2026-08-11-frontend-design.md) — code-grounded UI/UX
  proposal for the DL nodes: settings panel layout, live training-curve
  telemetry, image upload UX, and the exact converter/registry changes
  required, cited against real `frontend/ml-canvas/` files and line numbers.

All documents were independently validated: a rubber-duck review verified
every load-bearing architectural claim against the live codebase and caught
two blocking errors in the original training-execution design (now
corrected throughout — see the "Correction" notes in the findings and
architecture docs), and a separate agent explored the actual frontend
codebase to ground the frontend design in real, cited precedent rather than
generic React patterns.

## Decision Summary

DL models are a **parallel model family**, not a replacement: they register
with the existing `NodeRegistry`/`@node_meta` mechanism and appear as new
node types in the same ml-canvas pipeline (config-driven presets, not a
layer-by-layer architecture builder). PyTorch is the DL framework. Each data
modality (tabular, text, time-series, image) gets an isolated ingestion path
so formats never tangle. Training runs through a new **direct-fit dispatch
branch** in the pipeline engine (`_run_training`'s `is_deep_learning` check,
parallel to the existing clustering branch) — not a new job manager — since
every non-clustering node is normally routed through the sklearn-oriented
`TuningCalculator`, which cannot invoke a DL calculator's epoch loop. GPU
scheduling is deferred to the Ray migration (branch `080`), which requires
real (if modest) extension work to `resource_spec_for_job` for per-job-type
GPU differentiation — not automatic reuse.

Delivery order: **Phase 0 (shared infra) → Phase 1 (tabular DL) → Phase 2
(text DL) → Phase 3 (time-series DL) → Phase 4 (image DL) → Phase 5 (GPU
scheduling via Ray)**, each phase independently shippable and reviewable.
