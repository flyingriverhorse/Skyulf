# Deep Learning Integration — Implementation Roadmap

**Date:** 2026-08-11
**Status:** Ready for phase-by-phase execution

**Goal:** Coordinate independent, reviewable phases that add deep learning
model support without destabilizing existing pipelines. Each phase produces
working, shippable software and passes its own gate before the next begins.

## Global Constraints (apply to every phase)

- Never break existing sklearn/XGBoost/LightGBM nodes, jobs, or artifacts.
- Every DL calculator/applier implements the existing
  `BaseModelCalculator`/`BaseModelApplier` contract — no changes to
  `NodeRegistry`, `_node_runners.py` dispatch, or the public pipeline/job API
  shapes.
- New frontend dropdowns/enums are cross-checked against backend allow-lists
  before a phase is marked done (Backend/Core ↔ Frontend Sync Rule).
- After Python changes: `ruff check .`, `ruff format --check backend
  skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`, `ty check
  backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py
  celery_worker.py`.
- After frontend changes: `npm run lint`, `npx tsc --noEmit -p .`, `npm run
  build` from `frontend/ml-canvas/`.
- New heavy dependencies (`torch`, `transformers`) live behind the optional
  `dl` extra — never added to the default install.
- Every new function/method gets a 1-2 line docstring; target Python 3.11+
  idioms.
- Commits include `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

## Phase Order

| Order | Phase | Deliverable | Depends On |
|---|---|---|---|
| 0 | Shared DL infra | `deep_learning/` skeleton, `DLTrainingManager`, `.pt` artifact format, one working node (tabular MLP) end-to-end | None |
| 1 | Tabular DL | MLP classifier/regressor nodes with architecture presets, tuned-mode support | Phase 0 |
| 2 | Text DL | Transformer text-classification node (fine-tuned pretrained encoder) | Phase 0 |
| 3 | Time-series DL | Windowing transform + LSTM/TCN forecaster node | Phase 0 |
| 4 | Image DL | Image ingestion (new modality) + CNN/transfer-learning classifier node | Phase 0 |
| 5 | GPU scheduling via Ray | Wire `DLTrainingManager` to request GPU resources once Ray migration (branch `080`) lands | Phase 0-4, Ray migration plans 01+04 |

## Gates

1. **Phase 0 gate:** A tabular MLP node trains end-to-end on a synthetic
   dataset through the real job pipeline (submit → Celery → epochs →
   artifact saved as `.pt` → loadable → predict), with cancel and log
   behavior working identically to an existing sklearn node.
2. **Phase 1 gate:** MLP classifier/regressor pass accuracy/R² sanity checks
   on a known small dataset (e.g. iris/diabetes) comparable to a baseline
   sklearn model; tuned mode (Optuna over lr/batch-size/preset) completes.
3. **Phase 2 gate:** Transformer node fine-tunes on a small text dataset
   within an acceptable CI time budget (few minutes, capped epochs/dataset
   size) and produces sane accuracy; falls back cleanly to CPU.
4. **Phase 3 gate:** Forecaster node produces valid multi-step predictions on
   a synthetic seasonal series with the correct windowing semantics matching
   `preprocessing/time_series/`'s existing conventions.
5. **Phase 4 gate:** Image ingestion accepts a zip/folder + label CSV,
   streams via `DataLoader` without loading the full set into memory, and
   the classifier trains on a small image dataset (e.g. a few hundred
   thumbnails) via frozen-backbone transfer learning on CPU within a
   reasonable time budget.
6. **Phase 5 gate:** Only entered once the Ray migration's own compute gate
   (plan 04) has independently passed; DL jobs declaring `num_gpus=1` are
   observed running on a GPU-capable Ray worker with measurable speed-up
   over CPU for at least the image and transformer nodes.

## Execution Rule

Do not begin a phase while an earlier phase's gate has unresolved
correctness, security, or performance failures. Phase 5 is independently
gated on the Ray migration's own review process — do not couple DL delivery
timeline to Ray's; phases 0-4 ship and are useful standalone.

---

## Phase 0 — Shared DL Infra

**Objective:** Prove the whole plumbing path (registry → job → training loop
→ artifact → predict) with the simplest possible model, so phases 1-4 only
add modality-specific code.

**Create:**
- `skyulf-core/skyulf/deep_learning/__init__.py`
- `skyulf-core/skyulf/deep_learning/base.py` — `BaseDLCalculator`,
  `BaseDLApplier`, `resolve_device(preferred: str | None) -> torch.device`.
- `skyulf-core/skyulf/deep_learning/_training_loop.py` — `TrainingLoopConfig`
  (epochs, batch_size, lr, patience), `run_training_loop(model, train_ds,
  val_ds, config, progress_callback, log_callback) -> TrainingResult`.
- `skyulf-core/skyulf/deep_learning/tabular/mlp.py` — `MLPClassifierCalculator`
  / `MLPClassifierApplier` (classification only, in this phase — regression
  is Phase 1), registered as `"mlp_classifier"`.
- `backend/ml_pipeline/_execution/dl_training_manager.py` —
  `DLTrainingManager(TrainingJobManagerBase)`.
- `backend/ml_pipeline/artifacts/torch_format.py` — `save_torch`/`load_torch`
  helpers shared by `local.py`/`s3.py`.
- `skyulf-core/pyproject.toml` — `dl` optional-dependency group
  (`torch>=2.2,<3.0`).
- `requirements-dl.txt` (root) — mirrors `requirements-nlp.txt` style/comments.
- Tests: `skyulf-core/tests/deep_learning/test_mlp_classifier.py`,
  `tests/test_dl_training_manager.py`, `tests/test_torch_artifact_format.py`.

**Modify:**
- `backend/ml_pipeline/artifacts/local.py`, `s3.py` — add `.pt` save/load
  dispatch alongside existing `.joblib` methods.
- `backend/ml_pipeline/artifacts/discovery.py` — recognize `.pt` artifacts.
- `backend/ml_pipeline/_execution/strategies.py` — `JobStrategyFactory`
  picks `DLTrainingManager` when the resolved calculator is a
  `BaseDLCalculator` instance.

**Do not touch:** `NodeRegistry`, `StepType` enum, public API schemas,
frontend (no node exposed yet — Phase 0 validates internals only, optionally
behind a feature flag or dev-only registry entry).

---

## Phase 1 — Tabular DL

**Objective:** Ship the first user-facing DL nodes.

**Create:**
- `skyulf-core/skyulf/deep_learning/tabular/mlp.py` — extend with
  `MLPRegressorCalculator`/`Applier`, registered as `"mlp_regressor"`.
  Architecture presets: `"small"` (1 hidden layer), `"medium"` (2 layers),
  `"wide"` (2 wider layers) — fixed configs, no free-form layer editing.
- `frontend/ml-canvas/src/modules/nodes/modeling/NeuralNetworkClassifierNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/modeling/NeuralNetworkRegressorNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/modeling/DLTrainingSettings.tsx` —
  preset/epochs/batch-size/lr/early-stopping fields, `run_mode: basic|tuned`.

**Modify:**
- Tuning: extend the existing Optuna search-space handling (wherever
  `advanced_tuning_manager.py`/`_node_runners.py` inject search spaces) to
  accept a small DL search space (`lr`, `batch_size`, `architecture_preset`)
  when the calculator is a `BaseDLCalculator`.

**Verify:** frontend preset dropdown options == backend
`ARCHITECTURE_PRESETS` allow-list (sync-rule check).

---

## Phase 2 — Text DL

**Objective:** Add a fine-tuned transformer classifier as an option beyond
TF-IDF/Naive-Bayes.

**Create:**
- `skyulf-core/skyulf/deep_learning/text/transformer_classifier.py` —
  `TransformerTextClassifierCalculator`/`Applier`, using
  `transformers.AutoModelForSequenceClassification` +
  `AutoTokenizer`, registered as `"transformer_text_classifier"`, tagged
  `["text"]` (same tag convention as existing text-only models).
- `frontend/ml-canvas/src/modules/nodes/modeling/TransformerTextClassifierNode.tsx`

**Modify:**
- `skyulf-core/pyproject.toml` — `dl` extra gains `transformers>=4.x`.
- `TextClassificationNode.tsx`'s model dropdown (or a new sibling node,
  per design §4.5) to surface the transformer option gated behind the
  `text` tag filter already in place.

**Note:** cap default epochs/dataset size sensibly in tests/CI (small model,
few steps) since transformer fine-tuning is CPU-slow — document the expected
wall-clock budget in the node's description/tooltip.

---

## Phase 3 — Time-Series DL

**Objective:** Add sequence forecasting.

**Create:**
- `skyulf-core/skyulf/preprocessing/time_series/windowing.py` — sequence
  windowing transform producing `(sequence, target)` pairs, consistent with
  existing time-series node conventions (check `time_column`/`group_by`
  handling already present in that package).
- `skyulf-core/skyulf/deep_learning/timeseries/sequence_forecaster.py` —
  `SequenceForecasterCalculator`/`Applier` (LSTM or Temporal-Conv, preset
  driven), registered as `"sequence_forecaster"`.
- `frontend/ml-canvas/src/modules/nodes/modeling/SequenceForecasterNode.tsx`

**Verify:** windowing semantics match existing time-series preprocessing
(no leakage across the train/val split boundary — reuse existing
time-series-aware CV/splitting code rather than reimplementing).

---

## Phase 4 — Image DL

**Objective:** Add the first new data modality end-to-end.

**Create:**
- `backend/data_ingestion/connectors/image_manifest.py` —
  `ImageManifestConnector` (zip/folder + label CSV, or image-path column in
  an existing tabular dataset). Produces a manifest (path, label), not a
  `SkyulfDataFrame`.
- `backend/data_ingestion/schemas/image_ingestion.py` — request/response
  schemas for the new upload path.
- `skyulf-core/skyulf/deep_learning/vision/image_classifier.py` —
  `ImageClassifierCalculator`/`Applier`, frozen pretrained backbone (e.g.
  torchvision `resnet18`/`mobilenet_v3_small`) + trainable head, registered
  as `"image_classifier"`.
- `frontend/ml-canvas/src/modules/nodes/data/ImageDataLoaderNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/modeling/ImageClassifierNode.tsx`

**Modify:**
- `backend/data_ingestion/router.py` — new endpoint(s) for image
  upload/manifest creation.
- `skyulf-core/pyproject.toml` — `dl` extra gains `torchvision`.
- Storage: confirm `uploads/data/images/<dataset_id>/` path handling reuses
  the existing safe-path containment logic in
  `LocalFileConnector.resolve_safe_path` rather than duplicating it.

**Verify:** memory does not scale with dataset size (streamed via
`DataLoader`, not eagerly loaded); safe-path containment tested against
path-traversal attempts in the zip/manifest upload, consistent with the
existing connector's security posture.

---

## Phase 5 — GPU Scheduling via Ray

**Objective:** Give DL training real GPU acceleration, once available,
without a DL-specific scheduler.

**Precondition:** Ray migration plans 01 (execution backend foundation) and
04 (distributed compute / `ResourceSpec`) have passed their own gates and
been merged.

**Modify:**
- `backend/ml_pipeline/_execution/dl_training_manager.py` — request
  `ResourceSpec(num_gpus=1)` for image/transformer job types via
  `resource_spec_for_job("dl_training", settings)`, `num_gpus=0` for
  lightweight tabular MLP jobs (CPU is sufficient).
- `skyulf-core/skyulf/deep_learning/base.py` — `resolve_device()` honors a
  `CUDA_VISIBLE_DEVICES`/Ray-provided device hint instead of naive
  `torch.cuda.is_available()` auto-detection, so placement matches what Ray
  actually reserved.

**Do not:** build a parallel GPU queue, a DL-specific resource scheduler, or
a second `EXECUTION_BACKEND` value. This phase is wiring, not new
infrastructure.

**Verify:** an image-classifier job submitted with Ray backend enabled lands
on a GPU-capable worker (observable via Ray dashboard / job resource
metadata) and completes faster than the CPU baseline from Phase 4.
