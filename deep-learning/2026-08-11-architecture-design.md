# Deep Learning Integration — Architecture Design

**Date:** 2026-08-11
**Status:** Approved for planning
**Scope:** Add deep learning (DL) model support across tabular, text,
time-series, and image data, as a parallel model family alongside the
existing scikit-learn/XGBoost/LightGBM stack, without disrupting existing
pipelines.

## 1. Executive Summary

Skyulf's node/registry/execution architecture is already model-family
agnostic: `NodeRegistry` maps a string id to a `(Calculator, Applier)` pair
that implements a generic `fit()`/predict contract, and the frontend uses one
factory (`createModelingNode`) for every model node. DL support is added by:

1. A new `skyulf-core/skyulf/deep_learning/` subpackage providing
   PyTorch-backed calculators/appliers that implement the same contract as
   `SklearnCalculator`/`SklearnApplier`.
2. A new `DLTrainingManager` in `backend/ml_pipeline/_execution/` that
   understands epoch/batch training loops, early stopping, and checkpointing
   — the one genuinely new execution primitive — built against the
   backend-neutral execution interface so it isn't Celery-specific.
3. Modality-isolated data ingestion: tabular/text reuse existing ingestion;
   time-series adds windowing; image gets a wholly new connector + dataset
   abstraction.
4. A second artifact format (`.pt`/state-dict) alongside `.joblib` in
   `ArtifactStore`.
5. New config-driven (not layer-by-layer) frontend nodes built with the
   existing `createModelingNode` factory.
6. GPU scheduling deferred to, and reused from, the Ray migration
   (branch `080`) rather than a parallel Celery-GPU-queue design.

## 2. Goals

1. Add DL models as a first-class, drop-in alternative to existing model
   nodes, using the same registry/job/artifact/frontend plumbing.
2. Support four data modalities end-to-end: tabular, text, time-series,
   image — each with proper isolation so one modality's assumptions never
   leak into another.
3. Run correctly today on CPU-only Celery workers; automatically gain GPU
   scheduling once the Ray migration lands, without a rewrite.
4. Keep the frontend UX consistent with existing nodes: preset + hyperparameter
   dropdowns, not a layer-by-layer graph builder.
5. Every new frontend option has a verified backend counterpart (per the
   repo's sync rule) before a phase is considered done.

## 3. Non-Goals

1. A dedicated layer-by-layer neural architecture builder canvas (documented
   as a possible future phase in §8, not part of this plan).
2. Building a DL-specific GPU scheduler independent of the Ray migration.
3. Distributed multi-GPU / multi-node training (single-device training only,
   consistent with current single-worker execution model).
4. Audio/video data modalities (out of scope; image is the only new
   modality in this plan).
5. Replacing or retraining any existing sklearn/XGBoost/LightGBM node —
   they remain fully supported, unchanged.
6. AutoML/neural-architecture-search. Presets are curated, not searched.

## 4. Target Components

### 4.1 `skyulf-core/skyulf/deep_learning/` (new subpackage)

Mirrors `modeling/`'s structure:

- `base.py` — `BaseDLCalculator`/`BaseDLApplier`, subclassing/conforming to
  the same `BaseModelCalculator`/`BaseModelApplier` contract from
  `modeling/base.py` (so `_node_runners.py` needs zero changes to invoke
  them). Adds DL-specific hooks: `build_model(config) -> nn.Module`,
  `training_step`, `validation_step`, `device` resolution
  (`cuda`/`mps`/`cpu` auto-detect with explicit override).
- `tabular/mlp.py` — `MLPClassifier`/`MLPRegressor` calculators, architecture
  presets (`"small"`, `"medium"`, `"wide"`) mapping to fixed layer
  configurations — no arbitrary layer count from the UI in this phase.
- `text/transformer_classifier.py` — fine-tunes a small pretrained
  encoder (e.g. DistilBERT via `transformers`) with a classification head.
- `timeseries/sequence_forecaster.py` — LSTM/Temporal-Conv forecaster,
  consuming windowed sequences from `preprocessing/time_series/`.
- `vision/image_classifier.py` — transfer-learning CNN (frozen pretrained
  backbone + trainable head), consuming the new image dataset abstraction.
- `_training_loop.py` — shared epoch/batch loop, early stopping, gradient
  clipping, used by all four modalities so behavior (progress reporting,
  checkpoint cadence, seed handling) is consistent instead of
  reimplemented four times.
- Every node registers via the existing `@node_meta` + `NodeRegistry.register`
  — no registry changes needed.

### 4.2 Data ingestion additions (`backend/data_ingestion/`)

- **Tabular DL:** no changes — reuses `LocalFileConnector`/`S3Connector` and
  polars frames unchanged.
- **Text DL:** no changes — reuses existing text ingestion + the existing
  vectorization/tokenization nodes; the transformer node adds its own
  tokenizer internally (HF `AutoTokenizer`), not a new ingestion path.
- **Time-series DL:** no new connector; adds a windowing/sequencing
  transform in `skyulf-core/skyulf/preprocessing/time_series/` that produces
  `(sequence, target)` tensors instead of a flat frame, used only by
  time-series DL nodes.
- **Image DL (new modality):** new `ImageManifestConnector` in
  `backend/data_ingestion/connectors/`. Accepts either (a) a zip/folder
  upload of images with a label CSV mapping filename→label, or (b) an
  existing tabular dataset with an image-path column. It does **not**
  produce a `SkyulfDataFrame` — it produces a lightweight manifest
  (path, label) that the DL calculator wraps in a
  `torch.utils.data.Dataset`/`DataLoader` at train time, streaming images
  from `uploads/data/images/<dataset_id>/` lazily (never loading the full
  image set into memory).

### 4.3 Training execution (`backend/ml_pipeline/_execution/`)

- New `dl_training_manager.py` — `DLTrainingManager(TrainingJobManagerBase)`.
  Same external contract as `BasicTrainingManager` (job creation, cancel,
  log, status update) but internally runs an epoch loop via
  `deep_learning/_training_loop.py`, reporting progress per-epoch (not just
  per-node) through the existing `progress_callback`/realtime event
  mechanism (`backend/realtime/events.py`), so the frontend job-progress UI
  needs no changes beyond finer-grained percentages.
- Dispatch: registered in `JobStrategyFactory` alongside `basic`/`tuned`, so
  the existing `_node_runners.py` dispatch point picks `DLTrainingManager`
  when the resolved calculator is a `BaseDLCalculator` — no new `StepType`
  needed; DL nodes still submit `StepType.TRAINING` like every other
  training node, discriminated by `model_type`/`framework` in `params`
  (same pattern as `run_mode` today).
- **Execution-backend neutrality:** `DLTrainingManager` does not call Celery
  APIs directly — it goes through whatever execution backend interface
  Ray migration plan 01 establishes (`EXECUTION_BACKEND=local|celery|ray`).
  Until that lands, it runs exactly like today's managers (in-process on the
  Celery worker). Once Ray lands, DL training jobs (the ones that actually
  benefit from a GPU) declare `num_gpus=1` via `ResourceSpec`/
  `resource_spec_for_job`, and Ray's scheduler places them on a GPU-capable
  worker — no DL-specific scheduler is built.

### 4.4 Artifact storage (`backend/ml_pipeline/artifacts/`)

- Extend `ArtifactStore` (local + S3 implementations) with a
  `save_torch`/`load_torch` pair that writes `torch.save({"state_dict":...,
  "architecture_config": ...})` under a `.pt` extension, alongside the
  existing `.joblib` methods. `discovery.py` is extended to recognize `.pt`
  artifacts the same way it recognizes `.joblib` today.
- The DL applier's `predict()` reconstructs the `nn.Module` from
  `architecture_config` (the same preset/hyperparameter dict used at train
  time) before loading the `state_dict` — avoids pickling arbitrary class
  objects (a real security/versioning risk with raw `torch.save(model)`).

### 4.5 Frontend nodes (`frontend/ml-canvas/`)

- New node files following the exact `TextClassificationNode.tsx` pattern:
  `NeuralNetworkClassifierNode.tsx`, `NeuralNetworkRegressorNode.tsx`,
  `TransformerTextClassifierNode.tsx`, `SequenceForecasterNode.tsx`,
  `ImageClassifierNode.tsx` — all built via `createModelingNode`.
- Settings panels reuse `TrainingSettings`-style components, extended with
  DL-specific fields (architecture preset dropdown, epochs, learning rate,
  batch size, early-stopping patience) instead of the sklearn
  hyperparameter fields — but the same `run_mode: basic|tuned` discriminator
  pattern, so the "Advanced/tuned" path (Optuna-driven search over a small
  DL hyperparameter space: lr, batch size, preset) reuses existing tuning
  infra rather than a new one.
- `ImageClassifierNode` additionally needs a new upstream "Image Data
  Loader" node (parallel to the existing tabular Data Loader) since image
  data doesn't flow through the standard tabular data-loader node.
- Every new dropdown/enum (architecture presets, optimizer choices) is
  cross-checked against the backend's allow-list before the phase is marked
  done, per the repo's Backend/Core ↔ Frontend Sync Rule.

### 4.6 Dependencies & environment

- New optional extra in `skyulf-core/pyproject.toml`: `dl = ["torch>=2.x",
  "transformers>=4.x"]` (only pulled in when DL nodes are used, same pattern
  as `nlp`/`geo`). A root `requirements-dl.txt` mirrors `requirements-nlp.txt`'s
  documentation style.
- No Dockerfile/base-image changes required for CPU-only phases (0-3).
  Image phase (4) documents an optional CUDA-enabled image variant, applied
  only when a GPU worker is actually provisioned — not built speculatively.

## 5. Data Flow (representative: tabular DL training)

```text
Data Loader node (unchanged)
   -> polars/pandas frame
Feature Engineering nodes (unchanged)
   -> polars/pandas frame
Neural Network Classifier node (new)
   params: {model_type: "mlp", architecture_preset: "medium",
            epochs: 50, batch_size: 32, learning_rate: 1e-3,
            early_stopping_patience: 5, run_mode: "basic"}
   -> NodeRegistry.get_calculator("mlp_classifier")  [BaseDLCalculator]
   -> DLTrainingManager (epoch loop, progress per-epoch, device resolution)
   -> ArtifactStore.save_torch(state_dict, architecture_config)
   -> MLJob marked complete, artifact URI recorded (same as sklearn today)
```

Image training differs only in the data-loading step: an `Image Data Loader`
node replaces the tabular `Data Loader`, producing a manifest instead of a
frame; everything downstream (training manager, artifact store, job
lifecycle) is identical.

## 6. Error Handling & Testing

- **Error handling:** DL-specific failure modes (CUDA OOM, NaN loss,
  divergent training) are caught inside `_training_loop.py` and surfaced
  through the existing job error/log mechanism (`_append_job_logs`,
  `JobStatus`) — no new error-reporting path invented. NaN/divergence is
  treated as a job failure with a clear log message, not a silent partial
  result.
- **Testing:** Each phase adds calculator/applier unit tests mirroring
  `tests/` patterns used for sklearn nodes (fit/predict shape and type
  checks, registry round-trip, artifact save/load round-trip), plus one
  end-to-end pipeline test per modality (small synthetic dataset, 1-2 epochs,
  asserts job completes and produces a loadable artifact) so CI stays fast.
  GPU-dependent code paths are unit-tested with a mocked/forced-CPU device
  since CI has no GPU.

## 7. Relationship to the Ray Migration (branch `080`)

The Ray migration is independently planned, approved, but **unimplemented**
(docs only, on an unmerged branch). This DL plan deliberately does not
duplicate it:

- DL training is written against the same backend-neutral execution
  interface Ray plan 01 defines, so it needs no changes when Ray lands.
- GPU declaration/scheduling for DL reuses Ray plan 04's `ResourceSpec`/
  `entrypoint_num_gpus` — DL Phase 5 (§8) is "wire `DLTrainingManager` to
  request `num_gpus>0` for image/heavy-text jobs," not "build a GPU
  scheduler."
- If the Ray migration is delayed or rejected at one of its own gates, DL
  phases 0-4 are still fully functional CPU-only (with image models using
  frozen-backbone transfer learning, which is CPU-tractable), so DL delivery
  is not blocked on Ray's timeline.

## 8. Future Considerations (explicitly out of scope now)

- A dedicated layer-by-layer DL architecture-builder canvas, for power users
  who outgrow presets — only if requested after Phase 4 ships.
- ONNX export for faster/portable inference.
- Multi-GPU / distributed training (would ride on Ray Train, itself
  explicitly deferred by the Ray migration's own non-goals).
- Audio/video modalities.
