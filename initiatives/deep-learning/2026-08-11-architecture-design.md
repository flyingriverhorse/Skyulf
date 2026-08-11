# Deep Learning Integration — Architecture Design

**Date:** 2026-08-11
**Status:** Approved for planning
**Scope:** Add deep learning (DL) model support across tabular, text,
time-series, and image data, as a parallel model family alongside the
existing scikit-learn/XGBoost/LightGBM stack, without disrupting existing
pipelines.

> **Revision note:** this document was independently validated by a
> rubber-duck review against the live codebase after the first draft. Two
> blocking corrections are folded in throughout (see §4.3, §5): there is no
> `JobStrategyFactory`-based manager-selection seam, and every non-clustering
> training node — regardless of run mode — is normally routed through the
> sklearn-oriented `TuningCalculator`, which cannot invoke a DL calculator's
> `fit()`. The corrected design adds an explicit direct-fit dispatch branch
> instead of a new training manager. All other sections were confirmed
> accurate against the code, with additional scoping corrections noted
> inline in §4.4, §4.5, and §7.

## 1. Executive Summary

Skyulf's node/registry/execution architecture is already model-family
agnostic: `NodeRegistry` maps a string id to a `(Calculator, Applier)` pair
that implements a generic `fit()`/predict contract, and the frontend uses one
factory (`createModelingNode`) for every model node. DL support is added by:

1. A new `skyulf-core/skyulf/deep_learning/` subpackage providing
   PyTorch-backed calculators/appliers that implement the same contract as
   `SklearnCalculator`/`SklearnApplier`.
2. A new **direct-fit dispatch branch** inside `_run_training`
   (`backend/ml_pipeline/_execution/engine/_node_runners.py`), parallel to
   the existing clustering direct-fit branch, that calls a DL calculator's
   own `fit()` — implementing the epoch/batch training loop, early stopping,
   and checkpointing — bypassing the sklearn-oriented `TuningCalculator`
   entirely. No new training manager; the existing
   `BasicTrainingManager`/`AdvancedTuningManager` DB-row lifecycle is
   reused unchanged.
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

> **Correction (2026-08-11, post rubber-duck validation):** the original
> version of this section proposed a new `DLTrainingManager` selected by
> `JobStrategyFactory` based on calculator type. That mechanism does not
> exist: `BasicTrainingManager`/`AdvancedTuningManager` only manage the job
> DB row (create/cancel/status/list) and `JobStrategyFactory.get_strategy_by_job`
> dispatches purely on the job's `run_mode` column
> (`backend/ml_pipeline/_execution/strategies.py:191`) — it never sees a
> calculator instance, so it cannot "pick a manager based on calculator
> type." The actual model-fitting call happens inside the pipeline engine's
> `_node_runners.py`, and — critically — **every non-clustering training
> node, regardless of `run_mode`, is routed through `TuningCalculator`**
> (`_run_training` → `_run_training_tuned` →
> `StatefulEstimator.fit_predict` → `TuningCalculator.fit`,
> `_node_runners.py:497-533`, `_tuning/engine.py:271-357`). `TuningCalculator`
> converts X/y to NumPy, instantiates `calculator.model_class(**params)`
> (`engine.py:162-184`), fits/scores candidates with sklearn's CV/scorer
> machinery, and refits the best candidate — it requires a `model_class`
> attribute and **never calls the DL calculator's own `fit()` with an epoch
> loop.** Only the clustering path (`_run_training_direct`) calls
> `calculator.fit()` directly. The corrected design below reflects this.

- **No new training manager.** DL jobs use the existing
  `BasicTrainingManager`/`AdvancedTuningManager` DB-row lifecycle unchanged —
  cancel, log, and status plumbing require zero changes.
- **New direct-fit dispatch branch in `_run_training`.** Add an
  `is_deep_learning` check next to the existing `is_clustering` check
  (`_node_runners.py:514-520`): if `getattr(calculator, "is_deep_learning",
  False)`, route to a new `_run_training_dl` direct-fit path — analogous to
  `_run_training_direct` but keeping the target-column/split-dataset
  handling that clustering skips. This bypasses `TuningCalculator`/
  `SklearnBridge.to_sklearn` entirely: the DL calculator's own `fit(X, y,
  config, progress_callback, log_callback, validation_data)` (implementing
  the epoch loop from `deep_learning/_training_loop.py`) is called directly
  on the framework-native tensors, never converted to NumPy through the
  sklearn CV/scorer path. This is the single most important correction from
  the original design and removes the previously-assumed `model_class`
  requirement, the CV-multiplied-refit-training-cost risk, and the
  tuple-wrapped-artifact complication that `TuningCalculator`/`TuningApplier`
  would otherwise introduce.
- **Per-epoch progress:** `_run_training_dl` passes a real
  `progress_callback(current_epoch, total_epochs, score=val_metric)` straight
  through to the calculator's `fit()`, which the training loop invokes once
  per epoch — a genuine progress seam that does not exist on the tuning path
  (whose callback reports per-*trial*, not per-*epoch*: `_node_runners.py:756-760`).
- **"Tuned" (Advanced) mode for DL is a separate, lightweight
  implementation** — not a reuse of `TuningCalculator`'s sklearn CV/refit
  engine. A small `deep_learning/_dl_tuning.py` runs an Optuna study whose
  objective directly calls the calculator's own `fit()`/`fit()`-with-holdout
  once per trial (no k-fold CV multiplier, no NumPy bridge), searching only
  `architecture_preset`, `learning_rate`, and `batch_size` — consistent with
  the frontend's proposed constrained advanced search space
  (see the frontend design doc, §3.1). `StepType.TRAINING` and `run_mode:
  fixed|tuned` remain unchanged on the wire; the branch is selected the same
  way `is_deep_learning` is detected in `_run_training`.
- **Execution-backend neutrality:** the DL direct-fit path runs in-process on
  the Celery worker exactly like every other node today — no coupling to
  Celery APIs beyond what already exists. Once the Ray migration (plan 01)
  lands its execution-backend abstraction, DL jobs benefit the same way any
  other job does; §7 covers what additional work Ray GPU scheduling actually
  requires (more than "wiring").
- **Solo-pool caveat:** the Celery worker uses a **solo pool**
  (findings §2.6) — one job at a time. A multi-epoch DL job blocks the
  entire queue for its full duration, an existing limitation DL amplifies
  significantly more than a single sklearn `.fit()` call. Phase 0's gate
  must include an explicit wall-clock budget check for this reason (see
  roadmap M10).

### 4.4 Artifact storage (`backend/ml_pipeline/artifacts/`)

- Extend `ArtifactStore` (local + S3 implementations) with a
  `save_torch`/`load_torch` pair that writes `torch.save({"state_dict":...,
  "architecture_config": ...})` under a `.pt` extension, alongside the
  existing `.joblib` methods. `discovery.py` is extended to recognize `.pt`
  artifacts the same way it recognizes `.joblib` today. Call sites need an
  explicit format selector (today's `save(key, data)` takes no format
  argument) — add a `format: "joblib" | "torch"` parameter rather than
  sniffing the data type, so the choice is explicit and testable.
- The DL applier's `predict()` reconstructs the `nn.Module` from
  `architecture_config` (the same preset/hyperparameter dict used at train
  time) before loading the `state_dict` — avoids pickling arbitrary class
  objects (a real security/versioning risk with raw `torch.save(model)`).
- **Security requirement, not optional:** `local.py`/`s3.py` already
  document that `joblib.load` can execute arbitrary code
  (`local.py:36-38`, `s3.py:131-134`); `torch.load` carries the identical
  pickle-based risk. `load_torch` **must** call
  `torch.load(..., weights_only=True)` and the docstring must carry the same
  warning the joblib loaders already have — this is a direct requirement,
  not a nice-to-have, since artifacts may originate from S3 paths that are
  not fully trusted in every deployment.
- Since the DL direct-fit path (§4.3) bypasses `TuningCalculator`/
  `TuningApplier` for `fixed`-mode jobs, the artifact is the calculator's own
  return value, not the `(model, tuning_result)` tuple the sklearn tuning
  path produces (`skyulf-core/skyulf/modeling/base.py:123-136`,
  `_tuning/engine.py:357`). `save_torch`/`load_torch` only need to handle a
  bare model + architecture config for the `fixed` path. The "tuned" (DL
  Optuna) path in §4.3 does its own lightweight refit and must return the
  same bare-model shape — no tuple-unwrapping logic needs to be shared with
  the sklearn tuning artifact format.

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
- **Converter allow-list, a required change, not optional:** new frontend
  node `type` values (e.g. `neural_network_classifier`,
  `transformer_text_classifier`) must be added to `RUN_MODE_TRAINING_TYPES`
  in `frontend/ml-canvas/src/core/utils/pipelineConverter.ts:29-34` or they
  will never be recognized as training nodes and will not convert to the
  backend's canonical `training` step at all. This is exactly the kind of
  hand-duplicated allow-list the repo's own sync rule warns about, and is
  called out explicitly here so it isn't missed the way past
  `FeatureGenerationNode`/`InvalidValueReplacementNode` gaps were. The
  full, code-grounded frontend design — including this converter change,
  the exact settings-panel layout, live training-curve telemetry, and the
  image upload UX — is in
  [2026-08-11-frontend-design.md](2026-08-11-frontend-design.md).

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
   params: {model_type: "mlp_classifier", architecture_preset: "medium",
            epochs: 50, batch_size: 32, learning_rate: 1e-3,
            early_stopping_patience: 5, run_mode: "fixed"}
   -> NodeRegistry.get_calculator("mlp_classifier")  [BaseDLCalculator]
   -> _run_training's new is_deep_learning branch -> _run_training_dl
      (direct fit(), epoch loop, progress per-epoch, device resolution —
       bypasses TuningCalculator/SklearnBridge entirely; see §4.3)
   -> ArtifactStore.save_torch(state_dict, architecture_config,
      format="torch")
   -> MLJob marked complete, artifact URI recorded (same as sklearn today)
```

**Image training does not "differ only in the data-loading step" as
originally claimed.** An `Image Data Loader` node produces a manifest
(path, label), not a `SkyulfDataFrame` — and the engine's
`_get_training_input`/`_to_split_dataset` helpers, and every other
NumPy/frame-based helper in `_node_runners.py`, assume a frame. Image
training therefore requires explicit engine-level work beyond the ingestion
connector: either (a) a manifest-aware branch in `_run_training_dl` that
skips `_get_training_input` and hands the calculator a
`torch.utils.data.Dataset` built from the manifest directly, or (b) a
manifest object that satisfies just enough of the `SkyulfDataFrame` surface
to flow through existing plumbing unchanged. Option (a) is simpler and is
what roadmap Phase 4 now scopes explicitly (see roadmap Phase 4 "Modify").

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
duplicate it — and, per rubber-duck validation, the following is corrected
from the original "GPU scheduling falls out for free" framing:

- DL training is written against the same backend-neutral execution
  interface Ray plan 01 defines, so the direct-fit path (§4.3) needs no
  changes when Ray lands.
- **GPU declaration is not automatic.** Ray plan 04's
  `resource_spec_for_job(job_type, settings)`
  (`ray-migration/2026-08-10-04-distributed-compute-plan.md:179-190` on
  branch `080`) only branches on `job_type in {"tuning", other}` and returns
  a single **global static** `settings.RAY_ENTRYPOINT_NUM_GPUS` for every
  non-tuning job — there is no `"dl_training"` job type and no per-job GPU
  differentiation today. Getting `num_gpus=1` for image/transformer jobs and
  `num_gpus=0` for a cheap tabular MLP in the *same* deployment requires
  **extending `resource_spec_for_job`** with a new job-type branch and a new
  setting (e.g. `RAY_DL_ENTRYPOINT_NUM_GPUS`) — real, scoped work against
  the Ray migration's own code, not passive reuse. Phase 5 (roadmap) is
  updated to say so explicitly.
- If the Ray migration is delayed or rejected at one of its own gates, DL
  phases 0-4 are still fully functional CPU-only (with image models using
  frozen-backbone transfer learning, which is CPU-tractable), so DL delivery
  is not blocked on Ray's timeline.
- **Checks out unchanged:** Ray's own non-goal of deferring Ray Train
  genuinely does not conflict with DL's single-device training scope (§3,
  §8), and the Ray migration is confirmed docs-only on branch `080` — no
  `backend/ray_jobs/` or `backends/ray*.py` exists on any branch yet.

## 8. Future Considerations (explicitly out of scope now)

- A dedicated layer-by-layer DL architecture-builder canvas, for power users
  who outgrow presets — only if requested after Phase 4 ships.
- ONNX export for faster/portable inference.
- Multi-GPU / distributed training (would ride on Ray Train, itself
  explicitly deferred by the Ray migration's own non-goals).
- Audio/video modalities.
