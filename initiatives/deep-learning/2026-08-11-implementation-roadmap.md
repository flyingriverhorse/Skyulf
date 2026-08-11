# Deep Learning Integration — Implementation Roadmap

**Date:** 2026-08-11
**Status:** Ready for phase-by-phase execution

**Goal:** Coordinate independent, reviewable phases that add deep learning
model support without destabilizing existing pipelines. Each phase produces
working, shippable software and passes its own gate before the next begins.

## Global Constraints (apply to every phase)

> **Correction (2026-08-11, post rubber-duck validation):** the original
> constraint below claimed "no changes to `_node_runners.py` dispatch." That
> is false and has been corrected — Phase 0 requires one explicit,
> narrowly-scoped change there (a direct-fit branch parallel to the existing
> clustering branch). See Phase 0 below.

- Never break existing sklearn/XGBoost/LightGBM nodes, jobs, or artifacts.
- Every DL calculator/applier implements the existing
  `BaseModelCalculator`/`BaseModelApplier` contract. `NodeRegistry` and the
  public pipeline/job API shapes need no changes; `_node_runners.py`'s
  `_run_training` needs exactly one new dispatch branch (Phase 0) — no other
  changes to the engine's dispatch logic.
- New frontend dropdowns/enums are cross-checked against backend allow-lists
  before a phase is marked done (Backend/Core ↔ Frontend Sync Rule), and new
  frontend node `type` values are added to `RUN_MODE_TRAINING_TYPES` in
  `pipelineConverter.ts` (verified requirement, see findings §2.8/§2.10).
- `torch.load`/any DL artifact loader **must** use `weights_only=True` and
  carry the same arbitrary-code-execution warning docstring the existing
  `joblib.load` wrappers already have (`local.py:36-38`, `s3.py:131-134`).
- Every DL job's CI/gate time budget must account for the Celery **solo
  pool** (one job at a time per worker) — a multi-epoch job blocks the whole
  queue for its duration, more severely than a single sklearn `.fit()` call.
- `requirements-ci.txt`/CI config must explicitly install the `dl` extra
  wherever a gate requires actually running DL training, and the added
  install/runtime budget (torch is hundreds of MB) must be accounted for,
  not assumed free.
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
| 0 | Shared DL infra | `deep_learning/` skeleton, `_run_training`'s direct-fit dispatch branch, `.pt` artifact format, one working node (tabular MLP) end-to-end | None |
| 1 | Tabular DL | MLP classifier/regressor nodes with architecture presets, tuned-mode support | Phase 0 |
| 2 | Text DL | Transformer text-classification node (fine-tuned pretrained encoder) | Phase 0 |
| 3 | Time-series DL | Windowing transform + LSTM/TCN forecaster node | Phase 0 |
| 4 | Image DL | Image ingestion (new modality) + CNN/transfer-learning classifier node | Phase 0 |
| 5 | GPU scheduling via Ray | Extend `resource_spec_for_job` for per-job-type GPU sizing and route DL jobs through it once Ray migration (branch `080`) lands | Phase 0-4, Ray migration plans 01+04 |

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

**Objective:** Prove the whole plumbing path (registry → direct-fit dispatch
→ training loop → artifact → predict) with the simplest possible model, so
phases 1-4 only add modality-specific code.

> **Correction (2026-08-11, post rubber-duck validation):** replaced the
> originally-proposed `DLTrainingManager`/`JobStrategyFactory` dispatch (not
> implementable — `JobStrategyFactory` never sees a calculator, and every
> non-clustering node is normally routed through `TuningCalculator`, which
> cannot call a DL calculator's `fit()`) with a direct-fit branch in
> `_run_training`, parallel to the existing clustering branch. See
> architecture design §4.3 for the full rationale.

**Create:**
- `skyulf-core/skyulf/deep_learning/__init__.py`
- `skyulf-core/skyulf/deep_learning/base.py` — `BaseDLCalculator`,
  `BaseDLApplier`, `resolve_device(preferred: str | None) -> torch.device`.
  `BaseDLCalculator` exposes `is_deep_learning = True` (or equivalent marker)
  so `_run_training` can detect it without an `isinstance` import cycle.
- `skyulf-core/skyulf/deep_learning/_training_loop.py` — `TrainingLoopConfig`
  (epochs, batch_size, lr, patience), `run_training_loop(model, train_ds,
  val_ds, config, progress_callback, log_callback) -> TrainingResult`. The
  loop invokes `progress_callback(current_epoch, total_epochs, score=...)`
  once per epoch — this is the actual progress seam (there is no equivalent
  per-epoch callback anywhere in the existing tuning engine).
- `skyulf-core/skyulf/deep_learning/tabular/mlp.py` — `MLPClassifierCalculator`
  / `MLPClassifierApplier` (classification only, in this phase — regression
  is Phase 1), registered as `"mlp_classifier"`. `fit()` builds the model,
  runs `run_training_loop`, and returns a bare fitted model object (not a
  `(model, tuning_result)` tuple — that shape is specific to the sklearn
  tuning path this phase bypasses).
- `backend/ml_pipeline/artifacts/torch_format.py` — `save_torch`/`load_torch`
  helpers shared by `local.py`/`s3.py`. `load_torch` calls
  `torch.load(..., weights_only=True)` and its docstring carries the same
  arbitrary-code-execution warning `joblib.load`'s wrapper already has.
- `skyulf-core/pyproject.toml` — `dl` optional-dependency group
  (`torch>=2.2,<3.0`).
- `requirements-dl.txt` (root) — mirrors `requirements-nlp.txt` style/comments.
- Tests: `skyulf-core/tests/deep_learning/test_mlp_classifier.py`,
  `tests/test_run_training_dl_dispatch.py` (verifies `_run_training` routes
  a `BaseDLCalculator` to the new direct-fit branch and a sklearn calculator
  is unaffected), `tests/test_torch_artifact_format.py` (including a
  regression test that a maliciously crafted pickle payload is rejected
  under `weights_only=True`).

**Modify:**
- `backend/ml_pipeline/artifacts/local.py`, `s3.py` — add an explicit
  `format: Literal["joblib", "torch"]` parameter to `save`/`load` (or a
  parallel `save_torch`/`load_torch` pair) — do not infer format from the
  data type.
- `backend/ml_pipeline/artifacts/discovery.py` — recognize `.pt` artifacts.
- `backend/ml_pipeline/_execution/engine/_node_runners.py` — in
  `_run_training` (`:497-533`), add an `is_deep_learning` check immediately
  after the existing `is_clustering` check, routing to a new
  `_run_training_dl` method (parallel to `_run_training_direct`, `:535`)
  that still resolves `target_column`/`SplitDataset` (unlike clustering,
  which doesn't need a target) but calls `calculator.fit()` directly instead
  of going through `_run_training_tuned`/`TuningCalculator`.

**Do not touch:** `NodeRegistry`, `StepType` enum, `JobStrategyFactory`,
`BasicTrainingManager`/`AdvancedTuningManager` (the DB-row lifecycle is
reused unchanged), public API schemas, frontend (no node exposed yet —
Phase 0 validates internals only, optionally behind a feature flag or
dev-only registry entry).

---

## Phase 1 — Tabular DL

**Objective:** Ship the first user-facing DL nodes.

**Create:**
- `skyulf-core/skyulf/deep_learning/tabular/mlp.py` — extend with
  `MLPRegressorCalculator`/`Applier`, registered as `"mlp_regressor"`.
  Architecture presets: `"small"` (1 hidden layer), `"medium"` (2 layers),
  `"wide"` (2 wider layers) — fixed configs, no free-form layer editing.
- `skyulf-core/skyulf/deep_learning/_dl_tuning.py` — a small Optuna-driven
  search loop for DL "tuned" (Advanced) mode: calls the calculator's own
  `fit()` once per trial (holdout, not k-fold CV) over `learning_rate`,
  `batch_size`, `architecture_preset` only. This is **not** a reuse of
  `TuningCalculator`/`_tuning/engine.py` — see architecture design §4.3 for
  why that path is structurally incompatible with a DL calculator's `fit()`.
- `frontend/ml-canvas/src/modules/nodes/modeling/NeuralNetworkClassifierNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/modeling/NeuralNetworkRegressorNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/modeling/DLTrainingSettings.tsx` —
  preset/epochs/batch-size/lr/early-stopping fields. UI toggle is
  `basic|advanced` (matching the existing `RunModeToggle` convention in
  `TrainingSettings.tsx`); the converter emits `fixed|tuned` on the wire
  (see frontend design doc §3.1) — do not introduce a third UI value.

**Modify:**
- `backend/ml_pipeline/_execution/engine/_node_runners.py` — `_run_training`'s
  new `is_deep_learning` branch (Phase 0) additionally checks `run_mode`:
  `"fixed"` calls `calculator.fit()` once; `"tuned"` calls into
  `deep_learning/_dl_tuning.py`'s search loop instead of
  `_prepare_tuning_config`/`TuningCalculator`.
- `frontend/ml-canvas/src/core/utils/pipelineConverter.ts` — add
  `neural_network_classifier`/`neural_network_regressor` to
  `RUN_MODE_TRAINING_TYPES` (`:29-34`) so they convert to the backend's
  canonical `training` step — required, not optional (see findings §2.8).

**Verify:** frontend preset dropdown options == backend
`ARCHITECTURE_PRESETS` allow-list (sync-rule check); confirm the new node
types actually appear as `step_type: "training"` in a converted pipeline
(regression test against `pipelineConverter.test.ts`'s existing snapshot
pattern).

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

> **Correction (2026-08-11, post rubber-duck validation):** the original
> version of this phase scoped only ingestion/connector work and assumed
> "everything downstream is identical" to tabular DL. Verified false: the
> engine's frame-based helpers (`_get_training_input`, `_to_split_dataset`,
> the NumPy bridge) cannot consume a manifest. Explicit engine-level work is
> added below.

**Create:**
- `backend/data_ingestion/connectors/image_manifest.py` —
  `ImageManifestConnector` (zip/folder + label CSV, or image-path column in
  an existing tabular dataset). Produces a manifest (path, label), not a
  `SkyulfDataFrame`. Zip extraction validates **each member's resolved path**
  against the target directory before writing (zip-slip prevention) and
  rejects symlink entries — reusing `LocalFileConnector.resolve_safe_path`
  for the connector's *own* base path is necessary but not sufficient; it
  does not by itself prevent zip-slip during extraction.
- `backend/data_ingestion/schemas/image_ingestion.py` — request/response
  schemas for the new upload path.
- `skyulf-core/skyulf/deep_learning/vision/image_classifier.py` —
  `ImageClassifierCalculator`/`Applier`, frozen pretrained backbone (e.g.
  torchvision `resnet18`/`mobilenet_v3_small`) + trainable head, registered
  as `"image_classifier"`. Consumes a manifest directly via a
  `torch.utils.data.Dataset`, not a `SkyulfDataFrame`.
- `frontend/ml-canvas/src/modules/nodes/data/ImageDataLoaderNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/modeling/ImageClassifierNode.tsx`

**Modify:**
- `backend/ml_pipeline/_execution/engine/_node_runners.py` — the
  `is_deep_learning` direct-fit branch (Phase 0) additionally detects a
  manifest-typed input (e.g. via the node's declared input port type,
  `dataset` vs an `image_manifest` marker) and skips
  `_get_training_input`/`_to_split_dataset` entirely for that case, passing
  the manifest reference straight to `calculator.fit()`, which builds its
  own `Dataset`/`DataLoader`. This is the specific engine change the
  original phase description omitted.
- `backend/data_ingestion/router.py` — new endpoint(s) for image
  upload/manifest creation. Every manifest entry's image path — including
  the "image-path column in an existing tabular dataset" option — is
  validated against the upload root **before read at train time**, not only
  at ingestion time (a raw path column is a direct arbitrary-file-read
  vector otherwise).
- `skyulf-core/pyproject.toml` — `dl` extra gains `torchvision`.
- `frontend/ml-canvas/src/core/utils/pipelineConverter.ts` — add
  `image_classifier` to `RUN_MODE_TRAINING_TYPES` and make the new image
  source node a recognized traversal root/step alongside `dataset_node`
  (`:165-176`), serializing it to the image-manifest loader step.

**Verify:** memory does not scale with dataset size (streamed via
`DataLoader`, not eagerly loaded); zip-slip and manifest-path-traversal
tests run with `settings.TESTING=False` — `resolve_safe_path` skips
containment entirely when `TESTING` is true (`file.py:71-73`), so tests
under that flag would pass vacuously without actually exercising the
containment logic.

---

## Phase 5 — GPU Scheduling via Ray

**Objective:** Give DL training real GPU acceleration, once available,
without a DL-specific scheduler.

> **Correction (2026-08-11, post rubber-duck validation):** the original
> "wiring, not new infrastructure" framing overstated how ready
> `resource_spec_for_job` is. Verified: it returns a single global static
> `settings.RAY_ENTRYPOINT_NUM_GPUS` for every non-tuning job type — there is
> no per-job-type GPU differentiation today. This phase includes real,
> scoped extension work against the Ray migration's own code.

**Precondition:** Ray migration plans 01 (execution backend foundation) and
04 (distributed compute / `ResourceSpec`) have passed their own gates and
been merged.

**Modify:**
- `backend/ml_pipeline/_execution/resources.py` (Ray migration plan 04) —
  extend `resource_spec_for_job` with a `"dl_training"` branch and a new
  setting (e.g. `RAY_DL_ENTRYPOINT_NUM_GPUS`), distinct from the existing
  global `RAY_ENTRYPOINT_NUM_GPUS`, so image/transformer jobs can request
  `num_gpus=1` while a tabular MLP job requests `num_gpus=0` in the same
  deployment. This is joint work with whoever owns the Ray migration branch,
  not a unilateral DL-side change.
- The DL direct-fit dispatch path (Phase 0) requests a `ResourceSpec` via
  the extended `resource_spec_for_job("dl_training", settings)`, varying
  `num_gpus` by DL model type (image/transformer → 1, tabular MLP/forecaster
  → 0).
- `skyulf-core/skyulf/deep_learning/base.py` — `resolve_device()` honors a
  `CUDA_VISIBLE_DEVICES`/Ray-provided device hint instead of naive
  `torch.cuda.is_available()` auto-detection, so placement matches what Ray
  actually reserved.

**Do not:** build a parallel GPU queue or a second `EXECUTION_BACKEND`
value. Do extend `resource_spec_for_job` rather than inventing a competing
resource-declaration mechanism.

**Verify:** an image-classifier job submitted with Ray backend enabled and
`num_gpus=1` lands on a GPU-capable worker (observable via Ray dashboard /
job resource metadata) and completes faster than the CPU baseline from
Phase 4; a tabular MLP job in the same deployment is confirmed to request
`num_gpus=0` and does not consume a GPU slot.
