# Training visualizations — feasibility and delivery plan

**Date:** 2026-08-11  
**Decision:** Ship persisted post-fit diagnostics first. Add live epoch metrics
only with the proposed PyTorch direct-fit path. Do not promise live curves for
ordinary sklearn `fit()` calls.

## Executive verdict

### (a) Cheap, high-value, ship first

1. Make the existing post-fit evaluation automatically prominent in the job
   completion/result view: classification confusion matrix, ROC and PR;
   regression actual-vs-predicted and residual diagnostics; and feature
   importance where the estimator exposes it. The core evaluator already
   computes a confusion matrix and ROC/PR data after predictions
   (`skyulf-core/skyulf/modeling/_evaluation/classification.py:36-129`), caps
   residual payloads at 1,000 points (`regression.py:51-72`), and the engine
   persists raw evaluation data for the API (`backend/ml_pipeline/_execution/engine/_node_runners.py:330-340`).
   The frontend already renders the corresponding classification set
   (`ClassificationChartsForSplit.tsx:1-26`) and eight regression charts
   (`RegressionChartsForSplit.tsx:1-8`). This is **post-hoc, not live**, but
   appears immediately at completion and answers “what happened?” with almost
   no new ML computation.
2. Reuse current feature-importance extraction for tree models and linear
   coefficients (`backend/ml_pipeline/_execution/engine/_artifacts.py:60-97`)
   and its existing post-fit persistence (`_node_runners.py:419-426`). Do not
   claim importance for models that do not expose it; SHAP is already
   best-effort and sample-limited to 200 (`_artifacts.py:106-127`).
3. Add an optional, explicitly requested **final** PCA class-separation plot
   (2-D, sampled) only after the basic diagnostics. The UI already has a
   Plotly-backed grouped 3-D scatter component
   (`frontend/ml-canvas/src/components/eda/ThreeDScatterPlot.tsx:23-100`) and
   the slim Plotly bundle supports scatter/scattergl/3-D/heatmap
   (`frontend/ml-canvas/src/core/plotly.ts:1-21`). For classic ML this is a
   projection of transformed input features, not a learned representation;
   label it accordingly.

### (b) Valuable, but requires new plumbing

1. **DL live loss/metric/LR curves** are realistic once the approved DL
   direct-fit branch exists. That design deliberately gives the PyTorch
   calculator a shared manual epoch/batch loop
   (`initiatives/deep-learning/2026-08-11-architecture-design.md:78-101`) and
   passes `progress_callback(current_epoch, total_epochs, score=val_metric)`
   once per epoch (`...architecture-design.md:166-170`). Extend that callback
   to publish structured metric snapshots, persist them as an artifact, and
   draw an incremental Recharts line plot.
2. DL-only optional diagnostics—gradient norm, LR, and activation
   mean/std—are possible inside that manual PyTorch loop via
   `optimizer.param_groups`, parameter-gradient iteration after
   `backward()`, and selected forward hooks. They are not “free”: hooks,
   aggregation, sampling, and failure isolation must be implemented. Start
   with global gradient norm and LR; per-layer activation telemetry is an
   advanced debug mode, not default UI data.
3. A DL embedding-separation view is valuable after training. Capture
   penultimate-layer embeddings for a stratified, bounded holdout sample,
   project with PCA by default, and colour by true/predicted class. This is
   meaningfully interpretable for DL because the representation is learned;
   it is only an input-space diagnostic for classic ML.

### (c) Expensive or lower priority

1. A sklearn **learning curve** is not an in-progress trace. `sklearn.model_selection.learning_curve`
   retrains the estimator for every requested training-size/fold combination:
   roughly `n_train_sizes × n_CV_folds` fits, plus the normal final fit. Run it
   as a separately requested diagnostic job/artifact with caps, never inline
   in every training request.
2. A sklearn **validation curve** likewise fits each candidate value over CV
   folds. Skyulf already performs candidate/fold searches in `TuningCalculator.fit`
   before refitting (`skyulf-core/skyulf/modeling/_tuning/engine.py:321-357`);
   a validation curve is useful only as an opt-in post-tuning slice for one
   selected hyperparameter, not a free by-product of arbitrary Optuna trials.
3. t-SNE/UMAP is nonlinear, stochastic and can become expensive/memory-heavy
   as row count grows. Make it opt-in, stratify/sample (for example 2–5k
   points), run it after fit in a worker, cache the artifact, and use PCA as
   the reliable default. Do not stream evolving t-SNE/UMAP plots.
4. Per-epoch ROC/PR/confusion matrices require a full validation inference
   pass and, for curves, retaining every score/label. Permit a small fixed
   validation set and sparse cadence (for example every 5 epochs) only after
   live scalar curves; otherwise compute these once at completion.

## What training does today

Classic supervised “fixed” and “tuned” runs both go through a sklearn-oriented
`TuningCalculator`; fixed mode becomes a one-candidate search
(`backend/ml_pipeline/_execution/engine/_node_runners.py:497-533,712-774`).
The tuner converts data to NumPy (`skyulf-core/skyulf/modeling/_tuning/engine.py:297-319`),
uses the candidate/trial callback during tuning (`engine.py:321-337`), and
refits `model.fit(X_np, y_np)` once after selection (`engine.py:223-269`).
The only engine callback exposed at the pipeline layer labels updates as
**Trial current/total**, not iteration/epoch (`_node_runners.py:756-760`).
Thus no reliable sklearn epoch/batch metric stream exists today; model-specific
attributes such as boosting evaluation histories would need adapters and are
not a general contract.

The worker invokes `PipelineEngine.run()` synchronously for the job
(`backend/ml_pipeline/_services/pipeline_execution_service.py:199-225`).
The existing logging callback is throttled to DB/event publishing at two
seconds (`pipeline_execution_service.py:89-119`), and completion writes
progress=100 (`pipeline_execution_service.py:122-149`). Therefore “live” is
architecturally absent for ordinary classic fitting, but the process boundary
does not prevent it once training code publishes snapshots.

The approved DL design is explicitly PyTorch, not Keras/Lightning: its
calculator owns the epoch/batch loop and direct-fit dispatch avoids the
sklearn-only tuner (`initiatives/deep-learning/2026-08-11-architecture-design.md:125-180`).
That is the correct insertion point. Do **not** invent the previously rejected
`DLTrainingManager`; job lifecycle reuse is intentional
(`...architecture-design.md:127-150`).

## Graph inventory

| Graph | Family | Live during training? | Compute cost | Reuse / new work | Skyulf foundation |
|---|---|---:|---|---|---|
| Final confusion matrix | both classifiers | No need; optionally sparse DL snapshots | One validation prediction; matrix is cheap | evaluator/UI reuse; expose at completion | evaluator `classification.py:69-70`; UI `ClassificationChartsForSplit.tsx:57-100` |
| Final ROC / PR | probability classifiers | No need; optional sparse DL snapshots | One validation probability pass; curve payload downsampled | evaluator/UI reuse | `classification.py:72-123`; schema `schemas.py:15-35` |
| Final residual / actual-vs-predicted | regressors | No need | One prediction pass; capped at 1k points | evaluator/UI reuse | `regression.py:36-80`; `RegressionChartsForSplit.tsx:103-177` |
| Feature importance | classic; DL only with new method | No | Cheap if native; permutation/SHAP expensive | native classic reuse; DL new (permutation or attribution) | `_artifacts.py:60-97`; finalization `_node_runners.py:419-426` |
| Train/validation loss, accuracy, F1 | DL | Yes, epoch cadence | One train aggregate plus validation pass/epoch | new event/artifact/UI | proposed epoch callback `architecture-design.md:166-170` |
| Learning curve by train size | classic (also possible DL) | No—separate diagnostic | `sizes × folds` complete fits | new diagnostic worker/artifact/UI | tuner already does repeated candidate/fold fits: `_tuning/engine.py:321-357` |
| Validation curve by parameter | classic (also possible DL) | No—separate diagnostic | `parameter values × folds` fits | new diagnostic; may reuse tuning trial records only when exact match | tuner contract `engine.py:271-357` |
| LR / global gradient norm | DL | Yes, epoch cadence | negligible aggregate | new PyTorch loop instrumentation/UI | shared training loop planned at `architecture-design.md:97-100` |
| Layer activation mean/std | DL | Yes, sampled cadence | moderate; hook/transfer overhead | new, opt-in only | PyTorch loop new; no present Skyulf equivalent |
| PCA separation | both classifiers | Post-fit only | low/moderate; sample first | Plotly scatter reuse; projection artifact new | 3-D scatter `ThreeDScatterPlot.tsx:23-100`; EDA visualizer has PCA plotting (`profiling/visualizer.py`, `plot()` dispatcher) |
| t-SNE/UMAP embedding separation | DL preferred; classic diagnostic only | Post-fit only | high / stochastic; sample and cache | Plotly reuse; reducer/artifact new | same Plotly component; no training embedding artifact exists today |

“Live” in the final-metric rows should mean *visible immediately when complete*,
not a misleading animation of a single completed prediction. The existing
evaluation schema is designed for final structured curve/matrix data
(`skyulf-core/skyulf/modeling/_evaluation/schemas.py:8-50`), while the current
socket is intentionally an invalidator channel (`backend/realtime/events.py:1-6`).

## Real-time transport plan

Extend, rather than replace, `/ws/jobs`. A FastAPI manager already broadcasts
Redis/local-bus payloads as `{"channel":"jobs","data":...}`
(`backend/realtime/manager.py:22-29,95-105,148-165`); the socket client already
parses that envelope and routes it to subscribers
(`frontend/ml-canvas/src/core/realtime/jobEventsSocket.ts:70-84`). Add a
versioned event type—not a separate connection:

```json
{
  "channel": "jobs",
  "data": {
    "event": "training_metrics",
    "schema_version": 1,
    "job_id": "uuid",
    "node_id": "model-node",
    "sequence": 17,
    "timestamp_ms": 1786440000000,
    "step": {"kind": "epoch", "current": 17, "total": 50},
    "metrics": {"train_loss": 0.341, "val_loss": 0.402, "val_accuracy": 0.873},
    "system": {"learning_rate": 0.0003, "gradient_norm": 1.84},
    "partial": true
  }
}
```

Add `training_metrics` to backend `JobEventType`/Pydantic fields
(`backend/realtime/events.py:23-34`) and the TypeScript union/interface
(`jobEventsSocket.ts:14-27`). Keep status/progress fields unchanged. The
current channel broadcasts to every connected client and has no user filtering
(`events.py:19-21`), so payloads must contain no raw rows, labels, embeddings,
or feature values—only aggregate numbers. Final artifacts are fetched through
the normal authenticated job/evaluation API once authorization exists.

**Cadence and durability**

* DL: one scalar event per epoch, max one/sec; for very short epochs coalesce
  to one every 1–2 seconds while retaining all points in the job artifact.
* Batch updates: aggregate locally and emit at most every 2–5 seconds; never
  emit one event per batch by default.
* Classic tuning: optionally emit a separate `tuning_trial` summary after each
  complete trial, accurately labelled as trials—not a learning curve.
* Maintain `sequence` so the client drops duplicate/out-of-order messages on
  reconnect. Buffer only the active job's bounded series in Zustand/React
  state (for example 2,000 points), append it to
  `{job_id}_training_history` at completion/checkpoint, and reload it for
  historical review. The WebSocket remains best-effort, consistent with the
  publisher's existing “log and polling fallback” behaviour
  (`events.py:48-76`) and client reconnect behaviour
  (`jobEventsSocket.ts:86-105`).
* Do not overload `job.progress`: it is integer job-level status and currently
  becomes 100 only when the entire pipeline succeeds
  (`pipeline_execution_service.py:122-149`). Derive display progress from
  `step` for a training node, while retaining job-level progress separately.

## DL-specific wiring

1. Implement `TrainingTelemetry`/an `on_metrics(snapshot)` callable in the
   planned shared `_training_loop.py`; call it after the train aggregate and
   optional validation pass at epoch end. Pass it through the direct-fit
   `_run_training_dl` path described in the approved design, alongside the
   existing progress/log callbacks.
2. The callback must publish the compact event above and append the same
   validated scalar snapshot to an in-memory history. Check cancellation at
   epoch boundaries before beginning the next epoch.
3. Persist history at regular checkpoints and completion, so browser refresh,
   dropped WebSockets, and completed jobs retain the graph. Treat telemetry
   persistence failure as non-fatal to model training, but report it in logs.
4. Use a holdout/validation split for validation metrics. Never compute them
   against training examples while labelling them validation. Metric selection
   must follow problem type (loss universally; accuracy/F1 for classification;
   MAE/RMSE for regression).
5. Start with scalar curves, LR and global gradient norm. Add confusion
   snapshots every fifth epoch only for bounded validation samples. Add layer
   hooks only behind a debug option and capture mean/std/finite fraction, not
   tensors.
6. DL advanced tuning should emit trial summaries separately. The approved
   proposal uses a direct Optuna objective rather than sklearn CV
   (`architecture-design.md:171-180`); do not mix trial histories into a
   single model's epoch curve.

PyTorch provides the primitives (`Module` hooks, `loss.backward()`, parameter
gradients, optimizer param groups), but it does **not** provide Skyulf’s
transport, persistence, sampling policy or schema. Because Skyulf proposes a
manual common loop rather than Lightning/Keras callbacks, all of that plumbing
is deliberately new.

## Market precedent and priority signal

| Product | Validated capability | Live or post-hoc | Implication |
|---|---|---|---|
| Weights & Biases | `wandb.log` records time-series metrics; its workspace panels plot run histories and support system metrics/media. [Docs](https://docs.wandb.ai/guides/track/log/) | Live as metrics are logged | Adopt epoch scalar history and bounded telemetry first; do not try to clone a general experiment tracker. |
| TensorBoard | Scalar, histogram, image, graph, embedding/projector and profiler dashboards consume event files. [Get started](https://www.tensorflow.org/tensorboard/get_started) / [scalars](https://www.tensorflow.org/tensorboard/scalars_and_keras) | Event-file streaming/reload | Loss/accuracy/LR are baseline DL expectations; histograms/embeddings are optional diagnostics. |
| MLflow | Tracking logs timestamped, stepped metrics and renders metric history; artifacts preserve outputs. [Tracking docs](https://mlflow.org/docs/latest/ml/tracking/) | Logged incrementally; viewed from persisted run data | Make Skyulf history durable and job-addressable, not WebSocket-only. |
| Databricks AutoML | Produces MLflow experiments/runs plus model evaluation and feature-importance/explainability outputs. [AutoML training docs](https://docs.databricks.com/aws/en/machine-learning/automl/train-mlflow-model) | Primarily post-hoc/fast | Final evaluation and explainability are the right first classic-ML experience. |
| DataRobot | Model insights include ROC, lift/gains, confusion matrix, feature impact and residual-related diagnostics. [Model evaluation docs](https://docs.datarobot.com/en/docs/modeling/analyze-models/evaluate-models.html) | Predominantly post-hoc | Prioritize decision/evaluation charts over “live” decoration for classic ML. |
| H2O | Model output includes scoring history, variable importance, confusion matrices and ROC/performance diagnostics. [Performance and prediction](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/performance.html) | Scoring history may update during iterative training; diagnostics are model output | Scoring history is worth adding only for iterative adapters/DL; final diagnostics have broad value. |

These sources support a two-tier product: durable live scalar telemetry for
iterative DL, and strong post-fit diagnostics for all models. They do **not**
justify pretending that generic sklearn estimators expose meaningful epochs.

## Phased implementation plan

### Phase 0 — contract and safety (1 sprint)

* Define a versioned `TrainingMetricSnapshot` backend/TypeScript schema and
  event-size/cadence limits; add unit tests for serialization and sequence
  ordering.
* Add a training-history artifact key/version and retention/point cap. Specify
  failure behaviour, redaction, and cancellation semantics.
* Add no user-selectable model parameters in this phase, so frontend/core
  node-schema synchronization is not implicated.

**Exit:** a synthetic publisher can stream/reload a bounded line series without
regressing status events or polling.

### Phase 1 — completion-first classic diagnostics (1–2 sprints)

* Wire the already-saved evaluation data and metric/importance data into a
  prominent completed-job “Training results” panel; reuse the existing
  `ClassificationChartsForSplit` and `RegressionChartsForSplit` components.
* Show explicit availability states: ROC/PR need probabilities; importance may
  be unsupported; report evaluation failures without hiding the trained model.
* Add feature-importance bar rendering if the existing experiments view does
  not already surface the metric.

**Exit:** a completed classifier and regressor show their existing diagnostics
without a second training run or a hidden navigation path.

### Phase 2 — PyTorch scalar telemetry (with DL direct-fit delivery)

* Implement the DL direct-fit branch and shared loop already approved in the
  DL initiative; add epoch-end telemetry callbacks there, not to
  `TuningCalculator`.
* Extend the existing WebSocket schema/client and create a focused
  `TrainingTelemetryPanel` using Recharts (already a dependency in
  `frontend/ml-canvas/package.json`; existing charts import it at
  `RegressionChartsForSplit.tsx:10-15`).
* Persist/reload histories, test disconnect/reconnect and completed-job
  history, and throttle events.

**Exit:** an MLP job visibly updates train/validation loss and one task metric
per epoch, survives refresh, and has no batch-event flood.

### Phase 3 — selected DL diagnostics (1 sprint)

* Add LR and global gradient-norm series; validation confusion matrix only at
  a configured epoch cadence and only for bounded validation data.
* Add a PCA embedding plot from sampled final DL embeddings; provide true vs
  predicted colour selector and clear sampling/projection annotation.

**Exit:** plots remain bounded for the agreed dataset/epoch budget and do not
materially alter training throughput under benchmark.

### Phase 4 — opt-in expensive analysis

* Submit learning/validation curves as separate diagnostic jobs using explicit
  train-size/parameter/fold caps and artifact caching.
* Add sampled UMAP/t-SNE only if PCA proves insufficient in user research.
* Consider per-layer activation statistics only as an advanced debugging
  feature with framework/model allow-lists.

**Exit:** estimates (fit count, sample count, expected runtime) are shown
before execution; normal training remains unaffected.

## Open risks and decisions

1. There is no DL implementation in the current source tree; PyTorch telemetry
   depends on the approved DL initiative landing first.
2. The current socket is global broadcast without per-user authorization
   (`backend/realtime/events.py:19-21`); scalar-only events are acceptable
   short-term, while embeddings/raw predictions must wait for scoped channels.
3. The worker model is synchronous and the DL design notes a solo-pool,
   single-job queue constraint (`architecture-design.md:187-191`); live charts
   improve observability, not queue throughput.
4. Metric meaning requires a stable validation split, class-label handling and
   imbalanced-class defaults (PR/F1 should be available, not accuracy only).
5. Persisted histories need artifact-store versioning and retention policies;
   DB JSON is unsuitable for high-frequency points.
6. Decide product limits before Phase 2: maximum epochs/events, metric names,
   validation cadence, sample size, and whether users can opt into expensive
   diagnostics. These limits are required to make “near-live” reliable rather
   than a Redis/WebSocket load amplifier.
