# Deep Learning Frontend Design Proposal

**Date:** 2026-08-11  
**Scope:** `frontend/ml-canvas/` only; implementation proposal, not implementation  
**Sources reviewed:** the approved architecture and roadmap in this directory, plus the current frontend files cited below.

## 1. Executive decisions

1. Add six **separate canvas definitions**, not a layer-builder: five model nodes and one image source node. All five model nodes use `createModelingNode`; no new backend `StepType` is introduced.
2. Add one shared `DLTrainingSettings` form. It keeps the existing Basic / Advanced (Tuning) segmented control and job-submission path, but replaces sklearn's generic registry-driven parameter editor with a small, typed DL form.
3. Keep the existing Jobs Drawer as the primary running-job surface. Extend its existing `JobDetailsView` with a DL-only “Training” tab containing the live loss curve, device, epoch, and ETA. Do not put a full chart in the canvas card.
4. Reuse existing WebSocket invalidation and polling first. To render a live curve, the backend must additionally persist/return structured epoch telemetry; the current event and job payloads are deliberately insufficient.
5. Treat Image Data Loader as a distinct source/manifest type. It cannot be shoehorned into the tabular `DatasetNode` because the converter currently discovers only `dataset_node` roots and emits `dataset_id`/`data_loader`.

## 2. Existing frontend findings

### 2.1 Node definitions and `createModelingNode`

`src/core/factories/nodeFactory.ts:10-73` defines `createModelingNode<TConfig>`. A caller supplies the unique frontend `type`, display metadata, Lucide icon, settings component, optional default config, validation, preview, and optional ports. The factory:

- defaults to a `dataset` input (`Training Data`) and `model` output (`Model`) at lines 43-45;
- passes the settings component through unchanged (line 55);
- derives a compact fallback body preview from `model_type` and `target_column` (lines 56-63);
- requires `target_column` unless a caller overrides validation (lines 64-67);
- merges caller defaults over `target_column: ''` and `model_type: 'random_forest_classifier'` (lines 68-72).

This is the correct primitive for all five model nodes. Each DL node must override `model_type`, provide a DL settings component, and—where appropriate—override ports/validation. `ImageDataLoaderNode` must be a handwritten `NodeDefinition`, analogous to `DatasetNode`, because it is a source node with no input and a manifest output.

The existing task nodes are intentionally thin:

- `ClassificationNode.tsx:13-38`, `RegressionNode.tsx:12-37`, and `TextClassificationNode.tsx:15-40` call the factory, select an icon, bind the same `TrainingSettings` with a task filter, and supply defaults.
- `TextClassificationNode.tsx:7-14` documents a current constraint: text classifiers expect vectorized upstream text. The transformer node must explicitly supersede that expectation because its backend owns tokenization.
- Definitions are registered manually in `src/core/registry/init.ts:1-75`; a new file alone will not make a node appear in the palette.

### 2.2 Current training-settings UX and mode semantics

`src/modules/nodes/modeling/TrainingSettings.tsx` is the visual and behavioral precedent:

- `TrainingConfig` (lines 19-42) uses **frontend** `run_mode: 'basic' | 'advanced'`, fixed `hyperparameters`, advanced `search_space`, tuning fields, and CV fields.
- The `RunModeToggle` at lines 44-62 is the existing segmented Basic / Advanced (Tuning) control. “tuned” is **not** a UI mode value.
- Models come from `/pipeline/registry`, then are filtered by category and task tags (`TrainingSettings.tsx:191-226`). Classification uses `classification`, regression uses `regression`, and text classification maps to the `text` tag (lines 118-132).
- Basic mode fetches hyperparameter definitions from `/pipeline/hyperparameters/{modelType}` (lines 231-255); advanced mode fetches definitions and default search spaces (lines 261-305).
- The configuration column uses a Model Type select, Target Column select, and optional scaling advisory (`TrainingSettings.tsx:333-429`); advanced mode adds tuning strategy/metric/trials (lines 431-523); cross validation is an expandable card (lines 527-635).
- Fixed parameters use the “Customize” checkbox and individual cards (`TrainingSettings.tsx:641-755`); advanced search-space entries use typed comma-separated input or option chips (`SearchSpaceInput.tsx:1-148`).
- Wide properties panels use two columns; narrow panels use Configuration / Hyperparameters tabs (`TrainingSettings.tsx:862-900`). The gradient primary action submits via `useTrainingNodeContext` (lines 903-924).

The converter translates the UI values to backend values. `src/core/utils/pipelineConverter.ts:399-423` maps frontend `advanced` to backend `run_mode: 'tuned'` and `basic` to `run_mode: 'fixed'`; the same distinction is documented in `src/core/perf/perfThresholds.ts:39-48`. The approved architecture’s phrase `basic|tuned` therefore needs clarification before implementation: **retain the actual UI values `basic|advanced`; only emit `fixed|tuned` on the wire.**

### 2.3 Submission, converter, and task plumbing

`src/core/constants/stepTypes.ts:1-14` contains frontend definition constants which currently overlap backend names. It is not evidence that a new backend step type is required.

`pipelineConverter.ts` is the critical integration seam:

- it starts traversal only from nodes whose definition type is `dataset_node` (`lines 165-176`);
- it emits the tabular Dataset node as `data_loader` with `{ dataset_id }` (`lines 204-208`);
- it treats only `training`, `classification`, `regression`, and `text_classification` as unified training types (`lines 29-39`, `399-423`);
- every such node emits canonical backend `training`, never the frontend definition type;
- fixed parameters are currently whitelisted by `buildFixedTrainingParams` (`lines 126-142`), so DL fields would otherwise be silently dropped;
- advanced mode emits `algorithm` and `tuning_config.search_space` (`lines 405-416`).

This means new unique frontend IDs such as `neural_network_classifier` are needed, but no new backend `StepType` is needed. The converter must explicitly recognize those IDs and serialize the approved DL contract. It must also make the image source a traversal root and serialize it as the image-manifest loader step, rather than treating it as an unknown node.

`src/core/hooks/useTrainingNodeContext.ts:58-106` already handles schema lookup, upstream target inference, conversion, leakage validation, submission, notification, task tab selection, and drawer opening. DL settings should reuse it. Its upstream dataset resolver currently looks only for `datasetId`/`dataset_id` (lines 20-49), so it must be generalized to recognize an image manifest id or parameterized source id.

Task classification is registry-tag based in `src/components/pages/ExperimentsPage/utils/jobMeta.ts:150-163`; jobs history has only classification, regression, text classification, segmentation, and ensemble tabs (`components/panels/JobsDrawer.tsx:14-29`). DL should remain in its semantic task tab:

| DL node | Registry tags required | Existing task surface |
|---|---|---|
| Neural Network Classifier | `classification` | Classification |
| Neural Network Regressor | `regression` | Regression |
| Transformer Text Classifier | `text` (and preferably `classification`) | Text Classification |
| Sequence Forecaster | `regression` plus `time_series` | Regression |
| Image Classifier | `classification` plus `image`/`vision` | Classification |

No new `TaskType`, Jobs Drawer tab, or frontend `StepType` enum entry is recommended.

### 2.4 Current job progress/status behavior

The application already has a robust status transport but no numeric training telemetry:

- `src/core/realtime/jobEventsSocket.ts:14-25` defines events with optional `progress` and `current_step`; one singleton connects to `/ws/jobs` and reconnects with backoff (lines 30-155).
- `src/core/store/useJobStore.ts:44-133` uses WebSocket events as invalidation signals, then refetches job snapshots; it uses 30-second safety polling while connected and 3-second fallback polling while disconnected.
- `src/core/hooks/useJobPolling.ts:187-215` does the same for a selected job.
- `JobInfo` has no `progress`, `current_step`, device, ETA, or epoch-history field (`src/core/api/jobs.ts:4-38`).
- `JobDetailsView.tsx:593-600` explicitly says “Not reported” rather than draw a deceptive progress bar.
- Existing “Parallel Run” progress in `JobsDrawer.tsx:206-248` measures only completed branch count, not training progress.
- `JobDetailsView.tsx:89-98` already highlights `Epoch N/M` in live logs, and the Logs tab is real-time (`lines 531-555`), but parsing logs is not an acceptable source of chart data.

Canvas cards show completed trainer summaries, not live progress. `useNodeJobSummaries.ts:9-103` fetches completed per-node summaries after job events; `CustomNodeWrapper.tsx:364-497` renders that summary and labels it “previous run” while a new job is in flight. This is a good precedent for a small running-status chip, but not for a canvas chart.

### 2.5 Existing source/upload pattern

The existing tabular source consists of:

- `DatasetNode.tsx:15-136`: a settings component using React Query’s `useUsableDatasets`, a native select, an inline “New Upload” branch, and tabular schema preview;
- `DatasetNode.tsx:167-186`: a source node with no inputs, one `dataset` output, validation of `datasetId`, and a compact canvas component;
- `FileUpload.tsx:1-133`: drag-and-drop or browse, one file, 500 MB size check, upload progress, error panel, and CSV/XLSX/Parquet/JSON accept list;
- `core/api/datasets.ts:46-100`: `FormData` upload and XHR specifically to expose upload-byte progress;
- `core/hooks/useDatasets.ts:16-113`: React Query keys and mutation-based cache invalidation.

There is no existing image/folder/zip upload UX (`rg` found no image ingestion node or directory-upload control). A zip-first upload is therefore the reliable baseline; optional folder selection must be clearly browser-dependent and use `webkitdirectory`, not be the only way to import images.

### 2.6 Design system and chart inventory

This is a Tailwind + Radix/shadcn-style application, not MUI:

- `package.json` includes Tailwind, Radix primitives, `class-variance-authority`, Lucide, Recharts, Chart.js/react-chartjs-2, and Plotly.
- `tailwind.config.js:1-77` maps semantic colors to CSS variables and uses `tailwindcss-animate`; `index.css:20-105` supplies light/dark semantic tokens and reduced-motion behavior.
- Reusable `Button` and `Input` exist in `src/components/ui/button.tsx` and `input.tsx`; `FormField.tsx:24-81` supplies accessible label, hint, and error associations. Existing node settings mostly use local native controls with consistent Tailwind classes, as `TrainingSettings` does, rather than a shared Select/Slider/NumberInput primitive.
- `MetricsComparisonChart.tsx:1-178` and `pages/drift/DriftHistoryChart.tsx:1-100` establish the appropriate Recharts pattern: `ResponsiveContainer`, dark tooltips, `CartesianGrid`, `XAxis`, `YAxis`, `Legend`, and typed props.

Use **Recharts**, not a new chart package. It is already shipped, used for interactive metrics and time series, and is suitable for a two-line loss chart.

## 3. Proposed DL node architecture

### 3.1 Shared configuration contract

Define a frontend-only `DLTrainingConfig` in `DLTrainingSettings.tsx`:

```ts
type DLRunMode = 'basic' | 'advanced';
type DLArchitecturePreset = string; // narrowed per node by exported option arrays

interface DLTrainingConfig {
  run_mode: DLRunMode;
  model_type: 'mlp_classifier' | 'mlp_regressor' |
    'transformer_text_classifier' | 'sequence_forecaster' | 'image_classifier';
  target_column: string;
  architecture_preset: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  early_stopping_patience: number;
  optimizer?: string;
  execution_mode?: ExecutionMode;
  search_strategy: 'optuna';       // DL advanced UI permits supported strategy only
  n_trials: number;
  metric: string;
  random_state: number;
  search_space: Record<string, unknown>;
  strategy_params?: Record<string, unknown>;
  // modality-specific, explicit fields:
  text_column?: string;
  time_column?: string;
  group_by?: string[];
  forecast_horizon?: number;
  sequence_length?: number;
  image_label_column?: string;
}
```

The approved architecture’s representative parameters are top-level. Make that the wire contract as well: for fixed DL training the converter emits:

```ts
{
  run_mode: 'fixed',
  model_type,
  target_column,
  architecture_preset,
  epochs,
  batch_size,
  learning_rate,
  early_stopping_patience,
  optimizer,
  ...modalitySpecificFields,
  execution_mode
}
```

For advanced mode it emits `run_mode: 'tuned'`, `algorithm: model_type`, the same non-search “base” DL fields, and `tuning_config` containing `metric`, `n_trials`, `random_state`, `strategy_params`, and only the supported search dimensions: `architecture_preset`, `learning_rate`, and `batch_size`. This preserves the existing frontend/backend separation of UI `basic|advanced` vs wire `fixed|tuned`.

**Backend contract gate:** final preset IDs, optimizer IDs, min/max/step/default values, modality keys, and advanced search-space keys must be taken from the new backend registry/hyperparameter endpoints or a shared API manifest—not independently invented in TypeScript. The proposed initial UI values below are display/default recommendations pending that contract.

### 3.2 Shared settings component

Create `src/modules/nodes/modeling/DLTrainingSettings.tsx` with:

```ts
interface DLTrainingSettingsProps {
  config: DLTrainingConfig;
  onChange: (next: DLTrainingConfig) => void;
  nodeId?: string;
  isExpanded?: boolean;
  kind: 'tabular-classifier' | 'tabular-regressor' | 'text' | 'timeseries' | 'image';
}
```

It calls `useTrainingNodeContext(nodeId)` and uses the existing gradient submission button and `runJob(isAdvanced ? 'tuning' : 'training', task)`. It does **not** reuse `TrainingSettings` directly because that component fetches sklearn hyperparameter definitions, renders CV settings, and assumes tabular target/schema semantics.

Layout, deliberately matching `TrainingSettings.tsx:333-635` and `804-924`:

1. **Info notice**: “Architecture presets keep networks reproducible; layer-by-layer editing is not available.” Use the existing blue dismissible notice treatment.
2. **Training Mode**: copy `RunModeToggle` into an exported reusable `TrainingModeToggle`, or move it to `components/TrainingModeToggle.tsx`; retain labels exactly “Basic” and “Advanced (Tuning)”.
3. **Model configuration**: architecture preset select first, followed by task-specific data selectors (target/text/time/group/etc.). For tabular tasks, include the existing scale-data advisory; DL MLPs should carry/receive the same `requires_scaling` metadata.
4. **Training controls**: a responsive two-column grid in the expanded panel, one column in normal panel:
   - Epochs: numeric input, min 1, server-provided max/default;
   - Batch size: select of server-supported values (rather than arbitrary number);
   - Learning rate: numeric `step="any"` or preset select with a helper such as “Typical: 0.001”;
   - Early stopping patience: integer numeric input; a documented `0` policy must be agreed with the backend (recommend 0 means disabled only if backend adopts it).
5. **Advanced only**: preserve the purple advanced styling and the existing strategy configuration modal only if the backend supports it. The initial DL UI should select Optuna and limit search candidates to preset, learning rate, and batch size. Do not show sklearn-only grid/halving choices or generic CV configuration for epoch-trained DL jobs unless the backend actually supports them.
6. **Submit footer**: same disabled-until-source-connected behavior and same drawer-opening flow as `TrainingSettings`.

Use `FormField` for newly written controls where feasible; use the native select/input class strings already present in `TrainingSettings` when a form must exactly match it. Do not add sliders: the project has no established Slider primitive, values need precise entry, and DL batch sizes/presets are discrete.

### 3.3 Per-node definitions

| File | Factory definition | Defaults and settings specialization |
|---|---|---|
| `NeuralNetworkClassifierNode.tsx` | `createModelingNode<DLTrainingConfig>`; type `neural_network_classifier`; icon `BrainCircuit`; normal dataset input and trained-model output | `model_type: 'mlp_classifier'`, task classification, `architecture_preset: 'medium'`, recommended `epochs: 50`, `batch_size: 32`, `learning_rate: 0.001`, patience 5. Validate target and source. |
| `NeuralNetworkRegressorNode.tsx` | Same; type `neural_network_regressor`; icon `BrainCircuit` or `TrendingUp` | `model_type: 'mlp_regressor'`; regression task and matching metric defaults. |
| `TransformerTextClassifierNode.tsx` | Same; type `transformer_text_classifier`; icon `Languages` or `FileText` | `model_type: 'transformer_text_classifier'`; text task; add required `text_column` and target selection. Its description must say tokenizer is internal and surface an expected CPU-duration warning. It must not require TF-IDF/vectorizer upstream. |
| `SequenceForecasterNode.tsx` | Same; type `sequence_forecaster`; icon `ChartNoAxesCombined`/`ChartLine` | `model_type: 'sequence_forecaster'`; regression task; require target, time column, sequence length, and forecast horizon. Expose architecture presets only after backend names LSTM/TCN presets. |
| `ImageClassifierNode.tsx` | Same; type `image_classifier`; icon `Image` | `model_type: 'image_classifier'`; classification task; input port is `{ id: 'images', label: 'Image Manifest', type: 'dataset' }`; require an upstream image source rather than tabular target selection. |
| `ImageDataLoaderNode.tsx` | Handwritten `NodeDefinition<ImageDataLoaderConfig>`; type `image_data_loader`; icon `Images`; category Data Source | No input; output `{ id: 'images', label: 'Image Manifest', type: 'dataset' }`; custom card displays dataset name, image count, class count, and status when returned by the image API. |

Use explicit frontend definition string literals, not new `StepType` entries. This avoids falsely implying a new backend step. The converter maps all five model literals to canonical backend `training`.

## 4. DL telemetry and job UX

### 4.1 Recommended location

Put the full live experience in the existing **Jobs Drawer → Job Details** modal:

- add a `Training` tab between Overview and Live Logs when `job` contains DL telemetry (or registry metadata identifies a DL model);
- keep Overview’s status cards and Stop action as-is;
- add a compact, non-interactive “Running · 42%” chip to the model canvas card only after progress exists; the card remains short and the full chart is one click away through the job drawer/inspector link;
- after completion, render the same curve in the Training tab and retain it in the job result. Experiments can later consume it, but that is not a Phase 1 prerequisite.

This follows the application’s existing running-job pattern: submission opens the drawer (`useTrainingNodeContext.ts:77-101`), details already poll a single job (`JobDetailsView.tsx:403-412`), and logs are already live.

### 4.2 Required API/event extension

The current WebSocket `JobEvent` only invalidates; it does not safely carry a curve (`jobEventsSocket.ts:14-25`), and `JobInfo` cannot represent one (`api/jobs.ts:4-38`). Extend the backend response/event contract before rendering:

```ts
interface DLTrainingEpoch {
  epoch: number;
  train_loss: number;
  validation_loss?: number;
  learning_rate?: number;
  elapsed_seconds?: number;
}

interface DLTrainingProgress {
  current_epoch: number;
  total_epochs: number;
  percent: number;
  eta_seconds?: number;
  device: 'cpu' | 'cuda' | 'mps';
  epoch_history: DLTrainingEpoch[]; // bounded, append-only/replaced snapshot
}
```

Add `progress?: DLTrainingProgress` to `JobInfo`. Either include `training_progress?: DLTrainingProgress` in the existing status/progress WebSocket event or continue the current invalidation-plus-GET design and return it from `GET /pipeline/jobs/{id}`. Prefer the latter initially: it retains the established coalescing/refetch model and prevents reconnecting clients from missing points. A later optimization can carry the latest point in the event.

ETA must be backend-calculated from completed epochs (or omitted until at least one epoch has elapsed); the UI must display “Estimating…”/“—”, never a guessed duration. Device is an execution fact from the training manager, not browser GPU detection. Display `GPU` only for `cuda`, `Apple GPU` for `mps`, and `CPU` otherwise.

### 4.3 Components

Create:

- `components/panels/jobs/DLTrainingProgress.tsx`
  - props: `{ progress: DLTrainingProgress; status: JobStatus; compact?: boolean }`;
  - header chips: `Epoch 12 / 50`, `42%`, `CUDA`/`MPS`/`CPU`, and ETA;
  - accessible progress bar (`role="progressbar"`, `aria-valuenow`, min/max);
  - neutral empty state while queued or awaiting first epoch.
- `components/panels/jobs/DLTrainingCurve.tsx`
  - props: `{ epochs: DLTrainingEpoch[]; status: JobStatus }`;
  - Recharts `ResponsiveContainer` + `LineChart`, blue solid Train Loss and orange dashed Validation Loss, matching `DriftHistoryChart.tsx:42-95`;
  - render no validation series when absent; show a textual empty state for zero points; use a stable last-point indicator rather than animation churn during updates.

Wire `DLTrainingProgress` into the existing Overview Progress card, replacing “Not reported” only when the structured field exists. Wire the full component and curve into the new Training tab. Respect the app-level reduced-motion CSS (`index.css:91-105`) and disable/reduce chart animation when `prefers-reduced-motion` is set.

### 4.4 Failure/cancellation behavior

If a DL job fails (OOM, NaN loss, divergence), retain all received points, show the existing red status/error treatment, and add a short final-state line in the Training tab. Do not turn an incomplete curve into a “completed” metric chart. Cancellation likewise retains the curve and makes Stop unavailable once terminal; this reuses current `JobDetailsView` behavior at lines 495-524.

## 5. Image Data Loader UX

### 5.1 Settings flow

`ImageDataLoaderSettings` is a sibling to tabular `DatasetSettings`, not a modification of it. It has two deliberate ingestion modes:

1. **Upload images and labels (default):**
   - required image archive chooser/drop zone accepting `.zip`;
   - optional “Select folder” secondary action using a hidden `<input type="file" webkitdirectory multiple>`; show browser-support guidance and package the file list only if the backend API explicitly supports multi-file folders;
   - required label CSV chooser accepting `.csv`;
   - instructions: CSV must contain `filename` and `label`; offer selects for those headers after a lightweight parse/manifest response rather than hard-code column names;
   - upload byte progress, error panel, and cancel affordance matching `FileUpload.tsx:68-130`;
   - post-upload manifest summary: image count, label/class count, unsupported/skipped file count, and the selected filename/label mappings.
2. **Use an existing tabular image-path dataset:**
   - reuse `useUsableDatasets`, the native dataset select styling, and schema loading from `DatasetNode.tsx:54-134`;
   - require image-path-column and label-column selects from schema;
   - submit a manifest-creation request and then store the returned image manifest id.

The source node stores only a completed `image_dataset_id`/manifest identifier plus summary metadata, never browser `File` objects or an extracted folder path in canvas persistence.

### 5.2 API/query layer

Create a dedicated `core/api/imageDatasets.ts` and `core/hooks/useImageDatasets.ts`; do not overload `DatasetService` types that promise tabular schema/profile behavior. Proposed frontend-facing API contract (endpoint names must be confirmed with the Phase 4 backend router):

```ts
type ImageManifest = {
  id: string;
  name: string;
  image_count: number;
  class_count: number;
  labels: string[];
  status: 'pending' | 'ready' | 'failed';
};

uploadImageManifest({
  archive?: File,
  files?: File[],
  labelsCsv?: File,
  filenameColumn: string,
  labelColumn: string,
  onProgress,
}): Promise<ImageManifest>;

createImageManifestFromDataset({
  dataset_id: string,
  image_path_column: string,
  label_column: string,
}): Promise<ImageManifest>;
```

Mirror the React Query mutation/cache invalidation pattern at `useDatasets.ts:64-113`; use XHR/FormData for byte progress exactly as `DatasetService.uploadWithProgress` does (`datasets.ts:63-100`). Enforce size/type checks client-side for fast feedback, but present backend manifest validation as authoritative (zip safety, path containment, missing labels, unsupported images).

### 5.3 Image classifier settings

`ImageClassifierNode` reads the upstream manifest summary but does not expose a tabular target select. Its Basic form has:

- architecture/backbone preset (e.g., backend-defined `resnet18`/`mobilenet_v3_small` transfer-learning presets);
- epochs, batch size, learning rate, patience;
- read-only source summary with “N images · K classes” and a warning when missing/not-ready;
- optional image augmentations only if Phase 4 backend exposes a fixed allow-list; do not invent arbitrary transforms.

Its validation is stricter than factory default: require `image_dataset_id` discoverable upstream and a ready manifest, not `target_column`.

## 6. File-by-file implementation plan

### Phase 1: tabular MLP

| Action | File | Purpose / key implementation |
|---|---|---|
| Create | `src/modules/nodes/modeling/DLTrainingSettings.tsx` | Shared typed DL settings, Basic/Advanced mode, task/modality sections, `useTrainingNodeContext` submission, server-aligned defaults/search space. |
| Create | `src/modules/nodes/modeling/NeuralNetworkClassifierNode.tsx` | Factory definition for `neural_network_classifier`; MLP classifier default config and body preview such as `Medium MLP → churn`. |
| Create | `src/modules/nodes/modeling/NeuralNetworkRegressorNode.tsx` | Factory definition for `neural_network_regressor`; MLP regressor defaults. |
| Modify | `src/core/registry/init.ts` | Import and `registry.register()` both nodes after existing modeling registrations. |
| Modify | `src/core/utils/pipelineConverter.ts` | Add a `DL_TRAINING_DEFINITION_TYPES` set; serialize both node types to canonical `training` with the explicit DL fixed/tuned shapes; add them to source/terminal/model lists only where their behavior matches. Do **not** modify `StepType`. |
| Modify | `src/core/types/executionMode.ts` | Add the two definitions to `EXECUTION_MODE_AWARE_TYPES` if DL model nodes support existing Merge/Parallel behavior. |
| Modify | `src/core/hooks/useBranchColors.ts`, `src/core/store/useGraphStore.ts`, `src/core/perf/perfThresholds.ts`, and toolbar run-control type sets | Add the two model definition IDs wherever existing trainer IDs are enumerated, preserving canvas branching, source-to-model rejection, and trainer/tuner performance buckets. Consolidate these sets into a shared exported constant if practical to avoid a sixth duplicated list. |
| Modify | `src/core/utils/pipelineConverter.test.ts` and snapshots | Test fixed and advanced DL serialization, canonical `training` output, and preservation of every DL field. |
| Create | `src/modules/nodes/modeling/DLTrainingSettings.test.tsx` | Test defaults, mode translation inputs, controls, source-disabled submission, and advanced search-space restrictions. |

### Phase 2: transformer text classifier

| Action | File | Purpose / key implementation |
|---|---|---|
| Create | `src/modules/nodes/modeling/TransformerTextClassifierNode.tsx` | Factory node with transformer-specific description, default model id, target/text selectors. |
| Modify | `DLTrainingSettings.tsx` | Add `kind: 'text'`; require text column and show CPU-duration/tokenizer guidance. |
| Modify | `registry/init.ts`, `pipelineConverter.ts`, execution/branch/perf/run-control lists | Register and recognize the new frontend definition; canonicalize to backend `training`. |
| Modify | `core/templates/pipelineTemplates.ts` only if an existing text template should offer the transformer path | Keep existing vectorized-text template unchanged unless product explicitly chooses to add a second transformer template. |
| Modify/Create | converter and settings tests | Assert `text_column` survives conversion and no vectorizer-only UI requirement is imposed. |

### Phase 3: sequence forecaster

| Action | File | Purpose / key implementation |
|---|---|---|
| Create | `src/modules/nodes/modeling/SequenceForecasterNode.tsx` | Factory node with time-series description and forecast-specific default fields. |
| Modify | `DLTrainingSettings.tsx` | Add `kind: 'timeseries'`; use upstream schema to select target/time/group columns; require sequence length and horizon. |
| Modify | `registry/init.ts`, converter, execution/branch/perf/run-control lists | Register and canonicalize the node; preserve time-series-specific fields. |
| Create/Modify | tests | Assert validation for missing time/horizon, and exact wire serialization. |

### Phase 4: image ingestion and classifier

| Action | File | Purpose / key implementation |
|---|---|---|
| Create | `src/core/api/imageDatasets.ts` | Strict image manifest request/response types and XHR upload functions. |
| Create | `src/core/hooks/useImageDatasets.ts` | React Query keys, list queries, upload/create-manifest mutations, cache invalidation. |
| Create | `src/modules/nodes/data/ImageDataLoaderNode.tsx` | Source node, upload/existing-path settings, compact manifest card, source-specific validation. Split `ImageUpload` into a sibling file if the settings would otherwise exceed the existing `FileUpload` size. |
| Create | `src/modules/nodes/modeling/ImageClassifierNode.tsx` | Factory model node with image-manifest input and image-specific validation/defaults. |
| Modify | `src/core/registry/init.ts` | Register image loader and classifier. |
| Modify | `src/core/utils/pipelineConverter.ts` | Treat `image_data_loader` as a root; emit the Phase 4 backend image-manifest step with its manifest id/config; dispatch `image_classifier` as canonical training; use image root for `metadata.dataset_source_id` if backend contract requires it. |
| Modify | `useTrainingNodeContext.ts` | Resolve image manifest/source IDs and bypass tabular schema target assumptions for image jobs. |
| Modify | `executionMode.ts`, `useBranchColors.ts`, `useGraphStore.ts`, `perfThresholds.ts`, toolbar/run controls | Add image classifier to modeling behavior sets, but do not add image loader to trainer sets. |
| Create | `ImageDataLoaderNode.test.tsx`, API/hook tests, converter tests | Cover archive/CSV required validation, ready-manifest source validation, tabular-path mode, and image source → classifier pipeline serialization. |

### Cross-phase telemetry

| Action | File | Purpose / key implementation |
|---|---|---|
| Modify | `src/core/api/jobs.ts` | Add typed `DLTrainingEpoch` and `DLTrainingProgress` fields to `JobInfo`. |
| Modify | `src/core/realtime/jobEventsSocket.ts` | Type any newly delivered structured progress field; preserve compatibility with status-only events. |
| Create | `src/components/panels/jobs/DLTrainingCurve.tsx` | Recharts loss plot. |
| Create | `src/components/panels/jobs/DLTrainingProgress.tsx` | Progress/device/ETA cards and accessible progress bar. |
| Modify | `src/components/panels/jobs/JobDetailsView.tsx` | DL-only Training tab and structured Overview progress. |
| Modify | `src/components/canvas/CustomNodeWrapper.tsx` | Optional tiny live status chip only; retain summary/body priority. |
| Create | component tests | Verify CPU/GPU/MPS labels, missing ETA/history, curve update, failed/cancelled curve retention, and accessibility attributes. |

## 7. Compatibility and validation gates

1. **No accidental sklearn regression.** Existing `TrainingSettings` and all four existing frontend definition IDs continue to serialize byte-for-byte as they do now. DL serialization is a new explicit branch, not a change to `buildFixedTrainingParams`.
2. **Single source of supported options.** Before each frontend phase, compare backend node metadata/allow-lists with dropdowns and search spaces. Record the comparison in the PR; do not hard-code an option merely because it appears in this proposal.
3. **No fake live metrics.** Do not release a curve, device badge, numeric progress, or ETA until `JobInfo`/events return structured data. Logs remain supplemental.
4. **Image manifest isolation.** Never call tabular schema/profile endpoints for an image manifest, never persist browser paths/files in canvas nodes, and do not offer folder upload as the only path.
5. **Frontend checks.** For each implementation phase run from `frontend/ml-canvas/`: `npx eslint <changed files>` while iterating, then `npm run lint`, `npx tsc --project tsconfig.json --noEmit`, `npm run build`, and targeted Vitest tests.
6. **Backend/frontend sync.** Add a phase gate confirming all values/keys: presets, optimizer, epochs/batch/LR/patience bounds, text/time/image config keys, model registry tags, telemetry schema, and manifest API response fields.

## 8. Recommended implementation order

Implement the shared configuration/serialization first with the two MLP nodes, then telemetry (which benefits every DL modality), followed by transformer, sequence, and image. The image source must ship together with `ImageClassifierNode`; exposing a model node before a manifest-producing source produces an unusable canvas path. GPU presentation is telemetry-only and must accurately report the backend-selected device; GPU scheduling itself remains deferred to the Ray phase described by the approved architecture.
