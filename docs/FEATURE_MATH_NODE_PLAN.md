# Feature Math Node Design Plan

## 1. Goal & Scope

Add a new **Feature Math** node to the feature engineering canvas that lets users create multiple derived features in a single node. The node will support arithmetic combinations, statistical aggregations, ratios, fuzzy similarity scores, and datetime expansions. Users can configure several operations at once, enabling complex feature creation without duplicating nodes.

Key capabilities to support:

- **Arithmetic**: addition, subtraction, multiplication, division across 2+ columns.
- **Ratios**: numerator/denominator with safe division handling.
- **Statistical aggregations**: count, min, max, sum, mean, median, std, range.
- **Vector similarity**: fuzzy string similarity (token sort ratio) with optional normalization.
- **Datetime enrichment**: extract year, month, day, hour, weekday/weekend, quarter, season, time-of-day buckets (morning/afternoon/evening/night).
- **Flexible configuration**: each operation defines input columns, optional constants, output column name, fill/null handling, and metadata tags.
- **Multi-operation**: a single node instance may execute many operations of different types sequentially on the same dataframe.
- **Split-aware**: respect train/test/validation splits via existing `SplitAwareProcessor` conventions (fit-on-train where required, transform others).

## 2. Backend Architecture

### 2.1 New Module

Create `core/feature_engineering/nodes/feature_eng/feature_math.py` with:

- `FEATURE_MATH_ALLOWED_TYPES`: set of operation categories.
- `FeatureMathOperation` dataclass (parsed config per operation).
- Normalization helpers (coerce columns, constants, output names, math safely, timezone parsing).
- Execution helpers per operation type:
  - `_apply_arithmetic_operation` (handles add/sub/mul/div; supports broadcasting constants).
  - `_apply_ratio_operation` (safe division, optional smoothing epsilon).
  - `_apply_stat_operation` (count/min/max/std/median etc; per-row across columns or aggregated? requirement implies row-wise new feature; interpret as row-wise combination).
  - `_apply_similarity_operation` (string similarity; use `rapidfuzz` if available, fall back to difflib ratio; cache scorer import at module level to avoid repeated import).
  - `_apply_datetime_operation` (parse column to datetime using pandas; produce requested components; handle timezone and naive cases).

- Primary entry `apply_feature_math(frame: pd.DataFrame, node: Dict[str, Any], pipeline_id: Optional[str] = None)` returning tuple `(frame, summary, FeatureMathNodeSignal)`.

### 2.2 Config Schema

Expect node `data.config` shaped as:

```json
{
  "operations": [
    {
      "id": "op_1",
      "operation_type": "arithmetic",
      "method": "add",  // add|subtract|multiply|divide
      "input_columns": ["col_a", "col_b"],
      "constants": [1.5],
      "output_column": "total_score",
      "fillna": 0.0,
      "round": 3
    },
    {
      "operation_type": "datetime_extract",
      "input_columns": ["event_ts"],
      "datetime_features": ["hour", "weekday", "season", "time_of_day"],
      "timezone": "UTC",
      "output_prefix": "event_"
    },
    {
      "operation_type": "similarity",
      "input_columns": ["title_a", "title_b"],
      "metric": "token_sort_ratio",
      "output_column": "title_similarity",
      "normalize": true
    }
  ],
  "error_handling": "skip",   // skip|fail
  "epsilon": 1e-9               // optional global constant for safe division
}
```

Backend normalization should:

- Deduplicate column names, enforce minimum inputs per operation (>=2 for arithmetic, ratio, similarity; >=1 for datetime).
- Resolve defaults (output column naming if blank: e.g. `{method}_{col1}_{col2}`).
- Validate output column does not clash unless overwrite allowed via flag.
- Provide warnings in node signal for skipped operations.

### 2.3 Node Signal & Schemas

Add to `core/feature_engineering/schemas.py`:

- `FeatureMathOperationResult` (operation_id, output_column, status, message, created_columns).
- `FeatureMathNodeSignal` (node_id, executed_operations, skipped_operations, warnings).

Expose in API responses (pipeline preview signals, execution logs) similar to encoding nodes.

### 2.4 Pipeline Integration

Modify `routes.py`:

1. **Imports**: `from .nodes.feature_eng.feature_math import apply_feature_math, FEATURE_MATH_DEFAULTS` (if constants).
2. **Preview Flow**: In the loop around line ~900 add branch `if catalog_type == "feature_math": working_frame, _, _ = apply_feature_math(...)`.
3. **Full Execution**: In the main execution/preview sections (e.g. around line 1550) add `signals.feature_math.append(signal)` collection (update `PipelinePreviewSignals` schema with optional list `feature_math`).
4. **Node Catalog**: Add entry near other feature nodes with parameters:
   - Parameter `operations` type `json_config` (custom) or rely on front-end managed structure stored under `data.config` (node parameter definitions optional if UI handles state).
   - Provide high-level toggles: `error_handling`, `epsilon`, `timezone_default`, `allow_overwrite`.

5. **Recommendation/Auto-Detect**: (future) Could add suggestions; for initial version no recommendations endpoint necessary.

### 2.5 Transformer Storage

- Node is deterministic row-wise; no storage required. Ensure `SplitAwareProcessor` treats as transformer; category should be `"Feature Engineering" -> NodeCategory.TRANSFORMER` and safe to run per split.
- For datetime parsing, maintain consistent timezone; to preserve training/test parity, use same configuration without storing fit objects.

### 2.6 Dependencies

- Add optional dependency `rapidfuzz` for fuzzy similarity in `pyproject.toml` / `requirements-fastapi.txt` (backend) and ensure fallback to stdlib difflib.
- Document in README/plan to update Docker image if necessary.

### 2.7 Validation & Testing

New unit tests under `tests/feature_engineering/nodes/test_feature_math.py` covering:

- Arithmetic operations with numeric + fillna.
- Multiplication with constants.
- Ratio with zero denominator and epsilon fallback.
- Similarity metric fallback when `rapidfuzz` unavailable.
- Datetime extraction with timezone and categorical bins.
- Aggregation generating stats columns.
- Multi-operation node producing multiple columns.
- Split-aware behaviour (train/test) verifying deterministic outputs.

### 2.8 Metrics & Logging

- Log per operation success/failure.
- Node summary string summarising counts ("Feature math: 5 operations (4 succeeded, 1 skipped)").

## 3. Frontend Architecture

### 3.1 Node Catalog Entry

Update node catalog JSON builders (`routes.py` response) to expose UI metadata:

```json
{
  "type": "feature_math",
  "label": "Feature math & datetime",
  "description": "Create arithmetic combinations, similarity scores, and datetime buckets in one step.",
  "inputs": ["dataset"],
  "outputs": ["dataset"],
  "category": "Feature Engineering",
  "tags": ["math", "datetime", "aggregation"],
  "parameters": [
    { "name": "error_handling", "type": "select", "options": [...] },
    { "name": "epsilon", "type": "number", "default": 1e-9, "step": 1e-9 },
    { "name": "default_timezone", "type": "text", "placeholder": "UTC" }
  ],
  "default_config": {
    "operations": [],
    "error_handling": "skip",
    "epsilon": 1e-9,
    "default_timezone": "UTC"
  }
}
```

### 3.2 UI Component

Create new React component `frontend/feature-canvas/src/components/node-settings/nodes/feature_math/FeatureMathSection.tsx` with responsibilities:

- Render table/list of configured operations.
- Provide buttons to add new operation (modal or inline form).
- Support editing and reordering operations (drag or move up/down).
- Validate field-level errors (e.g. missing columns, duplicate output names) before saving.
- Provide presets for time-of-day buckets and seasons (auto-populate configuration).
- Show live preview of output column names and sample formulas.
- Manage toggles for `errorHandling`, `epsilon`, `defaultTimezone`, `allowOverwrite`.

Leverage existing shared utilities for column lists and dataset schema (e.g. `useDatasetColumns` hooks already used by encoding nodes).

### 3.3 State Management

- Extend `NodeSettingsModal` to register `feature_math` renderer, hooking into `renderNodeSpecificContent` map.
- Use local state to manage operations array, convert to backend format via serializer.
- Provide TypeScript types `FeatureMathOperationConfig` mirroring backend schema.
- Implement helper utilities `normalizeFeatureMathConfig`, `serializeFeatureMathConfig` similar to alias nodes.
- Ensure updates propagate via `onUpdateNodeConfig` to persist in pipeline graph.

### 3.4 Operation Editing UX

Each operation row includes:

- **Operation type select** (Arithmetic / Ratio / Aggregation / Similarity / Datetime Extract).
- Dependent inputs (for arithmetic show multi-select for columns and optional constants; for similarity show pair selector; for datetime show column and checkboxes for features).
- Output column text input (with auto-suggest).
- Optional advanced options per type (rounding, epsilon override, smoothing constant, normalization, timezone override).
- Delete button.

Consider using collapsible cards for readability when multiple operations exist.

### 3.5 Validation & Feedback

- Immediate inline errors for missing inputs or invalid combinations.
- Show summary chips for each operation ("age + tenure → age_tenure_sum").
- Provide preview badges ("Creates 4 columns" for datetime operations).

### 3.6 Frontend API Types

- Update `frontend/feature-canvas/src/api.ts` with new type definitions for node signals/responses.
- Ensure type-safe mapping in state selectors.

### 3.7 Tests & QA

- Add unit tests for serializer/normalizer utilities (using Jest existing patterns).
- E2E/regression: update cypress (if exists) or manual QA plan to confirm new node works in canvas, persists config, and preview executes.

## 4. Integration Points & Dependencies

- **Backend**: Node integrated with preview API, pipeline execution, node catalog, transformer categories.
- **Frontend**: Node settings registration, pipeline graph serialization, node palette update (new card in canvas sidebar).
- **Docs**: Update user guide and node catalog documentation to explain operations and examples.
- **Dependencies**: Add `rapidfuzz` (backend) and optionally `dayjs` or rely on existing libs for datetime UI formatting.

## 5. Implementation Plan

1. **Backend Skeleton (2-3 days)**
   - Add feature_math module with parsing and operations for arithmetic + aggregation.
   - Integrate into `routes.py` preview/execution flows and node catalog.
   - Update schemas, add tests, wire into `PipelinePreviewSignals`.

2. **Advanced Operations (2 days)**
   - Implement similarity (rapidfuzz + fallback) and datetime extraction logic (including season/time-of-day).
   - Extend tests for edge cases (nulls, timezones, string columns).

3. **Frontend UI (3-4 days)**
   - Build `FeatureMathSection` component with list + form.
   - Implement serializers, register node, update node palette.
   - Add tests for serialization and component behaviour.

4. **Integration & QA (2 days)**
   - Manual end-to-end validation (node preview, pipeline execution, exported dataset correctness).
   - Update documentation and changelog.

## 6. Open Questions

- **Performance**: Should large operation sets (50+) be split for performance? Consider chunking or vectorised pandas apply.
- **Timezones**: Allow per-operation timezone override vs global default?
- **Similarity metrics**: Support additional metrics (cosine, embeddings) in future iterations.
- **Aggregation semantics**: Current plan treats stats as row-wise across selected columns. Need confirmation if column-wise aggregates (groupby) or windowed stats are desired.
- **Error policy**: Default to `skip` with warning? Need product decision.

## 7. Next Steps

- Review plan with stakeholders.
- Confirm scope of initial release (which operation types are MVP-critical).
- Finalize dependency choices (rapidfuzz vs thefuzz; additional frontend libs).
- Schedule implementation sprints aligned with main export feature roadmap.
