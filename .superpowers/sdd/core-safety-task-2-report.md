# Task 2 Report: Preserve Every Preprocessing Step's Metrics

Status: implemented, repaired for follow-up review findings, and **pending independent review** (not yet accepted).

## Summary
- Refactored `FeatureEngineer.fit_transform()` to return:
  - top-level `summary`
  - top-level `steps`
  - only four top-level compatibility aliases: `fit_time`, `peak_memory_bytes`, `rows_in`, `rows_out`
- Namespaced every step as `"{index}:{name}"` with `name`, `transformer`, `fit_time`, `peak_memory_bytes`, `rows_in`, `rows_out`, and `details`.
- Kept node-specific metrics exclusively inside `steps[*].details`.
- Repaired the remaining frontend readers so wrapped single-transformer node results resolve `steps[*].details` without falling back to removed flat keys.

## Frontend selection policy
- `getNodeMetricDetails()` now auto-resolves nested metrics only when the payload has exactly one step with a `details` object.
- Multi-step payloads are treated as ambiguous and return `null` unless the caller passes an explicit `{ stepKey }` selector.
- Legacy flat payloads with no `steps` object still return the original flat metrics for historical backend results.

## Compatibility inspection
- Confirmed `backend/ml_pipeline/_execution/engine/_feature_eng.py::_run_feature_engineering()` forwards the preprocessing metrics dict untouched.
- Confirmed `backend/ml_pipeline/_execution/engine/_node_runners.py::_run_transformer()` forwards the dict returned by `FeatureEngineer.fit_transform()` untouched.
- Confirmed `frontend/ml-canvas/src/components/canvas/CustomNodeWrapper.tsx` only reads flat `fit_time`, `peak_memory_bytes`, `rows_in`, and `rows_out`, so the retained compatibility aliases remain sufficient there.
- Found real consumers of removed flat node-specific keys in frontend preprocessing node panels plus backend dropped-column rollups. Updated them to read nested step details (with legacy fallback for older payloads).
- Repaired the remaining direct flat-key readers in `ScalingNode.tsx` and `OutlierNode.tsx`; both now use the shared helper path.

## Files changed
- `skyulf-core/skyulf/preprocessing/pipeline.py`
- `skyulf-core/tests/test_preprocessing_pipeline.py`
- `tests/test_backend_strategies.py`
- `docs/user_guide/pipeline_quickstart.md`
- `frontend/ml-canvas/src/core/utils/preprocessingMetrics.ts`
- `frontend/ml-canvas/src/core/utils/preprocessingMetrics.test.ts`
- `frontend/ml-canvas/src/modules/nodes/processing/ScalingNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/OutlierNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/metricsFeedback.test.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/ImputationNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/EncodingNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/MissingIndicatorNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/FeatureSelectionNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/DropColumnsNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/DropRowsNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/FeatureGenerationNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/CastTypeNode.tsx`
- `frontend/ml-canvas/src/modules/nodes/processing/DeduplicationNode.tsx`
- `backend/ml_pipeline/_execution/strategies.py`
- `.superpowers/sdd/core-safety-task-2-report.md`
- `.superpowers/sdd/progress.md`

## Validation run
- `source .venv/bin/activate && pytest skyulf-core/tests/test_preprocessing_pipeline.py skyulf-core/tests/test_pipeline.py skyulf-core/tests/test_pipeline_integration_preprocessing.py tests/test_backend_strategies.py -q`
- `cd frontend/ml-canvas && npx vitest run src/core/utils/preprocessingMetrics.test.ts`
- `cd frontend/ml-canvas && npx vitest run src/core/utils/preprocessingMetrics.test.ts src/modules/nodes/processing/metricsFeedback.test.tsx`
- `cd frontend/ml-canvas && npx eslint src/modules/nodes/processing/ImputationNode.tsx src/modules/nodes/processing/EncodingNode.tsx src/modules/nodes/processing/MissingIndicatorNode.tsx src/modules/nodes/processing/FeatureSelectionNode.tsx src/modules/nodes/processing/DropColumnsNode.tsx src/modules/nodes/processing/DropRowsNode.tsx src/modules/nodes/processing/FeatureGenerationNode.tsx src/modules/nodes/processing/CastTypeNode.tsx src/modules/nodes/processing/DeduplicationNode.tsx src/core/utils/preprocessingMetrics.ts src/core/utils/preprocessingMetrics.test.ts`
- `cd frontend/ml-canvas && npx tsc --project tsconfig.json --noEmit`
- `cd frontend/ml-canvas && npm run build`
- `source .venv/bin/activate && ruff check .`
- `source .venv/bin/activate && ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py`
- `source .venv/bin/activate && ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py`

## Follow-up validation for the frontend repair
- `cd frontend/ml-canvas && npx vitest run src/core/utils/preprocessingMetrics.test.ts src/modules/nodes/processing/metricsFeedback.test.tsx` -> `9 passed`
- `cd frontend/ml-canvas && npm run lint` -> clean
- `cd frontend/ml-canvas && npx tsc --project tsconfig.json --noEmit` -> clean
- `cd frontend/ml-canvas && npm run build` -> success (pre-existing Vite circular chunk warning only)

## Notes
- Codacy CLI analysis was attempted after each edit, but the repository-local wrapper is malformed (`.codacy/cli.sh` is HTML and exits before analysis).
- No backend or core source files changed in this follow-up repair, so no additional Python-targeted tests or static gates were needed beyond the already-validated Task 2 contract commit.
