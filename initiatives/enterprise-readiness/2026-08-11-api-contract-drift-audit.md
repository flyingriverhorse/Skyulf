# API/frontend contract-drift audit

**Scope:** FastAPI backend (`backend/`) and React/TypeScript canvas (`frontend/ml-canvas/`). This excludes the already-documented hand-duplicated pipeline-node parameter risk. Citations are repository file:line locations inspected 2026-08-11.

## Executive assessment

The backend produces OpenAPI, but the frontend does not consume it. HTTP and WebSocket contracts are separately hand-authored TypeScript declarations; some Python endpoints are untyped. There is no API compatibility versioning. An incompatible backend/frontend or cached-asset rollout can therefore create a silent stale UI rather than a compile-time or CI failure.

## Findings

### 1. OpenAPI is generated but not used to create or verify frontend clients — High

FastAPI exposes `openapi_url=settings.API_OPENAPI_URL`, whose default is `/openapi.json` (`backend/main.py:282-291`; `backend/config/mixins/security.py:72-76`). The application additionally invokes FastAPI's `get_openapi()` to enrich that document (`backend/main.py:135-171`).

The canvas instead has hand-written Axios/fetch contracts: the Axios client is manual (`frontend/ml-canvas/src/core/api/client.ts:1-30`), as are `JobInfo` (`frontend/ml-canvas/src/core/api/jobs.ts:4-38`) and `PreviewResponse` (`frontend/ml-canvas/src/core/api/client.ts:126-148`). `frontend/ml-canvas/package.json:6-21,23-95` contains neither a generator dependency nor a generation script. The checked PR workflow runs Python static checks only (`.github/workflows/pr_check.yml:60-67`), not schema-diff or generated-client freshness checks.

**Concrete gap:** a Pydantic change can change `/openapi.json` without TypeScript compilation or CI comparing it to the manual interfaces. Findings 2–4 show existing examples.

**Likelihood:** High. The separate services change frequently; stale browser assets or independent deployment turns ordinary nullable/additive changes into real UI bugs.

**Minimal fix:** commit a generated OpenAPI snapshot and generate TypeScript types/client functions with `openapi-typescript` or `orval`. In CI, emit the backend schema, regenerate, and fail on a diff. Keep handwritten view-model adapters only for intentional response transformations.

### 2. `JobInfo` has confirmed nullability and enum drift — High

`GET /api/pipeline/jobs/{job_id}` returns backend `JobInfo` (`backend/ml_pipeline/_internal/_routers/jobs.py:105-111`). That model defines `job_type: Literal["training", "tuning", "preview"]` (`backend/ml_pipeline/_execution/schemas.py:20-27`), `created_at: datetime | None = None`, and `tuned_thresholds_enabled: bool | None = None` (`backend/ml_pipeline/_execution/schemas.py:43-52`).

The frontend's manually named interface instead requires `created_at: string`, allows job types only `training | tuning | eda | ingestion`, and omits `tuned_thresholds_enabled` (`frontend/ml-canvas/src/core/api/jobs.ts:6-38`). Preview jobs are not hypothetical: the backend job manager accepts and creates them (`backend/ml_pipeline/_execution/jobs.py:30-75`). Normal production mapping emits `tuned_thresholds_enabled` (`backend/ml_pipeline/_execution/basic_training_manager.py:94-121`; `backend/ml_pipeline/_execution/advanced_tuning_manager.py:91-125`).

**Concrete failure mode:** legacy/incomplete rows may serialize `created_at: null`, although a frontend caller is allowed by TypeScript to treat it as a string. A direct preview-job request yields a runtime `preview` value excluded by the frontend union. Axios' manual generic assertion (`frontend/ml-canvas/src/core/api/jobs.ts:74-76`) validates neither.

**Minimal fix:** generated types; meanwhile make `created_at: string | null`, include `preview`, add `tuned_thresholds_enabled?: boolean | null`, and contract-test a representative serialized response.

### 3. Dataset/ingestion preview contracts have actual identifier and status-set mismatches — High

The sample endpoint is Pydantic-modeled as `DataSourceSampleResponse(data: list[dict[str, Any]])` (`backend/data_ingestion/schemas/ingestion.py:91-92`) and returned by the router (`backend/data_ingestion/router.py:70-81`). The frontend safely but broadly extracts it as `unknown[]` (`frontend/ml-canvas/src/core/api/datasets.ts:151-159`).

Adjacent dataset/ingestion contracts demonstrably drift:

* backend `DataSourceRead.id` is `int` (`backend/data_ingestion/schemas/ingestion.py:49-64`), while frontend `Dataset.id` is `string` (`frontend/ml-canvas/src/core/types/api.ts:1-13`). `getAll()` returns `data.sources` without conversion (`frontend/ml-canvas/src/core/api/datasets.ts:27-35`);
* backend `IngestionStatus.status` is unconstrained `str` (`backend/data_ingestion/schemas/ingestion.py:41-46`) and actually returns `unknown` without stored metadata (`backend/data_ingestion/service.py:542-567`) and writes `cancelled` on cancellation (`backend/data_ingestion/service.py:122-154`). Frontend permits only `pending | processing | completed | failed` (`frontend/ml-canvas/src/core/types/api.ts:34-40`).

**Likelihood:** High for cancellation and old records; medium for numeric identifiers, which string interpolation happens to mask.

**Minimal fix:** make IDs consistently numbers or normalize every source ID to string; give backend ingestion status a `Literal`/`StrEnum`, include/normalize `unknown` and `cancelled`, then generate it. Validate sample rows as objects before presenting them.

### 4. Pipeline preview's per-node result property differs — Medium

The preview response says `node_results: dict[str, Any]` (`backend/ml_pipeline/_internal/_schemas.py:51-78`) and returns each dataclass' raw `__dict__` (`backend/ml_pipeline/_internal/_routers/preview.py:532-535`) from `POST /preview` (`backend/ml_pipeline/_internal/_routers/preview.py:661-733`). Backend `NodeExecutionResult` names its artifact field `output_artifact_id` (`backend/ml_pipeline/_execution/schemas.py:93-111`); frontend `NodeExecutionResult` instead declares optional `output` (`frontend/ml-canvas/src/core/api/client.ts:79-96`).

The primary preview fields currently align: the backend sends `pipeline_id`, `status`, data/branch fields, recommendations and warnings (`backend/ml_pipeline/_internal/_routers/preview.py:721-732`), and frontend declares them (`frontend/ml-canvas/src/core/api/client.ts:126-148`). The node field remains confirmed drift, obscured by `Any`.

**Likelihood:** Medium. No current use of `output` was found, but a future caller receives `undefined` with no compiler/server warning.

**Minimal fix:** define a Pydantic `NodeExecutionResultResponse` with `output_artifact_id` (or deliberately map it to `output`), declare `dict[str, NodeExecutionResultResponse]`, and generate the frontend type.

### 5. Status values are independently maintained and force-cast — High

Pipeline `JobStatus` itself is currently aligned for its six values (`backend/ml_pipeline/_execution/schemas.py:11-18`; `frontend/ml-canvas/src/core/api/jobs.ts:4`). But the frontend adds `pending` and force-casts EDA raw strings: EDA's endpoint has no `response_model` (`backend/eda/router.py:60-91`) and produces uppercase `PENDING`, `STARTED`, `RUNNING`, `CANCELLED` (`backend/eda/router.py:155-188`; `backend/eda/tasks.py:223-224`); frontend uses `as JobStatus` (`frontend/ml-canvas/src/core/api/jobs.ts:99-125`).

The shared badge covers several values but falls back to `Unknown` for omitted states (`frontend/ml-canvas/src/components/shared/StatusBadge.tsx:36-55`). Another rendering branch recognizes only lowercase completed/failed/running (`frontend/ml-canvas/src/components/panels/jobs/JobCard.tsx:14-24`). Thus `STARTED` or a newly added backend status silently takes default behavior.

**Likelihood:** High for lifecycle evolution; medium current impact because badge normalization masks some values.

**Minimal fix:** add a typed EDA response model and generated enum; explicitly normalize EDA values into the unified view model rather than assert-casting. Add exhaustive mappings/tests and fallback telemetry.

### 6. WebSocket messages are publisher-validated but frontend frames are not runtime-validated — Medium

Backend `JobEvent` validates `event`, `job_id`, and optional status/progress/current-step, including a four-value event literal (`backend/realtime/events.py:23-34`). The manager wraps it as `{"channel":"jobs","data":...}` (`backend/realtime/manager.py:22-29`).

Frontend redeclares this interface (`frontend/ml-canvas/src/core/realtime/jobEventsSocket.ts:14-25`) but uses `JSON.parse(msg.data) as Envelope` and checks only channel/data truthiness (`frontend/ml-canvas/src/core/realtime/jobEventsSocket.ts:70-83`). It never verifies event type, `job_id`, or numeric progress. Consumers use `evt.job_id` to decide whether to refresh (`frontend/ml-canvas/src/core/hooks/useJobPolling.ts:206-208`), so a renamed/missing field silently stops immediate invalidation and falls back to 30-second polling.

**Likelihood:** Medium. Current publisher and consumer agree; a change fails silently rather than in tests.

**Minimal fix:** use the already-installed Zod (`frontend/ml-canvas/package.json:61-64`) to `safeParse` a strict envelope/event before dispatch, record rejected frames, and test backend serialization against versioned frontend fixtures. OpenAPI does not cover WebSockets.

### 7. No API compatibility versioning — High

Routes are mounted at functional legacy prefixes such as `/api/pipeline`, `/api/eda`, and `/api` (`backend/main.py:372-391`), with no version prefix. `APP_VERSION` is OpenAPI metadata (`backend/main.py:282-289`), not route or request negotiation. Inspection found no `/v1`, `Accept-Version`, or equivalent handling. Built frontend assets are served by the backend (`backend/main.py:339-347`), but that does not protect cached assets, independent hosting, integrations, or rolling deployments.

**Minimal fix:** establish `/api/v1` (or explicit media-type/header versioning), publish a deprecation policy, and gate breaking OpenAPI diffs behind a version bump/migration.

## Response-shape comparison

| Endpoint | Backend | Frontend | Assessment |
| --- | --- | --- | --- |
| `GET /api/pipeline/jobs/{id}` | `JobInfo` (`backend/ml_pipeline/_internal/_routers/jobs.py:105-111`) | `JobInfo` (`frontend/ml-canvas/src/core/api/jobs.ts:6-38`) | **Drift:** nullable `created_at`, omitted `preview`, omitted `tuned_thresholds_enabled`. |
| `POST /api/pipeline/preview` | `PreviewResponse` (`backend/ml_pipeline/_internal/_schemas.py:51-78`) | `PreviewResponse` (`frontend/ml-canvas/src/core/api/client.ts:126-148`) | Main fields match; **drift:** `output_artifact_id` vs `output` in node result. |
| `POST /api/pipeline/run` | Four fields, including `job_ids` (`backend/ml_pipeline/_internal/_schemas.py:44-49`; `backend/ml_pipeline/_internal/_routers/run_pipeline.py:368-433`) | `jobsApi` matches (`frontend/ml-canvas/src/core/api/jobs.ts:61-71`) | **Match today**, but manual/no validation. A legacy helper declares only `{job_id}` (`frontend/ml-canvas/src/core/api/client.ts:194-196`). |
| `GET /data/api/sources/{id}/sample` | `{data: list[dict[str, Any]]}` (`backend/data_ingestion/schemas/ingestion.py:91-92`; `backend/data_ingestion/router.py:70-81`) | Extracts `unknown[]` (`frontend/ml-canvas/src/core/api/datasets.ts:151-159`) | Envelope aligns, rows are unvalidated; source ID/status drift is confirmed above. |

## Prioritized top 5

1. Generate TypeScript API clients/types from OpenAPI and fail CI on a schema/generated diff.
2. Correct `JobInfo`: nullable `created_at`, `preview`, and `tuned_thresholds_enabled`; contract-test serialized responses.
3. Type/version ingestion statuses and normalize IDs; eliminate assertion-casts of API values.
4. Introduce `/api/v1` and a breaking-change/deprecation policy before independent deployments or integrations.
5. Validate WebSocket envelopes with Zod and versioned fixture messages.
