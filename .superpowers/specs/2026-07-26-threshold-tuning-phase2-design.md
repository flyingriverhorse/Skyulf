# Threshold Tuning — Phase 2 (Product Integration) Design

**Status:** Approved for planning
**Depends on:** `docs/superpowers/specs/2026-07-26-threshold-tuning-design.md` (Phase 1, library-only, shipped)

## Context

Phase 1 added `optimize_thresholds()` / `apply_thresholds()` to `skyulf-core`
plus an opt-in `SkyulfPipeline.predict(use_tuned_thresholds=True)` wrapper.
It is library-only: nothing in the web product (backend API, database,
frontend) knows tuned thresholds exist, and — critically — the real
prediction path used by the deployed model
(`DeploymentService.predict()` in `backend/ml_pipeline/deployment/service.py`)
does not call `SkyulfPipeline.predict()` at all. It loads the bundled
artifact (`{"feature_engineer": ..., "model": ...}`) and calls the raw
sklearn estimator's `.predict()` directly. So Phase 1's pipeline wrapper has
zero effect on real deployed predictions today.

Phase 2 makes tuned thresholds a real, usable product feature: computed
from a job's existing evaluation data, persisted per model version, and
actually applied by the live `/predict` endpoint and reflected in the
Evaluation view's charts.

## Goals

- Let a user tune per-class decision thresholds for a completed
  classification job, from the Evaluation view, against a metric of their
  choice.
- Persist tuned thresholds at the job/model-version level so they survive
  redeployments of that same version, with an independent enable/disable
  toggle.
- Make the live `/predict` endpoint actually use saved, enabled thresholds
  automatically — with an optional one-off `override_thresholds` for ad-hoc
  testing that is never persisted.
- Prefer a genuine validation split when the user has configured one
  (`validation_size` on the Train/Test Splitter node — already supported
  end-to-end by the library and frontend); fall back to the test split
  otherwise, with no blocking and no warning banner (a hint on the splitter
  node nudges users toward configuring validation instead).
- Give both binary and multiclass jobs a way to see the *real* effect of
  tuned thresholds on the confusion matrix, not just a number.

## Non-goals

- No new "3rd split" plumbing beyond what the library/frontend already
  support (`validation_size` already exists) — Phase 2 does not touch the
  training pipeline's split logic.
- No automatic tuning during/after training — tuning is always an explicit,
  on-demand user action (per the earlier `on_demand` decision).
- No cross-job/cross-deployment threshold sharing — thresholds are scoped
  to one `TrainingJob` row.
- No changes to regression or clustering jobs.

## Architecture

### Data model

New columns on `TrainingJob` (`backend/database/models.py`), added via the
existing lightweight migration list in `backend/database/engine.py`
(`_MIGRATIONS`, `ALTER TABLE ... ADD COLUMN`, matching the `promoted_at`
precedent — no Alembic in this repo):

```python
tuned_thresholds: Mapped[Any | None] = mapped_column(JSON, nullable=True)
tuned_thresholds_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
```

`tuned_thresholds` shape (stored verbatim, echoed back by the preview
endpoint so the frontend can round-trip it into the save call unchanged):

```json
{
  "thresholds": {"0": 0.42, "1": 0.58},
  "classes": [0, 1],
  "metric": "f1",
  "split_used": "validation",
  "computed_at": "2026-07-26T10:00:00Z"
}
```

### Backend API

New endpoints in `backend/ml_pipeline/_internal/_routers/jobs.py` (same
module/style as the existing `promote_job` / `unpromote_job` pair):

- `POST /jobs/{job_id}/threshold-tuning/preview`
  Body: `{"metric": "f1" | "balanced_accuracy" | "mcc" | "precision" | "recall" | "accuracy"}`.
  Loads the job's evaluation data via the existing
  `EvaluationService.get_job_evaluation()`. Uses `splits["validation"]` if
  present and non-empty, else `splits["test"]`. Maps the metric name to a
  concrete `sklearn.metrics` scorer callable, calls
  `skyulf.modeling.optimize_thresholds(y_true, y_proba, metric=scorer)`.
  Returns `{"thresholds", "classes", "metric", "split_used", "achieved_score"}`.
  **Nothing persisted.** 400 if the job is not a classifier, has no
  `y_proba` in its evaluation data, or is not yet complete.

- `PUT /jobs/{job_id}/threshold-tuning`
  Body: the preview response's `{"thresholds", "classes", "metric",
  "split_used"}` (client echoes it back unchanged — the endpoint does not
  recompute). Writes `tuned_thresholds` (adding `computed_at` server-side)
  and sets `tuned_thresholds_enabled = True`.

- `PATCH /jobs/{job_id}/threshold-tuning/toggle`
  Body: `{"enabled": bool}`. Flips `tuned_thresholds_enabled` only; leaves
  `tuned_thresholds` untouched. 404/400 if no thresholds have ever been
  saved.

- `DELETE /jobs/{job_id}/threshold-tuning`
  Clears `tuned_thresholds` to `null` and sets `tuned_thresholds_enabled =
  False`.

Metric name → scorer mapping (backend-internal, not library-exposed):

| UI value            | scorer                                                            |
|----------------------|--------------------------------------------------------------------|
| `f1`                | `sklearn.metrics.f1_score(..., average="weighted")`                |
| `balanced_accuracy`  | `sklearn.metrics.balanced_accuracy_score`                          |
| `mcc`               | `sklearn.metrics.matthews_corrcoef`                                 |
| `precision`         | `sklearn.metrics.precision_score(..., average="weighted", zero_division=0)` |
| `recall`            | `sklearn.metrics.recall_score(..., average="weighted", zero_division=0)`    |
| `accuracy`          | `sklearn.metrics.accuracy_score`                                    |

### Deployment / prediction integration

`backend/ml_pipeline/deployment/schemas.py`:

```python
class PredictionRequest(BaseModel):
    data: list[dict[str, Any]]
    override_thresholds: dict[str, float] | None = None

class PredictionResponse(BaseModel):
    predictions: list[Any]
    model_version: str
    thresholds_applied: dict[str, float] | None = None
```

`DeploymentService.predict()` / `_predict_and_decode()`:

- Resolve which thresholds apply, in priority order: (1)
  `override_thresholds` from the request, (2) the active deployment's job
  `tuned_thresholds` if `tuned_thresholds_enabled` is true, (3) none (plain
  `.predict()`).
- When thresholds apply: call `estimator.predict_proba(X_transformed)`,
  then `skyulf.modeling.apply_thresholds(proba, thresholds, classes=...)`
  instead of `estimator.predict(X_transformed)`, then decode labels exactly
  as today.
- Fail-safe: if the stored `classes` don't match the live model's
  `classes_` (e.g. a swapped artifact), log a warning and fall back to
  plain `.predict()` rather than error out a live prediction request.
- `override_thresholds` keys must match the model's actual class labels;
  mismatched keys → 422 listing the expected classes.
- Response always reports `thresholds_applied` (whichever of the three
  cases fired, or `null`).

### Frontend

**Evaluation view** (`ExperimentsPage/components/EvaluationView.tsx`,
classification jobs only) — new "Threshold Tuning" panel below the existing
threshold slider / confusion matrix section:

- Metric dropdown (the 6 metrics above).
- "Preview" → calls the preview endpoint; shows the per-class thresholds
  table + achieved score + which split was used.
- "Save & Enable" → calls `PUT`; panel then shows saved state (thresholds,
  metric, computed_at) with an Enabled/Disabled toggle (`PATCH`) and a
  "Clear" action (`DELETE`).
- Hidden/disabled with a short explanation for non-classifier jobs or jobs
  without `predict_proba` support.

**Confusion matrix redraw** (`PerClassConfusionMatrix.tsx`,
`ClassificationChartsForSplit.tsx`): new optional prop
`thresholdsVector: Record<string, number> | null`. When set, per-row
predictions are recomputed via a new pure TypeScript function implementing
the same scaled-argmax rule as `apply_thresholds()`
(`classes[argmax(proba / thresholds)]`), applied against the already-loaded
`y_proba` matrix — no extra API call needed to redraw. A view toggle
("Default" vs "Tuned Thresholds") switches which rule drives the chart.
Binary jobs additionally snap the existing single-scalar slider to the
tuned positive-class threshold (reusing the existing "best-threshold badge"
snap mechanism `setThreshold(...)`), so the existing single-value slider UI
stays in sync too.

*Parity risk:* this duplicates `apply_thresholds()`'s algorithm in
TypeScript. Mitigated by unit tests asserting the TS function's output
matches fixed fixtures generated from the real Python function, plus a code
comment in each implementation pointing at the other.

**Train/Test Splitter node** (`TrainTestSplitNode.tsx`): a short hint near
`validation_size` noting that a non-zero value enables more robust
threshold tuning later (tuning otherwise falls back to the test split).

**Job/deployment list**: a small "Tuned thresholds active" badge wherever
job/deployment status already renders.

**Inference page** (`InferencePage.tsx`): collapsible "Advanced: override
thresholds" section — per-class number inputs (defaulting to the job's
saved thresholds if enabled, else placeholders) — populates
`override_thresholds` in the `deploymentApi.predict()` call. Results area
shows `thresholds_applied` from the response.

## Error Handling

- Non-classifier job → 400, panel hidden.
- No `predict_proba` in evaluation data → 400 with a clear message.
- Job not completed / not found → 400/404 (mirrors existing job-status
  checks).
- Invalid metric name → 422 (pydantic literal/enum validation).
- `optimize_thresholds()` raising (e.g. degenerate data) → surfaced as 400
  with the underlying message, no silent fallback.
- Deployment-time `classes` mismatch → warning-logged fallback to plain
  `.predict()`, never a hard failure of a live request.
- `override_thresholds` key mismatch → 422 listing expected classes.

## Testing Plan

- **skyulf-core**: none — Phase 1 coverage is sufficient; Phase 2 adds no
  new library behavior.
- **Backend** (`pytest`): endpoint tests for preview/save/toggle/clear;
  metric→scorer mapping; validation→test split fallback; `DeploymentService`
  tuned-threshold branch (mocked `predict_proba` estimator); `classes`
  mismatch fail-safe; `override_thresholds` validation.
- **Frontend** (`vitest`): TS scaled-argmax function vs. Python-generated
  fixtures; Threshold Tuning panel flows (mocked API); Inference page
  override round-trip.
- **Manual/e2e smoke**: one binary job with a validation split, one
  multiclass job without one → tune → save → confirm Inference page
  predictions actually change and the confusion matrix redraws correctly
  in "Tuned Thresholds" view.
