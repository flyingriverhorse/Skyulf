# Background Job Polling & Status Updates

## Problem Summary

The background full-dataset execution jobs were stuck at "Queued" status because:

1. **No frontend polling** - The UI wasn't checking for status updates after the initial job creation
2. **Missing status update mechanism** - Users had no way to see job progress

## Solution Implemented

### 1. Frontend Automatic Polling (DataSnapshotSection.tsx)

Added a `useEffect` hook that:
- Monitors active background jobs (`job_status === 'queued' || 'running'`)
- Automatically polls the backend every 5 seconds (or custom interval from `poll_after_seconds`)
- Updates the UI badge and context when status changes
- Stops polling when job completes/fails

```typescript
useEffect(() => {
  if (!fullExecutionSummary || !fullExecutionSummary.isActive) {
    return; // No active job
  }

  const pollInterval = signal.poll_after_seconds * 1000 || 5000;
  
  const timer = setTimeout(() => {
    fetchFullExecutionStatus(datasetSourceId, jobId)
      .then(updatedSignal => {
        // Update UI with new status
        setFullExecutionSummary(resolveFullExecutionSummary(updatedSignal));
      });
  }, pollInterval);

  return () => clearTimeout(timer);
}, [fullExecutionSummary, datasetSourceId]);
```

### 2. Expandable Job Details

Added a toggle button to show/hide detailed job metadata:
- **Collapsed by default** - Shows only essential status and reason
- **Expandable** - Click to reveal full job details (job ID, dataset rows, timestamps, etc.)
- **Clean UX** - Reduces visual clutter while keeping info accessible

### 3. Backend: Why `db_engine` Module Import?

```python
import core.database.engine as db_engine
```

**Purpose:**
- Background tasks run **after** the initial request completes
- They need database access to load the full dataset
- `async_session_factory` is initialized on app startup via `init_db()`
- Module-level import ensures tasks always reference the **current** session factory

**Flow:**
1. App starts → `init_db()` sets `async_session_factory`
2. Request creates background job → references `db_engine.async_session_factory`
3. Background task runs → uses initialized session factory to query database

## Backend Job Status Flow

1. **Queued** → Job created, waiting to start
2. **Running** → `_run_full_execution_job` executing
3. **Succeeded** → Job completed successfully
4. **Failed** → Error during execution (memory, exception, etc.)

## API Endpoints

### Create/Get Job Status
```
POST /ml-workflow/api/pipelines/preview
→ Returns PipelinePreviewResponse with full_execution signal

GET /ml-workflow/api/pipelines/{dataset_source_id}/full-execution/{job_id}
→ Returns FullExecutionSignal with current status
```

## UI Components

### Badge States
- **Queued** (blue) - Job waiting to start
- **Running** (blue, animated) - Job in progress
- **Succeeded** (green) - Completed successfully
- **Failed** (red) - Execution error
- **Deferred** (yellow) - Scheduled for background

### Context Section
```
┌─────────────────────────────────────┐
│ Full dataset run: Queued            │
├─────────────────────────────────────┤
│ • Processed rows...                 │
│ • Reason message...                 │
│                                     │
│ ▶ Show job details [TOGGLE]        │
│   ┌─────────────────────────────┐  │
│   │ Background job: Queued      │  │
│   │ Job ID: d38f55e...          │  │
│   │ Dataset rows: 466,285       │  │
│   │ Next check: 5s              │  │
│   │ Last updated: 4:16:42 PM    │  │
│   └─────────────────────────────┘  │
└─────────────────────────────────────┘
```

## Testing

1. **Trigger large dataset preview** (>200,000 rows)
2. **Observe badge** changes from "Queued" → "Running" → "Completed"
3. **Click toggle** to expand/collapse job details
4. **Check console** for polling activity (every 5s while active)

## Notes

- Jobs are **in-memory only** - not persisted to database
- Polling **auto-stops** when job reaches terminal state
- Backend logs job execution under logger `core.feature_engineering.routes`
- Session factory must be initialized before jobs can execute
