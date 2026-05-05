# Job Idempotency (Deduplication)

When you click **Run All Experiments** quickly more than once, the backend
automatically detects the duplicate requests and returns the existing
in-progress job instead of spawning new Celery tasks.

## What Gets Deduplicated

A submission is considered a **duplicate** if all three of these match
a job already in `queued` or `running` state created within the last
**30 seconds**:

| Key | Description |
|---|---|
| `dataset_source_id` | The dataset the pipeline is running on |
| `node_id` | The canvas node ID of the terminal (Training / Tuning) node |
| `branch_index` | Which parallel branch (0 = first, 1 = second, …) |

## Parallel Branches Are Not Deduplicated Against Each Other

If your canvas has two paths feeding the same training node (parallel
experiment mode), Branch 0 and Branch 1 get different `branch_index`
values from the partitioner and are each allowed to create their own
job.

```
Branch 0: Dataset → Encode → [Train]          branch_index=0
Branch 1: Dataset → Encode → Scale → [Train]  branch_index=1
```

Clicking twice fires two requests per branch.  Request 2 of Branch 0 is
deduplicated against Request 1 of Branch 0.  Branch 1 is unaffected.

## The 30-Second Window

The dedup window is **30 seconds**.  After 30 seconds have passed since
the last job was created, a new click always starts a fresh run.  This
prevents accidental blocking of legitimate re-runs.

## How It Works Internally

The backend uses a per-key `asyncio.Lock` to make the
"check if job exists → create if not" step **atomic**:

```
Click 1  →  acquires lock  →  no job found  →  creates job J1  →  releases lock
Click 2  →  acquires lock  →  J1 found (queued)  →  returns J1  →  releases lock
```

Without the lock, both clicks could see an empty table simultaneously
before either one committed its new row (a classic read-modify-write race).

## Known Limitation

The lock lives in-process.  If you run multiple uvicorn workers
(`--workers N`) each worker has its own lock dictionary and
cross-process races are not prevented.  The current default
single-worker configuration is fully protected.
