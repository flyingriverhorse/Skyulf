Status: IMPLEMENTED (awaiting independent review)
Base commit: 1e3d89a8

Repair summary:
- Added focused `StatefulTransformer.fit_transform()` regressions for both calculator-fit
  failure and applier-apply failure so the tracemalloc cleanup path is exercised before and
  after fitted params exist.
- `fit_transform()` now tracks explicit tracemalloc ownership: it starts and resets peak only
  when tracing was previously off, preserves caller-owned tracing/peak state, always records
  `fit_time`, and stops tracing only when this transformer started it.
- Success semantics remain aligned with the Task 1-safe train-transform hook: `rows_in` is
  captured before tracing work, `_fit_transform_inner()` stays untouched, successful calls set
  `rows_out` before return, and original exceptions still propagate unchanged.

Explicit ownership behavior:
- If tracing is already active, `fit_transform()` does not call `reset_peak()` and snapshots
  only the caller's current traced bytes as the subtraction baseline.
- If tracing is inactive, `fit_transform()` starts tracemalloc, resets the local peak, and
  treats the run as transformer-owned tracing.
- In the `finally` block, `fit_time` is always recorded and `peak_memory_bytes` is computed as
  `max(0, peak - baseline_current)` when tracing is still active.
- If tracing disappears mid-transform (for example, a caller stops it), no replacement trace is
  created, `peak_memory_bytes` stays at the defensive zero default, and the original exception
  or return path is preserved.
- Transformer-owned tracing is stopped in `finally`; caller-owned tracing is left running.

Files changed:
- `skyulf-core/skyulf/preprocessing/base.py`
- `skyulf-core/tests/test_preprocessing_base.py`
- `.superpowers/sdd/core-safety-task-3-report.md`
- `.superpowers/sdd/progress.md`

Red/green verification:
- RED: `source .venv/bin/activate && pytest skyulf-core/tests/test_preprocessing_base.py::test_fit_transform_stops_tracing_it_started_after_failure skyulf-core/tests/test_preprocessing_base.py::test_fit_transform_preserves_caller_owned_tracing_after_failure -q` -> `2 failed, 2 passed` on the pre-fix code because owned tracing stayed enabled after failure.
- GREEN: reran the same command after the code change -> `4 passed`.

Validation:
- `source .venv/bin/activate && pytest skyulf-core/tests/test_preprocessing_base.py skyulf-core/tests/test_preprocessing_pipeline.py -q` -> `78 passed`
- `source .venv/bin/activate && ruff check .` -> `All checks passed!`
- `source .venv/bin/activate && ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py` -> `569 files already formatted`
- `source .venv/bin/activate && ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py` -> `All checks passed!`

Notes:
- The new tests use an isolating fixture that stops tracemalloc before/after each ownership
  assertion so process-global tracing state cannot leak across test order.
- Codacy CLI analysis was attempted for each edited Python file, but the repository-local
  wrapper is malformed (`.codacy/cli.sh` starts with HTML and aborts before analysis).
