Status: COMPLETE (final independent review approved)
Base commit: 98f6e3b9

Follow-up summary (test-only, closing the two independent-review coverage gaps):
- Added a controlled calculator/applier regression that calls `tracemalloc.stop()` during
  `StatefulTransformer.fit_transform()` on both success and failure paths. The test proves
  the transformer keeps the original result/error semantics, avoids any secondary tracing
  exception, leaves `peak_memory_bytes` at the documented zero fallback, and exits with
  tracing cleanly off.
- Strengthened the caller-owned tracing failure coverage by creating a stable historical
  caller peak before entry and directly asserting the failing transform reports
  `peak_memory_bytes == 0` rather than claiming that prior peak as its own. Caller tracing
  remains active and the historical global peak is unchanged after the original failure.

Explicit peak policy:
- If tracing is already active, `fit_transform()` does not call `reset_peak()`. It snapshots
  the caller's entry global peak and reports only subsequent global-peak growth
  (`max(0, exit_peak - entry_peak)`), so an older caller-owned high-water mark is never
  attributed to this transformer.
- If tracing is inactive, `fit_transform()` starts tracemalloc, resets the local peak, and
  reports peak growth from that fresh local baseline, so transformer-owned runs still get an
  isolated local measurement.
- In the `finally` block, `fit_time` is always recorded and `peak_memory_bytes` is computed as
  `max(0, peak - peak_baseline)` when tracing is still active.
- If tracing disappears mid-transform (for example, a caller stops it), no replacement trace is
  created, `peak_memory_bytes` stays at the defensive zero default, and the original exception
  or return path is preserved.
- Transformer-owned tracing is stopped in `finally`; caller-owned tracing is left running.
- Final independent review approved both specification compliance and code quality through
  `6e036152`. Task 3 is complete.

Files changed:
- `skyulf-core/tests/test_preprocessing_base.py`
- `.superpowers/sdd/core-safety-task-3-report.md`
- `.superpowers/sdd/progress.md`

Coverage-gap verification:
- Added tests:
  - `test_fit_transform_handles_tracing_becoming_inactive_mid_run`
  - strengthened `test_fit_transform_preserves_caller_owned_tracing_after_failure`
- Targeted run: `source .venv/bin/activate && pytest skyulf-core/tests/test_preprocessing_base.py::test_fit_transform_handles_tracing_becoming_inactive_mid_run skyulf-core/tests/test_preprocessing_base.py::test_fit_transform_preserves_caller_owned_tracing_after_failure skyulf-core/tests/test_preprocessing_base.py::test_fit_transform_does_not_inherit_caller_peak_history_on_success -q` -> `5 passed`

Validation:
- `source .venv/bin/activate && pytest skyulf-core/tests/test_preprocessing_base.py skyulf-core/tests/test_preprocessing_pipeline.py -q` -> `83 passed`
- `source .venv/bin/activate && ruff check .` -> `All checks passed!`
- `source .venv/bin/activate && ruff format --check backend skyulf-core tests run_fastapi.py run_skyulf.py celery_worker.py` -> `569 files already formatted`
- `source .venv/bin/activate && ty check backend skyulf-core/skyulf skyulf-core/tests run_fastapi.py run_skyulf.py celery_worker.py` -> `All checks passed!`

Notes:
- The new tests use an isolating fixture that stops tracemalloc before/after each ownership
  assertion so process-global tracing state cannot leak across test order.
- The caller-owned peak helper uses a single large temporary allocation plus `gc.collect()`
  so the "historical peak with low current usage" state is deterministic and stable across
  both failure variants.
- Codacy CLI analysis was attempted for each edited Python file, but the repository-local
  wrapper is malformed (`.codacy/cli.sh` starts with HTML and aborts before analysis).
