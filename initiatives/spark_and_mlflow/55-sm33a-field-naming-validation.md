# SM-33A: direct record and result field names

Date: 2026-09-25. Branch: `090`. Committed baseline: `0c0fd17f` (SM-33).
This report covers the subsequent uncommitted SM-33A change.

## Decision and implementation

The user explicitly accepted recreating experimental projects and models and
rejected a compatibility module for a pre-production field rename.
The temporary `_workflow_fields.py` file and its alias-only tests were removed.

- `record_key_columns` is the actual field in Bundle configuration, Core
  `FrameSpec`, batch contracts, training/scoring calls and saved training specs.
- `result_available_at_column` is the actual availability mapping in config,
  training specs, source reads and approval evidence.
- The initializer has one `record_key_columns_json` array input for single or
  composite identities, and one `result_available_at_column` input. There are
  no hidden legacy field aliases or precedence rules.
- Existing validation rejects retired config field names as unknown settings.
  No alias adapter or automatic old-artifact conversion remains.
- Existing callers, examples, docs and rendered inference diagrams use the new
  names. Historical reports and rehearsal outputs remain historical evidence.

## Verification

The new canonical-name and retired-setting tests first failed against the alias
implementation (3 failures), then passed after its removal and direct rename.

- Integration suite plus frame/state unit contracts: **503 passed, 62 skipped**.
  Includes actual CLI template generation and local MLflow lifecycle tests on
  pandas and Polars. The base environment skips optional runtime/live checks.
- Dedicated Spark checks: **106 passed**. Run Spark and Delta in separate
  processes: the initial combined process reused a plain Spark session without
  Delta; a separate Windows Delta attempt exposed missing Hadoop native support.
  The existing WSL wrapper supplies Java and cached Delta jars.
- WSL Delta suite: **52 passed**, one existing test setup mismatch exposed.
  The full-rebuild test omitted `period_column` despite its shared fixture
  creating an `event_time` output column. Added the existing column to its three
  calls; the target-schema guard and product logic are unchanged. The focused
  rerun passed (**1 passed**): all 53 selected Delta cases are now verified.
  The passing tests cover native/Python Spark output, pandas/Polars
  local scoring, incremental new rows/replay and real training snapshot reads.
- Full repository `ty check`: passed.
- Ruff checks and formatting on changed Python files: passed.
- `mkdocs build --strict`: passed (build exit 0, 6.03 seconds). The wrapper's
  final log-print hit Windows console encoding; the captured build log confirms
  successful completion at `.cache/sm33a-docs.log`.
- Both affected Mermaid diagrams parsed and rendered to standalone SVGs.
- `git diff --check`: passed.

## Boundaries and next task

No Databricks deployment, resource deletion or new remote model training happened
in this slice. Existing deployed test jobs still use their previous matching
wheel/config. Regenerate projects and retrain experimental models before using
the updated library; old training evidence is intentionally unsupported.

SM-33B handles explicit date formats/timezones next. SM-33C adds date-free
training and optional per-row availability; SM-33D connects Core CV;
SM-33E provides combined live acceptance. SM-34 remains blocked on those tasks.
The rename itself does not make date fields optional or add new split behavior.
