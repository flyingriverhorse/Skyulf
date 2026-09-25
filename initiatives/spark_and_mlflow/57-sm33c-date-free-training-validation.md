# SM-33C - Date-free training and independent result availability

Date: 2026-09-25. Baseline: signed commit `7cec33a7` on branch `090`.
Status: DONE for local verification; changes remain uncommitted.
Not deployed to Databricks; combined live acceptance remains SM-33E.

## Delivered contract

- `split_strategy` explicitly chooses random (default) or temporal. Random
  training requires only source identity, features and a known target. Existing
  Core `DataSplitter` receives stable composite-key order with `test_size`,
  `random_state` and optional classification stratification; no duplicate
  splitting implementation or silent downgrade to unstratified sampling.
- Temporal training retains its observation window and ordered final holdout.
  Missing temporal inputs fail; they never select random mode implicitly.
- `filter_unavailable_results` independently enables per-row availability.
  `result_available_at_column` and `result_cutoff` govern known outcomes even
  without an event column. Unknown/late outcomes are counted and excluded;
  eligible null targets fail. With filtering disabled, all targets must be known.
- Manual training pins a Delta version. Monthly random training pins the latest
  bounded snapshot with no event window/lookback. Monthly temporal training
  retains UTC calendar windows. Availability uses invocation time as its
  independent cutoff. Explicit selection/window and CV options remain SM-33D.
- Training saves source, split, parsing, availability and holdout membership
  evidence. Approval replays the saved contract rather than today's config.
  Memory budgets can be tightened without changing dataset identity.
- Both fit engines retain their existing local pipeline/model behavior. Spark
  validates/reads bounded data; it does not distribute local feature/model fit.
- Bundle initialization exposes choices with conditional date prompts, leaves
  inactive mappings null, and preserves supplied conflicts for preflight errors.
  Three executable initializer examples and English guides cover date-free,
  delayed-result random and delayed-result temporal training. The operator
  walkthrough contains a Mermaid training flow.

## Review

An independent review found that the initial membership hash was not bound to
comparison evidence. The correction includes the post-split membership hash in dataset identity
before logging comparison proof. The real MLflow regression rejects a
membership-only saved-spec edit, then restores the original evidence and
completes approval. Independent re-review cleared the finding.
Earlier pre-production projects/evidence must be recreated; no alias adapter
or old-evidence reader is added.

## Verification

- Real Databricks CLI v1.17.0 template/initializer tests: 54 passed, including both
  engines/tasks, availability choices, explicit temporal mapping and all three
  new examples. These tests initialize local files; they do not deploy jobs.
- Final native integration regression suite: 165 passed. Includes real MLflow
  training/approval on both engines with `StandardScaler`, training-only fit
  input assertions, saved-source replay after config changes, membership
  tampering rejection, stratification guards, optional dates and existing
  temporal/lifecycle regressions.
- Real local Delta source tests via WSL: 11 passed. Date-free composite-key reads,
  version pinning after append, row/byte overflow, availability without events,
  delayed results in a later snapshot, plus the existing timezone/DST cases.
  An initial new-test failure passed a pandas Index as Spark's schema argument;
  converting it to a list fixed the fixture. No production Delta change was needed.
- Final full repository `ty check`, scoped Ruff check/format, and `git diff --check`: passed.
- Strict MkDocs build: passed, including the updated walkthrough.
- Current wheel built with `uv build --wheel --no-build-isolation`. Generated
  serverless dev Bundle passed `databricks bundle validate --strict -t dev
  --profile skyulf`. The initial validation correctly warned about a missing
  wheel; adding the real built wheel resolved it. No deployment followed.
- Independent scope review and follow-up review passed after the membership fix.

Native suite uses `--basetemp=.cache/sm33c-final` to avoid Windows temporary-folder
ACL failures. Real Delta uses `.cache/sm15-linux-run.sh`; do not combine its
session with plain Spark fixtures. Core test files are under
`skyulf-core/tests/integrations/`. Combined checked cases: 165 native + 54
CLI/template + 11 Delta = 230 passing tests.

## Boundary

No remote jobs, tables, models, aliases or schemas were created, changed or
removed in this slice. Real Delta tests run in the existing local WSL environment.
Combined personal-workspace acceptance remains SM-33E. SM-34 remains blocked on
that acceptance. Company production readiness is not established by local tests.


## Follow-up: readable input budget and sampling scope

At the user's request, Bundle settings now expose `max_input_mb` with a default
of 64. One unit is 1,048,576 bytes (MiB). Existing SDK specs keep byte budgets;
a single converter in the existing `_contracts.py` is reused by offline
validation, training, scoring and approval. No new adapter file or old-field
alias was added. Approval uses the stricter of its current configured budget
and the saved training byte budget; changing caps does not change dataset proof.
The real pandas/Polars approval tests tighten 2 MiB to 1 MiB and verify this.

Verified after the rename: 211 integration/runtime/template tests plus 41 real
CLI generation tests (252 passed). Full ty, scoped Ruff and formatting, strict
MkDocs, current wheel build and generated dev Bundle strict validation passed.
No source reader algorithm changed; the earlier 11 WSL Delta cases were not
rerun for this settings conversion. No live cloud mutation or commit occurred.

The user's example of selecting 10,000 random rows from 100,000 is tracked under
SM-33D as explicit deterministic training sampling. It is not implemented by this
rename. `max_rows` still fails on overflow. Sampling must operate on the pinned
eligible source before driver transfer and persist replayable selection evidence;
scoring must preserve prediction completeness rather than sample implicitly.
