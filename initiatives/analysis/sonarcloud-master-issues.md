# SonarCloud — `master` open issues triage (118)

Pulled live from the SonarCloud web API on 2026-09-05 for
`flyingriverhorse_Skyulf`, branch `master`, `resolved=false`.

> The `sonarqube` MCP server is registered in `~/.qoder/settings.json` but no
> MCP tools were loaded in the session (MCP tools only load at session start),
> so the issues were fetched directly from
> `GET https://sonarcloud.io/api/issues/search?componentKeys=flyingriverhorse_Skyulf&branch=master&resolved=false&ps=500`
> with the configured token, read from `settings.json` in-process so it never
> reached a command line. Same data, no MCP dependency. The raw payload and its
> TOON projection were written under `tmp/` — gitignored scratch, since deleted;
> re-run the call above to reproduce (TOON-encoded the flat 11-field record list
> at −44.4 % tokens, and the payload never entered context).

Every one of the 118 was read in the source tree and verified — including
running code where a verdict hinged on runtime behaviour. Verdicts below are
evidence-based, not inferred from the rule description.

## Headline

| Verdict | Count |
| --- | --- |
| **Genuine — fix** | **89** |
| **False positive / won't fix** | **29** |
| Total | 118 |

By type: `CODE_SMELL` 75, `BUG` 32, `VULNERABILITY` 11.
By severity: `MINOR` 69, `MAJOR` 30, `CRITICAL` 18, `BLOCKER` 1.
By software quality: `RELIABILITY` 106, `MAINTAINABILITY` 68, `SECURITY` 16.

Note that **62 of the 118 (53 %) are a single cosmetic rule** (`typescript:S7773`,
`MINOR`). Strip that out and the real backlog is 56 issues, of which the
highest-value cluster is **10 test-integrity bugs** (`python:S5779`) that empty
the failure messages of an integration suite guarding the node contract.

Branch context: working branch `0812` is 2 commits ahead of `origin/master`,
0 behind. Of the 58 files carrying issues, only
`skyulf-core/skyulf/profiling/_analyzer/multivariate.py` differs from `master`,
so all other line numbers match the working tree. That one file was located by
content instead of line number — and it turns out **2 of its 4 findings are
already fixed on `0812`**.

---

## Genuine — fix (89)

### 1. `python:S8414` — CORSMiddleware not outermost (1, **BLOCKER**)

`backend/main.py:360`

Starlette's `add_middleware` *wraps*, so the middleware added **last** is
**outermost**. `_add_middleware` currently adds TrustedHost → CORS → Logging →
ErrorHandler, giving an execution order of:

```
ErrorHandler → Logging → CORS → TrustedHost → app
```

CORS therefore sits **inside** `ErrorHandlerMiddleware`. Any exception converted
to a JSON error response by that outermost handler is produced *outside* the
CORS layer, so the response carries **no `Access-Control-Allow-Origin` header**.
The browser then blocks the frontend from reading the status or body of every
cross-origin error — 4xx/5xx responses surface as opaque network failures
instead of actionable messages. Same for a preflight rejected early by an outer
middleware.

Fix: add `CORSMiddleware` last so it is outermost.

### 2. `python:S7493` — sync file I/O in `async` routes (2, MAJOR)

`backend/ml_pipeline/_internal/_routers/pipelines_io.py:80, 154`

Both are inside `async def` handlers and call `file_path.open()` +
`json.dump`/`json.load` synchronously, blocking the whole event loop for the
duration of the disk write/read. Under concurrency this stalls every other
in-flight request on that worker.

Fix: run the blocking section in a thread (`anyio.to_thread.run_sync` /
`asyncio.to_thread`), keeping the JSON path helpers as they are.

### 3. `pythonsecurity:S5145` — log injection (3 of 4, MINOR/LOW)

CWE-117. User-controlled values reach a log record without CRLF scrubbing, so a
crafted value containing `%0A` can forge log lines and defeat audit trails.

| Site | Tainted value | Why genuine |
| --- | --- | --- |
| `backend/ml_pipeline/deployment/service.py:117` | `job_id` | `POST /deploy/{job_id}` declares `job_id: str` (`deployment/api.py:26-29`) — an unvalidated path param interpolated into an **f-string** log |
| `backend/monitoring/router.py:535` | `job_id` | `job_id: str` throughout the drift routes |
| `backend/data_ingestion/service.py:109` | `file_path` | from `source.config["file_path"]`, originally a user-supplied upload filename; filenames may contain newlines |

Fix: one shared scrubber (`sanitize_for_log`) applied to the tainted value at
each log call.

**SonarCloud under-reported this rule — fixing only the three flagged values
would have left two of the three statements still forgeable**, because each
interpolates a *second* tainted value the rule did not report:

| Site | Unflagged tainted value | Traced origin |
| --- | --- | --- |
| `deployment/service.py` | `artifact_uri` | on this branch it is always `str(db_job.node_id)`; `node_id` reaches the DB from the client-submitted pipeline graph via `run_pipeline.py:319,395` into the `NodeConfig` **dataclass**, which declares `node_id: str` with no validation at all |
| `data_ingestion/service.py` | `source_id` | `router.py:85` declares `delete_source(source_id: str)` — a bare unvalidated path param, not an `int` |

Both are now sanitized too. `monitoring/router.py` interpolates only `job_id`,
so it needed nothing further. (`data_ingestion/service.py` also logs the caught
exception `e`; an `OSError` message embeds the filename, which is a
second-order taint path. Sanitizing every exception object in every log call is
beyond this finding and was **not** done — recorded here as a known residual
rather than left implicit.)

`sanitize_for_log` escapes the C0 control block plus DEL to a visible `\xNN`
form rather than deleting it, so a forgery attempt stays readable in the record.
Verified: `x\r\ny\x00z\x7f` → `x\x0d\x0ay\x00z\x7f` (19 chars) — no raw control
char survives, output is single-line, the surrounding payload is preserved,
non-`str` inputs are handled via `str(value)` (`42` → `'42'`, `None` →
`'None'`), and clean text passes through unchanged.

**Not** fixing: `backend/eda/router.py:145` — see false positives.

### 4. `pythonbugs:S2583` — unreachable `or {}` (1 of 2, MAJOR)

`backend/ml_pipeline/_execution/jobs.py:250`

```python
summary = (job.metrics or {}).get("summary") if job.metrics else None
```

The ternary guard already proves `job.metrics` is truthy, so `or {}` can never
be taken — Sonar's "Unreachable code" range covers exactly `job.metrics or {}`.
Dead defensive code that obscures the real guard.

Fix: `summary = job.metrics.get("summary") if job.metrics else None`.

**Not** fixing: `skyulf-core/.../split.py:212` — see false positives.

### 5. `python:S6709` — unseeded `random_state` (2 of 4, MAJOR)

`skyulf-core/skyulf/preprocessing/bucketing.py:309, 332`

`KBinsDiscretizer(..., strategy="kmeans")` is constructed **without**
`random_state` at both sites (`_fit_kmeans` and `_fit_kbins`). K-means
initialisation is stochastic, so bin edges differ run to run on identical input.

This directly contradicts a core promise of the library: `skyulf-core` owns a
single `DEFAULT_RANDOM_STATE = 42` (`skyulf/types.py`) and a *semantic*
reproducibility seal (`SkyulfPipeline.fingerprint()`, `pipeline/seal.py`) that
hashes fitted weights. A k-means-bucketing node silently breaks that guarantee.

Fix: pass `random_state=DEFAULT_RANDOM_STATE` at both construction sites.

The other 2 `S6709` hits (`multivariate.py:197, 281`) are **already fixed** on
branch `0812` — now lines 284/380, both passing `DEFAULT_RANDOM_STATE`.

### 6. `python:S6729` — `np.where` with condition only (1, CRITICAL)

`skyulf-core/skyulf/profiling/_analyzer/multivariate.py:386` (line 375 on master)

```python
outlier_indices = np.where(preds == -1)[0]
```

`np.where(cond)` is a thin wrapper over `np.nonzero(cond)`; calling it with only
the condition builds the full index tuple and discards all but the first axis.
`np.nonzero` is the direct, intention-revealing call.

Fix: `np.nonzero(preds == -1)[0]`.

### 7. `python:S1764` — identical operands (2, MAJOR)

Both are the canonical `x != x` NaN test, so the *finding* is a style hit rather
than a logic error — but reading them showed one real bug hiding behind it.

**`backend/ml_pipeline/_execution/summary.py:426` — genuine bug.**

```python
def _first_finite(metrics, candidates) -> float | None:
    """Return the first finite numeric value found under any of ``candidates``."""
    ...
            if f == f:  # filter NaN
                return f
```

The function is *named* `_first_finite` and its docstring promises a **finite**
value, but `f == f` filters only NaN — `float("inf")` passes straight through.
Its 8 call sites (`accuracy`, `f1`, `auc`, `r2`, `rmse`, `mae`, `mse`, …) feed
human-readable node-card summaries, so an `inf` metric renders as `inf` in the
UI and can serialise to invalid JSON. The sibling `_train_only:455` (documented
"Same as `_first_finite`") has the identical defect and was not flagged.

Fix: `math.isfinite(f)` in both — matches the documented contract and clears the
finding.

**`backend/data_ingestion/serialization.py:453` — cosmetic.** `obj` is already
narrowed to `float` at line 451, so `math.isnan(obj)` is exactly equivalent to
`obj != obj` and reads better.

> Observation, deliberately **not** changed here: lines 453-461 of
> `serialization.py` are triply redundant for a Python `float` — `obj != obj`,
> `obj in (inf, -inf)`, then a `try/except`-wrapped `np.isinf(obj) or
> np.isnan(obj)` all cover the same ground, and the numpy branch can never add
> coverage because `isinstance(obj, float)` at 451 already rejects `np.float32`
> while accepting `np.float64` (a `float` subclass). Collapsing all nine lines
> to `if not math.isfinite(obj): return None` would be behaviour-identical. Left
> alone to keep this change surgical; worth a separate cleanup.

### 8. `python:S5779` — assertions swallowed by `except AssertionError` (10, CRITICAL)

`tests/integration/test_frontend_nodes.py:136, 143, 144, 210, 212, 213, 261, 263, 264`
`tests/unit/verify_polars_preprocessing.py:54`

Assertions placed inside a `try` whose handler catches `AssertionError` without
re-raising. `AssertionError` **is** a subclass of `Exception`, so these broad
`except Exception` handlers intercept assertion failures and re-report them
through `pytest.fail(...)`.

The tests do still fail — verified: `pytest.fail` raises `Failed`, which derives
from `OutcomeException(BaseException)`, so it propagates out of the handler. The
damage is to **diagnosis**, not to the pass/fail verdict:

- these are bare `assert` statements with no message, and
  `str(AssertionError())` is the empty string, so a real failure reports as
  `Node random_forest_classifier failed: ` with nothing after the colon;
- pytest's assertion rewriting is bypassed, so the usual expected-vs-actual diff
  never appears;
- "the engine raised" and "an assertion failed" collapse into one indistinguishable
  message.

In an integration suite guarding the frontend node contract, that turns a
five-second diagnosis into a debug session. `verify_polars_preprocessing.py:54`
is the milder unittest variant (`self.assertIn` → `self.fail(f"...: {e}")`),
which keeps the message but still conflates the two cases.

Fix: keep only the call that can legitimately raise inside the `try`, and move
the assertions after it.

### 9. `css:S4657` — shorthand clobbers longhand (1, CRITICAL) — visible UI bug

`frontend/ml-canvas/src/styles/components.css:162`

In the `select` rule, line 157 sets `padding-right: 2.5rem` to reserve room for
the dropdown-arrow background image (`background-position: right .5rem center`),
then line 162's shorthand `padding: .55rem .8rem` resets **all four sides**,
dropping `padding-right` back to `.8rem`. The arrow overlaps selected text in
every `<select>` using this rule.

Fix: reorder so the longhand wins, or fold the right padding into the shorthand.

### 10. `css:S4656` — duplicate property (1, MAJOR)

`frontend/ml-canvas/src/styles/layout.css:37`

`.feature-canvas-navbar__brand--gradient` declares `color: #f8fafc` (37) and
then `color: transparent` (41). Same specificity, so the later one always wins
and line 37 is **dead** — it does not act as a fallback for browsers lacking
`background-clip: text`, which is presumably why it was written. Removing it is
behaviour-identical; a real fallback needs `@supports`.

### 11. `typescript:S8786` — ambiguous regex quantifiers (3 of 6, MAJOR)

`frontend/ml-canvas/src/components/pages/InferencePage.tsx:291`
`frontend/ml-canvas/src/components/panels/jobs/JobDetailsView.tsx:133, 138`

All three share the shape `\d+\.?\d*`: two `\d` quantifiers separated by an
*optional* dot. On a failing long digit run the engine can split the run between
`\d+` and `\d*` in O(n) ways → quadratic backtracking.

Severity is **not** bounded, contrary to the "client-side only, so at worst a
UI stall" reading. Measured on a 60,001-char input (`'1'.repeat(60000) + 'x'`,
which forces the whole match to fail and so triggers the full backtrack):

| pattern | time |
|---|---|
| `^-?(?:\d+\.?\d*\|\.\d+)(?:[eE][+-]?\d+)?$` (old) | **1723.74 ms** |
| `^-?(?:\d+(?:\.\d*)?\|\.\d+)(?:[eE][+-]?\d+)?$` (new) | **0.11 ms** |

A repeat run measured 2029.77 ms → 0.09 ms, so the figure is stable at
"~2 seconds → ~0.1 ms", a **four-order-of-magnitude** difference. This is
user-reachable, not theoretical: `InferencePage` runs `NUMERIC_RE` over pasted
CSV cell values, so a long numeric-looking paste hangs the main thread for
~2 s per cell — and the two `JobDetailsView` patterns are unanchored and run
over log/duration strings in a `replace` loop, so they hit it per line. A
self-inflicted multi-second freeze of the whole tab is the realistic outcome.

Fix: `\d+(?:\.\d*)?` — **identical language**, no ambiguity, linear. Digits
after a dot become reachable only through the mandatory dot. Language identity
re-verified over 21 inputs (`''`, `0`, `42`, `-42`, `3.14`, `-3.14`, `.5`,
`-.5`, `5.`, `-5.`, `1e10`, `1.5e-3`, `1.e5`, `.5e2`, `abc`, `1a`, `' '`,
`+1`, `1.2.3`, `00`, `0.0`): zero mismatches.

The fourth `S8786`-shaped pattern in `JobDetailsView.tsx` — `\d+\.\d+` with a
*mandatory* dot — was deliberately left alone: it is already unambiguous and
linear, and rewriting it would be churn without a fix.

### 12. `typescript:S7773` — Number statics over globals (62, MINOR)

Cosmetic consistency, but a clean sweep: **62 sites across 27 files** —
41 `parseFloat`, 11 `parseInt`, 7 `isNaN`, 2 `isFinite`, 1 bare `NaN`.
(Reconciles exactly with the applied diff: 62 added `Number.*` tokens in the
27 files whose diff introduces one.)

The reason this needed auditing rather than a blind find/replace is that
**`Number.isNaN`/`Number.isFinite` are not drop-in replacements** — the globals
coerce their argument first (`isNaN("abc") === true`,
`Number.isNaN("abc") === false`), so a naive conversion on a string argument
silently changes behaviour. `Number.parseInt`/`Number.parseFloat`/`Number.NaN`
*are* the identical objects/values and are always safe.

All 9 `isNaN`/`isFinite` sites were checked individually: every argument is
statically already a `number` — a `Number(...)` wrapper, a `parseFloat(...)`
return, a `Math.max/min` result, or a `typeof x === 'number'` narrowing.
`tsconfig.json` sets `"strict": true` and the TS lib signatures take `number`,
so a string argument would not compile. **All 62 convert with zero behaviour
change.** No site needs the `Number.isNaN(Number(x))` coercion-preserving form.

Also confirmed: `tsconfig.json` targets `ES2022` with `lib: ["ES2022", "DOM",
"DOM.Iterable"]`, so all `Number.*` statics are available; no ESLint rule
enforces or forbids the conversion (`.eslintrc.cjs` has no
`no-restricted-globals` and `eslint-plugin-unicorn` is not installed), so
SonarCloud is the only source of this rule. The codebase already uses
`Number.isNaN`/`Number.isFinite` extensively, so this removes an inconsistency.

---

## False positive / won't fix (29)

### `python:S5863` — "assertions given twice the same argument" (6)

`skyulf-core/tests/unit/test_pipeline_card.py:27`
`skyulf-core/tests/unit/test_pipeline_coverage.py:45, 110, 121, 151`
`tests/integration/test_catalog_polars_ingestion.py:140`

These read like `assert x == x` copy/paste slips, and were triaged that way
initially. Reading them in context shows every one is **deliberate**: the
identical expressions are two independent *evaluations* of a function whose whole
contract is that it returns the same value every time.

`artifact_digest` (`skyulf-core/skyulf/pipeline/seal.py:19`) is documented as a
"stable semantic digest" that walks hyperparameters and fitted weights instead of
pickle bytes. `SkyulfPipeline.fingerprint()` is the same idea one level up. So
"calling it twice yields the same digest" is precisely the property under test —
a digest that leaked `id()`, a timestamp or a random salt would fail it. The test
names say so outright (`test_artifact_digest_is_deterministic_for_same_estimator`,
`test_fingerprint_is_deterministic_for_same_config`).

Verified by running it:

| Check | Result |
| --- | --- |
| `artifact_digest(est)` twice, same object | equal |
| `artifact_digest` of two independently-fit `LogisticRegression(random_state=42)` | equal |
| `artifact_digest` of two independently-fit `RandomForestClassifier(random_state=42)` | equal |
| `artifact_digest(_Marker)` twice, same class object | equal |
| `artifact_digest` of two *distinct* empty classes | **different** |

That last row is the reason `test_pipeline_coverage.py:151` must **not** be
"strengthened" into two separate `_Marker` classes: classes digest by identity, so
the rewrite would break a currently-passing test. Four of the six are also paired
with a real `!=` assertion against a genuinely different object
(`random_state=43`, `C: 10.0`, `int`), which is the half that pins sensitivity.

`test_catalog_polars_ingestion.py:140` is the canonical NaN test
(`assert pol["c"][0] != pol["c"][0]  # NaN != NaN`), pinning the F-13 engine
difference where polars keeps a literal `NaN` token as float NaN rather than null.

**Action:** five left exactly as they are — churning working determinism tests to
satisfy a linter would weaken them. The NaN check alone is rewritten to
`pd.isna(...)`, which is unambiguous, already imported in that file, and
behaviour-identical. All six should be marked FP in the SonarCloud UI.

> Optional future strengthening (not done here): the three same-object digest
> tests at `test_pipeline_coverage.py:45, 110, 121` could compare two
> *independently fitted* estimators instead, which the table above shows is safe
> and would additionally catch pointer/identity leakage — the exact ASLR hazard
> the neighbouring OC-62 test documents. It needs the test names updated too
> ("for same estimator" would no longer be accurate), so it belongs in a
> test-hardening pass rather than a lint cleanup.

### `typescript:S2871` — `sort()` without compare fn (6)

`ExperimentsPage.tsx:418,423` · `BranchComparisonCard.tsx:50` ·
`pipelineDiffLayout.ts:268` · `predictMergeConflict.ts:113` · `Jobs.tsx:445`

The real hazard of a bare `.sort()` is lexicographic ordering of **numbers**
(`[10,9,2].sort()` → `[10,2,9]`). Every one of these arrays is `string[]` —
metric keys, metric base names, node ids, contested column names, lowercased
status labels — and plain alphabetical order is exactly what the code wants.
`pipelineDiffLayout.ts` even carries a comment asking for deterministic
alphabetical layout. No numeric sort in the set. Adding comparators would change
behaviour (locale-sensitive ordering) for no benefit.

### `typescript:S6959` — `reduce()` without initial value (2)

`metricMeta.ts:122` · `predictMergeConflict.ts:93`

`reduce` with no seed throws on an empty array, but both inputs are guarded
upstream: `metricMeta.ts:120` returns early on `comparable.length < 2`, and
`predictMergeConflict.ts:86` returns early on `branchIds.length < 2` (the reduced
array is `branchIds.map(...)`). Both are provably ≥ 2 elements.

### `typescript:S8786` — remaining regex hits (3)

- `JobDetailsView.tsx:143` — `/-?\d+\.\d+(?:[eE][+-]?\d+)?/` has a **mandatory**
  literal dot between the quantifiers, so the digit runs cannot overlap → linear.
- `canvasExport.ts:59` — `/translate\(([^,]+)px,\s*([^)]+)px\)/`: the classes are
  bounded by `,` and `)`, no nesting; input is a React-Flow-generated
  `style.transform` (~30 chars), not user-controlled.
- `TrainTestSplitNode.tsx:203` — `/\.?0+$/`: dot vs zeros are disjoint classes, no
  nesting; input is `x.toFixed(2)`, ≤ 6 chars from a config fraction.

### `typescript:S2245` — `Math.random()` (5)

`CausalGraph.tsx:90` (×2) · `jobEventsSocket.ts:114` ·
`useNotificationsStore.ts:58` · `recentPipelines.ts:108`

Nothing security-relevant: a cosmetic fallback node position when dagre layout
fails, WebSocket reconnect **jitter** to avoid a thundering herd, and two local
UI id suffixes. No auth token, session id, CSRF nonce or key material anywhere in
the set. Three of the five already carry explicit `nosemgrep` annotations saying
so — the codebase has triaged these before.

### `pythonsecurity:S5145` — `backend/eda/router.py:145` (1)

```python
logger.info("Triggering analysis for dataset %s. Request: %s", dataset_id, body)
```

`dataset_id` is declared `dataset_id: int` in the route signature, so FastAPI
rejects any non-integer before the handler runs — it cannot carry a CRLF. `body`
is an `AnalyzeRequest` Pydantic model logged through `%s`, whose repr quotes and
escapes string fields, so no raw newline reaches the record. Not injectable.

### `pythonbugs:S2583` — `skyulf-core/skyulf/preprocessing/split.py:212` (1)

Sonar claims `not 0 < test_size < 1` "always evaluates to true", making the
`validation_size` check on line 214 unreachable. **Empirically false** — this
looks like the symbolic engine mishandling the chained comparison. Constructing
`DataSplitter` (whose `__init__` starts at line 204, matching the report):

| `test_size` | `validation_size` | Result |
| --- | --- | --- |
| 0.2 | 0.0 | OK — no raise |
| 0.2 | 0.1 | OK — no raise |
| 0.7 | 0.2 | OK — no raise |
| 0.0 | 0.2 | `ValueError: test_size must be between 0 and 1` |
| 0.9 | 0.2 | `ValueError: test_size + validation_size must be less than 1` |

The first three rows prove the condition is not always true, and the last proves
control flow reaches *past* line 214 to the third guard. All three branches are
live.

### `python:S6709` — `multivariate.py:197, 281` (2)

**Already fixed** on branch `0812`. Both `KMeans` (now line 284) and
`IsolationForest` (now line 380) pass `random_state=DEFAULT_RANDOM_STATE`.
These will clear on the next `master` scan after this branch merges.

### `python:S5332` — `backend/main.py:138` (2)

```python
servers = settings.API_DOCS_SERVERS or [f"http://{settings.HOST}:{settings.PORT}"]
```

A **default** OpenAPI docs server URL for local development, fully overridable
via `settings.API_DOCS_SERVERS`. No request is made to it; it is metadata
rendered in the Swagger UI server dropdown. Flagged twice because the f-string
contains two interpolations. Forcing `https` here would break local dev.

### `python:S1313` — `backend/data_ingestion/connectors/s3.py:112` (1)

```python
if "169.254.169.254" in msg:
    raise ValueError("S3 Connection Error: Could not find AWS credentials. ...")
```

`169.254.169.254` is the AWS EC2 instance-metadata endpoint. The code is not
contacting it — it is **string-matching it inside an exception message** to
recognise "botocore tried IMDS and failed" and translate that into an actionable
credentials error. Matching the literal is the entire point; extracting it to
config would obscure it.

---

## Fix order

Rationale: correctness and safety first, then the things that weaken the test
suite, then the large cosmetic sweep.

1. **CORS middleware order** (`S8414`, BLOCKER) — user-visible cross-origin
   failure mode.
2. **Test integrity** (`S5779` ×10) — assertion failures currently report with an
   empty reason and are indistinguishable from engine errors. Fix before the rest
   so the suite is trustworthy while other fixes land. (The 6 `S5863` hits in the
   same files are false positives; only the NaN check gets a clarity rewrite.)
3. **Core reproducibility** (`S6709` ×2) — violates a stated library guarantee.
4. **Blocking event loop** (`S7493` ×2).
5. **`_first_finite` inf leak** (`S1764`) + log injection (`S5145` ×3) + dead
   `or {}` (`S2583`) + `np.nonzero` (`S6729`) + `math.isnan` (`S1764`).
6. **CSS** (`S4657`, `S4656`) — one visible layout bug, one dead declaration.
7. **Regex** (`S8786` ×3) — behaviour-preserving rewrite.
8. **Number statics** (`S7773` ×62) — bulk sweep, last, so any fallout is
   isolated from the correctness work.

## Verification — all 89 fixes applied, gates run

| Gate | Result |
| --- | --- |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 661 files already formatted |
| `ty check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py` | All checks passed |
| `pytest skyulf-core/tests -q` | **3661 passed**, 70 skipped, 3 snapshots passed (90.5 s) |
| `pytest tests/ -q` | **1551 passed**, 7 snapshots passed (60.5 s) — 1547 + the 4 new S1764 tests |
| `npx tsc --noEmit` | exit 0 |
| `npm run lint` | clean (`--max-warnings 0`) |
| `npm run test -- --run` | **104 files / 872 tests passed** (39.1 s) |
| `npm run build` | exit 0, `✓ built in 17.45 s` → `static/ml_canvas/` |

Frontend rebuilt per the repo convention ("after working with the frontend
always rebuild it"); the build regenerated 61 content-hashed assets under
`static/ml_canvas/assets/`, which are committed.

Two gates are evidence of code correctness rather than feature correctness: the
CSS `padding-right` fix (S4657) and the middleware-order fix (S8414) change
runtime behaviour no unit test observes.

S8414 was verified directly by inspecting the built app's `user_middleware`
(`CORSMiddleware → ErrorHandlerMiddleware → LoggingMiddleware →
TrustedHostMiddleware`) — CORS is outermost.

S4657 was verified statically three ways, which settles the cascade question:
(i) `padding-right: 2.5rem` now follows the `padding: .55rem .8rem` shorthand;
(ii) no other `select` rule in `src/styles/` declares padding — `.dark select`
overrides only colours/background/border and `select:focus` only
outline/border/box-shadow; (iii) none of the **110** `<select>` elements in the
codebase carry a Tailwind padding utility (`p*-n`) that would outrank the rule,
so it is the sole padding source for every dropdown in the app and the fix is
load-bearing app-wide. What remains unverified is the rendered pixel output —
worth one visual pass over any dropdown before release, but no cascade ambiguity
is left.

The regex fix (S8786) is the one with a measured behavioural claim, reproduced
twice; see section 11.

Test coverage added by this sweep is deliberately narrow: 4 regression tests in
`tests/integration/test_node_summary.py`, all for the `_first_finite` /
`_train_only` infinity leak (S1764) — the only genuine finding that changed
observable behaviour without an existing test. Three pin the distinct pre-fix
renderings (`acc inf` instead of falling through to the next finite candidate;
`acc inf · f1 -inf` instead of no headline; `acc 0.80 · ▲inf` from an infinite
overfit gap, since `inf - 0.80 = inf` fails the `diff < 0.05` guard) and one
pins the NaN rejection that already worked, because `math.isfinite` is a
strictly wider filter than `f == f` and a future refactor could narrow it back.

Each test's discriminating power was **measured, not assumed**: the pre-fix
`f == f` check was replayed against the fixed module by substituting faithful
old versions of both helpers, and all three inf cases were confirmed to produce
different output before vs after. (An earlier attempt at this comparison
monkeypatched `_train_only` with a copy of `_first_finite`, dropping its
`train_`-only filter, and produced two bogus "does not discriminate" verdicts —
the faithful replay is the one to trust.) The NaN case genuinely does not
discriminate and is labelled a pin in its docstring.

Everything else in the sweep is either covered by the suites above or is
behaviour-preserving by construction (S7773, S4656, S6729, the S8786 rewrite,
S2583's dead branch). The two runtime-behaviour fixes with no automated coverage
are called out above: S4657 (CSS, needs a browser) and S8414 (verified by
inspecting the built middleware stack).

## False positives to report upstream

Worth marking in the SonarCloud UI so they stop reappearing, in rough order of
confidence:

- `split.py:212` (`S2583`) — demonstrably wrong; empirically disproved above.
- All 6 `S5863` — intentional determinism/idempotence assertions on the semantic
  digest, plus one canonical `x != x` NaN test.
- `eda/router.py:145` (`S5145`) — param is `int`, framework-validated.
- `s3.py:112` (`S1313`) — matching the IMDS literal in an error string, not
  connecting to it.
- `main.py:138` (`S5332`) — overridable local-dev docs default.
- All 6 `S2871` — string arrays, alphabetical intended.
- Both `S6959` — guarded non-empty by early returns.
- All 5 `S2245` — non-security `Math.random`, 3 already annotated.
- `JobDetailsView.tsx:143`, `canvasExport.ts:59`, `TrainTestSplitNode.tsx:203`
  (`S8786`) — linear patterns.
