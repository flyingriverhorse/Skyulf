# Problems panel review — 2026-09-08

Reviewed the supplied Problems export against working tree `f12dde9f`.
The export repeats entire blocks and includes five repeated locations within
each block. Counting each file/rule/location once gives **105 distinct
diagnostics**. Notebook locations include the cell identity.

This is a review, not a blanket application of automated suggestions. No
application source, dependency versions, example files, or editor settings were
changed. The actionable findings are recorded as OC-213–215 in
[the open queue](opus_core_analysis-open_queue.md).

## Complete disposition

| Group | Distinct diagnostics | Assessment |
|---|---:|---|
| ty in `09_leakage_safety.ipynb` | 16 | Reproduced; fix example configuration typing and use explicitly typed pandas partitions. |
| Trivy dependencies | 2 | Both reported versions are present in the lockfile and installed tree; patch releases are available. |
| Sonar in ty's cached `builtins.pyi` / `types.pyi` | 32 | Third-party type declarations outside Skyulf; not application defects. |
| Bandit B101 in `09_leakage_safety.py` | 11 | Correct observation about assertions, but these are example verification checks, not security enforcement. |
| Pylint W0221 in `CustomBinningCalculator.fit` | 1 | Decorator-related false positive; public calling convention works. |
| Codacy Ruff UP038 | 6 | Obsolete rule, removed from Ruff; retain valid tuple-based `isinstance` checks. |
| Sourcery in repository source and tests | 30 | Style/maintainability suggestions; two test-import warnings misclassify shared test utilities. See every location below. |
| Diagnostics in old reproduction scripts | 7 | Both referenced files no longer exist on disk; stale editor diagnostics. |
| **Total** | **105** | These are diagnostic locations, not 105 independent bugs. |

Repeated locations inside each copied block: `bucketing.py:531` (W0221),
`_node_runners.py:279,1271` (UP038), and `casting.py:268,270` (UP038).
Some export line numbers also precede changes in the current files.

## Actionable findings

### OC-213: leakage examples need accurate type information

Executed:

```powershell
.venv\Scripts\python.exe -m ty check skyulf-core/examples/09_leakage_safety.ipynb skyulf-core/examples/09_leakage_safety.py --output-format concise
```

Result: **22 diagnostics** — all 16 supplied notebook diagnostics plus six
additional diagnostics in the Python example at lines 128–131. The normal
configured ty scope covers backend, core source, and core tests, so a clean run
of that scope does not establish that examples are type-clean.

The notebook diagnostics reduce to three causes:

| Current notebook location | Count | Cause and correction |
|---|---:|---|
| Cell 18 / `leakage-17`: configuration filtering | 2 | An unannotated heterogeneous `safe_config` becomes a union of modeling dictionaries and preprocessing lists. Give the configuration the intentional `dict[str, Any]` type accepted by the pipeline API, or use a suitable typed configuration structure. |
| Cell 18 / `leakage-17`: external train/test statistics | 9 | `SplitDataset` slots allow frames from several engines and `(X, y)` tuples. Keep named pandas train/test frames before constructing the container and use those for pandas operations, or explicitly narrow the slot types. |
| Cell 22 / `leakage-21`: fold validation | 4 | The same broad split-slot union reaches `.drop(columns=...)`, prediction, and target indexing. Keep a named pandas validation frame before constructing the fold container. |
| Cell 24 / `native-tuning-run`: modeling replacement | 1 | Replacing the inferred modeling dictionary with tuning configuration conflicts with the original inferred value type. The configuration annotation fixes this too. |

The six Python-example diagnostics have the same split-slot cause. Do not
change `SplitDataset` to pandas-only or add blanket type ignores to clear them.

Runtime control: executed all **12 original notebook code cells in order in
one Python namespace**, then ran the original Python example with `runpy`.
Both completed, including all assertions. Expected warnings appeared only for
the deliberately unsafe examples. This was direct cell execution, not a test
of Jupyter kernel startup or rendering.

### OC-214 and OC-215: vulnerable transitive dependencies

`npm ls nanoid postcss-selector-parser --all` and a targeted lockfile inspection
confirmed:

| Dependency chain | Current version | Patch target |
|---|---|---|
| `postcss@8.5.26` → `nanoid` | 3.3.17 | 3.3.18 or a later compatible 3.x patch |
| `tailwindcss@3.4.18` → `postcss-selector-parser` | 6.1.2 | 6.1.3 or a later compatible 6.x patch |
| `tailwindcss` → `postcss-nested@6.2.0` → `postcss-selector-parser` | 6.1.2, deduplicated | Same parser update |

The recorded parent ranges (`^3.3.17`, `^6.1.2`, `^6.1.1`) permit these patch
updates. Major-version overrides are unnecessary for the reported fixes.

- **CVE-2026-67213:** the reviewed advisory identifies zero-size custom ID
  generators as a possible infinite-loop trigger and lists 3.3.18 as patched.
  See the [advisory](https://github.com/advisories/GHSA-2v37-7h3g-55p8) and
  [maintainer release](https://github.com/ai/nanoid/releases/tag/3.3.18).
- **CVE-2026-9358:** the maintainer's
  [6.1.3 release](https://github.com/postcss/postcss-selector-parser/releases/tag/6.1.3)
  explicitly backports the recursion fix.

This confirms affected dependencies, not an exploitable application endpoint.
The inspected PostCSS call uses `nanoid/non-secure` with a constant size of 6;
no direct imports of either package were found in `frontend/ml-canvas/src`.
Both packages are still worth updating. After updating, review the lockfile
diff, check the installed dependency tree, rerun the vulnerability scan, and
run the frontend lint/tests/build gates.

## Findings that should not drive behavior changes

### Sonar's 32 warnings in third-party type stubs

All these paths are under
`C:/Users/Murat/AppData/Local/ty/cache/vendored/typeshed/.../stdlib/`, outside
the repository. They describe Python interfaces rather than implementing them.
For example, the inspected iterator declarations specify `Iterator[str]` or
`Self` and use stub bodies; they are not broken runtime iterators.

| File | Export lines | Rules / count |
|---|---|---|
| `builtins.pyi` | 110; 3148,4604 | FIXME / TODO comments: 3 |
| `builtins.pyi` | 1478,1481,1954,2491,2654,2873,3036,3191,3294,3380,3455,3494,3543,4002,4266,4664,4831 | Iterator implementation rule on declarations: 17 |
| `builtins.pyi` | 4685 | `__round__` signature suggestion: 1 |
| `types.pyi` | 209,232,253,278 | Parameter-count suggestions: 4 |
| `types.pyi` | 314,465,798 | Iterator implementation rule on declarations: 3 |
| `types.pyi` | 471,525,586,808 | Parameter-kind suggestions on declarations: 4 |

Do not edit ty's cache to fix these. Close the external stub editors and refresh
Sonar/editor diagnostics; use analysis scope or Problems filtering to focus on
workspace-owned files if needed. No editor settings were changed in this pass.

### Bandit: 11 assertions in the teaching example

Locations: `skyulf-core/examples/09_leakage_safety.py` lines
51,64,69,72,75,83,88,101,115,118,138.

These statements verify that the demonstrated pipeline behaved as described.
The leakage gate itself uses exceptions. Bandit correctly explains that Python
optimization removes assertions, but this does not establish a security bypass
in Skyulf. Its [B101 documentation](https://bandit.readthedocs.io/en/latest/plugins/b101_assert_used.html)
also supports file-specific exceptions for verification code.

If the examples must validate themselves under `python -O`, use explicit
checks that raise. Otherwise, a documented B101 exception for this specific
example is reasonable; disabling B101 across production code is not warranted.

### Pylint: decorator adapts the fit signature

Location: `skyulf-core/skyulf/preprocessing/bucketing.py:531` (duplicated in
the supplied block).

`@fit_method` adapts implementation `(self, X, y, config)` to public
`(self, df, config)`. Inspected with `inspect.signature(...,
follow_wrapped=False)`, the bound wrapper exposes `(df, config)`.
Executed both pandas-frame and `(X, y)` inputs using `fit(df=..., config=...)`;
both returned the expected bin edges. Adjacent calculators already use a
targeted `pylint: disable=arguments-differ` for this pattern. Preserve the API.

### Codacy: six UP038 locations use a removed Ruff rule

Export locations:
`backend/ml_pipeline/_execution/engine/__init__.py:333,346`,
`backend/ml_pipeline/_execution/engine/_node_runners.py:279,1271`, and
`skyulf-core/skyulf/preprocessing/casting.py:268,270`.

Tuple-based `isinstance` is valid. Ruff removed UP038 in 0.13.0 because the
suggested union syntax is not recommended practice and is slightly slower.
See [Ruff's explanation](https://docs.astral.sh/ruff/rules/non-pep604-isinstance/).
The local version is 0.15.16; a fresh `ruff check .` passed. Refresh or align
Codacy's Ruff analyzer/rule selection instead of changing source to satisfy
this retired rule. Several exported engine line numbers have also moved.

## All 30 repository Sourcery suggestions

Line numbers below refer to the supplied export; review matched the current
code even where those lines shifted.

| File | Export lines | Count | Assessment |
|---|---|---:|---|
| `backend/ml_pipeline/_execution/_leakage_validation.py` | 396 | 1 | Assignment plus conditional could use `:=`; equivalent style choice. |
| Same | 424 | 1 | Raise/warn/append logic appears in both violation branches. A small shared helper could reduce duplication, but requires preserving verdict behavior. |
| Same | 435 | 1 | `not any(...)` versus `all(not ...)`: equivalent; current expression reads clearly. |
| Same | 321 | 1 | The 10% quality score is a maintainability heuristic, not a measured correctness or security score. |
| `skyulf-core/skyulf/preprocessing/bucketing.py` | 98,372 | 2 | Tuple-to-set membership is optional; no incorrect result demonstrated. |
| Same | 157,333 | 2 | Early returns versus conditional expressions: optional. Preserve the explanation of interval versus ordinal labels. |
| Same | 532 | 1 | Move assignment / use `:=`: optional. |
| Same | 504,559 | 2 | Dictionary update versus union: optional; current artifact construction is clear. |
| `skyulf-core/skyulf/preprocessing/cleaning/value_replacement.py` | 27,59,238 | 3 | Membership, early return, and named-expression suggestions; optional. Locations have shifted. |
| `skyulf-core/skyulf/preprocessing/vectorization/count_vectorizer.py` | 97 | 1 | Named-expression suggestion around warning text: optional. |
| `skyulf-core/skyulf/preprocessing/vectorization/tfidf_vectorizer.py` | 91 | 1 | Same warning-text style suggestion. |
| `skyulf-core/tests/unit/test_encoding_operation_leakage.py` | 146 | 1 | `not any` is already a meaningful assertion checking unseen vocabulary. |
| `skyulf-core/tests/unit/test_leakage_safety_validation.py` | 228,229 | 2 | `not any` log checks are meaningful assertions. |
| Same | 253 | 1 | Parametrization would improve per-strategy failure reporting; the current loop actually asserts rejection. |
| `skyulf-core/tests/integration/test_leakage_operation_contract.py` | 35,41 | 2 | Conditions construct parameterized cases and select expected rejection; both paths assert outcomes. Refactoring is optional. |
| `skyulf-core/tests/integration/test_value_replacement.py` | 15,16 | 2 | Imports are shared dataset/case-loader utilities under `tests/utils`, not imports of executable test cases. Keep the shared helpers. |
| `skyulf-core/tests/unit/test_leakage_enforcement.py` | 45,188 | 2 | Loops verify registry metadata and several leakage cases; assertions are present. Parametrization is optional. |
| `tests/integration/test_backend_fold_replay_audit.py` | 126,151,218 | 3 | Conditions prepare missing-value inputs, graph layouts, and tuning parameters; they do not remove the assertions. |
| Same | 164 | 1 | Repeated folds deliberately exercise state reuse on the same adapter. Blindly replacing this with isolated parametrized cases would lose that sequence check. |
| **Total** | | **30** | |

## Seven stale reproduction-script diagnostics

`tmp_repro_artifacts/core-review-20260908/modeling/repro_known.py:34` has one
Sourcery print suggestion. `repro_pipeline.py` has three BLE001 messages
(20,28,38), D100 and I001 at line 1, and D103 at line 9.

Both named files are absent on disk. These entries need an editor/analyzer
refresh, not changes to the application. No reproduction files were deleted
as part of this review.

## Scope of verification

- Reproduced all supplied ty notebook diagnostics and the six Python-example
  diagnostics using the local checker.
- Executed the original notebook cells and original Python example successfully.
- Probed the decorated calculator using its actual public keyword signature.
- Confirmed both dependencies in the lockfile, installed tree, and parent ranges;
  verified advisory/release information upstream.
- Ran fresh repository Ruff lint: passed.
- Reviewed each supplied source/rule group; Sonar, Sourcery, Bandit, Pylint, and
  Codacy/Trivy were not rerun through their editor integrations. Their live panel
  counts have not been refreshed by this agent.
- The targeted core audit-test failures reported in the previous review are a
  separate result; this export is not a list of those failing pytest cases.
