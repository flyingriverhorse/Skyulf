# Skyulf Deep Audit (Opus) — Tests, benchmarks, packaging & CI

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

**Scope:** `skyulf-core/tests/` (3,670 tests), `benchmarks/`, `examples/`,
`setup.py`, `MANIFEST.in`, `uv.lock`, `.github/workflows/`.

**The suite itself is in excellent shape** — 98.40% coverage, zero modules below
50%, and fully deterministic across repeated runs and `PYTHONHASHSEED` changes.
The problems are all *around* it: the environment it runs in, the gates that
guard it, and the metadata it ships.

---

## Findings

### OC-75
### 🔴 Critical — The prescribed dev environment's polars is below the library's own declared minimum

**Files:** `skyulf-core/setup.py:27`, `pyproject.toml:30`, `requirements-ci.txt:26`,
`skyulf-core/uv.lock:1160-1161`, `skyulf-core/skyulf/preprocessing/split.py:283-351`

All three dependency declarations require `polars>=1.43.2`, and
`DataSplitter._split_polars`/`_split_xy_polars` call `df.gather(idx)` — an eager
`DataFrame` method that does not exist below polars ~1.4x. But:

```text
installed venv:                polars 1.40.1   has gather: False
setup.py / pyproject / ci:     polars>=1.43.2
skyulf-core/uv.lock:           polars==1.36.1  (also below the floor)
```

Every polars-engine split therefore raises
`AttributeError: 'DataFrame' object has no attribute 'gather'`.

```text
10 failed, 3591 passed, 69 skipped in 82.92s   (run 1)
10 failed, 3591 passed, 69 skipped in 52.07s   (run 2 — identical failures)

FAILED tests/integration/test_split.py::test_data_splitter_split_polars_round_trips_back_to_polars
FAILED tests/integration/test_split.py::test_data_splitter_polars_split_preserves_dtypes
… 8 more across test_split.py / test_pipeline.py / test_wrapped_polars_frames.py

# Every example notebook (00–08), identical failure:
skyulf/preprocessing/split.py:350: AttributeError: 'DataFrame' object has no attribute 'gather'

# benchmarks/bench_engine_comparison.py:
TrainTestSplitter  -  SKIP AttributeError: 'DataFrame' object has no attribute 'gather'

# benchmarks/bench_roundtrip_removal.py — crashes uncaught, no try/except:
  File "benchmarks/bench_roundtrip_removal.py", line 66, in new
    splitter.split(df)
AttributeError: 'DataFrame' object has no attribute 'gather'
```

**Impact:** Anyone following the documented setup verbatim sees 10 failing tests,
**every example notebook broken**, and a benchmark script crashing — and will
reasonably conclude the library is broken. It isn't; the environment silently
violates the package's own version contract. `skyulf-core/uv.lock` reproduces the
same breakage from a clean `uv sync`, independent of this particular venv.

> **Caveat this places on the rest of the audit:** the cross-engine parity and
> determinism harnesses reported elsewhere in this audit ran in this same venv.
> Their preprocessing-node results stand (splitters were not in scope for them),
> but **any polars splitter behaviour in this report should be re-verified on
> polars ≥ 1.43.2.**

**Fix:** Upgrade the shared dev venv to `polars>=1.43.2` and regenerate
`skyulf-core/uv.lock`. Separately, wrap `bench_roundtrip_removal.py`'s splitter
benchmark in the same try/except-and-skip pattern `bench_engine_comparison.py`
already uses, so one broken node degrades a row instead of killing the script.

**Confidence:** 10/10 — independently re-verified.

---

### OC-76
### 🟠 High — Cross-engine parity tests cover 9 of 100 nodes and never compare applied output

**File:** `skyulf-core/tests/unit/test_engine_parity.py` (6 test functions, 234 lines)

```text
NodeRegistry.get_all_metadata()  -> 100 nodes
test_engine_parity.py            -> 6 test functions, 9 distinct calculators
```

Covered: `StandardScaler`, `MinMaxScaler`, `MaxAbsScaler`, `RobustScaler`,
`SimpleImputer`, `IQR`, `ZScore`, `Winsorize`, `WOEEncoder`.

Worse than the count: **every assertion compares `Calculator.fit()` artifact
dicts** (`_assert_artifacts_equal`, `_assert_bounds_equal`,
`_assert_fill_values_equal`). None compares the `Applier.apply()` **output
frame**. So even the 9 covered nodes can silently diverge in applied values,
dtypes, row order or null handling with no test noticing.

**Impact:** This is the direct cause of several findings in this audit. OC-04
(dtype divergence), OC-23 (polars `ratio` sign flip), OC-24 (null group keys),
and OC-58 (numeric→bool truthiness) are *all* apply-time cross-engine
divergences — exactly the class this test file is named for and does not test.

**Fix:** Extend the parametrized list to the remaining preprocessing calculators
(encoders, vectorizers, feature generation/selection, splitters), and add an
output-frame comparison alongside the artifact comparison. My own ad-hoc harness
(26 MATCH / 3 DIVERGE / 5 NOOP of 34 nodes) is a working proof this is cheap to
automate.

**Confidence:** 9/10

---

### OC-77
### 🟠 High — `--maxfail=1` hides the real failure count; `--cov-fail-under=45` sits 53 points below actual

**File:** `.github/workflows/skyulf-core-tests.yml:82-87`

```yaml
pytest skyulf-core/tests -q --maxfail=1 --disable-warnings \
  --cov=skyulf-core/skyulf --cov-branch --cov-report=term-missing \
  --cov-report=xml --cov-fail-under=45
```

```text
actual coverage:  TOTAL  13793  221  98%
enforced floor:   45%
```

`--maxfail=1` stops CI at the first failing test. In the OC-75 run above, that
would have reported **1 of 10** failures — turning triage into a
re-run-fix-re-run bisection loop. And a 45% gate on a 98.4% suite leaves a
53-point cushion in which large swathes of the codebase could regress to
completely untested before the gate ever fires.

**Fix:** Drop `--maxfail=1` (or raise it to e.g. `25`) so one run surfaces the
full list; raise `--cov-fail-under` to track reality with a small buffer (≈95%).

**Confidence:** 9/10

---

### OC-78
### 🟡 Medium — `py.typed` is declared in packaging metadata but does not exist

**Files:** `skyulf-core/setup.py:20-21` (`package_data={"skyulf": ["py.typed"]}`),
`skyulf-core/MANIFEST.in:3` (`include skyulf/py.typed`)

Both reference `skyulf/py.typed`; the file exists nowhere in the repo. setuptools
and `MANIFEST.in` both silently skip a referenced-but-missing data file, so the
build succeeds with no warning and the wheel simply ships without the marker.

```text
$ find skyulf -name "py.typed"                    -> (nothing)
$ unzip -l skyulf_core-0.8.8-py3-none-any.whl | grep -i py.typed   -> (nothing)
$ tar tzf skyulf_core-0.8.8.tar.gz | grep -i py.typed              -> (nothing)
# installed the built wheel into a clean venv:
os.path.exists(.../skyulf/py.typed)  -> False
```

**Impact:** Per PEP 561, mypy and pyright treat an installed `skyulf-core` as an
**untyped** third-party package and discard all of its inline hints — silently
degrading type checking for every downstream consumer, despite metadata implying
compliance. CI's `verify_skyulf_core_distribution.py` only imports 4 names, so
this is invisible to it.

**Fix:** Add an empty `skyulf-core/skyulf/py.typed`; re-verify it appears in the
built wheel.

**Confidence:** 10/10 — independently re-verified.

---

### OC-79
### 🟡 Medium — `joblib` is imported at module scope but not declared in `install_requires`

**Files:** `skyulf/core/serialization.py:22`,
`skyulf/modeling/_tuning/strategies/runner.py:14`, `skyulf-core/setup.py:22-29`

```text
$ grep -rn "^import joblib\|^from joblib" skyulf --include=*.py
skyulf/core/serialization.py:22:import joblib
skyulf/modeling/_tuning/strategies/runner.py:14:from joblib import parallel_backend
$ grep -c joblib setup.py
0
```

Both imports are eager and on the core import path.

**Impact:** Works today only because scikit-learn pulls `joblib` in transitively.
It is still incorrect PEP 508 metadata and is fragile against any future sklearn
release that changes that transitive dependency, or a resolver mode that doesn't
propagate transitive deps identically.

**Fix:** Add `"joblib>=1.2.0"` to `install_requires`.

**Confidence:** 9/10 — independently re-verified.

---

### OC-80
### 🟡 Medium — The three weakest-covered modules are untested precisely where silence is most dangerous

**Files:** `skyulf/modeling/_sklearn_compat.py:33-46`,
`skyulf/preprocessing/cleaning/value_replacement.py:22-57`,
`skyulf/config_validation.py:76-82`

```text
$ grep -rn "normalize_logistic_regression_params" tests/                  -> (nothing)
$ grep -rn "_coerce_key\|_coerce_mapping_keys\|_polars_dtype_kind" tests/ -> (nothing)

skyulf/modeling/_sklearn_compat.py                    16   5  69%  40, 43-46
skyulf/preprocessing/cleaning/value_replacement.py   124  16  87%  26-35, 53-57, 66
skyulf/config_validation.py                           61  10  84%  49, 62, 76-82, 101
```

1. `_sklearn_compat.py` — weakest module in the repo, **zero** test references
   anywhere. Its `l1`, `elasticnet` and `penalty=None` (→ `C=math.inf`) branches
   are all untested; only the incidental `l2` default is exercised.
2. `value_replacement.py`'s `_coerce_key`/`_coerce_mapping_keys`/
   `_polars_dtype_kind` — written, per their own docstring, *"so lookups actually
   match values instead of silently no-op'ing"* on polars — are **also**
   zero-referenced. The fix for a documented silent-failure bug has no regression
   test protecting it.
3. `config_validation.py`'s friendly-error formatter only exercises the
   `"missing"` branch; `"string_type"`, `"list_type"`/`"sequence_str"` and
   `"model_type"`/`"dict_type"` — the user-facing messages for 3 of the 4 common
   misconfigurations — never run.

**Fix:** Direct unit tests for all four `penalty` branches, for `_coerce_key`/
`_polars_dtype_kind` (int/float/bool coercion on valid and unparsable string
keys), and for each `_format_pydantic_error` branch.

**Confidence:** 9/10

---

### OC-81
### ⚪ Low — No `License ::` classifier or SPDX field, unlike the sibling root package

**File:** `skyulf-core/setup.py:81-85`

`skyulf-core`'s `LICENSE` is Apache 2.0, but `classifiers` contains only
`Programming Language :: Python :: 3`, `:: 3.12` and
`Operating System :: OS Independent` — no `License ::` classifier and no
`license=` field. The root `pyproject.toml` correctly sets
`license = { file = "LICENSE" }` plus a matching classifier for its own AGPLv3.

**Impact:** PyPI license filters, `pip-licenses` and SBOM scanners that key off
classifiers/SPDX rather than parsing LICENSE text report the license as UNKNOWN.

**Fix:** Add `license = "Apache-2.0"` (SPDX form) or the
`"License :: OSI Approved :: Apache Software License"` classifier.

**Confidence:** 8/10

---

## Suite health

| Metric | Value |
|---|---|
| Tests collected | **3,670** (3,591 passed / 10 failed / 69 skipped) |
| Overall coverage | **98.40%** — 13,793 statements, 221 missing, 188 files |
| Modules under 50% coverage | **0** — weakest is 68.75% |
| Determinism | ✅ Byte-identical results across 2 runs **and** under `PYTHONHASHSEED=1`; no wall-clock-, network- or dict-ordering-dependent tests found |
| Runtime | 82.9s / 52.1s |
| Slowest test | 16.5s — `test_select_silhouette_sample_indices_keeps_large_memory_bounded` (tracemalloc bound on a 1M-row array; passes consistently but is allocator-sensitive by construction) |
| Nodes with cross-engine tests | **9 / 100** (OC-76) |
| The 10 failures | 100% attributable to OC-75, not to source regressions |

### Weakest-coverage modules

| Module | Cover | Untested high-risk path |
|---|---|---|
| `modeling/_sklearn_compat.py` | 68.75% | `penalty=l1/elasticnet/None` — zero test references anywhere |
| `config_validation.py` | 83.61% | 3 of 4 `_format_pydantic_error` branches — the user-facing config errors |
| `preprocessing/cleaning/value_replacement.py` | 87.10% | `_coerce_key`/`_polars_dtype_kind` — added specifically to fix a silent no-op |
| `modeling/_explainability/shap_explanation.py` | 88.89% | SHAP masker / error-fallback branches |
| `modeling/_evaluation/clustering.py` | 93.04% | Clustering-metric edge-case guards |
| `preprocessing/dispatcher.py` | 93.18% | pandas-branch exception-log path (only the polars except-branch runs) |
| `preprocessing/encoding/target.py` | 93.26% | "no target column configured" guard on polars fit; `ValueError`→friendly-message branch |
| `preprocessing/vectorization/_common.py` | 93.52% | Lines 87-89, 167, 254-262 |

---

## Packaging integrity

| Check | Result | Evidence |
|---|---|---|
| sdist builds cleanly | ✅ | `skyulf_core-0.8.8.tar.gz`, 229 files |
| wheel builds cleanly | ✅ | `skyulf_core-0.8.8-py3-none-any.whl`, 193 files |
| Wheel installs standalone & imports | ✅ | Fresh venv; `from skyulf import SkyulfPipeline, EDAAnalyzer, DriftCalculator, NodeRegistry` succeeds; `__version__ == "0.8.8"` |
| No junk files (`.DS_Store`) leak | ✅ | Absent from both artifacts — `recursive-include skyulf *.py` correctly excludes it despite its presence in the working tree |
| `.egg-info` not committed | ✅ | `git ls-files skyulf-core/skyulf_core.egg-info/` empty |
| `py.typed` present in wheel | ❌ | **OC-78** |
| Declared deps actually satisfiable | ❌ | **OC-75** |
| `joblib` declared for direct import | ❌ | **OC-79** |
| License classifier present | ❌ | **OC-81** |
| CI wheel-verification depth | ⚠️ Shallow | `verify_skyulf_core_distribution.py` imports 4 top-level names; would not catch OC-78 or any packaging-data omission |

---

## What I checked and found sound

- **`run_check.sh`** correctly propagates the wrapped command's exit status
  (`set -uo pipefail`, `exit "$status"`). No `continue-on-error:` anywhere in the
  workflow tree — no silent-failure risk.
- **Docker smoke job** (`skyulf-core-docker-smoke`) installs via plain
  `pip install ".[dev,all]"`, **not** via the stale `uv.lock` — so that CI path
  resolves a compliant polars and is unaffected by OC-75. This is why CI is green
  while local setup is red.
- **Optional heavy deps** (`shap`, `sentence-transformers`, `imbalanced-learn`,
  `xgboost`, `lightgbm`, `optuna`) are all lazily imported inside functions with
  clean, actionable messages, correctly gated behind their `extras_require` groups.
- **All other direct imports** (pandas, numpy, polars, pyarrow, pydantic, scipy,
  statsmodels, sklearn) are declared; no unused core dependencies.
- **The 161-file `no_dedicated_tests.txt` list is not a real gap.** It is a
  naming observation. Spot-checking `engines/pandas_engine.py`,
  `engines/polars_engine.py`, `leakage.py` and `registry.py` shows 96–100% line
  coverage via integration and smoke tests despite having no
  `test_<module>.py` of their own.

---

## Improvement opportunities (not defects)

- Back-port `bench_engine_comparison.py`'s graceful `SKIP <error>` wrapping to
  `bench_roundtrip_removal.py`, which currently aborts entirely on the first
  broken benchmark (this is how OC-75 surfaced as a crash rather than a skip).
- `frontend-tests.yml`'s `npm test` has no coverage flag or threshold at all,
  unlike both Python suites — worth at least reporting the numbers, for parity.
