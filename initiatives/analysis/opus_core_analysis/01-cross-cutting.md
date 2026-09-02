# Skyulf Deep Audit (Opus) — Cross-cutting, packaging & registry

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

---

## Cross-cutting findings (verified directly)

### OC-01
### 🟠 High — `skyulf.__version__` always reports a stale, wrong version

**File:** `skyulf-core/skyulf/__init__.py:24-30`

`__version__` is resolved at import time via
`importlib.metadata.version("skyulf-core")`. The in-tree comment explains this
is deliberate, to avoid "a second copy that can drift out of sync". In practice
the mechanism drifts **silently and always**:

| Source | Version |
|---|---|
| `setup.py` (the real code version) | `0.8.8` |
| in-repo `skyulf_core.egg-info/PKG-INFO` | `0.8.5` |
| installed `skyulf_core-0.5.8.dist-info` | `0.5.8` |
| **what `skyulf.__version__` actually returns** | **`0.8.5`** |

```console
$ cd skyulf-core && python -c "import skyulf; print(skyulf.__version__)"
0.8.5
```

The value resolves to whichever stale metadata directory is found first — never
the version in `setup.py`. Because `importlib.metadata` succeeds, there is no
error to notice.

**Impact:** Any artifact, model card, run log, or reproducibility record that
stamps `skyulf.__version__` records a version that does not correspond to the
code that produced it. This defeats the purpose of version stamping.

**Fix:** Adopt a single source of truth. Either (a) put `__version__` in
`skyulf/__init__.py` and have `setup.py` read it, or (b) keep
`importlib.metadata` but add a build-time consistency assertion and delete the
stale `.egg-info` from version control. Add a test asserting
`skyulf.__version__ == <setup.py version>`.

---

### OC-02
### 🟠 High — Dev editable install is dangling; `import skyulf` fails outside the repo

`pip show skyulf-core` reports:

```
Editable project location: /private/tmp/skyulf-078k/skyulf-core
```

That directory **does not exist**. Consequently:

```console
$ cd ~ && python -c "import skyulf"
ModuleNotFoundError: No module named 'skyulf'
```

The test suite only passes because pytest inserts the rootdir onto `sys.path`.
Any script, notebook, or Celery worker run from a different working directory
cannot import the library.

**Impact:** The development environment does not reflect a real installation.
Import-time regressions, packaging errors, and missing-`__init__` problems are
invisible locally and only surface in deployment.

**Fix:** Re-install cleanly: `pip install -e ./skyulf-core`. Add a CI smoke step
that runs `cd /tmp && python -c "import skyulf; print(skyulf.__version__)"` so a
broken install fails the build.

---

### OC-03
### 🟠 High — Systemic `infer_output_schema` int→float misprediction across 22 nodes

**Files:** 22 preprocessing nodes using the `return input_schema` pass-through
pattern; consumed by `backend/ml_pipeline/_execution/_schema_graph.py:104`

`infer_output_schema` powers the frontend's downstream schema preview. 22 nodes
implement it as `return input_schema`, i.e. "I don't change the schema." For any
node that scales or transforms an integer column, that is false — the column is
promoted to `float64`.

I tested 6 of the 22 and **all 6 mispredicted**:

| Node | Predicted | Actual |
|---|---|---|
| `StandardScaler` | `{'a': 'int64'}` | `{'a': 'float64'}` |
| `MinMaxScaler` | `{'a': 'int64'}` | `{'a': 'float64'}` |
| `RobustScaler` | `{'a': 'int64'}` | `{'a': 'float64'}` |
| `MaxAbsScaler` | `{'a': 'int64'}` | `{'a': 'float64'}` |
| `Winsorize` | `{'a': 'int64'}` | `{'a': 'float64'}` |
| `PowerTransformer` | `{'a': 'int64'}` | `{'a': 'float64'}` |

`GeneralTransformation` (`transformations/general.py:160-165`) has the same
defect.

**Impact:** The canvas shows users an incorrect dtype for every downstream node
after a scaler. Any backend logic that plans or validates on the predicted
schema (type-compatibility checks, merge-conflict detection) is reasoning about
a schema that will never exist.

Note that returning `None` from `infer_output_schema` is a **documented,
intentional** "unknown / data-dependent" signal
(`preprocessing/base.py:97-113`). So the correct fix is cheap: these nodes
should promote the dtype, not return `None`.

**Fix:** Add a shared helper, e.g.
`schema.promote_to_float(cols)`, and use it in every node that produces
floating-point output from numeric input. Add a parametrised test asserting
`infer_output_schema(s) == SkyulfSchema.from_dataframe(apply(fit(df)))` for
every node that returns non-`None`.

---

### OC-04
### 🟡 Medium — Cross-engine dtype divergence in 3 nodes

I built an all-node cross-engine parity harness (fit + apply each registered
preprocessing node on both pandas and polars, then diff column sets, order,
shape, dtype, and values). Result: **26 match, 3 diverge, 0 error, 5 not
exercised, of 34 nodes.**

| Node | pandas dtype | polars dtype |
|---|---|---|
| `DummyEncoder` | `int64` (11 dummy cols) | `int8` |
| `GeneralBinning` | `int64` | `uint32` |
| `KBinsDiscretizer` | `int64` | `uint32` |

Values are identical; only the dtype differs. This matters because the existing
parity test (`tests/unit/test_engine_parity.py`) compares **artifacts only**,
never applied output frames, so it cannot catch this class of divergence.

The `uint32` case deserves attention: an unsigned bin index cannot represent a
negative sentinel. If a future change introduces `-1` for "out of range" (a
very common convention), the polars path would wrap to `4294967295` while
pandas gives `-1`.

**Impact:** A model trained through the pandas path and served through the
polars path receives different input dtypes. Strict schema validation would
reject; some estimators change behaviour on integer width.

**Fix:** Normalise output dtypes explicitly at the end of each dual-engine
applier. Extend `test_engine_parity.py` to compare **applied output frames**
(columns, order, dtypes, values) for every registered node, not just artifacts.

> **Checked and cleared:** I initially suspected that out-of-range values at
> transform time were silently becoming `NaN` in the binning nodes. They are —
> but this is **intentional and documented** at `bucketing.py:41-50`, and it is
> consistent across both engines. Not a bug. It is still worth surfacing a
> count of out-of-range rows to the user (see improvements).

---

### OC-05
### 🟡 Medium — `PowerTransformer` triggers a pandas deprecation that will become an error

**File:** `skyulf-core/skyulf/preprocessing/transformations/power.py:101`

```python
df_out.loc[:, valid_cols] = np.asarray(X_trans)
```

When applied to an integer column this emits:

```
FutureWarning: Setting an item of incompatible dtype is deprecated and will
raise in a future error of pandas.
```

I scanned the whole package; this is the **only** node with this pattern.

**Impact:** A future pandas release turns this warning into an exception,
breaking `PowerTransformer` on integer input.

**Fix:** Build the transformed block and assign with an explicit dtype-safe
construction, e.g. `df_out[valid_cols] = pd.DataFrame(X_trans, index=df_out.index, columns=valid_cols)`.

---

### OC-06
### 🟡 Medium — 9 registered nodes are unreachable from the UI

I diffed all 100 registry ids against every quoted string in
`frontend/ml-canvas/src`. These 9 registered, implemented, tested nodes have no
frontend affordance:

`CustomBinning`, `DataSnapshot`, `DatasetProfile`, `FeatureGeneration`,
`GeoDistance`, `H3Index`, `birch`, `gaussian_mixture`, `minibatch_kmeans`

Notably:
- The whole `geo/` package (348 lines + tests, fully implemented) is unreachable.
  `backend/ml_pipeline/_execution/_leakage_validation.py:27-28` even documents
  `GeoDistance`/`H3Index` behaviour.
- `SegmentationNode.tsx` offers only `kmeans`, hiding three implemented
  clustering algorithms.

**Impact:** Dead-but-maintained surface area. Contributors pay the cost of
keeping these nodes green with zero user benefit.

**Fix:** Decide per node — either expose it in the canvas or deprecate and
remove it. Add a CI check that every registry id is either referenced in the
frontend or on an explicit `INTENTIONALLY_HEADLESS` allow-list.

---

### OC-07
### 🟡 Medium — Node-id naming is split 55 PascalCase / 45 snake_case + redundant aliases

The single registry mixes `StandardScaler` with `random_forest_classifier`.
There is no rule a contributor can follow.

There is also redundant triple-aliasing:
- `FeatureGeneration` / `FeatureMath` / `FeatureGenerationNode` → the same applier (`feature_generation/generation.py:24-26`)
- `PolynomialFeatures` / `PolynomialFeaturesNode` → the same applier (`polynomial.py:88-89`)

**Fix:** Document the convention (preprocessing = PascalCase, estimators =
snake_case appears to be the de-facto rule) and enforce it in the `@node_meta`
decorator. Collapse the aliases behind `_DEPRECATED_ALIASES`, which already
exists for `Split → TrainTestSplitter`.

---

### OC-08
### 🟡 Medium — Public-API name collision: `DatasetProfile` means two different things

- `skyulf.DatasetProfile` → `skyulf.profiling.schemas.DatasetProfile` (a dataclass)
- registry node id `DatasetProfile` → `skyulf.preprocessing.inspection.DatasetProfileCalculator`

Same string, two unrelated concepts, both public.

**Fix:** Rename the node id to `DatasetProfileInspection` (keeping the old id in
`_DEPRECATED_ALIASES`), or rename the schema export.

---

### OC-09
### 🟡 Medium — Narrow `ruff select` hides substantial debt behind a green CI

**File:** `pyproject.toml:137-226`

```toml
select = ["E9","F63","F7","F82","I","UP","B","C4","SIM","PERF","BLE","PLC0415"]
```

This omits `D`, `ARG`, `S`, `PLR`, `C90`, `DTZ`, `N`, `RUF`, `PD`, `TRY`.
Running `ruff check --select ALL` on `skyulf-core/skyulf` reveals:

| Rule | Count | Why it matters here |
|---|---:|---|
| `ARG001`/`ARG002` unused arguments | **84** | This is the exact signature of "parameter accepted but silently ignored" — the dominant bug class in this audit |
| `D101`/`D102`/`D107` missing docstrings | **495** | Directly violates the repo's own `AGENTS.md` docstring rule |
| `PLR2004` magic values | 120 | Hardcoded thresholds like the `abs(skew) > 1.5` in OC-42 |
| `C901`/complexity | 15 | Worst: `pipeline/seal.py:34` `_feed_canonical` (C901=26, 67 statements, 17 returns); `_tuning/engine.py:458` `tune` (C901=16) |
| `ERA001` commented-out code | 7 | |
| `T201` `print()` in library code | 3 | `profiling/visualizer.py:29,496,516` |
| `S110` try-except-pass | 2 | `profiling/_analyzer/dates.py:92,103` |
| `TRY004` | 3 | Consistent with still-open `F-30` |

**Impact:** "CI is green" is not evidence of health. The 84 unused arguments in
particular are a mechanical detector for the very class of bug that dominates
this report.

**Fix:** Enable `ARG` immediately (highest signal-to-noise, directly finds
silent-no-op bugs). Then adopt `D` and `PLR2004` incrementally with
`per-file-ignores` for the existing backlog so new code is held to the standard.

---

### OC-10
### ⚪ Low — 4 dead `infer_output_schema` overrides

`vectorization/count_vectorizer.py:128`, `tfidf_vectorizer.py:122`,
`tokenizer.py:157`, `sentence_embedder.py:178` each override
`infer_output_schema` only to `return None`, which is already the base-class
default. Dead code that implies intent where none exists.

**Fix:** Delete the four overrides.

---

### OC-11
### ⚪ Low — Mega smoke test silently skips nodes with empty params

**File:** `skyulf-core/tests/unit/test_all_nodes_smoke.py`

The harness contains `if not params: return`. A node that regresses to a silent
no-op (returning an empty artifact) therefore **passes**. The test is also
pandas-only and asserts no semantics.

My parity harness found 5 nodes producing an empty fit under their own declared
default params: `AliasReplacement`, `InvalidValueReplacement`, `TextCleaning`,
`TargetEncoder`, `WOEEncoder`. For `TargetEncoder`/`WOEEncoder` this is
legitimate (they require `y`). For the other three it means the mega smoke test
never actually exercises them — and OC-19/OC-20 show that two of those three do
in fact have silent no-op bugs in production configurations.

**Fix:** Replace `return` with an explicit `pytest.skip(reason)` so skips are
visible in test output, and add per-node minimal configs that produce a
non-empty artifact.

**Verified sound:** determinism. I re-ran every preprocessing node twice
in-process and across `PYTHONHASHSEED=0/1`. **34/34 nodes produced
byte-identical output**, confirming no reliance on Python's salted `hash()`
(`HashEncoder` correctly uses `hashlib.blake2b`).

---
