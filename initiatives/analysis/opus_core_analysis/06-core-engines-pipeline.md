# Skyulf Deep Audit (Opus) — Core, engines, data, pipeline & registry

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

**Scope:** `skyulf/core/`, `skyulf/engines/`, `skyulf/data/`, `skyulf/pipeline/`, plus top-level `registry.py`, `types.py`, `utils.py`, `_validation.py`, `config_validation.py`, `leakage.py`, `__init__.py`. All read line-by-line.

---

## Findings

### OC-62
### 🔴 Critical — `fingerprint()` is not reproducible for any artifact holding an object-dtype NumPy array

**File:** `skyulf/pipeline/seal.py:57-59` (the `np.ndarray` branch of `_feed_canonical`),
consumed by `skyulf/pipeline/_pipeline.py:453-474` (`fingerprint()`)

`_feed_canonical` digests any `np.ndarray` via `arr.tobytes()`. For
`dtype=object` arrays, `tobytes()` serialises the raw `PyObject*` **pointers**
backing each element — not their values. Those pointers are ASLR/allocator
dependent and differ on every process run.

Every string-category encoder stores exactly such an array: sklearn's
`OneHotEncoder.categories_` is `dtype=object` for string categories, and is kept
verbatim as `encoder_object` in `OneHotArtifact`, `LabelEncoderArtifact`,
`OrdinalArtifact` and `TargetEncoderArtifact`.

```console
$ for i in 1 2 3; do PYTHONPATH=skyulf-core .venv/bin/python repro_e2e_fingerprint.py; done
# identical df, identical config (OneHotEncoder on "color" + random_forest_classifier,
# random_state=42), 3 separate processes:
fingerprint: c7766d0de9c30b41e43afd2ad1200416608c0b29edbd19ca937ea7f050dc9f88
fingerprint: 04e629cb5f90a5d1995fa5a4bd48db68ccb80a014b872e20bf5811a58fa2cace
fingerprint: 2d7854ce460ffcec0c3b14a214b3f0de438b00e31b403bdbfffbf1295795b5a4

# yet the predictions are byte-identical across all 3 runs:
preds hash: e0e1136e54a3fb71ec95af4cc67106106a46042e90b31a07b1f5f24bbcc845ec
```

**Impact:** This directly contradicts the docstring's promise that the digest is
"stable across library and pickle-protocol versions" and lets callers "prove
this prediction came from exactly this pipeline." Any pipeline using
OneHotEncoder / LabelEncoder / OrdinalEncoder / TargetEncoder / CountVectorizer /
TfidfVectorizer produces a `fingerprint()` and
`export_model_card()["fingerprint"]` that is **pure noise**. It is worse than
useless — it is actively misleading: two runs of an identical model on identical
data report different fingerprints, so anyone diffing model cards across a
re-deploy sees a false "model changed" signal.

**Why no test caught it:** `test_artifact_digest_is_deterministic_for_same_estimator`
fits only `LogisticRegression` on integer labels (so `classes_` is `int64`, not
object) and asserts `artifact_digest(est) == artifact_digest(est)` on the *same
live object in the same process* — trivially stable even with this bug.

**Fix:** For `dtype=object` (and generally non-contiguous/non-numeric) arrays,
don't use `tobytes()`; recurse per element exactly as the `list`/`tuple`
branches already do:
```python
h.update(f"ndarray-object:{arr.shape}|".encode())
for x in arr.flat:
    _feed_canonical(h, x)
```

**Confidence:** 10/10

---

### OC-63
### 🟠 High — `artifact_digest` raises `RecursionError` instead of the documented `TypeError` on cyclic graphs

**File:** `skyulf/pipeline/seal.py:34-140` (`_feed_canonical`)

The docstring states `artifact_digest` "Raises `TypeError` for anything it cannot
canonicalize: an artifact that cannot be digested must fail the seal, not
silently pass it." But `_feed_canonical` keeps no visited-object set, so any
self-referential attribute graph recurses to the stack limit:

```python
class Circular: pass
c = Circular(); c.self_ref = c
artifact_digest(c)
# RecursionError: maximum recursion depth exceeded
```

**Impact:** `SkyulfPipeline.fit()` → `export_model_card()` → `fingerprint()`
crashes with an unhandled `RecursionError` deep inside hashing code rather than
the documented, catchable `TypeError`, for any custom calculator artifact or
third-party estimator holding a reference cycle. It breaks the "must fail the
seal, not silently pass it" contract in the worst possible way — an opaque
low-level crash instead of an actionable error.

**Fix:** Thread a visited-`id(obj)` set through `_feed_canonical` (pushed/popped
around the dict/list/tuple/set/dataclass/`__dict__` recursions) and raise
`TypeError("Cannot digest object with a reference cycle: ...")`.

**Confidence:** 9/10

---

### OC-64
### 🟠 High — F-14 is only *partially* fixed: the engine registry global is still an unlocked race

**File:** `skyulf/engines/registry.py:60, 86-91`

Prior finding **F-14** (unscoped, unlocked global singletons) is recorded as
fixed. `core/compute.py` and `core/serialization.py` were indeed converted to
`ContextVar`. But the engines package was missed: `_active_engine: str = "pandas"`
is still a bare mutable class attribute and `set_active_engine()` still mutates
it with no lock and no contextvar.

The exact race F-14 described reproduces:

```text
two threads alternately calling set_active_engine("pandas")/set_active_engine("polars")
then resolve(None):
{'pandas_thread_saw_polars': 67, 'iterations': 2000}
```

**Impact:** 67 cross-contaminations in 2,000 iterations. F-14 explicitly called
this "not hypothetical" given Celery workers — and the backend does run pipelines
under Celery. A worker thread can execute a pipeline against the *other*
thread's engine, which given OC-58 (numeric→bool cast divergence) and OC-60
means silently different results, not just a crash.

**Fix:** Convert `_active_engine` to a `ContextVar` exactly as `compute.py` and
`serialization.py` already were. **This finding should reopen F-14 rather than
be tracked separately.**

**Confidence:** 9/10

---

### OC-65
### 🟡 Medium — The polars `to_numpy()` zero-width "parity fix" does not achieve parity

**File:** `skyulf/engines/polars_engine.py:130-136`

The comment claims: *"polars' `to_numpy()` raises… on a 0-column frame; pandas
yields `(n, 0)` float64, so mirror that to keep engine parity"*, implemented as
`if df.width == 0: return np.empty((df.height, 0), ...)`.

But a polars frame reduced to zero columns (e.g. via `.select([])`) **always**
reports `shape == (0, 0)` — polars cannot retain the row count once all columns
are dropped. So `df.height` is already `0` when the shim runs, and it returns
`(0, 0)`, not `(n, 0)`.

```text
pandas 0-col frame shape:               (3, 0)   # row count preserved
polars 0-col shape (via .select([])):   (0, 0)   # row count lost natively
polars 0-col to_numpy ("fixed"):        (0, 0)   # shim changes nothing

# end-to-end via SklearnBridge.to_sklearn((X, y)), y of length 5:
pandas -> X (5, 0)  y (5,)
polars -> X (0, 0)  y (5,)

LogisticRegression().fit(X, y):
  pandas -> ValueError: Found array with 0 feature(s) (shape=(5, 0)) ...
  polars -> ValueError: Found array with 0 sample(s)  (shape=(0, 0)) ...
```

**Impact:** When a feature-selection node legitimately drops every column (see
[OC-32](./03-feature-generation-selection-vectorization.md), where
`VarianceThreshold` does exactly this on all-constant input), the polars pipeline
reports **"0 samples"** — a false diagnosis pointing the user at a data-loading
bug that doesn't exist — while pandas correctly reports "0 features". Both fail,
but with contradictory root causes. The fix is also *documented as working*,
which is how it survived.

**Fix:** Track the pre-selection row count upstream rather than trusting
`df.height` after the columns are gone.

**Confidence:** 7/10

---

### OC-74
### 🟡 Medium — `NodeRegistry.list_models()` hides all 4 Ensemble models and its `category` argument is dead

**File:** `skyulf-core/skyulf/registry.py:101-108`

```python
return [
    node_id
    for node_id, metadata in cls.get_all_metadata().items()
    if metadata["category"] == "Modeling"
    and (category is None or metadata["category"] == category)
]
```

The hardcoded `metadata["category"] == "Modeling"` runs *before* the caller's
filter, so the `category` parameter can only ever match `"Modeling"` or return
nothing — and the four `"Ensemble"`-category models are invisible to this helper
entirely.

```text
categories: {'Modeling': 34, 'Ensemble': 4, 'Preprocessing': 30, 'Data Operations': 5,
             'Cleaning': 6, 'Feature Engineering': 9, 'Feature Selection': 5,
             'Inspection': 2, 'Text': 5}
Ensemble ids:              ['voting_classifier', 'stacking_classifier',
                            'voting_regressor', 'stacking_regressor']
list_models()           -> 34 ids
list_models("Ensemble") -> []          # dead parameter
[i for i in list_models() if 'voting' in i or 'stacking' in i] -> []
```

**Impact:** Any registry-driven UI or tooling that uses `list_models()` (rather
than `get_all_metadata()`) silently offers 34 of the 38 available models. The
shipped model dropdowns happen to use `get_all_metadata()` via `/registry`, so
this is latent today — but it is a trap for exactly the
generate-the-contract-from-the-registry direction this audit recommends (see
[R1](../opus_core_analysis.md#r1)). It is also the kind of asymmetry that made
OC-06 hard to pin down.

**Fix:**

```python
model_categories = {"Modeling", "Ensemble"}
return [
    node_id
    for node_id, metadata in cls.get_all_metadata().items()
    if metadata["category"] in model_categories
    and (category is None or metadata["category"] == category)
]
```

**Confidence:** 9/10 — behaviour executed and confirmed above.

---

## Engine-dispatch & schema table

| Concern | file:line | pandas | polars | Same? |
|---|---|---|---|---|
| `EngineRegistry.resolve()` module detection | `engines/registry.py:117-131` | `__module__` → `"pandas"` | `__module__` → `"polars"`, unwrap via `to_native()` | ✅ |
| `to_numpy()` on a normal frame | `pandas_engine.py:98-107` | `df.to_numpy()` | `df.to_numpy()` | ✅ |
| `to_numpy()` on a 0-column frame | `polars_engine.py:130-136` | `(n, 0)` | `(0, 0)` despite the shim | ❌ **OC-65** |
| `SplitDataset.copy()` dispatch | `data/dataset.py:29-35` | `.copy()` | correctly falls through to `.clone()` | ✅ |
| `SkyulfSchema.from_dataframe()` dtype extraction | `core/schema.py:82-89` | `.dtypes.items()` | falls back to `.schema.items()` | ✅ |
| `artifact_digest()` on object-dtype arrays | `pipeline/seal.py:57-59` | same bug | same bug | Shared correctness bug, not a parity gap — **OC-62** |
| `EngineRegistry._active_engine` mutation | `engines/registry.py:60,86-91` | unlocked global | unlocked global | Both racy — **OC-64** |

---

## Prior-finding re-verification

| Prior ID | Claim | Still fixed? | Evidence |
|---|---|---|---|
| **F-14** | Unscoped, unlocked global singletons | **❌ Partially — engines package NOT fixed** | `compute.py`/`serialization.py` now use `ContextVar`, but `engines/registry.py:60` is still a bare mutable class attribute. Race reproduced: 67/2000 cross-thread contaminations. **See OC-64.** |
| F-19 | `pipeline.py` mixes four responsibilities | ✅ Fixed | Now a package: `_pipeline.py` (orchestration), `seal.py` (digest), `diagram.py` (rendering); docstrings reference "Extracted from `pipeline.py` (F-19)" |
| F-07 | `._df` private unwrapping in 9 modules | ✅ Fixed | `git log` shows `e8c0cf50 Close F-07: public to_native() unwrap replaces private ._df reach-ins`; verified in `utils.py`, `_pipeline.py` |
| F-08 / F-09 / F-30 | — | Open (excluded from scope by instruction) | — |

---

## What I checked and found sound

- Read every in-scope file line-by-line: `core/` (schema, protocols, compute,
  model_registry, deprecation, serialization, warnings, artifacts,
  meta/decorators), `engines/` (protocol, registry, pandas_engine, polars_engine,
  sklearn_bridge), `data/` (dataset, catalog), `pipeline/` (_pipeline, seal,
  diagram), plus `registry.py`, `types.py`, `utils.py`, `_validation.py`,
  `config_validation.py`, `leakage.py`, `__init__.py`.
- `SkyulfSchema.rename()` duplicate-collision check, `assert_compatible()`
  missing/unexpected/dtype/order diffs, and `from_dataframe()`'s pandas→polars
  fallback — all traced by hand and by targeted scripts; no defects.
- `SplitDataset.copy()`'s `.copy()`-then-`.clone()` fallback order is correct:
  raw `pl.DataFrame`/`pl.Series` genuinely lack `.copy()`.
- `NodeRegistry.register()`/`get_all_metadata()`: `category` is a required
  non-optional field on `NodeMetadata` across all ~55 real registration sites, so
  `list_transformers()`/`list_models()`'s direct `metadata["category"]` indexing
  is safe in practice.
- `validate_leakage_safety()` and its four `is_explicit_*` predicates behave per
  docstring, including the fail-closed default for unregistered transformers.
- `EngineRegistry._detect_top_level_package()`'s unwrap-via-`to_native()`
  discriminator does not false-positive on third-party modules whose names merely
  contain "pandas"/"polars".
- `utils.py`'s `_pack_pandas_output`/`_pack_polars_output`, `resolve_columns` and
  `detect_numeric_columns` — including the NaN-vs-null polars distinction
  (`fill_nan(None)` correctly applied before `drop_nulls()` for float columns at
  both call sites). A contrived `dict`-as-`y` case raises a clear `AttributeError`
  rather than silently corrupting data.
- **Verified the package import graph is acyclic with clean layering**
  (`core→engines`, `data→engines`, `preprocessing→{core,registry,utils,engines,types,data}`,
  `modeling→{engines,types,core,registry,data}`, `pipeline→{modeling,preprocessing,…}`).
- Confirmed `_feed_canonical`'s complexity matches the brief: 17 `return`
  statements, 92 AST statement nodes.

---

## Improvement opportunities (not defects)

- `core/compute.py` (`ComputeBackend`), `core/serialization.py`
  (`ModelSerializer`), `core/model_registry.py` (`ModelRegistry`) and
  `core/warnings.py` (`SkyulfWarning`) are all exported from the top-level public
  API, but grep confirms **none is referenced anywhere outside its own defining
  module** — they are forward-looking seams for the Databricks/MLflow work. Fine
  as placeholders, but release notes should state they are inert so users don't
  assume wiring exists.
- `NodeRegistry.get_all_metadata()` returns `dict(cls._metadata)` — a *shallow*
  copy. Each per-node metadata dict is still the live shared object; a future
  caller mutating `metadata[node_id][...]` in place would corrupt registry state
  for the process lifetime. (The current `backend/` call site defensively does
  `dict(meta)` first.) Prefer `{k: dict(v) for k, v in cls._metadata.items()}`.
- `pipeline/diagram.py`'s `mermaid_escape()` only escapes `"`; a param value
  containing a literal newline would break the Mermaid label syntax. Cosmetic.
- Refactor `_feed_canonical` (C901 = 26, 67 statements, 17 returns) while fixing
  OC-62/OC-63 — its complexity is precisely why both bugs went unnoticed.
