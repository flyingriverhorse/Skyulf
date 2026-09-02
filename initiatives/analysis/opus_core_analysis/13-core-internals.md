# 13 — Core internals (`skyulf/core/`, `skyulf/engines/registry.py`)

**Scope:** all 11 modules of `skyulf/core/` (1,395 lines) plus the engine
registry that `core` coordinates with.

New ids continue at **OC-91** (OC-90 is used by [12-splitters](./12-splitters.md)).

| ID | Severity | Issue | Location |
| --- | --- | --- | --- |
| [OC-91](#oc-91) | 🟡 Medium | Three public `core/` seams (263 lines) have zero call sites, and one duplicates a differently-shaped backend class of the same name | `core/deprecation.py`, `core/model_registry.py`, `core/serialization.py` |

Plus a re-verification of [OC-64](./06-core-engines-pipeline.md#oc-64) with a
narrower blast radius and a much cheaper fix than filed — see
[below](#oc-64-re-verified-merged-not-re-filed).

---

<a id="oc-91"></a>
### OC-91
### 🟡 Medium — Three public `core/` seams are unused; one collides with a backend class name

**Files:** `core/deprecation.py` (89), `core/model_registry.py` (83),
`core/serialization.py` (91) — 263 lines, all exported from `core/__init__.py`.

Each module describes itself as an *"Additive, non-breaking seam ahead of the
Databricks/MLflow phases"*. Building seams ahead of need is defensible, so this
is Medium, not High. But a whole-repo search finds **no call sites at all** for
any of them:

```bash
grep -rn "@deprecated\|warn_deprecated("        skyulf-core/skyulf backend/   # (none)
grep -rn "get_model_serializer\|set_model_serializer\|model_serializer(" ...   # (none)
grep -rn "InMemoryModelRegistry"                skyulf-core/skyulf backend/   # (none)
```

Three specific consequences:

**1. The deprecation policy is declared but never applied.** `deprecation.py`'s
docstring states a concrete, sensible policy — *"a public symbol marked
deprecated in minor `X.Y` keeps working through `X.(Y+1)` and may be removed in
`X.(Y+2)`"*. Not one symbol in the codebase uses `@deprecated` or
`warn_deprecated`. Meanwhile the audit found several symbols that *should* be
deprecated — the dead `infer_output_schema` overrides
([OC-10](../opus_core_analysis.md#oc-10)), the unreachable node aliases
([OC-07](../opus_core_analysis.md#oc-07)), and the dead search-space dicts
([OC-100](./14-hyperparameters.md#oc-100)). The tool exists; it just is not used.

**2. `ModelVersion` means two different things.** `core/model_registry.py`
defines a `ModelVersion` dataclass (`name`, `version`, `model`, `metadata`),
while the backend independently defines its own `ModelVersion` in
`backend/ml_pipeline/model_registry/schemas.py`, consumed by a completely
separate `ModelRegistryService`. Two unrelated classes, same name, same domain
concept, different shapes. This is the same defect class already filed as
[OC-08](../opus_core_analysis.md#oc-08) (`DatasetProfile` meaning two different
things) — which suggests a naming-collision pattern rather than a one-off.

**3. `serialization.py`'s docstring overstates its role.** It says the default
serializer *"preserves today's joblib behaviour"* and *"matches current backend
behaviour"*, which reads as though call sites route through the seam. They do
not — the backend calls joblib directly. A maintainer installing a custom
serializer via `set_model_serializer()` would see **no effect whatsoever**.

**Fix:** either wire the seams in (route backend model persistence through
`get_model_serializer()`; start using `@deprecated` for the symbols above), or
mark them explicitly as reserved-for-future API so nobody assumes they are
load-bearing. At minimum, rename one of the two `ModelVersion` classes.

---

<a id="oc-64-re-verified-merged-not-re-filed"></a>
### OC-64 re-verified — *merged, not re-filed*

> **Merged, not re-filed:** already filed at 🟠 High as
> [OC-64](./06-core-engines-pipeline.md#oc-64) ("F-14 only partially fixed — the
> engine registry global is still an unlocked race"). Two corrections to the
> write-up follow.

**Confirmed.** `EngineRegistry._active_engine` (`engines/registry.py:60`) is a
plain mutable class attribute, and `set_active_engine` assigns it directly
(`:86-91`) with no lock, no thread-local, and no context scoping. In a threaded
server one request calling `set_active_engine("polars")` changes the engine for
every other concurrent request.

**Correction 1 — the blast radius is narrower than the finding implies.**
`_active_engine` is only consulted on two of `resolve()`'s three paths
(`:104-114`):

```python
if data is None:
    return cls.get(cls._active_engine)          # consulted
top_level = cls._detect_top_level_package(data)
engine_name = cls._TOP_LEVEL_TO_ENGINE.get(top_level)
if engine_name is not None and engine_name in cls._engines:
    return cls.get(engine_name)                 # NOT consulted — detected from data
cls._warn_unknown_data_type(data)
return cls.get(cls._active_engine)              # consulted (unknown-type fallback)
```

Whenever real pandas/polars data is passed, the engine is auto-detected from the
data and the global is irrelevant. So corruption is confined to calls made
*without* data, or with an unrecognised type. That is a real but bounded
exposure — worth reflecting in the fix priority.

**Correction 2 — the fix is far cheaper than the finding suggests, because the
correct pattern already exists three files away.** `core/serialization.py` solves
exactly this problem properly, and even documents why:

> *"The active serializer is held in a `contextvars.ContextVar`: a change is
> visible to the current thread/asyncio task and anything spawned from it
> afterwards, but **concurrent pipelines in other contexts cannot reconfigure
> each other mid-run**."*

with a token-resetting context manager:

```python
_DEFAULT_SERIALIZER: ContextVar[ModelSerializer] = ContextVar(...)

@contextmanager
def model_serializer(serializer):
    token = _DEFAULT_SERIALIZER.set(serializer)
    try: yield serializer
    finally: _DEFAULT_SERIALIZER.reset(token)
```

Porting that ~10-line pattern to `_active_engine` (plus an `active_engine()`
context manager) resolves OC-64 and closes F-14 properly. The irony worth noting
in the fix ticket: the seam that would have prevented this bug is one of the
unused ones filed above as [OC-91](#oc-91).

---

## Checked and found sound

- **`InMemoryModelRegistry` locking is correct.** `register()` holds
  `self._lock` across the whole read-modify-write of `_store[name]`, so two
  concurrent registrations of the same name cannot both read the same
  `len(versions)` and collide on a version number. The inline comment states
  precisely this invariant. Reads (`get`, `versions`) are lock-free, which is
  safe here: `versions()` returns a copy (`list(...)`), and appends to a Python
  list are atomic under the GIL, so a reader sees either the old or new tail,
  never a torn state.
- **`core/serialization.py` is exemplary.** See above — correct `ContextVar`
  usage, token-based restore, and a docstring that explains the concurrency
  reasoning. This is the model the rest of the codebase's global state should
  follow.
- **`_detect_top_level_package` is carefully written.** It compares the *top-level*
  module component rather than doing a substring match, explicitly to avoid
  false positives from third-party modules like `fake_polars_stub` or
  `my_pandas_wrapper`. It unwraps Skyulf's own engine wrappers only via the
  public `to_native()` discriminator, deliberately avoiding polars' private
  `._df`. Both decisions are documented and correct.
- **`core/deprecation.py` is well-implemented** (even if unused). The message
  builder handles every combination of optional `since`/`removed_in`/
  `replacement`, `functools.wraps` preserves the wrapped signature, and the name
  lookup uses `getattr(func, "__qualname__", getattr(func, "__name__", "callable"))`
  — correctly defensive for callables that are not plain functions.
- **`@node_meta` (`core/meta/decorators.py`)** makes `learns_from_data` a
  keyword-only *required* argument, so omitting it is a decoration-time error
  rather than a silent opt-out of the leakage gate. That is the right default
  for a safety-relevant flag.

## Improvements

- **Sweep for other unscoped mutable globals.** `_active_engine` was found by
  F-14 and is still open; a targeted search for module- and class-level mutable
  state that outlives a request would confirm whether it is the last one.
- **Adopt `@deprecated` for the symbols the audit already identified** as dead
  or superseded — it converts several "dead code" findings into a managed
  removal path at almost no cost.
