# Plan — Pipeline persistence: pickle → joblib

**Status:** 🟨 planned (needs a go/no-go decision)
**Created:** 2026-08-29 · **Tracker:** [`skyulf-core-findings-tracker.md`](skyulf-core-findings-tracker.md)
**Touches:** skyulf-core (primary), backend (small), frontend (none)

---

## 1. Motivation

`SkyulfPipeline.save()` / `SkyulfPipeline.load()` (`skyulf-core/skyulf/pipeline/_pipeline.py:510-520`)
use raw `pickle.dump` / `pickle.load`. Everywhere else the stack already
standardized on joblib:

- Backend artifact stores (`backend/ml_pipeline/artifacts/local.py`, `s3.py`)
  are joblib-only (`.joblib` extension, `joblib.dump/load`).
- The F-14 serializer seam (`skyulf-core/skyulf/core/serialization.py`)
  already defaults to `JoblibModelSerializer` and was designed exactly so a
  format change doesn't touch call sites.
- The notebook export cells teach `pipeline.save(...)` / `SkyulfPipeline.load(...)`
  as the user-facing persistence API (`_notebook_builders.py:257,321-327`).

So the migration is about making the **one remaining pickle site** consistent
with the rest of the stack, not about replacing a pickle-heavy codebase.

### What this migration is NOT

Be honest about the limits — joblib is pickle-protocol internally:

- **Not a security fix.** `joblib.load` executes the same pickle protocol.
  The existing nosec waivers ("trusted artifacts written by this same process")
  remain accurate under joblib. Real untrusted-input safety requires a
  non-pickle format — that is the ONNX plan
  ([`skyulf-core-onnx-support-plan.md`](skyulf-core-onnx-support-plan.md)),
  not this one.
- **Not a fingerprint change.** The F-15 reproducibility digest
  (`skyulf/pipeline/seal.py::artifact_digest`) is content-addressed over
  topology + weights + hyperparameters; it never looked at serialization
  bytes. Switching formats changes **zero** fingerprints.

### What it buys

1. **One serializer seam.** `save`/`load` route through `ModelSerializer`,
   so the MLflow/cloud-serializer phase (the reason the seam exists) covers
   whole pipelines, not just bare models.
2. **Efficiency for big artifacts.** joblib compresses large numpy arrays
   (mmap + optional zlib) — a pipeline carrying a big fitted estimator or
   cached arrays serializes noticeably better.
3. **Consistency.** One format word across the stack: `.joblib` artifacts,
   joblib models, joblib pipelines.
4. **Version metadata.** A small envelope (`{"skyulf": version, "kind":
   "pipeline", "payload": ...}`) enables forward-compat decisions instead of
   "does this pickle happen to load".

---

## 2. Current state (verified 2026-08-29)

| Site | Format | Notes |
|---|---|---|
| `SkyulfPipeline.save/load` | **raw pickle** | the migration target |
| `backend/ml_pipeline/artifacts/local.py` / `s3.py` | joblib | models + node artifacts |
| `backend/ml_pipeline/deployment/service.py` | resolves `.joblib`/`.pkl` URIs | deployment path |
| `core/serialization.py` seam | `JoblibModelSerializer` default | ContextVar-scoped (F-14) |
| Notebook export cells | teach `pipeline.save("skyulf_pipeline.pkl")` | API surface stays identical |
| F-15 seal (`fingerprint()`) | content-addressed | unaffected |

Only one grep hit for raw pickle in `skyulf-core/skyulf` outside the two
pipeline methods: none. The blast radius is genuinely small.

---

## 3. Design

### 3.1 Route through the existing seam

```python
# skyulf/pipeline/_pipeline.py (sketch)
def save(self, path: str) -> None:
    get_model_serializer().dump(self, path)

@classmethod
def load(cls, path: str) -> "SkyulfPipeline":
    obj = get_model_serializer().load(path)
    if not isinstance(obj, SkyulfPipeline):
        raise TypeError(...)
    return obj
```

The seam's `dump(model, path)` signature already fits; a whole pipeline is
just a bigger object to the serializer.

### 3.2 Backward compatibility — load old pickle files forever-ish

Users and notebooks have `.pkl` pipelines on disk. Load path:

1. Try the active serializer (joblib).
2. On protocol/format failure, fall back to legacy `pickle.load` with a
   logged deprecation warning.
3. Save path always writes the new format.

No mass-migration tooling needed — read-your-old-files, write-the-new-one.

### 3.3 Format envelope (optional, recommended)

Wrap the payload so loaders can reason about versions:

```python
{"format": "skyulf-pipeline", "schema_version": 1, "skyulf_version": "0.8.7",
 "payload": <SkyulfPipeline>}
```

Keeps `SkyulfPipeline.load` the only place that knows about the envelope.
Skip this if we want the absolute-minimum change; it is cheap insurance.

### 3.4 What stays unchanged

- Public API: `save(path)` / `load(path)` signatures.
- `fingerprint()` / seal semantics.
- Backend artifact stores (already joblib; pipelines don't flow through them).
- Frontend — zero impact.

---

## 4. Phases

| # | Phase | Work | Est. |
|---|---|---|---|
| 1 | Core swap | `save`/`load` → seam; legacy-pickle read fallback + deprecation log; envelope decision | half day |
| 2 | Tests | round-trip (new→new), legacy pickle → new loader, serializer-swap via `model_serializer()` context, fingerprint stability across formats | half day |
| 3 | Docs + notebook cells | `serialization.md` user guide; notebook cell wording ("joblib-backed" instead of "pickles"); keep `skyulf_pipeline.pkl` filename in examples or rename to `.joblib` | 2 h |
| 4 | Gates | core suite + backend unit + ruff/ty; changelog entry (next release) | — |

**Total: ~1.5 days** including docs. Small enough to fold into any branch.

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Old `.pkl` files in user notebooks | Legacy-read fallback (3.2); deprecation warning only |
| Envelope breaks third-party `pickle.load` of our files | Envelope is opt-in; document the format |
| joblib version skew between writer/reader | joblib artifacts carry the same pickle-protocol constraints as today — no regression; `skyulf_version` in envelope helps diagnose |
| Tests pinning pickle bytes | None found — tests pin behavior, not bytes |

## 6. Decision points (need user call)

1. **Envelope or bare joblib dump?** (recommend envelope)
2. **Keep `.pkl` example filenames in notebooks** or rename to `.joblib`?
3. **Ship in 0.8.8 or hold for the MLflow phase** (the seam exists for it —
   doing both together avoids two changelog entries)?
