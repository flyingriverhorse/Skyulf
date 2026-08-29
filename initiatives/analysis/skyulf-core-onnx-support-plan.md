# Plan — ONNX export & serving (core + backend + frontend)

**Status:** 🟨 planned (needs a go/no-go decision)
**Created:** 2026-08-29 · **Tracker:** [`skyulf-core-findings-tracker.md`](skyulf-core-findings-tracker.md)
**Touches:** skyulf-core, backend, frontend — full-stack by design
**Depends on:** nothing hard; pairs well with the
[joblib migration plan](skyulf-core-joblib-migration-plan.md) (ONNX is the
answer to "untrusted-input safety" that joblib deliberately is not)

---

## 1. Motivation

Today a trained model leaves Skyulf only as a Python pickle/joblib blob.
That means:

- **Serving requires Python.** Deployments (`backend/ml_pipeline/deployment/`)
  must run the same interpreter + sklearn versions that trained the model.
- **No language-neutral artifacts.** Consumers in Java/Rust/edge can't load
  our models.
- **Version fragility.** A sklearn bump can silently break loading of old
  artifacts (pickle binds classes by module path).

ONNX gives a portable, versioned graph artifact that an optimized runtime
(`onnxruntime`) can serve without Python-in-the-loop, and that other stacks
can consume. It is also the standard bridge format for sklearn models
(`skl2onnx`), XGBoost/LightGBM (`onnxmltools`), and text pipelines.

### Why careful planning matters

- **Not every estimator converts.** The registry has ~100 nodes; the model
  families that convert cleanly are a subset. Shipping a button that fails on
  half the models is worse than no button.
- **Preprocessing is the hard half.** Our preprocessing chain is a custom
  `FeatureEngineer`, not sklearn `Pipeline` — it has no automatic ONNX path.
  Phase 1 must scope to *model-only* export with a documented input contract.
- **Numerical parity is non-negotiable.** The F-15 seal contract
  ("same hash ⇒ same predictions") must hold across formats: a parity gate
  compares joblib-model predictions against the ONNX runtime before an export
  is declared successful.

---

## 2. Support matrix (feasibility, verified against converter docs)

| Model family | ONNX path | Phase |
|---|---|---|
| Linear/Logistic regression, Ridge/Lasso/ElasticNet | `skl2onnx` (mature) | 1 |
| Tree ensembles (RF, ExtraTrees, GBDT) | `skl2onnx` | 1 |
| XGBoost classifier/regressor | `onnxmltools.convert_xgboost` | 1 |
| LightGBM (if/when registered) | `onnxmltools.convert_lightgbm` | 2 |
| Naive Bayes (Gaussian/Multinomial/Bernoulli) | `skl2onnx` | 1 |
| SVC/SVR (RBF) | `skl2onnx` with `svm` ops (works, larger graphs) | 2 |
| KNN | `skl2onnx` (limited distance metrics) | 2 |
| Calibrated classifier | usually **not** convertible | — (report unsupported) |
| Voting/Stacking ensembles | partial (convert members, manual assembly) | 3 / research |
| Text vectorization (TfidfVectorizer + model) | `skl2onnx` text ops | 3 |

Preprocessing export (imputers/scalers/encoders as ONNX ops) is **out of
scope for phases 1-2** — the export contract is "numeric feature matrix in,
prediction out", with the expected column list recorded in the artifact's
metadata.

---

## 3. Design

### 3.1 Core — export seam, optional dependency

```
skyulf-core/skyulf/modeling/_export/
    __init__.py
    base.py        # ModelExporter protocol: can_export(model) / export(model, path)
    onnx.py        # OnnxExporter (skl2onnx / onnxmltools dispatch)
    _parity.py     # predict-parity check vs the in-memory model
```

- **Optional extra, not a hard dep:** `skyulf-core[onnx]` =
  `skl2onnx`, `onnxmltools`, `onnxruntime`. Import probes use
  `importlib.util.find_spec` (the F-28 pattern) — core without the extra
  raises a helpful `SkyulfOptionalDependencyError`, never crashes at import.
- **Registry-tag driven:** each model node's registry metadata gains
  `export_formats: ["joblib"]` → `["joblib", "onnx"]` where supported, so the
  backend/frontend ask *what a model can do* instead of maintaining a second
  hardcoded map (the F-10 lesson).
- **Parity gate:** `export()` writes the graph, runs a sample through
  `onnxruntime`, compares against `model.predict`/`predict_proba` with a
  tolerance, and **fails the export** (not warns) on mismatch. The exported
  artifact carries metadata: input schema (column names/dtypes), opset,
  converter versions, and the source pipeline `fingerprint()` — tying the
  ONNX file back to the F-15 seal.

### 3.2 Backend — artifact + serving

- **Artifact store:** on a successful job with `export_onnx: true` (node
  param, advanced tuning section), `_artifacts.py` writes `model.onnx`
  alongside the joblib artifact and records it in job metrics/artifact
  listing (`onnx_supported`, `onnx_path`).
- **Download endpoint:** extend the existing artifact-download router so the
  frontend can fetch the `.onnx` file (content-type
  `application/octet-stream`, same auth as other artifacts).
- **Deployment/serving (phase 3):** `deployment/service.py` gains an
  `ONNXRuntimeDeployment` option — load `.onnx` via `onnxruntime`, serve
  predictions without sklearn in the serving env. This is the payoff for
  version fragility; it is deliberately the last phase.

### 3.3 Frontend

- **Training node settings:** "Export ONNX after training" checkbox
  (Advanced mode; shown only when the selected model type declares onnx in
  its registry metadata — the schema-preview machinery already ships
  per-node capability data).
- **Experiments/Model Registry:** an artifact row shows `model.onnx` when
  present, with a download button; unsupported models show a muted
  "ONNX not available for this model type" hint (honest, not a silent gap).
- **Deployments page (phase 3):** runtime choice `Python (joblib)` vs
  `ONNX Runtime` when both artifacts exist.

---

## 4. Phases

| # | Phase | Deliverable | Est. |
|---|---|---|---|
| 0 | Spike | Convert one RF + one logistic regression to ONNX in a scratch script, measure parity, confirm dependency footprint | half day |
| 1 | Core seam | `ModelExporter` + `OnnxExporter` + parity gate + registry `export_formats`; optional extra; phase-1 families | 2 days |
| 2 | Backend wiring | `export_onnx` job param, artifact write + metrics, download endpoint | 1 day |
| 3 | Frontend | checkbox, registry/experiments artifact rows, capability hints | 1 day |
| 4 | Wider matrix | SVC/KNN/LightGBM families + text pipeline research | 2 days |
| 5 | ONNX serving | `ONNXRuntimeDeployment` in deployment service | 2-3 days |

**Phases 1-3 ≈ 1 week** of focused work, shippable incrementally (core with
tests is independently valuable; each phase has its own gate).

---

## 5. Risks

| Risk | Mitigation |
|---|---|
| Converter gaps surprise users | Registry metadata drives visibility — unsupported types never show the button |
| Numerical drift (converter versions) | Parity gate at export time; opset+converter versions recorded in artifact metadata |
| Dependency bloat in core | Optional extra + `find_spec` probes; CI installs the extra only in the export test job |
| Preprocessing expectations mismatch (user feeds raw columns) | Export metadata carries the exact input contract; docs + notebook cell showing the expected frame |
| F-15 seal confusion ("does the ONNX hash match?") | Seal stays Python-side (identity of the learned model); the ONNX artifact *references* the fingerprint rather than replacing it |
| skl2onnx API churn | Pin converter versions per release; single `_export/onnx.py` owner |

## 6. Decision points (need user call)

1. **Start with the phase-0 spike** before committing to the full plan? (recommended)
2. **Where does `export_onnx` live** — per-model node param (recommended)
   or a global job setting?
3. **Is ONNX serving (phase 5) actually wanted now**, or is portable-export +
   download the real need? This halves or doubles the effort.
4. **LightGBM:** not registered in the core today — do we add it (also closes
   an ecosystem gap) or keep the matrix sklearn+XGBoost?
