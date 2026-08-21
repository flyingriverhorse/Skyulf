# Backend Polars Migration — Plan

**Date:** 2026-08-11
**Branch:** `078` → `080`
**Status:** Phase 0 complete (F-07–F-14 shipped red-green on `080`, 2026-08-21). Phase 1a landed:
`SKYULF_ENGINE` setting exists — and by decision of 2026-08-21 it **defaults to `polars`**, not
pandas. This pulls the Phase 3 fixes (F-31/F-32/F-30/F-43, drift consumer, serving frame) forward
as mandatory before the default is safe for users; they ship in the same release.
**Companion:** [`2026-08-11-audit-findings.md`](2026-08-11-audit-findings.md), [`2026-08-11-leakage-enforcement-plan.md`](2026-08-11-leakage-enforcement-plan.md)

---

## 1. The gap

**Skyulf is presented as Polars-backed. The backend does not use Polars at all.**

Verified:

- `backend/data/catalog.py` reads **exclusively** through pandas — `pd.read_csv` at lines 90, 280,
  291 and `pd.read_parquet` at 95, 126, 281, 293, 297. There is no Polars reader anywhere in the
  catalog.
- `backend/ml_pipeline/_execution/engine/_node_runners.py:154-158` raises `TypeError` if a
  non-pandas frame reaches the engine. (This guard was added deliberately in `7485ade6` to turn
  silent degradation into a loud failure — it is correct, and it must be *removed as the last step*
  of this migration, not the first.)

So for every canvas user, the pipeline is pandas from ingestion to prediction. Polars is only
exercised by people importing `skyulf-core` directly as a Python library.

**Consequence.** This is a marketing/reality gap, and it also explains a pattern in the audit: the
Polars correctness bugs (F-01, F-04, F-06, F-07, …) do not affect canvas users today, because
Polars never runs there. Migrating the backend to Polars **activates every one of those bugs**.

> **Hard sequencing constraint:** the Polars correctness bugs are Tier 1/Tier 2 of the audit. This
> migration must not start until they are fixed and covered by NaN-aware parity tests. Switching
> the backend to Polars first would ship silent data corruption to every user.

---

## 2. Which nodes leave Polars, and why

The audit found **zero nodes that silently downgrade Polars to pandas** — every node returns a
Polars frame when given one. But internally, some convert. There are 31 `.to_pandas()` sites in
`skyulf-core`. They fall into three very different categories.

### Category A — `fit`-only statistics (11 nodes) · **not a data conversion, low priority**

These convert a **column subset** during `fit` purely to compute statistics with sklearn/scipy,
then return a plain `dict` artifact. **The data frame itself is never converted; `apply` stays pure
Polars.** Verified — every one of these call sites is inside a `def fit(...)`:

| Node | Site |
|---|---|
| IQR | `outliers/iqr.py:91` |
| ZScore | `outliers/zscore.py:96` |
| Winsorize | `outliers/winsorize.py:86` |
| EllipticEnvelope | `outliers/elliptic.py:110` |
| PowerTransformer | `transformations/power.py:136` |
| GeneralBinning | `bucketing.py:440, 485` |
| UnivariateSelection | `feature_selection/univariate.py:56` |
| ModelBasedSelection | `feature_selection/model_based.py:56` |
| FeatureInteraction | `feature_generation/interaction.py:169` |
| GeoDistance | `geo/distance.py:170` |
| H3Index (fit) | `geo/h3_index.py:127` |

Helpers: `_helpers.py:52` (`to_pandas`), `:73` (`resolve_columns_then_to_pandas`), `:119`
(`select_then_to_pandas`). These already narrow to the selected columns before converting, which
was a deliberate optimisation.

**Verdict:** mostly fine. The underlying math is sklearn/scipy-bound and needs numpy regardless.
The realistic improvement is `→ numpy` directly instead of `→ pandas → numpy`, which
`_helpers.py` already offers via `resolve_columns_then_to_numpy` / `select_then_to_numpy`.
Worth doing for wide frames; **not** a correctness issue. The codebase already carries 5
`TODO(pandas-removal)` markers acknowledging this (e.g. `outliers/iqr.py:84`,
`outliers/winsorize.py:84`).

### Category B — full round-trip in `apply` (**Polars → pandas → Polars**) · **the real target**

Here the actual user data is converted and converted back. This is where memory, time and dtype
fidelity are lost.

| Node(s) | Site | Native-Polars feasible? |
|---|---|---|
| **H3Index** | `geo/h3_index.py:75` — `_h3_index_apply_polars` does `X.to_pandas()` then `pl.from_pandas()` on the **whole frame** | **Yes, easily.** Only `lat_col`/`lon_col` are needed. Extract those two columns to numpy, compute H3 cells, attach with `with_columns`. No frame round-trip. |
| **EllipticEnvelope** | `outliers/elliptic.py:62` | **Yes.** The sklearn model only produces a boolean **mask**. Compute the mask from numpy, then `X.filter(mask)` natively. The round-trip exists only to reuse `_elliptic_filter_pandas`. |
| **TfidfVectorizer, CountVectorizer, HashingVectorizer, Tokenizer, SentenceEmbedder** (5 nodes) | `vectorization/_common.py:49` `apply_text_pandas_only` | **Partly.** sklearn vectorizers need `list[str]` input and emit a sparse matrix. But you can pull `list[str]` straight out of Polars and build the output with `pl.from_numpy`/`with_columns` — no pandas needed in between. |
| **Over/Under-sampling** | `resampling.py:115` | **No (keep pandas).** `imblearn` is pandas/numpy-bound. Low value — resampling is a training-only step. |
| **TrainTestSplitter** | `split.py:44` `_to_pandas_remember_engine` | **Yes.** It already *remembers* the engine and converts back, so the contract is right — but `train_test_split` on indices plus a native Polars slice avoids the round-trip entirely. |

**Also relevant, outside preprocessing:** `modeling/_evaluation/clustering.py:22`,
`profiling/visualizer.py:692`, `profiling/expect.py:43`, `utils.py:143-144, 316`,
`pipeline.py:69-74`, `preprocessing/pipeline.py:501-511`, `feature_selection/_common.py:144`,
`encoding/woe.py:185`.

### Category C — legitimate and permanent

The **model boundary**. `SklearnBridge` converting to numpy before `estimator.fit` is correct and
must stay — sklearn requires numpy. The audit proved this handoff is bit-identical across engines.
This is not something to "fix".

### Category D — dispatcher (not a conversion at all)

`dispatcher.py:95, 134, 159` — these `to_pandas()` calls sit on the **pandas branch** of an
`if engine == POLARS / else` split. When the engine is Polars they never execute. Harmless; do not
"optimise" them.

---

## 3. Migration plan

### Phase 0 — Prerequisite: fix Polars correctness first *(blocking)*

Ship audit **Tier 1 + Tier 2**. Specifically F-01 (DummyEncoder), F-04/F-05 (SimpleImputer),
F-06 (IQR/ZScore/ManualBounds), F-07 (OrdinalEncoder), F-08 (KBins), F-09 (wrapper), F-11
(HashEncoder), F-13 (drift NaN), plus the NaN-aware and wrapped-frame parity tests.

**Do not begin Phase 1 until this is done.** Everything below assumes Polars is correct.

### Phase 1 — Make the engine choice explicit and observable

1. ~~Add an explicit engine setting (config/env, e.g. `SKYULF_ENGINE=pandas|polars`), defaulting to
   **pandas**~~ **Done on `080` (2026-08-21), with the default overridden to `polars`** —
   `backend/config/mixins/core.py` declares `SKYULF_ENGINE: Literal["polars", "pandas"] = "polars"`
   (case/whitespace-normalized by a validator in `backend/config/base.py`; covered by
   `tests/unit/test_settings_engine.py`). The original plan deferred the polars default to Phase 5;
   the 2026-08-21 decision pulls it forward, which makes every Phase 3 fix a blocking item for this
   release.
2. Record the engine on the job record and in the deployment bundle (this is audit finding **F-25**
   — nothing currently detects "trained on Polars, served on pandas"). **Done on `080`
   (2026-08-21)** — deployment bundles and job `job_metadata` record the training engine.
3. Surface it in the UI (job details / experiments) so the engine in use is never a guess.
   **Done on `080` (2026-08-21)** — engine tile in job details, Polars badge on job cards.
4. Add engine-parity CI: run a representative pipeline set under both engines and assert identical
   artifacts and metrics. **Done on `080` (2026-08-21)** — `backend-tests.yml` now matrices the
   backend suite over `SKYULF_ENGINE=[polars, pandas]`, so neither engine can regress while the
   other stays green.

### Phase 2 — Polars ingestion behind the flag

**Done on `080` (2026-08-21)** — `backend/data/catalog.py` reads CSV/parquet natively with Polars
under `SKYULF_ENGINE=polars` (items 5–7 all landed), including the `NaN`-token parity coverage and
the engine-aware ingestion guard.

5. Add `pl.read_csv` / `pl.read_parquet` readers to `backend/data/catalog.py`, selected by the
   Phase 1 flag.
6. **Watch the CSV `NaN` token.** `pl.read_csv` produces float `NaN` where `pd.read_csv` produces
   `NaN` that `isna()` matches — this is the exact trigger for F-13 (drift silently reporting "no
   drift"). Ingestion parity tests must include a literal `NaN` token, empty fields, and mixed
   int/null columns.
7. Relax the `_node_runners.py:154` guard to accept Polars **only when the flag is on**, keeping
   the loud failure for the mismatch case.

### Phase 3 — Fix the backend's pandas-only assumptions

**Done on `080` (2026-08-21)** — every finding in the table below is fixed: F-12/F-11 shipped in
skyulf-core 0.6.0 (F-11's cross-engine hash parity makes the "serving builds a `pd.DataFrame`"
crossing harmless); F-31/F-32/F-30 fixed in the backend engine (`_artifacts.py`,
`_node_runners.py`, `deployment/service.py`); F-43 fixed in
`skyulf-core/skyulf/modeling/_evaluation/clustering.py` (Polars NaN is a valid value, not a null,
so float reference columns also filter `is_nan()`). The drift consumer
(`backend/monitoring/router.py`) accepts a Polars reference frame directly. Full backend suite
passes under both `SKYULF_ENGINE=polars` (default) and `SKYULF_ENGINE=pandas`.

The engine currently assumes pandas in several places the audit already documented. All must be
fixed before Polars can be enabled by default:

| Finding | What breaks | Location |
|---|---|---|
| **F-12** | `hasattr(train_frame, "select_dtypes")` silently skips the numeric filter → Polars clustering deployment is completely broken | `_node_runners.py:260` |
| **F-31** | `_normalize_train_frame` returns `None` → drift reference data never saved | `engine/_artifacts.py:154-178` |
| **F-32** | `_feature_names_for_importance` pandas-only → `feature_importances = None` | `engine/_artifacts.py:38-58` |
| **F-30** | `_pretty_dtype` pandas-only → deployment input schema shows `unknown` for Date/Duration/Categorical | `deployment/service.py:613-638` |
| **F-43** | Clustering reference crosstab invents a `"nan"` segment | `skyulf-core/skyulf/modeling/_evaluation/clustering.py` |
| **F-11** | HashEncoder buckets differ per engine, and serving always builds a `pd.DataFrame` | `encoding/hash.py:34` vs `:53` |

F-31/F-32 are currently rated LOW **only because they are unreachable**. This migration makes them
live and user-visible — the Experiments page has SHAP and Feature Importance tabs that render
exactly these artifacts (§6 of the findings). **Reclassify both to HIGH the moment Phase 2 lands.**

Also note `backend/monitoring/router.py:224` does `pl.from_pandas(ref_data)` — the drift consumer
currently *expects* pandas. It must accept a Polars reference frame directly.

### Phase 4 — Remove the Category B round-trips in `skyulf-core`

In value order: H3Index → EllipticEnvelope → TrainTestSplitter → the 5 vectorizers.
Leave resampling on pandas. Each must be verified with a before/after parity test proving output
values are unchanged.

*Done (2026-08-22)* — all four sites are Polars-native; resampling stays on pandas as planned.
Each site landed with a red-green dtype-preservation test (the round-trip silently upcast nullable
`Int64` → `Float64`) plus the existing value-parity suites, and the full core suite passes under
both `SKYULF_ENGINE` values.

### Phase 5 — Flip the default

*Default flip done early by decision of 2026-08-21* — `SKYULF_ENGINE=polars` is the default from
Phase 1a onward, with pandas kept supported and documented as a first-class option. The remaining
obligation stands: publish a benchmark showing the actual gain — otherwise the migration has no
evidence behind it.

*Benchmark published (2026-08-22)* — `skyulf-core/benchmarks/bench_roundtrip_removal.py` measures
the Phase 4 removals against a reconstruction of the old round-trip path; results are in
`docs/performance.md` (splitter 2.79x, elliptic envelope 3.47x, count vectorizer 1.13x).

---

## 4. Honest cost/benefit

**Real wins:** the Polars claim becomes true; genuine memory and speed gains on large CSV/Parquet
ingestion; the Category B round-trips (whole-frame `to_pandas()` + `from_pandas()`) disappear.

**Real costs:**
- The whole Tier 1/Tier 2 bug list must be fixed **first** — this is not optional.
- F-31/F-32 change from latent to live and become user-visible blank panels if missed.
- Polars is already a hard dependency (`setup.py:27`, `pipeline.py:12`), so no new dependency —
  but `polars_engine.py`'s `HAS_POLARS` fallback is unreachable dead code (**F-26**) and should be
  deleted or made real as part of this work.
- Dtype differences become live: Polars `Int64`-with-nulls `.to_numpy()` upcasts to float64;
  pandas nullable `Int64` crashes 4 nodes that Polars handles (**F-10**). Both directions need
  tests.

**What this plan does not do:** it does not touch the model boundary (Category C is correct as-is),
and it does not claim a performance number. Phase 5 must measure one.

---

## 5. Interim: fix the claim, not just the code

*Resolved (2026-08-22):* the backend now defaults to `SKYULF_ENGINE=polars` (Phase 1a decision,
2026-08-21) with Phases 2–3 landed, so the "canvas/backend runs on Polars" claim is accurate —
the README's dual-engine marketing line matches reality again. The discipline stands for any
future gap: fix the code so the claim becomes true, and in the meantime do not assert something
the code does not do.
