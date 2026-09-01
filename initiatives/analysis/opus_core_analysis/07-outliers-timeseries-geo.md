# Skyulf Deep Audit (Opus) — Outliers, casting, bucketing, resampling, time series & geo

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

**Scope:** `skyulf-core/skyulf/preprocessing/` — `outliers/`, `casting.py` (429 lines), `bucketing.py` (539 lines), `resampling.py` (377 lines), `time_series/`, `geo/`, `inspection.py` (153 lines). All read in full; every finding proven with executed dual-engine repros.

---

## Findings

### OC-58
### 🔴 Critical — Numeric→boolean casting silently reinterprets any nonzero value as `True` on polars

**File:** `skyulf-core/skyulf/preprocessing/casting.py:143-178`
(pandas comparison: `:218-232`, `:322-330`)

`_build_polars_cast_exprs` special-cases only **string**→bool casts. A
**numeric** source column cast to `"bool"` falls through to the generic
`pl.col(col).cast(pl.Boolean, strict=not coerce_on_error)`. Polars' numeric→
Boolean cast is C-style truthiness (`x != 0`) and **never raises, even with
`strict=True`**.

Pandas' `_cast_bool`/`_coerce_boolean_value` treats anything other than literal
`0`/`1` as ambiguous — masking to `<NA>` when coercing, and **raising** when not.

```text
config = {"type_map": {"x": "bool"}, "coerce_on_error": True}    # x = [1, 0, 2, NaN]
pandas: {'x': [True, False, None, None]}   BooleanDtype   # 2 -> null  (ambiguous)
polars: {'x': [True, False, True, None]}   Boolean        # 2 -> True  (WRONG)

config = {"type_map": {"x": "bool"}, "coerce_on_error": False}   # strict mode
pandas: raised TypeError "Need to pass bool-like values"          # correctly rejects
polars: {'x': [True, False, True, None]}                          # silently succeeds

# direct polars cast, strict=True:
x = [1.0, 0.0, 2.0, -1.0, 0.5, null]  ->  [true, false, true, true, true, null]
```

**Impact:** This is the most dangerous finding in the audit alongside
[OC-12](./02-encoding-cleaning-imputation-scaling.md). `coerce_on_error`
defaults to `True` and **is not exposed in the UI**, so every user hits the
default path. Any pipeline casting a numeric column to boolean on polars turns
every nonzero value — negative numbers, fractions, and out-of-range sentinels
like `-1` or `999` — into `True`. The identical config on pandas nulls those
same values out. A boolean flag feature can therefore carry *opposite* semantics
depending only on which engine ran, with no warning or error on either side.
Strict mode, the one mechanism a user has to demand loudness, is silently
inert on polars.

**Fix:** Route numeric-source boolean casts through the same validation as int
casts: build `pl.when(src == 0).then(False).when(src == 1).then(True).otherwise(None)`
mirroring `_coerce_boolean_value`, and when `coerce_on_error=False`, check for
any value outside `{0, 1, null}` and raise, mirroring
`_validate_polars_string_bool_casts`.

**Confidence:** 10/10 — reproduced on both engines, both `coerce_on_error` settings.

---

### OC-59
### 🟠 High — `DatasetProfile` numeric-column coverage is completely different between engines

**File:** `skyulf-core/skyulf/preprocessing/inspection.py:44-58` (polars) vs `:60-71` (pandas)

`_profile_fit_polars` hand-lists **4** dtypes (`Float64, Float32, Int64, Int32`)
to decide which columns get `numeric_stats` — omitting `Int8/Int16/UInt8/UInt16/
UInt32/UInt64` — and applies no other filtering. `_profile_fit_pandas` instead
calls `detect_numeric_columns(X)`, which additionally **excludes binary (0/1-only)
and constant columns**.

The two paths share no logic, even though the codebase already has a canonical,
dtype-complete helper for exactly this (`auto_detect_numeric_columns` /
`POLARS_NUMERIC_DTYPES` in `engines/polars_engine.py:33`) that neither uses.

```text
df: a=int8[1..5], b=binary int64[1,0,1,0,1], c=constant int64[7]*5, d=float64
PANDAS numeric_stats keys: ['a', 'd']       # int8 kept, binary/constant excluded
POLARS numeric_stats keys: ['b', 'c', 'd']  # int8 DROPPED, binary/constant INCLUDED
```

The two column sets are **almost entirely disjoint**.

**Impact:** The same dataset yields a profile artifact describing a different set
of columns per engine. An `Int8`/`UInt*` column — very common immediately after
an upstream `Casting` node — simply vanishes from the polars profile with no
indication anything was skipped.

**Fix:** Have `_profile_fit_polars` use the existing
`auto_detect_numeric_columns`/`POLARS_NUMERIC_DTYPES`, and either replicate the
exclude-binary/exclude-constant filtering or explicitly document the decision not
to, so both paths select the same logical column set.

**Confidence:** 9/10

---

### OC-60
### 🟠 High — `GeneralBinning`'s `missing_strategy: "label"` is a silent no-op on polars

**File:** `skyulf-core/skyulf/preprocessing/bucketing.py:134-142` vs `:197-231`

`missing_strategy`/`missing_label` are read and applied **only** in the pandas
path (`_apply_missing_strategy`, called from `_bin_one_column_pandas`). The
polars path (`_bucketing_apply_polars` → `_build_polars_exprs` →
`_polars_one_col_expr`) never reads `missing_strategy` at all, so nulls and
out-of-range values are left as `null` — the `"keep"` behaviour — regardless of
configuration.

```text
config: {"strategy":"equal_width","n_bins":3,
         "missing_strategy":"label","missing_label":"MISSING_TAG"}
x = [1.0, 2.0, NaN, 4.0, 100.0]

pandas: x_binned = [0.0, 0.0, "MISSING_TAG", 0.0, 2.0]   dtype -> object
polars: x_binned = [0,   0,   null,          0,   2]      dtype stays u32
```

**Impact:** A pipeline configured to tag missing/out-of-range values with a
sentinel produces that label on pandas but plain nulls on polars. Downstream
logic branching on the label string, or expecting a non-null flag, behaves
completely differently per engine. Not currently reachable from `BinningNode.tsx`
(no UI control exists), but it is a first-class documented backend config key
that will misbehave the moment it is wired up or set via direct API JSON.

**Fix:** Implement `"label"` (and any other non-`"keep"` strategy) in
`_polars_one_col_expr`, including widening the ordinal/bin-index dtype to string
when a label is injected, to match pandas' `astype(object)` widening.

**Confidence:** 9/10

---

### OC-61
### 🟡 Medium — `BinningNode`'s "Precision (Decimals)" UI field is never sent to the backend

**Files:** `frontend/ml-canvas/src/modules/nodes/processing/BinningNode.tsx:19,163-166`
vs `frontend/ml-canvas/src/core/utils/pipelineConverter.ts:334-345`

`BinningConfig.precision` is a real, stateful UI control (shown when
`label_format === 'range'`, default 3, range 0-10) stored in `node.data.precision`.
The converter's `BinningNode` branch builds `params` as an **explicit object
literal that omits `precision`** — unlike sibling branches (`outlier`,
`ResamplingNode`, `TimeSeriesNode`) which forward `node.data` wholesale. The
backend's `precision` key (`bucketing.py:222`) is real and functional; it is
simply unreachable.

```ts
// BinningNode.tsx:164-165
value={config.precision ?? 3}
onChange={(e) => onChange({ ...config, precision: parseInt(e.target.value) || 0 })}

// pipelineConverter.ts:336-345 — no `precision` key:
params = { columns, strategy, n_bins, label_format, output_suffix,
           drop_original, custom_bins, custom_labels };
```
```console
$ grep -c precision frontend/ml-canvas/src/core/utils/pipelineConverter.ts
0
```

**Impact:** Changing "Precision" from 3 to 0 or 6 to control interval-label
rounding is accepted by the UI but always executes with the backend default.
Structurally identical to OC-15 and OC-13.

**Fix:** Add `precision: node.data.precision` to the `BinningNode` params object.

**Confidence:** 10/10

---

> **Merged, not re-filed:** this agent independently confirmed that `GeoDistance`
> and `H3Index` have **zero** frontend representation — no `*Node.tsx`, no palette
> entry in `core/registry/init.ts`, no converter branch
> (`grep -rln "GeoDistance\|H3Index" frontend/ml-canvas/src/` → no output).
> Already filed as **[OC-06](./01-cross-cutting.md)**.

---

## pandas-vs-polars parity table

| Operation | Same result? | Notes |
|---|---|---|
| IQR / Z-Score / Winsorize / ManualBounds / EllipticEnvelope apply | ✅ | Mask-building and NaN preservation symmetric |
| Casting: fractional float → int (coerce) | ✅ | `[1.5,-2.5,3.0,NaN]→int32` gives `[None,None,3,None]` on both |
| Casting: out-of-range int8 (coerce / strict) | ✅ equivalent | Both mask when coercing and raise when strict; exception *type* differs (`OverflowError` vs polars `InvalidOperationError`) |
| **Casting: numeric → bool** | ❌ | **OC-58** — `2.0`→`None` (pandas) vs `True` (polars); strict mode doesn't raise on polars |
| Casting: string → bool | ✅ | Same alias table both engines; `["yes","no","maybe",None]` identical |
| Casting: empty frame / all-null column → int | ✅ | |
| Resampling (SMOTE) balance + row values | ✅ | Seeded run: identical 35/35 class counts and identical sorted feature values |
| `LagFeatures` incl. null `group_by` key | ✅ | `dropna=False` deliberately matches polars' `.over()` null-group semantics; documented in code |
| `RollingAggregate` (mean/std/sum) incl. null group | ✅ | Verified including `std` ddof |
| `DateFeatures` — all 12 features | ✅ | Verified across leap year, year boundary, invalid string, null |
| `GeneralBinning` core cut logic | ✅ | Out-of-range→NaN and int64-vs-uint32 dtype already known (OC-04) |
| **`GeneralBinning` `missing_strategy:"label"`** | ❌ | **OC-60** |
| **`DatasetProfile` numeric_stats coverage** | ❌ | **OC-59** |
| Geo haversine / euclidean | ✅ | Formulas algebraically identical; pure math, no engine-specific handling |

---

## Frontend/backend option parity table

| Node | Mismatch |
|---|---|
| Outlier (`IQR`/`ZScore`/`Winsorize`/`EllipticEnvelope`) | None. `ManualBounds` exists in backend but has no `method` option in the UI dropdown (dead, not a mismatch) |
| `GeneralBinning`/`CustomBinning`/`KBinsDiscretizer` | **`precision` dropped by converter (OC-61)**. `missing_strategy`, `missing_label`, `include_lowest`, `duplicates`, `column_strategies`, `kbins_*` are backend-only unused capacity |
| `Casting` | `coerce_on_error` not exposed (defaults `True` server-side) — note this is what makes OC-58 unavoidable for UI users |
| `Oversampling`/`Undersampling` | None — fully in sync (all method and tuning param names match 1:1) |
| `LagFeatures`/`RollingAggregate`/`DateFeatures` | None — fully in sync |
| `GeoDistance`/`H3Index` | Entire node unreachable (OC-06) |

---

## What I checked and found sound

- **Outlier nodes** (5 files + `_common.py`, read in full): mask-building,
  NaN-preservation and `y`-filtering wrapper logic verified symmetric between
  engines; `OutlierNode.tsx` field names match backend `config.get(...)` exactly.
- **Casting** (429 lines, read in full): executed dual-engine repros for
  fractional→int coercion, out-of-range int8 coercion *and* strict-mode raise,
  empty-frame cast, all-null-column cast, and the string→bool alias table — all
  consistent except OC-58.
- **Resampling** (377 lines): ran a seeded SMOTE oversampling end-to-end through
  both engines (`fit` → `apply`); identical class balance and identical sorted
  feature values. Frontend method/param names cross-checked 1:1 against
  `_import_over_samplers`/`_import_under_samplers` — no mismatches.
- **Time series**: executed parity tests over 12 calendar features across
  leap-year, year-boundary, invalid-string and null dates (all identical), and
  `LagFeatures`/`RollingAggregate` with a `None` group key mixed among named
  groups (all identical).
- **Geo**: haversine/euclidean algebraically equivalent; the H3 path correctly
  isolates only the two coordinate columns to numpy, preserving unrelated dtypes;
  invalid-coordinate handling identical on both engines.
- **Bucketing** (539 lines): all five strategies' edge-fitting and label
  formatting verified; the two already-known issues (out-of-range→NaN
  consistency, int64-vs-uint32 dtype) confirmed present and unregressed.
- **Inspection** (153 lines): `DataSnapshot` head-N passthrough consistent;
  missing-count logic (`isna().sum()` vs `null_count()`) consistent.
- Confirmed none of the still-open F-08 / F-09 / F-30 recur in a new form here.

---

## Improvement opportunities (not defects)

- `ManualBounds`, `CustomBinning` and `KBinsDiscretizer` are reachable only
  indirectly or not at all — surface them or remove the dead paths.
- `Casting`'s strict mode (`coerce_on_error=False`) and its alternate
  `{columns, target_type}` config shape are backend-only. Given OC-58, a strict
  toggle in the UI would have real safety value.
- `GeneralBinning`'s `missing_strategy`, `duplicates`, `column_strategies` and
  `kbins_strategy` are fully implemented with no UI surface — worth exposing,
  especially since fixing OC-60 needs a control anyway.
- Unify `inspection.py`'s two ad-hoc numeric-dtype filters on the shared
  `auto_detect_numeric_columns`/`POLARS_NUMERIC_DTYPES` helper, both to fix OC-59
  and to stop a *third* divergent definition of "numeric" from appearing later.
