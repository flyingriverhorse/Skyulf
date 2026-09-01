# Skyulf Deep Audit (Opus) — Shared helpers & dtype coverage

> Part of [`opus_core_analysis`](./README.md). Severity: 🔴 Critical · 🟠 High · 🟡 Medium · ⚪ Low. Finding IDs use the `OC-` prefix.

**Scope:** `skyulf/preprocessing/_helpers.py`, `_artifacts.py`, `_schema.py`,
`fold_adapter.py`, and the shared dtype-detection helpers in `skyulf/utils.py` /
`skyulf/engines/`. These modules sit under every preprocessing node but were in
no audit agent's assigned scope. Audited directly by the lead auditor.

**Theme:** the dual-engine dtype allow-lists are *hardcoded enumerations*, so any
dtype not explicitly listed is silently invisible. Two real dtypes fall through.

---

## Findings

### OC-120
### 🟠 High — `Decimal` columns are silently skipped by every auto-numeric node, and crash pandas when selected explicitly

**Files:** `skyulf/engines/__init__.py` (`POLARS_NUMERIC_DTYPES`),
`skyulf/utils.py:300-316` (`detect_numeric_columns`),
`skyulf/preprocessing/_helpers.py:160-166` (`auto_detect_numeric_columns`)

`POLARS_NUMERIC_DTYPES` is a hardcoded frozenset of the 10 int/float types.
`Decimal` is not in it. On the pandas side a decimal column arrives as `object`
dtype, which `select_dtypes(include=["number"])` also excludes.

```text
POLARS_NUMERIC_DTYPES:
  frozenset({Float32, Float64, Int16, Int8, Int32, Int64, UInt8, UInt16, UInt32, UInt64})

polars dtypes: {'price': Decimal(precision=10, scale=2), 'qty': Int64}
polars detect_numeric_columns -> ['qty']      # price missing
pandas dtypes: {'price': dtype('O'),          'qty': dtype('int64')}
pandas detect_numeric_columns -> ['qty']      # price missing
```

**Consequence 1 — silent half-processing.** With an empty `columns` config —
which the UI contract defines as "apply to all numeric columns" — the decimal
column passes through completely untouched, with no error and no warning:

```text
config: {}   (empty = scale ALL numeric columns)
fitted columns: ['qty']

INPUT                          OUTPUT
┌───────────────┬─────┐        ┌───────────────┬───────────┐
│ price         ┆ qty │        │ price         ┆ qty       │
│ decimal[10,2] ┆ i64 │        │ decimal[10,2] ┆ f64       │
╞═══════════════╪═════╡        ╞═══════════════╪═══════════╡
│ 100.00        ┆ 1   │        │ 100.00        ┆ -1.341641 │
│ 200.00        ┆ 2   │        │ 200.00        ┆ -0.447214 │
│ 300.00        ┆ 3   │        │ 300.00        ┆ 0.447214  │
│ 400.00        ┆ 4   │        │ 400.00        ┆ 1.341641  │
└───────────────┴─────┘        └───────────────┴───────────┘
```

**Consequence 2 — engines disagree when the user works around it.** The natural
next step for a user who notices the column wasn't scaled is to select it
explicitly. That works on polars and **crashes on pandas**:

```text
config: {'columns': ['price']}

POLARS explicit:  ✅ scales correctly to f64
  ┌───────────┐
  │ price     │
  │ f64       │
  │ -1.341641 │ …

PANDAS explicit:  ❌ TypeError: unsupported operand type(s) for -:
                     'decimal.Decimal' and 'float'
                     at scaling/standard.py:89  vals = vals - np.array(mean)[col_indices]
```

**That the polars path succeeds proves the operation is well-defined for
decimals.** The auto-detect exclusion is an oversight, not a deliberate safety
choice.

**Reachability — this is not exotic.** `backend/data/catalog.py:104` reads
uploaded files with `pl.read_parquet` (`:106` is the pandas branch), and Parquet
round-trips the decimal logical type exactly:

```text
parquet round-trip dtype: [Decimal(precision=10, scale=2)]
```

Decimal is the standard warehouse type for currency, so **any uploaded Parquet
with a money column hits this.** SQL `NUMERIC`/`DECIMAL` via a database connector
is the same story.

**Blast radius:** every node that auto-selects numeric columns — all scalers
(`scaling/_common.py:10,18`), outlier detectors (`outliers/elliptic.py` and
siblings), `transformations/power.py:136`, `feature_selection/{variance,
correlation,_common}.py`, `feature_generation/polynomial.py:105`, and
`preprocessing/inspection.py:68`.

**Fix:** Add `pl.Decimal` to `POLARS_NUMERIC_DTYPES` (it needs an `isinstance`
check, since `Decimal(10,2)` is a parameterized instance, not a bare class), and
in the pandas branch detect object columns whose values are `decimal.Decimal`.
Then cast to float64 at the numeric-conversion boundary so the pandas arithmetic
path stops crashing. Add a cross-engine test with a decimal column.

**Confidence:** 9/10 — every step above executed and reproduced.

---

### OC-121
### ⚪ Low — polars `Enum` columns are invisible to text auto-detection, diverging from the pandas `Categorical` equivalent

**File:** `skyulf/preprocessing/_helpers.py:148-157` (`auto_detect_text_columns`)

The polars branch tests `t in [pl.Utf8, pl.Categorical, pl.Object]`. `pl.Enum` —
polars' fixed-category string type — is absent. The pandas branch uses
`select_dtypes(include=["object", "string", "category"])`, which **does** catch
the equivalent `pd.Categorical`.

```text
polars dtypes: {'utf8': String, 'enum': Enum(categories=['x','y']),
                'cat': Categorical, 'dec': Decimal(10,2), 'i64': Int64}
  polars text -> ['utf8', 'cat']              # enum missing
  pandas text -> ['utf8', 'enum', 'cat']      # all three found
```

End-to-end, the same node with the same config produces different output per
engine:

```text
TextCleaning, operations=[{'op':'case','mode':'lower'}]

POLARS:  plain='aa','bb'   enu='XX','YY'   <- enum untouched
PANDAS:  plain='aa','bb'   enu='xx','yy'   <- categorical lowercased
```

Affects `cleaning/text.py:199` and `cleaning/alias.py:139`.

**Reachability is low, and I want to be straight about that.** Nothing in
`skyulf` constructs a `pl.Enum` (grep confirms zero occurrences), the `Casting`
node's `type_map` has no enum option, and neither CSV nor Parquet ingestion
produces one by default. It is reachable only by an SDK caller passing a
hand-built polars frame. Filed as Low for that reason — but it is the *same root
cause* as OC-120 (hardcoded dtype allow-lists that miss newer polars types), and
should be fixed in the same pass.

**Fix:** Add `pl.Enum` to the list, using `isinstance(t, pl.Enum)` since `Enum`
is parameterized by its category list.

**Confidence:** 9/10

---

### OC-122
### ⚪ Low — `TextCleaning` silently ignores an unrecognised operation name

**File:** `skyulf/preprocessing/cleaning/text.py:151-153`

```python
handler = _TEXT_OPS_POLARS.get(op.get("op", ""))
if handler is not None:
    expr = handler(expr, op)
```

An unknown `op` is dropped with no error, no warning, and no log line:

```text
operations=[{'op': 'nonsense_xyz'}]  ->  frame returned completely unchanged
```

The node's `@node_meta` declares only `params: {'columns': [], 'operations': []}`
— giving no indication of the valid op names, or even that `operations` is a list
of *dicts* rather than a list of strings. A caller who reasonably guesses
`{'op': 'lowercase'}` (instead of the real `{'op': 'case', 'mode': 'lower'}`)
gets a silent no-op.

**I verified the shipped frontend is not affected:** `TextCleaningNode.tsx:13`
types `op` as `'trim' | 'case' | 'remove_special' | 'regex'` and all 14 of its
mode values match the backend exactly (table below). This is a robustness and
SDK-ergonomics gap, not a live contract break — but it is precisely the shape of
[OC-19](./02-encoding-cleaning-imputation-scaling.md#oc-19) and
[OC-60](./07-outliers-timeseries-geo.md#oc-60), which *were* live.

**Fix:** Raise on an unrecognised `op`, and enrich the `@node_meta` params to
describe the operation schema — which is exactly what
[R1](../opus_core_analysis.md#r1) requires anyway. This node is a good worked
example of why R1's `@node_meta` enrichment must describe nested structures, not
just flat scalars.

**Confidence:** 9/10

---

## `TextCleaning` cross-engine mode parity — all clean

Every op/mode combination the UI can emit, run on both engines over the same
input (leading/trailing spaces, an embedded date, a double space, non-ASCII, an
empty string and a null):

| op | mode | Parity |
|---|---|---|
| `case` | lower / upper / title / sentence | ✅ ✅ ✅ ✅ |
| `trim` | both / leading / trailing | ✅ ✅ ✅ |
| `remove_special` | keep_alphanumeric / keep_alphanumeric_space / letters_only / digits_only | ✅ ✅ ✅ ✅ |
| `regex` | collapse_whitespace / extract_digits / normalize_slash_dates | ✅ ✅ ✅ |

**14 / 14 match**, including null and empty-string handling. The polars
`str.extract` vs pandas regex paths and the `map_elements`-based date
normalisation agree exactly. Only the *dtype-detection* layer (OC-121) diverges,
never the transformation logic.

---

## What I checked and found sound

- **`safe_scale`'s in-place mutation is safe.** Its docstring claims "callers
  always pass a slice/copy". Verified: all three call sites
  (`scaling/maxabs.py:51`, `standard.py:91`, `robust.py:66`) pass
  `np.array(scale)[col_indices].copy()`. No caller can have sklearn's internal
  `scale_` mutated out from under it.
- **`POLARS_NUMERIC_DTYPES` correctly excludes `Boolean`**, matching pandas'
  `select_dtypes(include=["number"])`, which also excludes `bool`. A real
  cross-engine trap, correctly avoided.
- **`auto_detect_datetime_columns`** handles parameterized `Datetime` instances
  via a belt-and-braces `t in [pl.Date, pl.Datetime] or isinstance(t, pl.Datetime)`,
  so timezone-aware columns are caught. It excludes `Duration`/`Time`, matching
  pandas excluding `timedelta64` from `include=["datetime"]`.
- **`resolve_valid_columns` dedupes order-preservingly** via `dict.fromkeys`,
  deliberately preventing the polars `DuplicateError` that a repeated column name
  in `.select()` would raise (pandas silently duplicated instead). The comment
  documents the divergence it exists to paper over — good practice.
- **`resolve_columns_then_to_numpy` handles nullable extension dtypes.**
  `Int64`/`Float64` masked arrays `to_numpy()` as object arrays full of `pd.NA`,
  which crash sklearn; the code forces `dtype="float64", na_value=np.nan` to match
  what polars produces natively. This is prior finding F-10, correctly fixed.
- **The `_helpers.py` / `dispatcher.py` boundary is documented and observed** —
  the module docstring states the split (dispatcher owns control flow, helpers own
  leaf utilities) and no helper dispatches a full node.

---

## Improvement opportunities (not defects)

- The three dtype detectors in `_helpers.py` duplicate the dtype allow-lists that
  also exist in `utils.py`'s `detect_numeric_columns` and in
  `engines/POLARS_NUMERIC_DTYPES`. Three copies of the same knowledge is why
  OC-120 and OC-121 could each go unnoticed. Consolidate to one source.
- Prefer `isinstance`-based dtype predicates over `in [list_of_classes]`
  membership throughout. Membership silently fails for every *parameterized*
  polars dtype (`Decimal(p,s)`, `Enum([...])`, `Datetime(tu,tz)`, `List(inner)`),
  which is the direct mechanism behind both findings above.
