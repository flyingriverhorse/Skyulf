# Native Polars Correlation-Threshold Fitting Design

## Context

`CorrelationThresholdCalculator.fit()` currently calls `to_pandas(X)` for
every input, then computes `DataFrame.corr(method=...).abs()` and derives the
upper-triangle drop list. The apply path is already dual-engine: it drops the
precomputed list natively for Pandas and Polars.

This is an evidence-gated Wave 2 candidate from
`temp/skyulf-core-pandas-polars-audit-2026-08-05.md`. Skyulf remains
dual-engine. The goal is to remove an avoidable fit-time Polars-to-Pandas
conversion without changing the public artifact, accepted configuration, or
Pandas behavior.

The frontend already sends the matching
`correlation_method: "pearson" | "spearman" | "kendall"` configuration key
from `FeatureSelectionNode.tsx`; no cross-layer contract change is needed.

## Decisions

The approved design is:

1. Migrate only fit-time raw and wrapped Polars inputs. Do not change the
   Pandas fit path, the artifact shape, or the existing dual-engine apply
   path.
2. Use native Polars pairwise correlation for `pearson` and `spearman`.
3. Preserve the current Pandas compatibility route for `kendall`, callable
   methods, unsupported selected dtypes, generic frame types, and unavailable
   native capability.
4. Retain the declared `polars>=1.36.0` dependency floor. Do not add a
   dependency upgrade solely for this candidate; native capability must be
   checked before choosing the native route.
5. Add a concise source comment beside the Pandas fallback explaining that it
   should be revisited if Polars gains native Kendall or callable-correlation
   support.
6. Repair native automatic numeric-column detection so floating `NaN` has the
   same missing-value meaning as it has in the legacy Pandas route.

## Evidence

- PyPI reported Polars 1.43.2 as the latest release on 2026-08-06; the active
  project environment used 1.40.1.
- `pl.corr` supports `method="pearson"` and `method="spearman"`. It does not
  accept Kendall or arbitrary callables. `DataFrame.corr()` is not the chosen
  route because it is Pearson-only and delegates to NumPy.
- Polars' default NaN behavior differs from Pandas. A jointly filtered pair
  (`not null` and `not NaN` in both columns) reproduced Pandas'
  pairwise-complete Pearson and Spearman values in clean, tied, null, NaN,
  Boolean, constant, sparse-pair, and 100 randomized fixtures.
- The existing Polars numeric selector calls `drop_nulls()` but leaves float
  `NaN` values in the candidate set. Pandas `dropna()` removes both. This
  changes automatic selection for a one-finite-value-plus-NaN column and for
  binary or constant columns with NaNs. Normalizing float NaNs to null before
  the current checks matched the legacy Pandas selection in all targeted
  probes.
- An exploratory 100k x 50, 5%-NaN Pearson fit probe measured 0.080 seconds
  for the native pairwise Polars prototype versus 0.169 seconds for the
  current raw-Polars-to-Pandas route. This is directional evidence only; it
  is not the final promotion benchmark.

Reference APIs:

- https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.corr.html
- https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.corr.html

## Architecture

### Compatibility branch

Factor the current body into an explicit Pandas compatibility helper. It
continues to:

1. convert through `to_pandas(X)`;
2. resolve candidate columns;
3. call Pandas `corr(method=...)`;
4. use the existing upper-triangle comparison; and
5. return the unchanged artifact.

The public Pandas route therefore has no behavioral change.

For native-capable raw `pl.DataFrame` and `SkyulfPolarsWrapper` inputs, use a
small module-local normalizer analogous to the established
`profiling.expect._as_polars()` pattern. It exposes the underlying raw
Polars frame without calling `to_pandas()`.

Native eligibility is determined before computation. It requires:

- an available Polars correlation API with the required method capability;
- `correlation_method` equal to `pearson` or `spearman`; and
- selected columns that can retain legacy numeric/Boolean correlation
  semantics after a Float64 cast.

Any input outside that predicate goes directly to the compatibility helper.
There is no catch-all "try native, then silently fall back" behavior.

### Native correlation calculation

For eligible inputs:

1. Resolve columns against the native frame. An explicit `columns` list keeps
   its current order. Automatic selection uses the repaired native numeric
   detector.
2. Return the existing empty artifact when fewer than two columns remain.
3. Build an internal Polars frame containing only the selected columns cast
   to `Float64`. This lets numeric and explicitly selected Boolean columns
   share the same valid-row and correlation semantics.
4. Generate one aggregate expression for every ordered upper-triangle pair,
   preserving the current candidate-column order. For each pair, retain only
   rows where both cast columns are non-null and non-NaN, then calculate
   `pl.corr(..., method=method).abs()`.
5. Evaluate all pair expressions in one Polars `select`. Use ordinal internal
   aliases rather than user column names, avoiding alias collisions.
6. Walk pair results in the same upper-triangle order as the existing Pandas
   matrix. A right-hand column is dropped if any left-hand correlation is
   strictly greater than `threshold`.

Null or NaN correlation results never exceed the threshold, matching Pandas'
comparison behavior for constant or too-sparse pairs. Threshold equality
continues to keep the column because the comparison is strictly `>`.

The native path returns the current artifact exactly:

```python
{
    "type": "correlation_threshold",
    "columns_to_drop": [...],
    "threshold": threshold,
    "method": method,
    "drop_columns": drop_columns,
}
```

### Numeric-selection parity repair

`utils._polars_column_excluded()` currently treats only Polars nulls as
missing. For floating series, it will normalize NaN to null before
`drop_nulls()`, binary detection, and unique-value counting. Non-floating
series retain the existing path.

This small shared correction is in scope because the new native fit path
would otherwise choose different automatic candidates than the current
Polars-to-Pandas implementation. It is not a general selector redesign.

## Error and Compatibility Contract

- `kendall`, callables, invalid method strings, and unsupported selected
  dtypes use the existing Pandas route, preserving its error behavior.
- Missing columns, too few resolved columns, `drop_columns`, artifact keys,
  column order, and apply output behavior remain unchanged.
- The native path must not alter the frontend enum or backend pipeline
  payloads.
- The fallback receives a short source comment such as: "Retain this
  compatibility route until Polars supports Kendall and callable
  correlations." It is an implementation note, not a user-visible warning.

## Verification

### Contract tests

Add focused tests for Pandas, raw Polars, and `SkyulfPolarsWrapper` inputs:

1. The audit fixture's exact Pearson artifact, apply schema/order, and strict
   `threshold=1.0` boundary.
2. Pearson and Spearman parity with ties, nulls, float NaNs, Boolean columns,
   constants, and sparse pairwise-complete observations.
3. Automatic-column parity when float NaNs make a column empty, binary, or
   constant after missing values are removed.
4. No native route conversion: patch the local conversion helper to fail and
   show that eligible raw and wrapped Polars fits still succeed.
5. Compatibility fallback for Kendall, callables, invalid methods, and
   unsupported selected types, including the established Pandas error text.

Run the focused feature-selection and utility suites, then the repository's
required Ruff, formatting, and Ty checks for the touched Python files.

### Promotion benchmark

Extend the existing opt-in `pytest-benchmark` convention and use isolated
RSS measurement without adding a dependency. Compare the retained legacy
helper against the native helper for raw and wrapped Polars inputs at:

- 100k x 50;
- 1M x 20; and
- 50k x 500.

Each frame has a 5% missing-value rate and a correlated block. Measure
fit time and peak process memory after frame construction.

Promote the native implementation only if it achieves at least one of:

- 25% lower peak memory; or
- 20% lower fit time,

without a Pandas regression or any artifact, candidate-selection,
schema/order, supported-method, or error-contract divergence. Otherwise
retain the current all-Pandas fit implementation and record that result in
the audit.

## Non-Goals

- Removing Pandas as a public engine.
- Supporting native Kendall or arbitrary callable correlation before Polars
  provides an equivalent API.
- Changing frontend controls, backend payloads, artifact schemas, or
  apply-time behavior.
- Using `DataFrame.corr()` as a superficial replacement for the conversion.
- Raising the package's Polars minimum version for this migration alone.

## Rollback

The explicit Pandas helper remains the rollback path. If contract tests or
the promotion benchmark fail, remove or disable only the new native branch;
do not weaken the public contract to make the migration pass.
