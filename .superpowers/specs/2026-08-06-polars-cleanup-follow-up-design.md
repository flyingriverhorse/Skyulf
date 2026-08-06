# Polars Cleanup Follow-up Design

## Goal

Complete the remaining verified low-risk Polars-first and duplication cleanups
in `skyulf-core` without changing public APIs, supported dataframe inputs, or
model/preprocessing results.

## Scope

1. `preprocessing/feature_generation/polynomial.py`
   - Keep Pandas and Polars inputs supported.
   - Narrow to the configured feature columns and pass numpy directly to
     sklearn `PolynomialFeatures` during apply, matching the existing fit path.

2. `preprocessing/vectorization/_common.py`
   - Resolve valid text columns against the native input frame first.
   - Convert only those columns to Pandas for CountVectorizer and
     TfidfVectorizer, which still require their existing Pandas text assembly.

3. `preprocessing/transformations/general.py`
   - For Polars input, pass a reshaped numpy column directly to
     `PowerTransformer.fit()` and perform the Box-Cox positivity check on the
     same numpy array.
   - Preserve the existing Pandas branch and its behavior.

4. `preprocessing/encoding/_common.py`, `woe.py`, and `target.py`
   - Add one shared target-extraction helper that reads an explicit `y` or
     extracts `target_col` from either Polars or Pandas input.
   - Replace WOE's and TargetEncoder's duplicate extraction implementations.
   - Use the existing `select_then_to_pandas()` helper in WOE fit paths instead
     of maintaining separate manual Polars and Pandas narrowing logic.

## Compatibility and Error Handling

- Existing public function and calculator signatures remain unchanged.
- Pandas callers remain supported; this work removes internal Pandas hops only
  where downstream sklearn code consumes numpy.
- Existing missing-column, missing-target, dtype, and sklearn exceptions remain
  visible through the current error paths.
- No broad coercion abstraction or unrelated preprocessing refactor is included.

## Testing

- Add or adjust focused tests to cover Polars and Pandas inputs for each changed
  path.
- Assert that vectorizer fit returns only the resolved text columns.
- Assert polynomial apply and PowerTransformer fit preserve output values and
  schemas.
- Assert WOE and TargetEncoder preserve explicit-`y` and `target_col`
  extraction behavior for both engines.
- Run targeted preprocessing tests, Ruff, type checking, and the full
  `skyulf-core` test suite.

## Success Criteria

- The five verified findings are removed without output changes.
- No full-frame conversion remains in the affected narrow-column paths.
- Target extraction has one shared implementation.
- The existing full-suite baseline remains green.
