# PR196: release version alignment and CI repair

Branch: `092`; base: `master`; pull request: https://github.com/flyingriverhorse/Skyulf/pull/196.

The requested publication includes the completed SM-23 implementation, earlier
SM-57/58 work, Databricks/MLflow module organization and the project landing page.
Commit `cbbc43cf` published the pending project changes. This follow-up repairs
the first PR checks and aligns backend, frontend and standalone Core at **0.9.2**.
No merge or package release is part of this request.

## Findings and corrections

| Finding | Cause and correction |
| --- | --- |
| Core collection fails before executing tests | Four Spark test filenames collide with integration filenames under pytest's existing import mode. Add `skyulf-core/tests/spark/__init__.py`; preserve global import mode and coverage. Eight explicit files collect 168 nodes with exact before/after identity parity. |
| Dependency vulnerability scan fails | Locked fsspec, Mako and Werkzeug versions have published advisories. Raise the matching direct/optional dependency floors and regenerate the lock: fsspec/s3fs 2026.9.0, Mako 1.4.3, Werkzeug 3.1.9. Preserve scanner rules and ignores. |
| Backend reports a development version in a checkout | The root uv project has no installed distribution metadata. Prefer installed metadata when present, then read the root manifest relative to the module. Preserve the explicit `APP_VERSION` environment override and development fallback for missing/invalid metadata. |
| Five tuning error-translation tests no longer inject failures | The test replaces `HalvingGridSearchCV`, while production constructs `_CoverageHalvingGridSearchCV`. Update the test replacement to the actual constructor; preserve all five expected errors and production behavior. |
| Backend Polars job fails at `GeneralBinning` | Polars 2 rejects direct categorical-to-integer casts. Supply explicit numeric labels to `cut`, then decode their text into `Int64`. Preserve fitted edges, nulls, out-of-range values and custom labels independently of physical category codes. |
| Runtime assertions vanish under optimized Python | Benchmark prediction validation, required KS probability and required revisit policy now raise explicit `ValueError` with actionable messages. Validation order and valid-path values remain unchanged. |
| Codacy flags safe serialization, generated SQL and public imports | Document narrowly scoped rule exceptions only after tracing each path. Pickle is used solely for byte sizing, never deserialization; SQL names are validated and quoted, SQL values are typed or generated, and one canonical query is compared rather than executed. Two imports are intentional compatibility exports. No global exclusion or raised threshold. |

## Verification evidence

- The frozen dependency export used by CI scans **293 packages, zero vulnerabilities**.
- Version regression tests: four reproduced failures, then **15 passed**; separate
  collection confirms 15 nodes. Fresh backend settings, frontend package metadata,
  installed editable Core and root manifest all report **0.9.2**.
- Dependency-consumer batch: 151 passed and five tuning fixture failures reproduced.
  After correcting that fixture, the complete explicit tuning test file reports
  **126 passed**. S3 connector/cache and MLflow tracking consumers passed unchanged.
- Runtime guards: **73 passed** across seven explicit affected files; four focused
  guard cases also pass under `python -O`. Review confirms unchanged SQL/serialization
  semantics and preserved public exports.
- Frontend version synchronization/check, production build and all **12** size
  budgets pass. Existing lint/complexity verification remains applicable because
  no frontend source code changed in this follow-up.
- Binning: 20 added regression cases reproduce ten failures on Polars 2.0.0.
  After the fix, **90 affected/consumer tests pass on both Polars 2.0.0 and
  1.44.1**. The original backend `test_all_transformers` also passes on 2.0.0.
  Boundary/null/empty/sparse batches and custom label precedence are covered.

- Full CI Ruff/format/Ty scopes and Lizard CCN 10 pass on the final repair tree.
  The explicit KS probability helper keeps the existing complexity limit; no
  threshold was raised. Six affected drift/guard cases and two optimized guards
  pass after that extraction.
- The built **0.9.2** wheel contains **519 Python source files**, byte-identical to
  the checkout. SHA-256:
  `2ce014a713ea5bf0f8f2ada37497b50f191a0b4c23f7f2cf978ef4ff451483dc`.

At the repair commit, native verification of this new wheel is blocked before
upload or job submission: the selected `skyulf` profile reports an invalid OAuth
refresh token. The user explicitly deferred this native test. Resuming it will
require `databricks auth login --profile skyulf`. The prepared check covers six Spark
drift cases, two delayed-window cases, two runtime guards and compatibility
exports without writing shared tables or changing existing jobs. No fresh native
success is claimed. Remote PR acceptance is tracked on the exact pushed head;
the earlier failed run is not reused as evidence for these repairs.

Local diagnostic output lives in ignored `tmp_repro_artifacts/pr092/`; no test
data, model artifacts, personal settings or credentials are included in commits.
The original SM-23 native rollout remains documented in
[delivery177](177-sm23b-online-monitoring.md); validation of this follow-up is
separate from that earlier deployed 0.9.1 wheel.
