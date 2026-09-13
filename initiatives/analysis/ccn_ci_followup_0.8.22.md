# 0.8.22 CI follow-up — 2026-09-13

The three reported drift failures came from request counters shared between
tests. The original six-file coverage report also matched the saved pre-follow-up
coverage exactly. This change fixes test isolation and adds 35 Core regression
cases without modifying production code, rate limits or coverage exclusions.

## Rate-limit reproduction and repair

Running these files together reproduced the same three failures:

```powershell
.venv/Scripts/python.exe -m pytest tests/integration/test_ccn_release_pipeline.py tests/integration/test_drift_reference_space.py tests/integration/test_drift_target_columns.py -q
```

Result before repair: **3 failed, 26 passed, 4 xfailed**. The failing cases were
the two `include_target=False` drift-target cases and the missing/new-feature
case reported by CI.

The new pipeline tests make eight drift requests across four engine/search
combinations. The other drift tests use the same implicit client address,
`127.0.0.1`. Their requests accumulated in the shared SlowAPI limiter until the
20/minute budget was exhausted. Running a small subset alone did not reveal
that cross-test dependency.

`tests/conftest.py` now calls the limiter's public `reset()` before and after
each test. It does not disable throttling or change the production quota.
The added `test_drift_rate_limit_isolation.py` invokes the actual decorated
drift route twice in separate parameterized cases. Each case verifies:

- Twenty requests reach the route's expected missing-reference response.
- The twenty-first request from that address raises `RateLimitExceeded` before
  the route records another alert.
- Another client address still has its own budget.

Before the fixture change, the second case failed on its first request. After
the change, the combined group passes **31 tests**, with the four existing
OC-320 expected failures unchanged.

## Coverage on the original refactor patch

Measured executable lines added by `e1bc7b44..08dc1503`, using the saved baseline
and a fresh complete Core run after these tests. A partially covered branch
line is excluded from the "fully covered patch lines" percentage. Values below
are rounded locally; Codecov may truncate its displayed percentages.

| File under `skyulf-core/skyulf/` | Before: fully covered patch lines | After: fully covered patch lines | Executed patch lines | Covered patch branch exits |
|---|---:|---:|---:|---:|
| `modeling/_explainability/shap_explanation.py` | 23/34 (67.65%) | 33/34 (97.06%) | 34/34 | 15/16 |
| `preprocessing/casting.py` | 25/29 (86.21%) | 29/29 (100%) | 29/29 | 14/14 |
| `preprocessing/vectorization/_common.py` | 14/18 (77.78%) | 18/18 (100%) | 18/18 | 6/6 |
| `modeling/_tuning/engine.py` | 72/75 (96%) | 75/75 (100%) | 75/75 | 30/30 |
| `modeling/_tuning/refit.py` | 6/9 (66.67%) | 9/9 (100%) | 9/9 | 6/6 |
| `leakage.py` | 52/53 (98.11%) | 53/53 (100%) | 53/53 | 28/28 |

All **12 missing executable lines** are now exercised. Partial lines decreased
from **14 to 1**. Across these files, **218/218 changed executable lines** and
**99/100 changed branch exits** are covered.

The remaining SHAP partial is `_display_shap_samples`'s `resolved is None`
branch (line 324 → 339). The public path first accepts only valid 2D/3D SHAP
values; its per-row resolver returns a tuple for both accepted ranks. Thus this
guard has no reachable input through the current public computation path.
The guard remains in production, and no exclusion was added to hide it.

The new tests exercise observable behavior:

- SHAP: exact-tree additivity retries, legacy exceptions, failed retries,
  malformed optional interactions, signed aggregation and zero display limit.
- Casting: strict integral/fractional conversions, nullable integer width,
  row/label preservation and native/wrapped Polars behavior.
- Text vectorization: non-string fallback preserves spelling, counts, dtypes,
  engine/wrapper type, row order and targets.
- Tuning/refit: convergence callbacks, unrelated warning provenance and real
  halving holdout scores when named preprocessing frames are absent.
- Leakage: malformed casting maps remain conservative; first-split selection,
  warning order and warn/raise/ignore policies remain consistent.

## Final verification

| Check | Result |
|---|---|
| Full Core with branch coverage and the existing 90% gate | **8,885 passed, 80 skipped**, 3 snapshots; **97.42%** coverage |
| Full backend, pandas | **3,656 passed, 1 deselected, 4 xfailed**, 7 snapshots |
| Full backend, Polars | **3,656 passed, 1 deselected, 4 xfailed**, 7 snapshots |
| Ruff check/format on all five changed/added Python files | Passed |
| Configured global ty check | Passed |
| Git whitespace check | Passed |

Full runs used scikit-learn 1.9.1, imbalanced-learn 0.14.2, NumPy 1.26.4,
pandas 2.3.2 and Polars 1.44.1. Core and backend ran as separate pytest suites
with filesystem-based coverage sources, unique temporary/coverage paths and
Hugging Face offline flags.

The backend's one deselected test is the existing local-only
`test_full_inference_pipeline`, which attempts to delete artifacts in the
external `skyulf-mlflow` workspace. It is skipped on CI when that workspace is
absent; it was not modified or treated as inference evidence here. OC-246
remains open. All four xfails belong to the previously reproduced OC-320
serving defect; neither finding is fixed by this follow-up.

Ignored local evidence under `tmp_repro_artifacts/`:

- `ccn_rate_limit_red.log`, `ccn_rate_fixture_red.log`, `ccn_rate_limit_green.log`.
- `ci_0822_backend_pandas.log`, `ci_0822_backend_polars_final.log`.
- `ci_0822_core.log`, `ci_0822_core.coverage`, `ci_0822_core_coverage.xml`.
- `analyze_0822_patch_coverage.py` reads the coverage data and original patch;
  it does not alter source files or coverage exclusions.

Queue status remains **59 open / 4 parked**. These are test-only repairs; no
frontend changes or model retraining are required.
