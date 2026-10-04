# Clean Databricks acceptance, 2026-09-29

## Destination and scope

The user requested a clean area and comprehensive live verification of completed
Databricks work. Existing `workspace.skyulf_lifecycle_test` objects were preserved.
Catalog creation using Default Storage was rejected by Databricks with a UI-only
instruction. Created the empty schema **workspace.skyulf_validation_20260929**.
Workspace payloads and experiment folders use the same new name.

The user explicitly approved the wheel, notebooks and 14 serverless tasks.
Run: **161612155854062**. Payload: `rehearsals/clean_validation_20260929/`.
Wheel: `e8700764c7879cb1bfb3b1c53ca57875abdefdb0851f685993a262d4585d980a`.
All 281 installed Python module hashes were checked by the search matrix tasks.

## Observed results

| Task | Result |
| --- | --- |
| 34 models x grid | 34 passed, 1,726 trials |
| 34 models x random | 34 passed, 101 trials |
| 34 models x Optuna | 34 passed, 102 trials |
| 34 models x halving-grid | 34 passed, 1,726 trials |
| 34 models x halving-random | 34 passed, 101 trials |
| Optuna settings | 30 passed |
| Halving settings | 30 passed |
| Ensemble settings | 110 passed |
| CV/metrics settings | Initial 80 passed/4 failed; corrected-wheel rerun: **84 passed** |
| Nested temporal/group/threshold, pandas | 8 passed; MLflow replay verified |
| Nested temporal/group/threshold, Polars | 8 passed; MLflow replay verified |
| Real UC lifecycle and Delta | Passed; v1 approval, v2 replacement, rollback, replay, failure preservation |
| Four-branch project pipeline | Passed; distinct recipes, three output modes, UPDATE/DELETE rebuild, next insert |
| Broad contract and real Delta pytest | **516 passed, 4 CLI-only skips**; all 20 real Delta tests passed |

The four skipped tests require a local Databricks CLI opt-in. Ran those exact
four branch-template generation tests locally with the `skyulf` profile:
**4 passed**. No Spark/Delta test was skipped in the cloud contract suite.

The four branches are pandas regression, Polars ensemble regression, Polars
classification and pandas ensemble classification. Named `frequency_only`,
`imputer_only`, `combined` preprocessing and independent pre-split selections run
from the current template. The test breaks editable feature code after packaging
and still approves/scores the saved set. Each component remains without a champion
alias while the set is approved as one unit. All/combined-only/separate-view targets
are real Delta outputs. The source UPDATE/DELETE recomputes the same set, removes
deleted keys and changes all consumer views; a later insert with an unseen category
appends successfully. This verifies notebook entrypoints, not a newly deployed
persistent Bundle scheduler or time-trigger behavior.

## Defect found and verified on the corrected wheel

Explicit halving `n_samples` maxima were reused unchanged inside nested searches.
For example, an outer training population of 64 rows inherited max_resources=96;
sklearn eventually tried to sample more rows than the inner fold contained.
The four failures cover halving-grid/random x regression/classification.

Added a small shared search-population bound before building a halving searcher:
numeric maxima are ceilings capped to the current search population; explicit
minima still fail if impossible. Tree-count resources are unaffected. Requested
configurations remain unchanged for other folds and the final search.

Four new regression cases reproduced the same sampling failure before the fix.
After the fix, **176 nested/halving/tuning-engine tests passed**. Full Ruff,
formatting, Ty and production CCN <= 10 passed.

Corrected wheel: `d814b3a22ea0964ac3bcd6157e3ab3ab12eebc7715e7a0bfc550bf3adcddcd07`.
Five focused cloud tasks run under the same destination's `r2` subdirectory.
Automatic review initially rejected the changed payload; the user explicitly
approved it and run **475680264133492** started. On the corrected wheel,
**84 CV/metrics settings**, **30 halving settings**, **34 halving-random models**
and **34 halving-grid models** passed. The regression task passed 172 tests but
failed four `g_score` cases because its test environment omitted imbalanced-learn.
Run **807427751644166** added imbalanced-learn 0.14.1 and exposed a transitive
dependency incompatibility: imblearn's `sensitivity_specificity_support` could
not unpack `_check_targets` results. Run **476456648913994** explicitly installed
sklearn 1.8.0 but reproduced the same four failures, ruling out the initial
hypothesis that the Databricks base sklearn implementation alone caused it.

Downloaded **sklearn-compat 0.1.6** into an isolated local directory. Merely putting
that directory before the otherwise passing local environment reproduced the
exact unpacking exception in a direct `geometric_mean_score` call. Existing
sklearn-compat **0.1.5** passes. Added that pin to root dependencies, both requirements
files, Core's imbalanced/all extras and uv.lock. No Core algorithm change for this
second failure. The 176 local regressions, full Ruff/format/Ty passed again.

Run **769662790160519** repeats the approved r2 wheel and 176 tests with sklearn
1.8.0, imbalanced-learn 0.14.1 and sklearn-compat 0.1.5: **SUCCESS, 176 passed**,
31 warnings, 49.83 seconds. Task run: **1093734795023109**. The failed g_score
cases and the nested-halving regression all pass on Databricks.
The dependency metadata changes are local; this run provisions the exact pin
through its environment, without replacing the already-approved r2 wheel.
Initial overall runs remain FAILED; individual passing task results above and
corrected reruns are the acceptance evidence.

The local dependency declarations and uv.lock now agree on the compatibility pin.
Full Ruff, formatting, Ty and production Lizard CCN <= 10 passed after these edits.
No full monorepo pytest or MkDocs rerun; no frontend changes. These are bounded
model/strategy and representative settings tests, not every possible configuration.

## Reading the registry test set

`sm36c_20260929_r1_set` is an MLflow PyFunc package of exact model components and
business rules. Its v1/v2 are release versions of that package, not the versions
of two different algorithms. Both releases use `sm36c_20260929_r1_left` v1 and
`sm36c_20260929_r1_right` v1. The combined rule changes between the two releases.
The lifecycle test approves v1, replaces it with v2, then rolls back to v1.
Consequently the set's champion is v1 at completion. Component aliases in this
particular test are deliberate setup and are asserted unchanged throughout;
the four-branch project test separately verifies components without champion aliases.

The user requested a signed delivery commit after acceptance. No push requested.
Rehearsal artifacts are excluded from commits.

Pre-commit verification: **322 passed, 14 skipped** in the combined model-set and
nested/halving/tuning suite (144.24 seconds). Skips are local Spark/Delta and
Windows symlink capability gates; real Delta execution is recorded above.
The initial local invocation could not access the OS pytest temp directory;
the passing rerun used a fresh temporary directory inside the repository.
Full Ruff/format and staged hooks passed, including full Ty and Lizard CCN 10.
Frontend hooks correctly skipped because no frontend source changed.
