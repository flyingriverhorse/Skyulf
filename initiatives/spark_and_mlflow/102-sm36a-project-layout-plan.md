# SM-36a: project source layout and packaged feature recipes

## Agreed scope

Organize generated src files by responsibility and make custom pre-split and
preprocessing code work across relative imports. Preserve existing single-file
projects and artifacts. Row eligibility at scoring, temporal history retrieval,
post-prediction rules and automatic dependency installation remain separate work.

## Layout and contracts

- src/jobs/: existing notebook entrypoints; keep the two jobs and graph unchanged.
- src/features/__init__.py: export the two recipe builders.
- src/features/pre_split.py: fixed cleanup and declared training row filters.
- src/features/preprocessing.py: fitted feature transformations, refit per fold.
- src/features/custom/: optional project Calculator/Applier pairs and helpers.
- src/modeling/: existing tuning.py and ensemble.py hooks.
- src/tools/preview.py: offline configuration validation.

Snapshot all Python source beneath features under the existing 64 KiB source
budget. Load relative imports from saved source under a digest-specific package,
without relying on the editable directory or changing sys.path. Keep external
dependencies explicit in the deployment environment. Do not introduce a second
transformation engine or permit learned pre-split operations.

## Execution and verification

- [x] Pin package imports, source identity and fresh-process artifact reload.
- [x] Implement bounded package snapshots and isolated import loading.
- [x] Move generated files, update resource paths, hooks and preview.
- [x] Connect custom pre-split and preprocessing implementations to both builders.
- [x] Run affected Python suites, real CLI generation/strict validation, demos,
      Ruff, formatting, full Ty, CCN10 and documentation checks.

Existing uncommitted CV matrix tests are separate verified work. No deployment,
commit or push is required by this layout request.

## Verified local delivery (2026-09-28)

- 331 related Python tests passed, including source versions, lazy relative
  imports, import-failure cleanup, fixed filter guards, actual per-fold means,
  and pandas/Polars MLflow fresh-process loading of both file/package recipes.
- The registered custom eligibility test now changes cwd to its temporary
  directory. Its follow-up check passed and generated artifacts stay out of
  the repository root. The two artifacts from earlier runs were preserved in
  `rehearsals/project_layout_orphan_artifacts/`.
- 74 real CLI generation tests passed. An actual generated project's custom
  pre-split and preprocessing examples were enabled and previewed together.
- Generated jobs passed `bundle validate --strict --target dev --profile skyulf`.
  This used the existing wheel solely to satisfy local artifact references;
  no deployment or cloud execution of the new package implementation occurred.
- The external demo returned centered values [4, 6] and surviving IDs [1, 4]
  on both engines. The examples remain opt-in in generated projects.
- Full Ruff, CI formatting scope, full Ty, backend/Core CCN10 and strict MkDocs
  passed. No frontend changes were needed.
- CLI fixtures were updated for directories/globbed source paths. An old
  template `__pycache__` was removed before regenerating; only Python source
  is explicitly included in the new source sync patterns.

Remaining SM-36a scope is still open: historical feature context, keyed scoring
exclusions, output rules, and external dependency/non-Python asset delivery.

## Follow-up: general custom steps (2026-09-28)

The latest user steering replaces business-specific demonstrations entirely.
`custom/pre_split_custom.py` implements minimum row completeness over selected
columns; `custom/preprocessing_custom.py` implements training-only frequency
encoding for selected string categories. Following the user's final correction,
parent recipes show each custom factory call directly inside the returned step
list beside Core operations. Uncomment its import and step, then adapt columns.
There are no separate column-list variables, append logic or enable flags.

The old external preprocessing demos and generated custom/examples.py are removed.
Tiny synthetic contract helpers live only in a test fixture. New integration tests
configure the shipped parent recipes and exercise both engines, independent phase
selection, real split/training/CV and local/MLflow fresh-process model reload.

Final verification: 129 tests passed together (74 real CLI generation cases,
21 general custom-step cases and 34 project/source-replay cases). Ruff, CI format
scope, full Ty, Core/custom CCN10 and strict MkDocs passed. The earlier related
lifecycle/template regression run also passed 453 tests. No cloud run was made
for these general custom steps.

Inline-step follow-up: 22 tests passed, including activation of the displayed
steps in a real CLI-generated project and both-engine train/CV/model reload.
Ruff, CI formatting, full Ty, Core CCN10 and strict MkDocs passed again.
