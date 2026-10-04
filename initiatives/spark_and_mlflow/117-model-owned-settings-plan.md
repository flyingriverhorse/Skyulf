# Model-owned Bundle settings implementation plan

> For agentic workers: execute the approved design task by task; independent
> template catalog and branch-generation slices can proceed in parallel.

**Goal:** Generate editable model-owned ensemble/search settings for single-model,
competition and multi-target projects, without new shared override hooks.

**Architecture:** Keep runtime workflow/model contracts and legacy hook support.
Use Databricks native library templates to render JSON model definitions from
wizard answers; keep generated search catalogs synchronized with Core. Candidate
and branch definitions own their individual settings and populated search spaces.

**Tech stack:** Python, JSON schema, Databricks Go templates, existing Core APIs.

**Spec:** User-approved conversation: all model settings live in each model's
definition, initialized by Bundle; Core supplies initial search spaces. Users can
edit one model without affecting another. Branches can mix tasks and ensembles.

## Constraints

- No competition inside model sets; preserve champion lifecycle and isolation.
- No new runtime dependencies. Legacy ensemble.py/tuning.py projects still work.
- Generate real spaces from Core, never hand-maintain alternative model defaults.
- Shared competition CV/metric/pre-split stay shared; branch policies independent.
- Keep optional-dependency behavior and validation failures explicit.
- No commit/push without request; preserve existing staged work.
- Clean the task's temporary output after verification.

## Task 1: Core-derived editable search spaces

- [x] Add failing tests for materialized standalone and nested ensemble spaces.
- [x] Add build_model_spaces.py and generated library/model_search_space.tmpl.
- [x] Respect strategy, selected bases, calibration prefixes and halving resource.
- [x] Add freshness verification alongside the existing schema gate.

## Task 2: Wizard-owned models and branches

- [x] Add failing CLI tests for distinct branch targets/models/recipes/settings.
- [x] Add per-branch wizard definitions and branches.py.tmpl.
- [x] Reuse ensemble/search rendering across single, competition and branches.
- [x] Emit independent explicit model settings and search spaces.
- [x] Remove ensemble.py/tuning.py from new template output, retaining loaders.

## Task 3: Integration and evidence

- [x] Update intentional old template expectations and user documentation.
- [x] Run real CLI generation, training tests and legacy-hook regression tests.
- [x] Run Ruff, full Ty scope, Lizard, schema/catalog checks and pre-commit.
- [x] Strict-validate generated Bundle; record live/local evidence separately.
- [x] Update active queue/report and remove temporary verification output.

## Delivery evidence - 2026-09-29 (uncommitted)

Single-model settings live in `config/workflow.json -> pipeline.modeling`.
Competition settings live in each `candidates.py` entry, and independent targets
in each `branches.py` workflow. Wizard answers generate executable definitions,
including ensemble members/params, search strategy and editable finite spaces.
Branch target, features/recipes, CV, split and quality controls are independent.
Offline preview now resolves and displays the actual branches.

Catalog values come from Core's existing parameter definitions, with compact grid
spaces and richer random/Optuna spaces. Only selected ensemble members contribute
nested keys, including calibration and stacking-final prefixes. Halving reserves
its resource axis. Explicit spaces remain user-owned; `{}` selects runtime Core
defaults. After changing ensemble membership/calibration, update nested keys or
clear the space. Explicit initialization `model_params` keeps `{}` unless a custom
space is supplied, preserving fixed-parameter behavior.

New projects no longer install shared ensemble/tuning hook files. Existing hook
loaders and saved artifact behavior are unchanged and compatibility tests pass.

### Verified

`SKYULF_BUNDLE_CLI_TEST_PROFILE=skyulf .venv/Scripts/python.exe -m pytest` on the
following integration files passed **273 tests**, 607 warnings, in 133.69 seconds:

- `test_competition_guided_setup.py`, `test_competition_template.py`
- `test_databricks_bundle_template.py`, `test_databricks_bundle_generation.py`
- `test_databricks_tuning_template.py`, `test_databricks_project_ensemble.py`
- `test_databricks_layout_prompts.py`, `test_databricks_schema_build.py`
- `test_databricks_branch_template.py`, `test_guided_branches.py`
- `test_model_owned_template.py`, `test_bundle_model_spaces.py`

This includes actual CLI generation, 188 catalog render cases compared with Core,
all four single-ensemble families, all five branch search strategies, independent
recipes/policies, duplicate-name rejection and offline branch preview. Generated
mixed branches trained and scored held-out rows with pandas and Polars (ridge,
logistic regression, voting regression and stacking classification). Generated
competition training and legacy hooks also passed. Catalog render coverage is
not a claim that every model/strategy combination was newly trained.

Passed analysis commands:

```text
.venv/Scripts/ruff.exe check . --extend-exclude .tmp-review-model,.verification-model-settings
.venv/Scripts/ruff.exe format --check backend skyulf-core tests run_skyulf.py celery_worker.py
.venv/Scripts/ty.exe check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py
.venv/Scripts/lizard.exe backend skyulf-core/skyulf --CCN 10 -w
.venv/Scripts/python.exe skyulf-core/templates/databricks/build_schema.py --check
.venv/Scripts/pre-commit.exe run --files <existing changed source/test/doc/config files>
```

One test file needed formatting and was formatted before the successful hook run.
Pre-commit included schema/catalog freshness, whitespace, JSON/YAML, Ruff/format,
Lizard, full Ty and frontend complexity. Frontend lint had no changed files.
No frontend source changed, so no frontend rebuild was required.

`databricks bundle validate --strict --target dev --profile skyulf` passed for
three generated projects: single ensemble, competition and four mixed branches.
The previously verified SM-54 r3 wheel was copied locally to satisfy their wheel
references; nothing was deployed. No new paid Databricks job ran for this template
follow-up. Existing cloud runtime evidence remains in report116. MkDocs was not
run; documentation changes are prose and configuration examples.

Temporary verification outputs were removed after recording these results.
Both `.tmp` and `.verification-model-settings` are absent. Existing staged work
was preserved; this follow-up has not been committed or pushed.
