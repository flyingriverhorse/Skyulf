# Readable generated model files - 2026-09-29

Approved scope: name generated files single_model.py, model_competition.py and
multi_model.py; keep model_set.py for coherent release/scoring policy. Generate
editable Python dictionaries instead of embedded JSON documents. Move single
model settings out of workflow.json into build_modeling(). Retain old project
filenames and legacy hooks without silently combining two definitions.

Implementation map:

1. Runtime: project.py resolves single_model.py at its public boundary;
   competition_project.py and branch_notebook.py accept new filenames with legacy
   fallbacks. Tests cover conflicts, saved config and independent layouts.
2. Templates: library/modeling.tmpl and Core-derived catalog render Python values;
   renamed templates retain existing factory APIs and independent settings.
3. Schema: build_schema.py compactly formats generated properties without changing
   question semantics. Editable schema sources stay split by topic.
4. Verify actual CLI generation, Python factories, training/heldout, legacy
   loading, preview, schema equivalence, full Ty/Ruff/Lizard and pre-commit.
   Strict-validate projects. No deploy or cloud run is part of this follow-up.
5. Update docs/queue/evidence and remove .verification-model-layout. Preserve
   staged work, no commit or push without request.

## Delivered behavior (uncommitted)

- `single_model.py` owns `MODELING`; `build_modeling()` is loaded during training
  and preview into the existing pipeline contract. Effective parameters persist
  in existing artifacts/training plans. Source edits cannot alter saved replay.
- `model_competition.py` owns `MODELS` for one-target competitors, retaining the
  `build_candidates(task)` API. `multi_model.py` owns independent target workflows,
  retaining `build_training_branches()`. Both use editable Python dictionaries.
- `model_set.py` still controls joint model version selection/promotion and
  scoring destinations. For example revenue/cost training lives in multi_model;
  their coherent release lives in model_set; profit rules live in features/scoring.
- Old candidates.py/branches.py are accepted when their replacement is absent.
  Two competing files, or a single_model.py plus nonempty JSON model, fail before
  executing user code. Old ensemble/tuning hooks remain supported.
- Empty model declarations are accepted only for single-model saved-artifact
  actions. Training still requires resolution; malformed/nonempty invalid model
  definitions, preprocessing and explanation settings are still checked.
- Native template helpers emit Python booleans/None and preserve quoted JSON
  keywords, escaped slashes, backslashes and Unicode in explicit init overrides.
- Schema formatting preserves parsed values and recursive member order while
  reducing 84,008 lines to 1,785 (about 2.41 MB to 1.15 MB). Readable topic files
  remain the editable source. No questions or defaults were removed.

## Verification

- New template-name regression was observed failing before changes. Runtime
  ownership/compatibility tests also failed before their implementation.
- Actual Databricks CLI 1.17.0 generation, Python declaration loading, selected
  standalone/ensemble fits and held-out scoring on pandas and Polars: 36 tests
  passed in 22.26 seconds. Includes 188 Core catalog render combinations; these
  are catalog checks, not 188 newly trained models.
- Saved-action regression was found during integration: empty JSON modeling was
  rejected before approve/reject/rollback/score. After the fix, 188 focused runtime,
  notebook and workflow tests passed. Tests with raising editable model/feature
  files prove saved actions do not execute those sources.
- Full Ruff lint/format, full CI Ty scope, backend/Core Lizard CCN 10,
  schema/catalog freshness and changed-file pre-commit passed. Frontend complexity
  hook passed; no frontend source changed, so no frontend build was needed.
- `uv build --wheel --no-build-isolation --out-dir .verification-model-layout/dist
  skyulf-core` succeeded. Final wheel SHA256:
  `2c980748a205bd42e20321ec6942845761a3c307375b3d86ff35f59954cc58aa`.
- `databricks bundle validate --strict --target dev --profile skyulf` passed for
  single ensemble, competition and mixed multi-model projects referencing that
  wheel. No files were uploaded and no paid cloud job was run for this follow-up.
- MkDocs was not run; documentation changes are prose, names and examples.

### Integration run and corrections

The broad 20-file integration run completed with 586 passed and 20 failed in
500.59 seconds. All failures were in test_databricks_bundle_generation.py:
date-free/manual-example/nested tests still validated or prepared the raw JSON
without resolving the new Python model file first. Tests now call the same
project loader as notebook training before applying their existing assertions.
No training validation was weakened to accept unresolved models.

The final regression command reruns the complete affected CLI suite together
with runtime/workflow/notebook/lifecycle contracts:

```powershell
$env:SKYULF_BUNDLE_CLI_TEST_PROFILE = 'skyulf'
.venv/Scripts/python.exe -m pytest `
  skyulf-core/tests/integrations/test_databricks_bundle_generation.py `
  skyulf-core/tests/integrations/test_model_file_loading.py `
  skyulf-core/tests/integrations/test_databricks_workflow_config.py `
  skyulf-core/tests/integrations/test_databricks_job_runtime.py `
  skyulf-core/tests/integrations/test_databricks_lifecycle_tasks.py -q
```

The preceding 36-test CLI/training/catalog run covered test_readable_model_template,
test_model_owned_template, test_guided_branches and test_bundle_model_spaces.
The broad run additionally covered competition setup/project/template,
bundle/template/tuning/ensemble/layout/schema/branches, project preprocessing and
project package regressions. CLI tests were explicitly enabled, not skipped.

Final regression result: **356 passed**, 63 warnings, 418.11 seconds. This
includes all 20 previously failing CLI cases and the expanded saved-action
contracts. No unresolved test failure remains from the integration run.
Final test/doc pre-commit rerun also passed, including full Ty scope. No commit
or push was made; previously staged user work remains staged as before.
Temporary `.verification-model-layout` output, including the local test wheel,
was removed and directory absence verified. Final Ruff and diff whitespace
checks passed after cleanup.
