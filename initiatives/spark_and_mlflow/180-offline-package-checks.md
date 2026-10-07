# Offline project package checks

Date: 2026-10-07. Follow-up to the user-requested reference review in
[delivery 179](179-reference-template-improvements.md).

## Problem and change

`src/tools/smoke.py` previously validated workflow settings and Python syntax
only. A project could pass despite an absent declared lookup file, an invalid
dependency pin, or a missing package boundary. These errors then surfaced when
training attempted to capture the package for the saved model.

`projects/project_checks.py` now passes present `src/features/` and
`src/composition/` packages through the existing inert `project_source` validator.
This shares training's package, asset, exact dependency declaration and size
rules instead of introducing another implementation. The snapshot is constructed
in memory and discarded; no hook runs, dependency is installed, or file is written.
Legacy projects without these package directories retain their existing checks.

The generated smoke entry point prints JSON for expected validation failures and
exits with code 1. Messages identify the configuration file, Python source or
package. Syntax failures additionally expose `file`, `line` and `column`.
Success retains the existing fields and adds the `project_packages` list.
The scope identifier is preserved for existing consumers.

The generated `START_HERE.md` and README explain the checks, output and limits.
The changelog also corrects the prior artifact-version sentence: the shipped
artifact declaration still requires Core 0.9.2; this task does not change versions.

## Usage

From the generated project root, with its matching Core environment installed:

```powershell
python src/tools/smoke.py
```

For example, an absent declared feature asset now yields:

```json
{
  "status": "failed",
  "error_type": "ValueError",
  "message": "src/features: Project asset/metadata must be a file inside its root: missing_lookup.csv.",
  "project_hooks_executed": false,
  "remote_operations": false
}
```

Generated tools do not automatically update existing Bundle projects. To adopt
this output there, install the updated Core wheel and apply the changed smoke
entry point deliberately, preserving edited project configuration and recipes.

## Validation

- Before implementation, 19 new cases failed while the three existing smoke
  checks passed. Missing assets and invalid metadata were incorrectly accepted;
  malformed configuration and CLI failures lacked the new output contract.
- Added UTF-8 and syntax-location probes reproduced two further diagnostic gaps.
- Independent review reproduced a configuration type error that bypassed JSON
  handling (`training_layout` given as a list). A focused regression failed,
  then passed after normalization inside the configuration-check boundary.
  The reviewer inspected the correction and reported no remaining findings.
- Final affected execution: **33 passed** across `test_project_checks.py` and
  `test_databricks_project_package.py`. This covers inert hooks, unchanged files,
  valid uninstalled dependency declarations, malformed config/source, asset and
  dependency rejection, legacy projects and the existing training consumer.
- Test import/collection check: **24 collected** in `test_project_checks.py`;
  this is collection evidence, separate from execution above.
- Actual CLI **1.17.0** generated single-model, competition and model-set projects
  against a loopback identity fixture. All three smoke commands succeeded; each
  then rejected a deliberately missing declared asset with JSON and exit code 1.
  This is local CLI validation, not native Databricks execution.
- Root Ruff, full CI formatting scope, full CI Ty scope and Lizard CCN 10 passed.
- Pre-commit passed whitespace, EOF, Ruff, formatting, Lizard and Ty checks.
  Schema/YAML/JSON/frontend hooks had no applicable changed files. No frontend
  rebuild or MkDocs build was needed for this library/generated-template change.

Reproduction commands and generated JSON evidence are under ignored
`tmp_repro_artifacts/task180/`. No employer documents, credentials, wheels or
generated test projects are included in this change.

## Remaining scope

This advances the existing **SM-40** offline-check work; its broader generated
project CI/CD scope remains PARTIAL. Smoke does not execute builders, resolve
candidate/branch semantics, check installed dependency versions or wheel
readiness, resolve deployment targets, verify cloud access, or prove model
behavior. Preview/build/Bundle validation and runtime acceptance remain separate.
Native Databricks testing is still deferred at the user's request.
