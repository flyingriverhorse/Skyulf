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

## PR delivery and first CI repair

GitHub writes recovered. Implementation `bcd30ea1` was pushed on branch `093`
and [PR 197](https://github.com/flyingriverhorse/Skyulf/pull/197) opened against
`master`, including delivery 179. Its first Core shard completed in about eleven
minutes and exposed two failures rather than timing out (4,178 passed, 393 skipped).

- The empty pandas calendar fixture inferred `float64`, correctly violating the
  explicit numeric epoch-unit contract. Only that empty fixture now declares
  a nonnumeric dtype. Numeric fit and legacy replay tests cover `Int64`/`Float64`,
  ordinary epochs, all-null and empty columns on both engines. The DateFeatures
  runtime is unchanged. The original failing case was reproduced; the final
  two-file calendar/feature union passed **127 tests**, with four existing warnings.
  Root independently inspected the test-only correction and its type controls.
- Unsupported evaluation splits already returned no report, but the later
  coverage attachment still called `len()` on their payloads. Coverage inference
  now requires an evaluated report; saved exclusion evidence remains authoritative,
  including fully excluded populations. Regression cases for train/test/validation
  all failed before the fix. The final modeling-base/coverage union passed
  **61 tests**, with eight warnings; **52 modeling tests collected** separately.

Full CI-scope Ruff, formatting, Ty and Lizard passed again after these repairs.
No local full suite was run. Remote checks are tracked on the PR; pending checks
are not treated as passed, and the native Databricks deferral remains in effect.

The third shard then completed with one additional stale expectation (3,910
passed, 62 skipped). `test_collect_trials_from_cv_results` omitted the existing
`evaluation_coverage` field from trial summaries. After reproducing that failure,
the expectation now includes an empty list when legacy search results provide
no fold counts. Runtime collection behavior is unchanged; all **18 tests** in
`test_tuning_engine_failure_branches.py` passed. Ruff/format and full CI Ty passed
again for this test-only change.

The last shard finished in about twenty minutes with six parameterized failures
from one registry-transport expectation (3,883 passed, 160 skipped). Resolution
now intentionally downloads only `MLmodel`; the old test expected the complete
package URI. All six failures were reproduced locally. The corrected assertion
requires exactly one metadata download with the bound registry, while retaining
the concrete model URI and digest checks. All **11 tests** in
`test_review_batch16_mlflow.py` passed. An independent reviewer accepted both
the trial-summary and registry-transport test corrections without rerunning them.

The first full CI pass through the four partitions completed without timeouts.
Its nine failed cases comprised the four causes above; the other CI test and
security gates passed. Combined coverage and Sonar were skipped behind the
failed test dependencies, so they still require the corrected PR run.

## Include native runtime tests in the coverage aggregate

The first CI run's four coverage databases were downloaded and combined against
their exact source revision, `bcd30ea1`. Combined statement/branch coverage was
**88.48%**, below the existing 90% floor. The separate Spark/Delta workflow passed,
but did not record coverage for that aggregate; the missing measurements were
concentrated in Spark execution, preprocessing and monitoring modules.

Core CI now calls that existing workflow and waits for its two runtime lanes.
Each records branch coverage for the same complete Core source scope. The
combiner requires exactly six nonempty, valid branch databases: partitions
`0` through `3`, `spark` and `delta`. Missing or corrupt runtime data cannot
silently disappear. The 90% floor and production source scope are unchanged.
The old independent PR trigger is removed to avoid duplicate native executions;
manual Spark/Delta workflow dispatch remains available. These are JVM tests on
GitHub runners, not cloud Databricks jobs.

Both new native-contribution cases failed against the four-input combiner.
The final CI-helper test file passed **25 tests**, covering each runtime's unique
branch contribution and missing/corrupt/empty/statement-only data. Targeted Ruff
and formatting, full workflow actionlint, root Ruff and full CI Ty passed.
The root CI formatting scope and Lizard CCN 10 also passed. Independent review
accepted workflow dependencies/concurrency, six artifact identities, input guards
and the unchanged coverage scope and floor without repeating the test batch.
Actual aggregate production coverage is still pending the six-lane CI run.

That run then passed 53 Delta tests and 419 Spark tests, but skipped the pyfunc
test module: the Spark-only requirements do not install optional MLflow. The
Spark workflow now installs the existing MLflow requirements alongside its Spark
requirements and explicitly imports both before running tests. The two existing
regression/classification pyfunc parity cases must therefore execute in the
dedicated lane. Base and Delta environments retain their dependency boundaries;
no production package requirement or version constraint changes.
Combined Python 3.12 dependency resolution passed with `uv pip compile`;
actionlint and independent dependency/workflow review passed. Execution of those
two parity cases remains pending the corrected CI environment.
