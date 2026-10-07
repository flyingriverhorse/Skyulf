# Direct YAML defaults — delivery 182

The user requested a fresh template reset after delivery 181: no existing projects
need migration. New Bundle projects now generate `config/features.yml`,
`config/training.yml` and `config/inference.yml` directly for all three layouts.
There is no generated `workflow.json`, `src/modeling/`, or migration command.
The migration implementation and its migration-only tests were removed. Public
legacy JSON/Python loaders and saved-model replay remain supported.

## Configuration contract

- `training.yml`: `version: 1`, optional `defaults`, required named `models`.
  Shared workflow settings include source, keys, inputs, split/CV, resource limits
  and quality policy. Each model supplies `model: {type, params?}` and optional
  sibling `tuning`. Nested overrides replace the whole value. `feature_lookup`
  uses the workflow schema and may be shared or specified per target.
- `inference.yml`: `version: 1`, scoring policy/source/output settings, plus
  optional `model_set` destinations/publication/composition. Declarative
  `weighted_sum`/`weighted_mean` composition remains supported.
- `features.yml`: optional upstream feature-group configuration; generated
  `groups: {}` adds no feature job. Its graph/tooling remain separately opt-in.
- Python functions remain in `src/features/` and optional upstream transforms in
  `src/feature_groups/`. No model/tuning settings are duplicated in Python.
- `pipeline` may express empty preprocessing/modeling structure and optional
  explainability. Model declarations and Python preprocessing retain their
  respective ownership. Duplicate pipeline/model explainability is rejected.
- Runtime jobs use `config/training.yml`; tools use `project_config_path()` and
  the shared strict reader. Legacy JSON remains selectable when YAML is absent.
- Inactive optional fields are omitted. The default single-model training file
  has about 39 lines and inference about 12. YAML retains explicit custom nulls,
  escaped strings, Unicode and numeric exponents from initializer JSON.
- Parsing is bounded to 64 KiB and 32 nesting levels; aliases, duplicate keys,
  unknown fields and conflicting JSON/Python owners fail explicitly.

Single-model resolution preserves target-bound UC names. Multi-target resolution
removes representative branch overrides before applying independent model entries,
so optional CV/date/native-lookup fields cannot leak into a sibling. Saved models
retain resolved settings and frozen custom source rather than rereading YAML.

## Local verification

All Python commands use `.venv/Scripts/python.exe`. CLI rendering used
`SKYULF_BUNDLE_OFFLINE_CLI=1`; the final union also sets
`SKYULF_BUNDLE_CLI_TEST_PROFILE=offline` to enable existing opt-in cases. Generator
subprocesses receive dummy localhost credentials. Strict Bundle validation uses
the existing loopback HTTP fixture for identity/path/policy reads. No Databricks
workspace operations, deployment, commit or push were performed.

Test-driven reproductions:

1. The initial direct-generation tests failed for the old JSON/Python layout;
   six layout/compute combinations and the direct pipeline case then passed.
2. Four valid identifiers (`on`, `null`, `true`, `No`) changed type in YAML;
   generated identifiers are now quoted.
3. Existing generated consumers exposed single-model defaults overwriting bound
   UC names; workflow fields now resolve before model materialization.
4. Duplicate pipeline/model explainability was accepted, and JSON exponent/emoji
   values changed after YAML parsing; focused reproductions now pass.
5. Three mixed nested-CV branch cases leaked the first branch's optional settings;
   all now pass. A separate assertion covers native lookup isolation.
6. Final review found shared `pipeline.explainability` was omitted from competition
   and multi-target adapters. Two focused failures reproduced the issue; shared
   model resolution now preserves it in every layout.

Intermediate evidence: initial consumer union 74 failures/131 passes identified
the shared UC-binding issue; after correction 246 passed with 13 obsolete explicit
default-key assertions. Remaining generation/deployment consumers passed 83 cases
with one Unicode failure; that parser issue was fixed and focused six-case
parser/branch verification passed. These are diagnostic runs, not final acceptance.

The final deduplicated affected union completed with **614 passed and one test
fixture failure** in 429.41 seconds: the new target-binding unit fixture omitted
required scoring table names. Only that fixture changed; the complete YAML file
then passed **22 tests**. Following the shared-explainability review correction,
the YAML file plus generated SHAP and branch-isolation cases passed **40 tests**
in 4.91 seconds (including three new regression cases). All **618 distinct cases**
are covered by passing evidence at their relevant final states. The large union
was not repeated after the narrow repair. Logs are in the ignored
`tmp_repro_artifacts/task182/yaml/` directory. Exact main-union scope:

```text
tests/integration/platforms/
  test_databricks_yaml_config.py
  test_databricks_yaml_defaults.py
  test_databricks_bundle_generation.py
  test_databricks_bundle_template.py
  test_databricks_tuning_template.py
  test_databricks_branch_template.py
  test_competition_guided_setup.py
  test_competition_template.py
  test_feature_bundle_generation.py
  test_selected_bundle_generation.py
  test_databricks_daily_generation.py
  test_databricks_deployment.py
  test_readable_model_template.py
  test_template_review_batch14.py
  test_guided_branches.py
  test_training_graph_generation.py
  test_simple_weight_artifacts.py
  test_model_owned_template.py
  test_model_file_loading.py
  test_project_checks.py
  test_competition_project.py
  test_databricks_bundle_notebook.py
  test_databricks_project_preprocessing.py
  test_databricks_project_ensemble.py
  test_platform_review_regressions.py::test_d11_3_public_refresh_preserves_notebooks_and_repeated_output
```

Paths above are relative to `skyulf-core/`. Invocation flags:
`-q -o addopts= -p no:cacheprovider --basetemp=tmp_repro_artifacts/task182/yaml/final --tb=short`.
This union covers every changed template consumer and direct legacy loader/graph
consumers; it is not a whole-layer/full-repository suite.

Scoped Ruff, formatting, CCN ≤ 10 and `build_schema.py --check` passed before
production freeze; Ruff/format/CCN passed again for the final narrow correction.
Root owns final full CI Ty/static checks. The offline default
tests use checked-in actual YAML output fixtures; real CLI tests compare generated
defaults to those fixtures instead of maintaining a second template interpreter.

## Limits and review

Native cloud execution remains user-deferred. Static smoke does not validate the
deployment environment's optional Feature Engineering SDK dependency: operators
must add the documented exact pin to shared `deployment/requirements.txt` when
enabling native lookup. It does validate project package dependency declarations.

The YAML agent inspected native training/table checks and fitting, competition,
and approval callers without rerunning another agent's suites. No additional
confirmed defect was found: lookup controls survive source projection, overrides
are rejected, snapshot guards surround SDK reads/logging, prepared frame binding
is verified, and native single/set approval retain their saved label contracts.
