# Delivery 179 — reference review and Bundle operator improvements

Date: 2026-10-07. Starting revision: `8432a6ba`, branch `093`.

## Starting state

PR #196 was merged into `master`. Security, backend and frontend checks passed,
but the Core job exceeded its 45-minute limit and Codecov patch coverage failed.
The merged tree was therefore not fully verified. Delivery 178 records the
previous repairs and the user's explicit deferral of fresh Databricks native
testing after OAuth expired. That deferral remains in force.

SM-23's batch/online monitoring delivery remains recorded in Delivery 177.
This work completes specific deployment/documentation omissions; it does not
reimplement monitoring or claim that every production acceptance item is closed.

## Reference coverage

Read all **99 Markdown files** in the user-confirmed local
`codes-main (6)/codes-main/leadgen_reg/docs` corpus:

| Reader scope | Files |
| --- | ---: |
| Root, governance, references, learning and how-to guides | 63 |
| Direct explanation pages | 16 |
| Nested monitoring-service and dashboard explanation pages | 20 |

Compared these with the original `dbml-mlops-template` and the working
`leadgen_reg` example in the supplied `(5)` checkout. The example's changes are
not treated as the original template contract. The earlier `docs_mlops` directory
is not the requested reference corpus.

Read inventories, source hashes and detailed review notes are retained locally
under ignored `tmp_repro_artifacts/task179/`. Employer documents, internal URLs,
identities and environment values are not copied into the generic template.
The supplied pages explicitly identify empty drafts, dated examples and diagrams
whose contents are unavailable. Reading their Markdown does not establish review
of linked websites, attachments or unseen diagrams.

## What transfers usefully

| Reference pattern | Skyulf comparison | Decision |
| --- | --- | --- |
| Task-oriented onboarding: where to edit, first run, troubleshooting | Existing generated README contains extensive contracts but is difficult to enter | Add a short generated `START_HERE.md`, conditional on the selected layout |
| Separate runtime identity from deployment identity and resource permissions | Train/score targets cover this; monitoring was omitted | Apply the same explicit identity and optional ACL contract to all three jobs |
| Compute/cost controls by workload | Monitoring intentionally runs independent serverless, but its runtime was hardcoded | Add monitoring-specific environment version and budget-policy variables |
| A model card joins registry, training and operating evidence | Skyulf already records these in separate artifacts and monitoring datasets | Keep consolidated card/lineage work under SM-47 |
| Freshness and unavailable evidence are explicit | Observation freshness and keyed batch provenance exist; source-event-age SLA is a separate gap | Clarify current behavior and diagnostic limits; do not label unknown data healthy |
| Small generated project tests and credential-free CI | Local smoke/preview and CLI fixtures exist; generated-project CI remains incomplete | Retain SM-40, including generated recipe/key/type tests and a dedicated offline CLI gate |
| Online features and endpoint lifecycle | Useful for key-based serving; not required to improve this batch flow | Keep under SM-21a/b; do not enable stores or billable resources by default |
| Scoped cleanup and backfill operations | Separate operational lifecycle, requiring ownership and dependency evidence | Keep SM-45/SM-50; do not import drop/recreate defaults |

Preserved Skyulf contracts: fold-local learned preprocessing, CV selection before
the reserved holdout, fail-closed candidate/champion comparison, immutable model
versions and set releases, keyed scoring provenance, delayed-label eligibility,
and guarded retraining. The reference's scoring-side self-promotion, selection on
the final holdout, permissive `ALL_DONE` publication, overwrite-all monitoring
tables and failed-query-as-zero examples are not adopted.

## Selected template changes

All paths below are relative to
`skyulf-core/templates/databricks/template/{{.project_name}}/`.

- `START_HERE.md.tmpl` provides the selected layout's editable model files,
  local smoke/preview/build steps, explicit target/profile commands, training to
  scoring sequence, monitoring evidence and a symptom-to-stage troubleshooting
  table. The main `README.md.tmpl` links to it.
- `deployment/artifact.json` and the generated README now agree with release
  **0.9.2**. The old 0.9.1 artifact requirement rejected the current release wheel.
  A regression test builds from the actual shipped settings and current manifest.
- `deployment/targets.yml.tmpl` applies shared Run-as identities and opt-in
  permissions to monitoring. Separate-identity mode explicitly reuses the scoring
  principal for monitoring. Personal targets retain their own deployment identity
  and resource names. No principal or Unity Catalog grant is provisioned.
- `deployment/variables.yml.tmpl` and `resources/monitoring.job.yml.tmpl` expose
  `monitoring_environment_version` and `monitoring_budget_policy_id` for the
  independent serverless job. Defaults preserve environment version 4 and an
  unspecified budget policy. Training/scoring compute choices remain separate.
- `deployment/README.md.tmpl` explains train → score, score → monitoring and
  optional monitoring → train execution permissions. Job ACLs do not grant data
  access, model ownership or permission to use a service principal.
- `src/monitoring/README.md` describes all five dashboard pages, including Online
  serving, captured-traffic limits, asynchronous log availability, missing/failed
  evidence, and observation versus scoring/source freshness.

Wizard source changes also remove two duplicate prompt orders. Scoring input
follows training input, and the snapshot question precedes CV. Source tenth slots
become integer CLI orders; schema generation rejects duplicate final orders with
both field names. Question values, defaults and skip conditions are preserved.

## CI repair findings

Detailed reproductions and commands are in the ignored `task179/ci/evidence.md`.

| ID | Finding | Correction |
| --- | --- | --- |
| CI179-01 | Backend version tests execute a copied module outside measured coverage | Exercise the actual resolver with temporary manifests and patched metadata; keep startup/environment checks |
| CI179-02 | Two tests allow evaluation-row expansion in ordinary transform | Assert the established rejection, including the step that expands rows |
| CI179-03 | Four XGBoost tests expect encoded labels from public prediction | Assert original labels and equality with the saved artifact |
| CI179-04 | Three layout tests omit the shared inference-mode question | Include the actual shared choice |
| CI179-05 | Fresh-process competition test imports an obsolete path | Use the canonical training/fitting module |
| CI179-06 | Spark worker-wheel metadata test hardcodes 0.9.1 | Compare against the installed release metadata while retaining exact-source checks |
| CI179-07 | Two wizard prompt-order collisions | Correct source positions, regenerate schema and reject future collisions |
| CI179-08 | Temporal Polars MLflow test bypasses documented nullable transport | Prepare the pyfunc request before checking continuation semantics |
| CI179-09 | Shared runtime helpers still cross private-symbol boundaries | Define shared helpers directly under meaningful internal names; migrate exact callers |
| CI179-10 | Architecture test confuses canonical-module aliases with function aliases | Resolve actual modules; retain private-symbol/attribute and function-alias rejection |
| CI179-11 | Approximately 18,000 tests exceed a single 45-minute CI job | Partition by stable file identity, then combine all raw coverage with the existing 90% branch floor |
| CI179-12 | Generated artifact settings reject release 0.9.2 | Synchronize template metadata and test the shipped configuration |

Both previous Core jobs kept advancing and stopped at different progress ranges
(PR 65%, master 61%) before their 45-minute deadline. Quiet pytest output does not
identify the exact active test; no terminal infinite hang was observed. Sharding must
preserve all collected tests and the aggregate coverage gate; a higher timeout,
excluded tests or a lower floor is not the fix.

## Validation and delivery state

Focused verification (deduplicated within each affected batch):

- Version resolution: 15 tests passed; real-module line/branch coverage is 100%.
- Six stale-contract test files: 120 tests passed with isolated Polars 2.0.0.
- Artifact packaging: 14 passed, 16 opt-in CLI cases skipped in that run. The
  release regression executes the generated wheel-preparation command against
  shipped settings and a current-release metadata fixture, not a deployed wheel.
- Wizard build/order checks: 14 passed, including duplicate-order and unsupported
  precision reproductions. Generated diff changes only two order values; all
  2,353 fields, defaults, enums and skip conditions are preserved.
- Existing online dashboard contract: 2 passed.
- Monitoring deployment contracts: 38 cases passed with actual CLI 1.17.0 using
  loopback fixtures; all three generated layouts have valid guide links and no
  unresolved Go template tokens. Four missing-setting/identity cases were first
  reproduced failing. The monitoring guide's 14 canonical library paths exist.
- Internal API repair: 609 passed, 36 skipped across 37 explicit affected files
  (645 collected). Skips: 31 require PySpark/Java, two require optional Delta,
  three require CLI opt-in. Legacy module/class identity, saved artifact loading,
  boundary positive/negative cases, monitoring, drift and retraining passed.
  Inverse AST comparisons match the original 17 runtime files and seven directly
  changed consumer test files after reversing the declared identifier changes.
- Four-way test allocation was checked against the existing 18,011-node inventory:
  4,571 / 4,021 / 5,449 / 3,970, with an exact disjoint union and whole-file ownership.
  This is an inventory check, not a full test execution.
- Shard/aggregate tests: 15 passed. Real miniature pytest collection, exact
  partitioning, malformed/missing/statement-only data, complementary branches,
  source-relative XML, uncovered files and the 90% failure exit are exercised.
  Independent review accepted both aggregate fixes without repeating that batch.
- Ruff, CI-scope Ty, Lizard CCN 10 and full-workflow actionlint 1.7.12 passed.
  Formatting was corrected in the new schema regression and its scope rechecked.

Independent review also reproduced two aggregation defects before delivery:
standalone XML lost the original source-relative paths, and coverage's permissive
combine could skip a corrupt nonempty shard. The final combiner validates and
merges all four branch databases explicitly and configures the Core source root;
missing or malformed input fails before an aggregate report can pass.

Final Ruff, formatting, CI-scope Ty and actionlint passed after all source edits.
Pre-commit passed schema freshness, whitespace, YAML/JSON, Ruff, formatting,
Lizard and Ty. Frontend hooks had no changed files; no frontend build was needed.
This document records local acceptance; the PR checks provide the remote result.
Full local test suites are not authorized and
are left to CI. Fresh native Databricks acceptance remains user-deferred; local
CLI generation/resolution against a loopback fixture is not cloud validation.

Separate existing issue: master demo promotion has cherry-pick conflicts against
the intentionally divergent `deploy/demo-mode` branch. This work does not modify
that branch or its deployment.
