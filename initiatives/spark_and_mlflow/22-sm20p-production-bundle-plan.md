# SM-20P production Bundle implementation plan

Updated: 2026-09-24. SM-20a is a live-validated **test/development Bundle
template**, not a company production deployment. This plan compares it with
`C:\Users\Murat\Downloads\codes-main (2)\codes-main\dbml-mlops-template`
and defines the work needed before calling a Skyulf-generated project
production-ready. No company production workspace has been deployed to.

**Goal:** Generate a small-data Skyulf project that can be promoted through
test, system test and production environments without changing its ML code or
silently reusing test tables in production.

**Architecture:** Keep Skyulf Core as the training/inference implementation.
Use Databricks Bundle targets and explicit target-specific catalog/schema
bindings. Keep recurring scoring separate from candidate training and manual
alias approval. Give production one prediction table for each model; keep
admission controls in a metadata schema and provision them once.

**Tech stack:** Databricks Declarative Automation Bundles, serverless Python
jobs for the personal `dev` target, company-policy job clusters for
`test/syst/prod`, Unity Catalog Delta, Skyulf Core 0.9.0, MLflow 3.16.1.

**Current status:** Target-specific name binding and both generated compute
variants have local tests and strict CLI shape validation. Company policy,
catalog permissions, cluster runtime and end-to-end execution are unverified.
The supplied company catalog/schema examples appear plausible to the user but
are not confirmed workspace settings. The CLI profile and `PayingRegNo` must be
verified with the company team before any company deployment.

## What the company reference does differently

| Concern | Company reference | SM-20a today | SM-20P decision |
| --- | --- | --- | --- |
| Environments | `ci/test/syst/prod` targets and catalog mapping | One `dev` target | Add `test/syst/prod` targets with explicit catalog mapping; retain `dev` for compatibility until migration |
| Isolation | Developer suffix on shared targets, empty suffix on shared production | Bundle workspace is isolated, UC table names are manually chosen | Separate non-prod names and control rows; fixed names in production |
| Data roles | Distinct raw/refined, metadata and result schemas | One initial catalog/schema prompt; JSON edited afterward | Ask for input, metadata and result schema mapping; allow explicit full table overrides |
| Compute | Company job clusters, policy lookup and cost tag | Serverless environment 4 | Personal `dev` is serverless; company targets use policy lookup and `PayingRegNo`, without a hardcoded policy ID |
| Workflow | Independent jobs plus an optional orchestration DAG | Independent train/compare/stage/promote/score jobs | Keep independent jobs; add a safe train→compare orchestration only if useful, never auto-promote |
| Extras | Feature store, endpoint, monitoring, dashboards, online store | None | Keep optional and later; do not generate unusable resources or 25 startup questions for the first small-data path |

The company template's `databricks_template_schema.json` is a *questionnaire*
for `bundle init`, not the training or inference runtime configuration. Its
regression Bundle uses target variables such as `catalog_name` and
`resource_suffix`. Synced YAML/JSON files are not automatically rewritten with
target variable values; the job passes those values to its notebook. SM-20P
must preserve this distinction or it could write a test table from a prod job.

## Resource contract

One production model can use these UC objects, possibly in different schemas:

| Object | Example role | Created by Skyulf project? |
| --- | --- | --- |
| Training table | Existing labeled input | No |
| Scoring source | Existing append-only input with CDF | No; only validate CDF and keys |
| Registered model | `metadata.customer_model` | Training registers it |
| Prediction table | `mlresult.customer_predictions` | One-time explicit setup, then score appends to the same table |
| Score admission table | `metadata.customer_score_admission` | One-time internal control table, not a second prediction output |
| Alias admission table | `metadata.customer_alias_admission` | One-time internal control table only if alias jobs are enabled |

The two admission tables hold ownership/coordination rows. They do not contain
features or predictions and should live in an operations/metadata schema.
The SM-20a `skyulf_sm20a_r1_*` prefix was a unique live-test namespace, not a
required production naming convention. A production project should not ask an
operator to run every job in sequence for each new batch: the score job alone
handles each new source increment; candidate training is separate, and stage
and promote are deliberate review actions.

## Implementation tasks

### P1. Target-specific binding and generation

**Files:** `templates/databricks/databricks_template_schema.json`, generated
`databricks.yml`, `config/workflow.json`, `resources/workflow.jobs.yml`,
`src/workflow.py`; extend `tests/integration/test_sm20a_bundle_template.py`.

- [x] Define separate personal and company questionnaires. Personal has only
  project/engine/catalog/schema; company adds host, test/syst/prod catalogs,
  input/metadata/result schemas, policy lookup and cost tag. The company
  example values require review.
- [x] Add target-specific Bundle variables and distinct workspace roots.
  Non-prod has a per-developer suffix; production has no suffix. Production
  must never inherit a test catalog; the company workspace host must be
  explicitly reviewed for every target.
- [x] Resolve table/model names from the active target at the notebook boundary
  before invoking Skyulf; test that the same generated source maps to disjoint
  test/syst/prod identities. Allow an existing fully qualified source table
  when training and scoring inputs are owned by another team.
- [x] `bundle init` both variants and `bundle validate --strict` the generated
  personal `dev` and company `test/syst/prod` shapes. Company validation used
  a **dummy policy ID and personal workspace host**, solely to validate Bundle
  syntax. It does not prove company policy/compute, catalog access or execution.
- [ ] Confirm the actual company profiles, host/catalog mappings, policy and
  tag, then validate against those profiles. No company deployment until then.

### P2. One-time provisioning and normal job usage

**Files:** generated `src/workflow.py`, `resources/workflow.jobs.yml`, README,
new focused provisioning tests and a revised Databricks user guide.

- [ ] Provide one explicit setup/preflight action that checks input tables,
  CDF, key/feature schema, permissions, local size budget and pinned model
  output schema; then creates only the approved empty prediction table and
  internal admission rows if absent. Never overwrite existing tables.
- [ ] Keep `score` as the only routine inference job. Add an optional paused
  schedule (SM-28b owns cadence); no manual month or source version per run.
- [ ] Keep `train` separate. `compare` is read-only. `stage` and `promote` stay
  manual with expected-version checks; no all-in-one automatic promotion.
- [ ] Explain that a new model version does not automatically rewrite earlier
  predictions. Full-history rescore is SM-27, not a side effect of promotion.

### P3. Company compute and deployment gate

**Files:** generated target resources, docs, a new live validation report.

- [x] Use serverless in the personal workspace and a looked-up cluster policy
  plus `PayingRegNo` tag for company targets, per the user's direction. No
  policy ID or credentials are stored in generated files.
- [ ] Validate the generated project against the designated **test** profile;
  deploy and run setup, train, score initial/new-row/no-op and alias jobs in an
  isolated test area. Check target model identity, row counts and Delta history.
- [ ] Validate `syst` and `prod` target configuration read-only with their
  designated profiles. Actual deployment/compute in those environments needs
  the user's environment details and explicit rollout decision. Do not mark
  production-ready based only on the personal `skyulf` workspace.

## Exit criteria and current gaps

SM-20a already proves the Skyulf services and five Bundle jobs on personal
serverless compute. SM-20P closes the target mapping, provisioning and
company-compute gaps. The company reference's feature store, online serving,
monitoring, dashboard and cleanup resources are not implied by this small-data
Bundle; they remain separately tracked optional work. The current template
should be described as a validated starting point until P1-P3 pass in the
intended company environment.
