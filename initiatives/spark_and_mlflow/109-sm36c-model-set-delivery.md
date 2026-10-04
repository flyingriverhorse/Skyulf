# SM-36c: coherent model sets — design and implementation plan

Date: 2026-09-29. Status: DONE for the documented scope. Base commit: `20dfa004`.
Delivery commit requested after the latest [acceptance112](112-clean-databricks-acceptance.md).
Spec: [report58, SM-36c](58-custom-fe-and-multi-model-bundle-plan.md).

## Design

Package a complete set of pinned component artifacts under one versioned registry
identity. Its manifest records component names/versions/payload digests, ordered
input/output schemas, record keys, composition source/rules and package identity.
Copy validated local pipeline artifacts into the set so prediction needs neither
editable project code nor mutable component aliases nor network access.

One set champion points at one complete package. Component aliases remain
independent; updating a component or rule creates a new set version and requires
explicit validation/activation. Reuse shared alias admission, pending-event
handling and durable receipts. Rollback selects the prior complete set.

Score each bounded component sequentially, project its own raw input columns,
and join component outcomes by explicit unique non-null record keys. Namespace
all component outputs. Preserve each component's exclusions; required-component
eligibility governs optional composed outputs. Composition uses the existing
versioned custom output-rule contract, with recorded source and declared types.
No component/rule exception produces a successful partial final result.

Append receipts and rows identify the complete set version/digest. Full refresh
uses one atomic Delta overwrite only after successful complete scoring. Readers
retain the preceding committed snapshot until that transaction succeeds.
Temporal carry state, when used, belongs to each component and is committed with
the same final receipt. A repeated successful publication is a no-op; uncertain
writes require receipt verification. Existing single-model paths stay compatible.

## Alternatives and decisions

- Chosen: self-contained set package and one set alias; coherent selection and
  offline/future-serving parity, at the cost of copying component payloads.
- Rejected: resolve each component champion on every scoring run; permits mixed
  releases and cannot establish a single rollback identity.
- Rejected: independently publish branch tables before composition; leaves partial
  final state on component failure. Intermediate computation stays uncommitted.
- Ruling: continue on the user's existing clean feature branch `091`, preserving
  their workspace and temporary directories; no automatic commit or push.
- User selected per-rule dependencies: each rule declares `required_components`;
  an unrelated excluded component does not suppress that rule. Preserve component
  outputs and record each skipped rule's reason. Missing predictions are never zero.
- Ruling: publish full rebuilds with atomic Delta overwrite of the same compatible
  table, preserving identity/history and avoiding a separate view activation step.

## Implementation tasks

Use test-driven-development and subagent-driven-development for bounded tasks.
Keep production CCN <= 10, preserve optional dependency boundaries, and run the
same Ruff/format/Ty scopes as `.github/workflows/pr_check.yml`.

### 1. Self-contained package and keyed local execution

Files: new `skyulf/inference/model_set.py`, focused private helpers as needed,
and `tests/integrations/test_model_set_{artifact,scoring}.py`.

- [x] Write failing tests for two fitted component pipelines, separate engines,
  independent schemas, prediction parity, key reorder, duplicate/missing keys,
  composition output types, source deletion and tampered artifact rejection.
- [x] Implement strict manifest plus `save_model_set`, `load_model_set`,
  `predict_model_set`; namespace outputs and preserve component outcome columns.
- [x] Prove isolated-process replay and bounded package/row memory; no alias reads
  occur during local inference. Review task before dependent adapters use it.

### 2. MLflow package, release and rollback

Files: new `integrations/mlflow/model_set.py` and focused tests; reuse registry
publication, digest resolution and controlled alias receipts where appropriate.

- [x] Log/register a set PyFunc with key-inclusive signature and frozen assets.
- [x] Resolve a set alias once and validate every component/rule before activation.
- [x] Exercise initial activation, replacement, rollback, expected-version conflict,
  failed validation and unknown alias mutation; component aliases stay unchanged.

### 3. Databricks publication and project integration

Files: new `integrations/databricks/model_set_batch.py`, narrow existing entrypoint
integration, template `src/modeling/model_set.py` and focused integration tests.

- [x] Build sets from complete SM-36b results; reject partial-parent receipts.
- [x] Read one bounded source snapshot/increment for all components. Join by keys,
  persist set provenance and all component history only with final output commit.
- [x] Test append retries, component/rule failure before write, all-excluded rows,
  set changes, full refresh and rollback. Validate generated two-job configuration.

### 4. Acceptance and handoff

- [x] Read-only review and relevant regression tests; full Ruff/format/Ty/CCN10.
- [x] Actual Databricks full pipeline: component fit/register, set packaging,
  approval, independent/composed parity, source append/no-op, controlled failure,
  new set activation and rollback, fresh-process reload.
- [x] Record exact wheel hash, successful/failed runs and tested boundaries.
- [x] Update authoritative queue/handoff only to the verified completion level.

## Verification recorded so far

- Combined local artifact/scoring/batch/MLflow/import suite: 106 passed,
  1 Windows symlink privilege skip. Project/branch regressions: 59 passed,
  4 ordinary-run CLI opt-in skips; actual CLI template run: 16 passed.
- Expanded project integration: 9 passed, including authentic completed branch
  training, original registered component bytes and immutable earlier set replay
  after repackaging the same parent with new rules.
- Final MLflow set suite: 15 passed, including composition-only dependency pins
  in logged requirements and conflicting component pins rejected.
- Full CI Ruff, format (1109 files), Ty and backend/Core CCN <= 10 passed.
  Generated multi-target serverless Bundle: actual `bundle validate` passed
  without warnings once the tested wheel was present. No deployment occurred.
- Fresh subprocess with MLflow imports explicitly blocked still loads the
  legacy branch configuration API. Optional dependencies remain optional.

### First live Databricks acceptance

Run `191435347102413`: SUCCESS, both tasks successful.
Wheel SHA256 `2af252893e449e4165b81627dff4abbb922b71e10784cf980dfa667e75622595`.

- Task `871130004628319`: 110 passed, no failures/skips, including four real
  Delta cases, both engines, saved rules, per-rule exclusions and temporal carry.
- Task `342866247568652`: actual UC versions 1/2, first approval, expected-version
  conflict, replacement, full rebuild, rollback and isolated-process PyFunc load.
  Set `workspace.skyulf_lifecycle_test.sm36c_20260929_r1_set` ended at champion 1;
  component aliases stayed at 1. Delta output advanced through versions 1/2/3/4
  (initial, append, replacement, rollback), retaining five rows. A controlled rule
  failure preserved output version 4 and its exact receipt.
- MLflow producing run: `643bfbde90534fdeb74d5ea200dd3d82`.
- Raw evidence: `rehearsals/sm36c_20260929/{tests-output-r1,registry-output-r1,run-r1}.json`.

### Final candidate acceptance

Run `357086522081045`, final wheel SHA256
`6240df62310363e60bb6898b6832b67b6d427ccf38608b7e66c0a0079664ab08`.
Adds project notebook entrypoints, digest-addressed immutable set publication,
composition dependency capture, real empty rebuild and INT-key widening cases.
Both tasks SUCCESS:

- `948407249224482`: **114 passed, zero failures/skips**, including six real Delta
  cases. Duration 304.23 seconds. This is the final wheel's contract test result.
- `875594516965375`: actual `run_branch_training_notebook` trained two independent
  pandas/Polars branches, packaged the authentic completed parent, then approved
  and scored through the Bundle entrypoints. Training/composition files were
  removed and editable rules changed before saved-only approval/scoring.
  Forty exact profit outputs passed; repeat scoring kept Delta version 1; one
  source append produced one new row and Delta version 2 (41 total rows).
- Parent MLflow run `5e7b74962b7944e5bcb63c2fcaa61aa0`, set
  `workspace.skyulf_lifecycle_test.sm36c_20260929_project_r1_set` version 1,
  digest `dd54af77ac09f45cc1df1d2e8ad4da620a133275e179cacbfb4cef4dc4072113`.
- Raw evidence: `rehearsals/sm36c_20260929/{tests-output-r2,project-output-r2,run-r2}.json`.

Final explicit Delta acceptance run `648117691633410`, task `691480190189601`:
SUCCESS, same final wheel. All-excluded publication saved two rows and a receipt;
repeated scoring was a no-op. Two temporal components committed their complete
continuation with three predictions over two source batches. Stored predictions
and final history exactly matched uninterrupted full-snapshot scoring; retry was
a no-op. Evidence: `rehearsals/sm36c_20260929/{run-edges,edges-output}.json`.

## Review fixes

Reproduced before fixes: empty Spark snapshots lost input dtypes; transition from
stateless to temporal components could bootstrap only from new rows; local rules
could read undeclared extra inputs absent from batch/PyFunc; repackaging one parent
could overwrite old registered assets; composition-only dependency pins were absent
from logged MLflow requirements. Each now has regression coverage. Concrete set
version changes also refresh full-rebuild row provenance even for identical bytes.

## Delivery boundaries

Whole-frame bounded local execution, with components processed sequentially.
Explicit approval proves functional execution, not predictive quality. Existing
single-model behavior stays separate; no same-target winner selection is added.
Set operators are approve/rollback; reject fails explicitly. One exclusive alias
writer and one prediction-table writer remain required, enforced identities belong
to SM-37. Schema changes need a compatible new output table; incremental CDF accepts
inserts only. Trusted source/pickle hashes detect mutation, not malicious producers.
No persistent test jobs were deployed and no code was committed or pushed.
Static checks and affected tests were run; no full 11k-test suite or MkDocs build
was repeated for this delivery. Generated wheels, test caches and `mlruns/` are
verification output and must not be included in a later commit.

## User follow-up: naming and configuration explanation

New MLflow packages now write `bundle_digest`, `local_pipeline_digest` and
`model_set_digest` without the `skyulf_` prefix. One registry read boundary
normalizes historical keys so already-published Databricks artifacts remain usable.
Focused registry, local-model, model-set and batch suites: 86 passed, including
new-name round-trips, legacy set replay and rule-error prevention of Delta writes.
Full Ruff/format/Ty/CCN10 passed. This metadata rename has local evidence;
the live wheel hashes above describe the preceding cloud-tested implementation.

A rule exception propagates and fails the scoring job before any new Delta commit;
existing successful output remains intact. Intentional per-row filter exclusions
retain the separately agreed per-rule dependency behavior. Each component runs its
saved feature scoring policy before the optional cross-model composition rules.

Core/PyFunc defaults (100,000 rows, 256 MiB) apply only without explicit limits.
Bundle entrypoints pass `config/workflow.json` values instead: initializer defaults
are 10,000 rows and 64 MiB. Limits bound individual source/result frames and do
not cap total process RAM, sample records or automatically split oversized input.
