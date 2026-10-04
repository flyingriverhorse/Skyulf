# SM-36g / SM-36h / SM-36i acceptance

Date: 2026-09-28. DONE for the documented scope; independent persisted audit passed.
Baseline commit: `42bc0d37` (graph/resource work). Nested changes are included in the delivery commit.

## Delivered behavior

- Temporal nested CV: explicit clock, stable order, past-to-future inner/outer
  folds; row-count gap/test/expanding-or-rolling windows; isolated final holdout.
- Group and stratified-group CV: aligned identifiers, disjoint inner/outer and
  final holdout groups; missing/insufficient/class-incomplete partitions rejected.
- Time/group metadata is removed before learned preprocessing and final model
  fitting. Inference preserves row order and uses the fitted feature schema.
- Nested binary thresholds: fresh inner OOF probabilities of the selected recipe,
  threshold-aware outer hard-label scores, unchanged ranking/probability scores,
  and a separately selected final threshold. Persisted provenance is `inner_oof`.
- Core, backend Basic/Advanced, Canvas controls/results and Databricks Bundle
  settings share the policy. Switching UI methods clears hidden incompatible
  window settings. Temporal custom windows require an explicit clock.
- Existing two train/score jobs retained. No third permanent operator job.

## Review defects found and closed

1. SDK final test partition was missing temporal/group isolation validation.
2. Threshold wrapper needed the legacy sklearn classifier marker for 1.4/1.5.
3. Stateful evaluation reported default-threshold scores despite returning
   threshold-aware predictions; reports now use the actual predictions.
4. Seeded nested thresholds incorrectly claimed validation-split provenance.
5. Existing manual template renderer and historical dataset-ID test fixtures
   needed the new optional fields. Production legacy identities remain stable.

## Local verification

- Broad related Python suites: 1,881 passed, 16 skipped; 12 stale fixture failures
  above. Both affected files then passed completely: 129/129.
- Final combined changed-feature suite: 291/291 passed (includes all 12 repaired
  failures); 40.07 seconds. This is not a claim of rerunning the entire repository.
- Additional threshold seed-store lifecycle/service tests: 59/59 passed.
- Final frontend focused tests: 134/134 passed; earlier expanded run 190 passed.
- Independent sklearn outer winners/scores; exact scaler training membership at
  inner/outer/final levels for both engines; rolling window/gap boundaries;
  all five strategies with real bounded fits; missing/group/class guards.
- Frontend lint, CCN10, build and size check passed. Final main bundle:
  `index-C3giKtYM.js`, 331.6 KiB gzip within 340 KiB budget.
- Full Ruff lint/CI formatting scope, full CI Ty scope, backend+Core Lizard CCN10,
  `git diff --check`, and `mkdocs build --strict` passed.
- Real Databricks CLI generation: 3 cases passed; all 3 generated projects passed
  `bundle validate --strict` after adding the verified wheel.
- Same cloud acceptance notebook passed 16/16 locally with persisted MLflow
  download/reload parity before upload.

## Live Databricks acceptance

Workspace: `dbc-45604623-c18b.cloud.databricks.com`, explicit profile `skyulf`.
Isolated root: `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_lifecycle_test/graph3_20260928_r1/nested_policies`.

Wheel SHA256:
`715a1f559753f41bf4a4aca6d9e3f6fb3d5020f4ccc69f36a0bbbe3bd5e8e202`.
All 259 packaged Python source files matched the working tree before upload;
remote wheel bytes were downloaded and their SHA verified.

Successful run: [329510165594261](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/1032997994981331/run/329510165594261).
Both pandas and Polars tasks succeeded, eight cases each:

| Case | Model | Strategy | Policy / threshold |
|---|---|---|---|
| temporal_grid | Ridge regression | grid | temporal |
| group_random | Ridge regression | random | group |
| temporal_voting_optuna | Voting regressor | Optuna | temporal |
| group_stacking_halving_grid | Stacking regressor | halving grid | group |
| group_threshold_halving_random | Logistic regression | halving random | stratified group + threshold |
| temporal_threshold_grid | Logistic regression | grid | temporal + threshold |
| group_voting_threshold | Voting classifier | grid | stratified group + threshold |
| temporal_stacking_threshold | Stacking classifier | grid | temporal + threshold |

Every case checks filtered split membership, staged Parquet metadata, real fitted
scaling/search, nested evidence, expected feature manifest, local artifact reload,
MLflow upload/download/reload prediction equality, and exact saved tuning JSON.
Each predicts 24 holdout rows; 384 replay predictions across the 16 cases.
Runtime recorded in MLflow: sklearn 1.6.1, NumPy 2.1.3, SciPy 1.15.1,
pandas 2.2.3, Polars 1.44.2. This also differs from the local sklearn 1.8 runtime.

### Failed first attempt retained

Run `417057667951020` passed its first seven pandas cases, then the Python process
segfaulted during temporal stacking classification. Diagnostic single-case run
`595790100383979` passed with the exact same wheel/settings; the full two-task
repeat above passed all 16. No production workaround or skipped case was applied.
The native crash's exact cause is unresolved; it is not reported as a fixed bug.

The crashed parent/last-child MLflow runs (`0ff64d53d40a4204bb389f7426ae645e`,
`95276780d38a4e078eafcda5116cee9b`) remained RUNNING after process death. Automatic
approval review rejected changing their remote status; no mutation was performed.
Successful acceptance and its persisted audit are separate completed runs.

Independent read-only audit re-downloaded both parent summaries and all 16
tuning receipts, checked FINISHED run states and SHA256 equality. Audit result:
`rehearsals/nested_policies_live/audit-summary.json` (16 cases, 384 replay rows).
The PowerShell redirected wrapper returned 1 because MLflow wrote progress/hints
to stderr; its saved audit completed without an exception, and a separate local
receipt verifier returned 0.

## Scope and evidence

This is bounded representative coverage, not every model x strategy x policy x
setting combination. Nested threshold supports binary probability classifiers;
multiclass, regression and hard voting reject threshold selection. Combined
row-changing preprocessing plus target re-encoding remains explicitly rejected
where a reliable class mapping cannot be established. Existing default ordinary
CV behavior remains compatible.

Raw evidence: `rehearsals/nested_policies_live/` (requests, run/output JSON,
per-engine summaries, installed runtime, audit artifacts and source hash report).
Broad regression log: `rehearsals/nested-combined-r1.log`.
Plan: [100-nested-policy-implementation-plan.md](100-nested-policy-implementation-plan.md).

## Initializer follow-up (same delivery)

CV method/policy is now asked before the data window. Ordinary or nested temporal
selection derives a temporal final holdout, exposes clock/calendar questions and
hides unused split/shuffle/seed questions. Existing workflow JSON is unchanged.
Normal/group selections retain their explicit final split choice.

Fresh pre-commit checks: 201 focused Python tests; all 73 real CLI generation tests;
73 schema/template tests after equivalent predicate flattening; 82 focused frontend
tests; frontend build/size, full Ruff/Ty/format/CCN10 and strict MkDocs passed.
Two newly generated default temporal projects passed strict Bundle validation.
This prompt-only follow-up did not change the previously cloud-verified Core wheel.
