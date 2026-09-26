# SM-33E guided setup and live acceptance

Status: PASSED on 2026-09-26 for the documented local/personal-serverless scope.
Baseline: `fe3897c4`, branch `090`; SM-33D/SM-33E uncommitted.

## Implemented locally

Ordered initializer sections: basics/data, preprocessing, model, evaluation CV,
lifecycle and compute. The initializer offers a numeric imputation/scaling
preset and task-compatible Core model ID; custom ordered steps and parameters
remain editable in the generated workflow. CV, independent windows, date parsing
and training sampling map directly into the existing Core integration contracts.
The generated offline preview reuses job preflight and Core registry metadata.
It does not train, read data or check cloud permissions. Monthly preview reports
runtime selection instead of stale manual pins (review finding fixed).

## Personal workspace acceptance plan

Profile `skyulf`, existing schema `workspace.skyulf_lifecycle_test`, existing jobs
train `155738051514173` and score `684955889505992`. Both were checked idle before
preparation. No new schema or persistent job; no deletion of prior test resources.
Schedules stay paused/absent; every compute run has a 900-second timeout. Explicit retry prevention was
strengthened after discovering serverless auto-optimization (see findings below).
Existing generated Bundle files are backed up before changing the test configuration.

- One deterministic 240-row source `sm33e_source_r1`, with numeric features,
  classification/regression targets, Copenhagen event strings and Vilnius result
  date strings. CDF enabled for incremental scoring. Append exactly three new
  records after both initial scores. No real customer data.
- Two new models: `sm33e_pandas_classification_r1` and
  `sm33e_polars_regression_r1`, each with a matching `_predictions` table.
- Pandas classification: date-free random split, stratification, seeded sample,
  imputation/scaling, random forest and stratified evaluation CV.
- Polars regression: temporal holdout, non-UTC source/result parsing, delayed
  results filtered before seeded sampling, imputation/scaling, random forest,
  time-series evaluation CV with no shuffle.
- Each generated configuration runs through the actual existing train job,
  saved-evidence manual approval and conditional child score. No alias UI edits.
- After the append, both score jobs must add three records; another score must
  be a no-op without changing the target Delta version.
- Audit MLflow heldout/CV metrics, training selection and membership artifacts,
  concrete UC champion versions and output provenance/counts.

Wheel and helper notebooks use a fresh workspace path under
`/Workspace/Users/edwardwolfe99@gmail.com/skyulf_lifecycle_test/sm33e/r1`.
Rehearsal scripts/results are retained under `rehearsals/sm33e_live/`.
Live acceptance passed separately from local tests; SM-34 is now READY.

## Live findings and repairs

1. Fixture run `557892789655787` failed before writing because this serverless
   Spark Connect writer rejects the `errorifexists` alias. The harness uses
   `error` instead, retaining non-overwrite semantics. Corrected fixture run
   `165653745370927` created exactly 240 records.
2. Pandas train `205363783667118` succeeded (128 train / 32 holdout,
   heldout accuracy 0.96875). Approval `47276728690499` initialized champion v1
   and its child score wrote 240 rows to target version 1.
3. Polars train `7414061191563` registered candidate v1 but failed before
   completing its comparison receipt: repeated metric values differed at roughly
   1e-15 because parallel Random Forest predictions accumulate tree outputs in
   scheduler-dependent order. No Polars champion was approved.
4. Read-only diagnostic `74818126666445` proved identical sample/holdout hashes
   and changes only in metric last bits across four replays. Local same-artifact
   evaluation reproduced different metrics in 25/25 parallel calls; sequential
   joblib evaluation produced 0/25 differences without changing model n_jobs.
5. The repair confines sequential joblib prediction to `evaluate_local_holdout`.
   Normal training and batch scoring retain configured parallelism. Exact
   evidence comparison and receipt hashes are unchanged; no tolerance, rounding
   or bypass was added. This is not a general determinism guarantee for every
   numerical runtime. New tests reproduce the regression on both engines and
   verify context restoration after errors. A new r2 wheel was used for
   retraining; failed v1/v2 are retained as evidence.

6. The original failed Polars run contains two task attempts (0 and 1), each
   registering a candidate (v1/v2), despite the harness setting `max_retries=0`.
   Serverless auto-optimization can add retries independently. The generated
   serverless notebook tasks now set `disable_auto_optimization: true`, and both
   compute modes set `max_retries: 0`. Controlled/manual recovery remains explicit;
   these settings do not make training registration an exactly-once transaction.
   [Official behavior](https://docs.databricks.com/aws/en/jobs/run-serverless-jobs).
7. Corrected Polars run `812856395324358` succeeded as v3 with 106 training rows,
   54 temporal holdout rows, 17 unavailable results and RMSE 2.8916158447221196.
   Saved-evidence approval `737987074188078` succeeded, initialized champion v3
   and triggered the child score, which wrote 240 rows to target version 1.
8. Test harness configuration switching originally used `shutil.copy2`, restoring
   an older timestamp when switching back to pandas. Bundle incremental file
   sync kept the newer remote Polars file. Run `436364096149894` therefore used
   Polars and correctly returned a no-op; it is **not** pandas acceptance evidence.
   The harness now uses a fresh copy timestamp and exports/compares the deployed
   workflow before starting any model-specific run. The corrected pandas
   deployment passed that exact read-back check. No product runtime change was
   needed for this test harness issue.

## Local verification

- 164 focused native integration tests passed for config, previews, templates,
  CV, training windows and job runtime (`.cache/sm33e-native-final.log`).
- 48 actual CLI generation tests passed, including the published examples,
  pandas/Polars and classification/regression fit/save/reload/predict, conditional
  prompts, CV and retry settings (`.cache/sm33e-retry-green.log`). Strict Bundle
  validation passed for all four engine/task combinations and both live configs.
- 110 evaluation/lifecycle/retraining regression tests passed; one native Spark
  module was skipped because this environment lacks PySpark. Live serverless
  runs provide the cloud execution evidence below; this is not a claim that the
  entire Core test suite was run (`.cache/sm33e-evaluation-green.log`).
- New evaluation regression tests first failed with parallel predictions, then
  passed with the scoped sequential context for both engines/tasks. Independent
  review also caught and verified the monthly-preview correction.
- Full ty, scoped Ruff/format, strict MkDocs and wheel build passed. These suites
  overlap; their counts should not be added as a count of unique tests.

The deployed r2 wheel SHA-256 is
`27d4334d826049cef5ee54161e6c894173bfd9b27fe6d213510146a688c57f7e`.
It lives under the sibling `sm33e/r2` workspace directory; helper notebooks
remain under `sm33e/r1`. Both deployed job definitions were read back and retain
`max_retries: 0` and `disable_auto_optimization: true` on notebook tasks.

## Incremental scoring evidence

Append run `994570221320332` succeeded, increasing the source from 240 to 243
records. Polars incremental run `430905474194324` read/wrote exactly three new
records from source version 1 and committed target version 2. Repeated run
`527354609455267` returned `noop=true`, input/output counts zero, and retained
target version 2. The returned manifest on a no-op describes the previous
successful commit; this invocation's counts are the top-level result counts.

Pandas incremental run `386997334301578` then read/wrote exactly three new
records with champion v1 and committed its own target version 2. The preceding
misconfigured harness run is excluded from this evidence.

Pandas repeat `256505142332761` also returned `noop=true`, input/output counts
zero, and target version 2. Both workflows therefore passed first-score,
append-three and repeat-without-duplicates checks.

| Accepted step | Pandas classification | Polars regression |
| --- | --- | --- |
| Train | `205363783667118` | `812856395324358` |
| Approve | `47276728690499` | `737987074188078` |
| Child first score | `469263567962230` | `1058174921855309` |
| Append score | `386997334301578` | `430905474194324` |
| No-op score | `256505142332761` | `527354609455267` |
| MLflow run | `0101f75a3cbe4db38c396794011fc302` | `6f5de92b880740ff873976d819203614` |
| Champion | v1 | v3 |
| Final-holdout primary metric | Accuracy 0.96875 | RMSE 2.8916158447221196 |

The [train job](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/155738051514173)
and [score job](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/684955889505992)
retain these run histories.

The [final read-only audit](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/40622625485911/run/626395227690495)
passed. Both tables contain 243 unique record keys, no null predictions, the
expected model/version provenance and Delta version 2. Both model training runs
contain heldout/CV metrics and `cross_validation.json`, `training_selection.json`,
`holdout_membership.json`, and `candidate_training_spec.json`. The audit verified
128/32 train/holdout rows for pandas and 106/54 for Polars, plus 17 unavailable
Polars results excluded before sampling. A separate local `verify_results.py`
checked exact model identities, successful terminal states and all three score
phases against saved API responses; it passed after the live audit.

## Scope and operator handoff

This acceptance uses a small synthetic fixture in the personal serverless
workspace. It does not certify company policies/identities, production data,
all Core nodes, arbitrary custom models, tuning or serving. Four engine/task
combinations were exercised locally; live cases are pandas classification and
Polars temporal regression. Quality thresholds are fixture-specific, not
recommended production thresholds.

Existing train/score jobs are retained and currently point to the pandas
classification workflow. Both models, prediction tables and the source remain
inspectable under `workspace.skyulf_lifecycle_test`. Failed Polars candidates
v1/v2 remain for diagnosis; the accepted Polars champion is v3. No new persistent
Bundle jobs or schemas were created. One-time fixture/diagnostic/audit runs are
separate from those two user-facing jobs.

Source snapshot 0 was pinned for training and approval; its later append is
source version 1. Sampling applied only to training (160 selected rows including
the final holdout). Scoring used every source record, independently of temporal
training windows or delayed result availability.
