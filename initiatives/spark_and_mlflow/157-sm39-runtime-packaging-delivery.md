# SM-39: generated runtime packaging and compute controls

Date: 2026-10-04. Implementation is uncommitted; no push performed.

## Delivered behavior

- A generated Bundle declares a Core source checkout or release wheel and expected
  version once in `deployment/artifact.json`. `artifacts.skyulf` runs the checked
  builder during deployment. Name/version validation happens before replacement.
- Both jobs use the same prepared wheel and `deployment/requirements.txt`.
  Initial active single/competition/multi-target model choices, ensemble members,
  weighted menus and Optuna selections generate their optional dependencies.
  Inactive branches do not add packages. Training-only chart/SHAP packages live
  in `train-requirements.txt`; custom feature requirements reach both jobs.
- MLflow and optional packages have centralized exact direct pins. This is not a
  transitive dependency lock. Changing Python model declarations after generation
  still requires updating the project's runtime requirements.
- Targets can override job tags, serverless environment/budget policy, and
  policy-cluster runtime/node type/minimum/maximum workers. Existing lifecycle
  graphs and personal-development monitoring isolation are preserved.
- The wheel's SHA-256 is a standards-compatible filename build tag, preserving
  its distribution version and bytes. A changed wheel gets a different uploaded
  dependency identity, avoiding serverless reuse of an earlier same-version
  package. `build.json` records both source/deployed filenames and the digest.
  Inputs inside generated `dist/skyulf` are rejected before any deletion.

## Independent local installation proof

Two empty Python environments installed only the generated wheel and training
requirements; neither borrowed an editable checkout or its packages. The final
content-tagged wheel was installed in both, retaining Core version `0.9.1`.

| Runtime | Exercise | Independent comparison |
| --- | --- | --- |
| pandas / XGBoost | StandardScaler, SMOTE, Optuna, fit, save/load | 120/120 predictions match direct scaler + SMOTE + cloned estimator fit |
| Polars / LightGBM | Same selected workflow with Polars input | 120/120 predictions match the independent reference |

The local full-fit comparison deliberately checks transformation/estimator and
serialization equivalence. It does not claim heldout performance; the real
Databricks check below independently reconstructs its train/holdout split.

Generated project requirements also installed custom feature dependency
`h3==4.3.1`, Optuna `4.5.0`, integration `4.5.0`, CMA-ES `0.13.0`, and
imbalanced-learn `0.14.1`. Local sklearn was `1.8.0`; cloud sklearn was `1.6.1`.
Comparisons were against the matching runtime, not across those versions.

## Actual Databricks execution and mathematical check

Isolated personal-development schema:
`workspace.skyulf_sm39_20261004_0b3bd08d`.
CLI `1.17.0`, profile `skyulf`, generated serverless environment `4`.
No extra runtime packages were manually appended to the deployed jobs.

| Case | Full train run | Automatic score child | Result |
| --- | --- | --- | --- |
| pandas / XGBoost | [910876814670626](https://dbc-45604623-c18b.cloud.databricks.com/jobs/736686023345768/runs/910876814670626) | 981241001505761 | SUCCESS; registered v1 and wrote 180 keyed predictions |
| Polars / LightGBM | [15708757119883](https://dbc-45604623-c18b.cloud.databricks.com/jobs/805533600250458/runs/15708757119883) | 913420670379625 | SUCCESS; registered v1 and wrote 180 keyed predictions |

Configuration: binary classification, three-fold stratified CV, Optuna two trials,
20% stratified holdout with seed 42, StandardScaler and SMOTE. Sample weights
were absent and decision threshold tuning was off. This acceptance checks the
ordinary unweighted flow with optional estimator/tuning packages.

Independent run `111307299686456` passed. It checked 335 installed Python module
hashes against the built wheel, reloaded both registered models, reconstructed
the split/scaler/SMOTE outside Skyulf, and refit a cloned selected estimator.
For each model:

- All 180 saved Delta predictions and all reloaded-artifact predictions matched.
- Probability arrays matched at absolute tolerance `1e-10` (zero relative tolerance).
- Delta keys were identical, unique and complete.
- MLflow heldout balanced accuracy matched the independent calculation within
  `1e-12`: XGBoost `0.8890909090909092`; LightGBM `0.8690909090909091`.

The estimator hyperparameters came from the fitted model. This proves the selected
model's fitted transformation/prediction behavior, not global optimality of the
two-trial search or exhaustive correctness of every estimator/layout combination.

No-new-data score runs `427100231541300` and `129576817515413` both succeeded:
`noop=true`, zero output rows, and original Delta commit version 1 retained.

## Redeployment and cache regression proof

Review identified that overwriting an unchanged wheel filename could leave a
serverless environment using cached code. Changing the distribution version on
every deployment would instead break exact saved-model requirements. The final
fix changes the wheel build tag, preserving the release version.

Run `438256998835078` used a changed, isolated test wheel with one added sentinel
module. It verified that new module really executed, checked 336 module hashes,
and repeated both existing v1 models' numerical checks without retraining.
Run `889974655805009` followed deployment of the original wheel with the final
builder: the sentinel was absent, all 335 original module hashes matched, and
both existing v1 models still passed the same numerical checks.

Both jobs now reference the original bytes with SHA-256
`ca1a169317fc3cf45554828b759897548920aa48230b65f87d99997c8d6a4f14`
in their wheel filenames. The isolated sentinel wheel is no longer selected.

## Verification and limits

The four affected test files passed 262 cases in one full CLI-enabled run.
The subsequently added source/output-collision regression passed alongside the
other four builder cases: 263 distinct affected cases passed in total.
The last two real score runs after the final wheel redeployment also succeeded:
`961483697997751` and `1085406684111662`; both reported `noop=true`, zero new rows
and unchanged commit version 1. All acceptance runs are finished; generated
personal-development jobs have paused schedules.
Ruff check/format, full CI Ty scope, backend/Core CCN-10 gate and generated schema
checks passed. Wrong package/version, changed same-version bytes, repeatable
identity and input/output source collision have regression coverage.

The initial strict documentation build was blocked by pre-existing broken JSON
links in `sampling_weight_acceptance.md` and `weighted_training.md`. The review
follow-up removed those obsolete links and the related support-matrix link;
`mkdocs build --strict` now passes. Targets removed by HEAD `83e89955` were not
restored.
No frontend code changed. No full unrelated backend/Core suite was run for this
delivery. Pre-commit checks passed before the delivery commit; frontend hooks
were correctly skipped because their source scope was unchanged.

Company policy compute and nonempty budget-policy acceptance need the actual
company target (SM-43b). Separate run-identity grants/denials and concurrent writer
acceptance remain SM-37. Generated-project upgrade/CI and broader source/retention
work retain their existing queue IDs. No universal production-readiness claim.

Raw commands, generated projects, receipts, logs and JSON API outputs are under
`tmp_repro_artifacts/sm39/`; affected-suite logs are
`tmp_repro_artifacts/sm39-tag-final.log`. Those temporary artifacts are not committed.

## Threshold and packaging review follow-up (2026-10-04)

The wheel-reference test now checks the declared artifact version, builder and
shared output glob. Eight multi-target graph cases also had stale expectations
predating the monitoring registration branch. They now verify both conditional
monitoring routes and the `NONE_FAILED` scoring join. All 44 packaging/branch
cases passed with real local CLI template generation. No job graph was changed
for these test corrections.

The threshold/lifecycle suites passed 95 cases. Two added pandas/Polars cases
independently reconstruct weighted sklearn fitting and unweighted threshold
selection, then change only calibration weights. Coefficients and threshold
scores match the reference; a 25% calibration split leaves 180 fitting rows and
60 calibration rows. The documented policy remains unweighted evaluation with
no final full-training refit. Weighted business-cost threshold objectives are
not implemented. Documentation now makes these boundaries and positional
probability-column mapping explicit.

Ruff, formatting, full CI Ty scope, CCN-10 and schema checks passed. This follow-up
changes tests/docs only and was not separately deployed or cloud-tested. SM-23c
now explicitly includes separate performance-degradation display acceptance;
that policy/dashboard implementation remains open. Logs are under
`tmp_repro_artifacts/claude-review-*` and are not committed.
