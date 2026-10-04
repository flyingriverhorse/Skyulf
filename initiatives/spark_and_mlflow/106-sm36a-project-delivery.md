# SM-36a: project scoring rules and complete bounded delivery

Date: 2026-09-29. Completes the remaining approved SM-36a slice after temporal
history commit `84a7dcb4` and project packaging/custom steps commit `9cc81304`.
Status: DONE for the approved scope. Local and live acceptance passed.
This report is included in the scoring-policy delivery commit after `84a7dcb4`.

## Contract and implementation map

- `inference/project_scoring.py`: saved `build_scoring()` configuration with
  named/versioned eligibility and output callbacks, finite JSON parameters,
  declared scalar output schemas and strict row/index/column validation.
- `inference/local_scoring.py`: scoring-only wrapper; the original
  `predict_local_pipeline` remains the predictor used for training evaluation
  and candidate/champion comparisons. Both engines use defensive pandas rule
  inputs; model/feature processing uses the recorded fit engine.
- `integrations/databricks/project.py`, `local_sdk.py` and MLflow `local_model.py`:
  capture editable policy once, then replay saved source/parameters on scoring,
  metadata probes and pyfunc loading. Model digests bind the entire policy.
- Delta incremental and period writers keep every requested key. Policies add
  `scoring_status` and `exclusion_reason`; excluded estimates/business outputs
  are null. Receipts record predicted/excluded counts and total outcome count.
  All-excluded increments publish their outcomes and cursor atomically. Failure
  before commit preserves the previous receipt; replay does not duplicate keys.
- Eligibility runs before temporal FE. Excluded rows do not enter lag/rolling
  history; excluded-only batches preserve the prior tail. Raw schema validation
  applies even when every row is excluded. Training-only filters remain separate.
- `_project_files.py`, `project_package.py`, `project_dependencies.py`: explicit
  `assets.json` and exact pins in `requirements.txt`, embedded in the versioned
  source snapshot. `read_project_asset(__package__, path)` provides immutable
  bytes after original project files disappear. Verify exact installed pins
  before project code or pickle loads, including cached modules. Artifact and
  MLflow environment metadata contain the declared pins.
- Template `src/features/scoring.py`, `custom/scoring_custom.py`, empty
  `assets.json` and commented `requirements.txt`: opt-in recipe and usable
  missing-field eligibility/prediction-band rules. Preview displays saved rule
  declarations; job reports display prediction/exclusion counts.

## Deliberate bounds

Code, encoded assets and pins share the existing 64 KiB snapshot limit. Explicit
asset paths cannot escape the package; asset symlinks are rejected. Requirements
are exact installed distribution versions, without options/URLs/markers/extras.
Runtime loading verifies but never installs them: job environments must already
provide dependencies. Large downloaded sentence models and optional H3 execution
are not claimed by this delivery. New generic predicate/data-quality gates and
custom pre-split value edits remain the separate scopes in the active queue.

Scoring callbacks are trusted producer code, like existing custom preprocessing.
They must be deterministic; this is not a sandbox or a proof of Python purity.
Introducing new output columns requires an exact compatible target schema; a
fresh target is the supported migration path. Business output nulls are allowed;
missing/nonfinite estimator outputs must fail rather than count as predictions.
No new frontend/backend API controls were required for these project Bundle
hooks. Multi-model training/composition remains SM-36b/36c.

## Verification

- Final new project/scoring/delivery selection: **111 passed, 6 skipped**.
  Five skips require local Delta and are covered in the cloud payload. The
  native Windows symlink test lacks privilege; simulated regular/broken symlink
  rejection tests passed. The final guard adds 20 missing/nonfinite estimator
  output tests across prediction/probability columns and both engines.
- Wider regression: 343 passed, 5 local Delta skips. The skipped cases are in
  the cloud acceptance payload, with real source/target tables.
- After the cloud adapter fix: 29 focused MLflow/scoring tests passed, including
  the red/green preloaded-adapter regression. The existing MLflow promotion
  regression suite also passed: **64 tests**.
- MLflow tests log to real SQLite tracking, inspect persisted dependency pins
  and reload in fresh isolated Python processes after source deletion.
- Regression/classification and selected two-member voting ensembles run with
  both engines, preserving probability values and nullable exclusion outcomes.
- Ruff, CI format scope, full CI Ty scope and backend/Core CCN <= 10 passed.
  Strict MkDocs and all applicable pre-commit hooks passed. No frontend code
  changed in this slice. Actual CLI project generation and offline preview
  passed; focused template/output rendering checks: **76 passed** (overlaps
  the broader regression suite).
- Review corrected metadata probes bypassing scoring rules and legacy period
  record keys named `scoring_status` being confused with policy status.

## Live acceptance

Existing personal test workspace: dbc-45604623-c18b.cloud.databricks.com.
Payload: `skyulf_lifecycle_test/sm36a_delivery_20260929_r1`.
Initial run: `682479310287443`, task `863806902184704`.
Initial wheel SHA256: `8e11796b0a5a29cb1ff3e8d09ee1658eda5f94a0b8d8303c0bffbcf451ffa530`.
This initial wheel precedes the final missing-estimator-output guard.
Initial payload result: **96 passed, 1 failed, zero skipped**. All five actual
Delta cases passed; the failure was an absent SQLAlchemy dependency for the
local SQLite MLflow test (not a production writer failure).
Scoring-guard wheel SHA256: `a6224e643cf9f49e9807b9f0c3839efa9458cdb048373cc43e5d5b5458bd69fa`.
Matrix run: `300127116664540`, task `778599775886620`.
Payload result: **116 passed, 1 failed, zero skipped**, again with all five
actual Delta cases passing, plus the missing/nonfinite estimator guards.
The single failure was the SQLite harness dependency noted above.

MLflow-only retry `1119642384131604` supplied SQLAlchemy/Alembic, but an
unbounded SQLAlchemy 2.x selection lacked the legacy pool class imported by
the workspace MLflow runtime. Pinning the verified `sqlalchemy==2.0.43`
resolved that harness issue. Retry `316064635985728` then exposed a real
production defect: Databricks MLflow serialized a preloaded PythonModel whose
cached pipeline referenced digest-qualified custom modules. Fresh-process
unpickling failed before `load_context` could install those modules.

A regression reproduced that exact sequence locally. `SkyulfLocalPythonModel`
now serializes all existing instance metadata with `_artifact=None`, leaving
the live instance untouched. `load_context` restores verified source/assets
and fitted state from the separately saved pipeline. The fix also avoids
serializing a duplicate cached model. Independent review found no blocker.

Final adapter-fix wheel SHA256:
`4a17b5bd5b548e8352da7ccad93f7b2e65d36c6e1fb6d9618d0930babb48f7a6`.
Final focused run: `626703200367513`, task `994131779929034`.
**6 passed, zero failed, zero skipped** on Databricks, including actual MLflow
log/download/load in a fresh isolated subprocess after source deletion and
preloaded-adapter pickle replay with preserved metadata. Exact dependency
pins are verified in the logged environment and before source loading.

The final run targets the changed MLflow delivery path; it does not claim a
new full Delta matrix. The prior 116 passing cases include both-engine Delta
publication/retry tests; the only production change afterward is adapter cache
serialization. At that acceptance, all wheel Python sources matched the working tree after
newline normalization. The later follow-ups are covered by the final combined acceptance below. Raw outputs are retained in
`rehearsals/sm36a_delivery_20260929/result_*.json`. Notebook SUCCESS alone was
not used as acceptance: each JSON pytest failure list was inspected.

Test tables use isolated `workspace.skyulf_lifecycle_test.sm36a_delivery_20260929_`
names and are removed by the fixture after verification. Tests do not register
models or move registry aliases. Rehearsal wheels/payloads/results stay untracked.


## User follow-up: reuse pre-split or choose custom scoring (2026-09-29)

The generated `src/features/scoring.py` now exposes two independent switches:

- `USE_CUSTOM_SCORING=False`: capture/reuse the pre-split recipe as prediction
  eligibility. True selects the separate custom eligibility/output builders.
- `SKIP_TARGET_PRE_SPLIT_STEPS=False`: reject target-reading steps when reusing.
  True skips target-dependent filters, retaining other steps. It never changes
  the model target. Mixed target/feature filters are skipped whole, while fixed
  target/feature edits are projected onto feature fields.

`scoring_pre_split.py` resolves and freezes the effective recipe. The shared
`apply_pre_split_step` executes existing Core calculators/appliers and the same
survivor validation in training and scoring. Selection happens on a copy; original
accepted inputs go to the saved model, avoiding double normalization. Custom
registrations restore from saved project code in a fresh process. First exclusion
reasons identify their pre-split step. All requested keys remain represented.

Scoring dependencies must be declared in `input_columns`; missing fields fail
at configuration time, rather than silently skipping a rule. Deduplication is
batch-local, not cross-batch global uniqueness. Empty recipes preserve the normal
schema. Existing artifacts retain their saved policy; these switches affect newly
trained versions. Returning None explicitly disables policy processing as before.

Template documentation now explains BEFORE eligibility versus AFTER output rules,
separates their builders/implementations, and includes required-value, numeric-range
and prediction-band usage. This change was requested after the cloud runs above.

Verification: **285 affected tests passed** (including 18 pre-split reuse cases
and the new template cases). Both engines, target skip/custom selection, bounds,
dedup, fixed edit order, all-excluded inputs, exact-once normalization and isolated
fresh-process custom loading after deleting the project folder are covered.
Full Ruff, full CI Ty, Lizard CCN 10, formatting and strict MkDocs passed.
This follow-up was subsequently verified in the final combined acceptance below.


### Combined scoring follow-up

The user requested both filter sources together. SCORING_MODE now replaces the
editable USE_CUSTOM_SCORING boolean with pre_split/custom/combined; the default
remains pre_split. SKIP_TARGET_PRE_SPLIT_STEPS stays independent and applies to
both reuse modes. Previously saved boolean-based source/policies still load.

Combined mode evaluates pre-split first, retains its exclusion reasons and only
passes surviving original inputs (fresh callback indexes) to custom eligibility.
Predictions and output rules run on survivors of both checks. All-excluded batches
skip custom callbacks/model; empty pre-split recipes retain custom rules. Invalid
mode values fail before training. Source keys and original pandas indexes survive.

Final affected selection: **200 passed** across scoring, reuse, saved delivery,
project preprocessing and Bundle template tests. Tests verify order with callbacks
that fail if pre-split-excluded rows reach them, both engines, keyed results,
all-excluded handling, independent target skipping, invalid modes, custom-only
behavior and fresh-process combined-model reload after removing source files.
Ruff, formatting, full CI Ty, Lizard CCN 10 and strict MkDocs passed.
This follow-up is included in the delivery commit and final live acceptance below.


## Final scoring-mode Databricks acceptance (2026-09-29)

Final wheel SHA256:
`be5b352bde51ad2bd018261d3adae84dc393c9a247082e9ef327cdf282798d53`.
All Core Python wheel sources match the committed sources after newline normalization.
Payload: `skyulf_lifecycle_test/sm36a_delivery_20260929_r1/combined_final`.
Run: `530905329987142`; task: `150335010820829`.
**156 passed, zero failed, zero skipped**, with notebook result JSON inspected.

The six test modules cover saved assets/pins, isolated project reload, MLflow
log/load, scoring outcomes, pre-split reuse and combined ordering on both engines.
Seven actual Delta tests verify incremental custom/combined rules on both engines,
period publication/replay on both engines, and a legacy key named scoring_status.
Assertions check persisted keys, nullable excluded predictions/business columns,
first exclusion reasons, count receipts, injected pre-commit failure with unchanged
Delta history, successful retry, all-excluded cursor advancement, and no-op stability.
Combined mode exercises both pre-split and custom exclusions in the same batch,
with the target-reading training filter explicitly skipped.

Initial run `909124804746854` failed before test collection because the notebook
referenced a different directory from its upload destination. Only the rehearsal
notebook path was corrected; the wheel was unchanged. No product test failure
occurred in the final run. The serverless test environment uses client 4, pytest 8,
SQLAlchemy 2.0.43 and Alembic 1.x with the workspace MLflow runtime.

Fresh local commit selection: **161 passed, 1 skipped** (Windows symlink privilege).
The six cloud modules also collect **156 tests** locally. Ruff, formatting, full
CI Ty and backend/Core Lizard CCN 10 passed; applicable pre-commit hooks passed.
Tables use isolated `workspace.skyulf_lifecycle_test.sm36a_combined_20260929_`
names and fixture cleanup. No registered models or aliases are changed.
Raw rehearsal outputs and the wheel remain untracked under
`rehearsals/sm36a_combined_20260929/`.
