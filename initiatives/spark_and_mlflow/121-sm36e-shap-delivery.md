# SM-36e - Optional SHAP setup and readable results

Status: DONE for the approved scope, 2026-09-30.

## Delivered

- Bundle initialization asks whether to enable SHAP, default off. The shared
  limits apply to single models, competition candidates and generated branches.
  Branch declarations contain their own editable `pipeline.explainability`.
- Training environments explicitly pin `shap==0.49.1` and `matplotlib==3.10.0`
  for serverless and policy clusters. Scoring dependencies remain unchanged.
- `max_samples` limits training rows; `max_features` rejects excessive transformed
  width rather than selecting top features. `max_display_samples` limits saved
  sample waterfalls, with zero keeping only the global chart.
- Saved fitted preprocessing is reused; target, holdout and temporal metadata
  are not explanation features. Existing disabled/unavailable/completed states
  remain explicit, with a separate report status when plotting is unavailable.
- MLflow stores `explanations.json` and portable `explanations.html` containing
  embedded PNG charts, a run link and fitted model URI. Notebook output retrieves
  these reports separately from task values and exit JSON.
- Charts show global mean absolute importance and signed sample waterfalls with
  transformed feature values. Classifier output units may be raw scores/log-odds;
  the report does not label every classifier explanation as a probability.
- Competition children keep their reports. The winner's evidence/report is also
  copied to the parent, preserving the originating child run and model URI.
- Multi-target and legacy notebook adapters use the same default Databricks
  tracking endpoint for report display as their training service.

Frontend SHAP views were inspected for interpretation and contribution direction;
no frontend files were changed. Reports use headless Matplotlib without a CDN.

## Local verification

- Final related regression suite: **193 passed**, 17 opt-in CLI tests skipped.
- Additional search/lifecycle tests: **5 passed** (both engines, phased/legacy
  graph and optional explanation evidence).
- Real CLI SHAP matrix: **12 passed** (three layouts x two compute modes x on/off).
- Strict `dev` Bundle validation: single model, competition and multi-target,
  each including the built wheel. No Bundle was deployed by these validations.
- Real SHAP tests cover fixed regression/classification plus Voting/Stacking
  regression/classification on pandas and Polars. Signed contributions are checked
  against actual model outputs using unrounded transformed inputs.
- Tests cover training-only projection, fitted transform reuse, feature guard,
  unavailable states, escaped labels, report logging, display deduplication by run,
  competition winner provenance and default branch endpoint wiring.
- Full CI Ruff lint/format and Ty scopes passed. Backend/Core Lizard CCN <= 10,
  generated schema freshness and `git diff --check` passed.
- An existing notebook fixture was updated for the previously delivered
  `feature_recipes` metadata; no production recipe behavior was changed here.

## Live Databricks evidence

Workspace: `dbc-45604623-c18b.cloud.databricks.com`, selected profile `skyulf`.
Remote directory:
`/Workspace/Users/edwardwolfe99@gmail.com/skyulf_validation_20260929/shap_20260930_r1`.

1. Initial run `622982689998339` stopped during pytest collection: Workspace files
   do not support the attempted `__pycache__` creation. No source table or model
   had been created. The harness was corrected to copy tests to temporary disk.
2. Same-wheel run **1028237054653182**, task **775992642086359**, succeeded:
   **41 installed-wheel tests, zero failures/errors/skips**, then real Delta read,
   random-search Voting regression, three-fold CV and UC registration.
3. Final-wheel replay **22297034072471**, task **1093777148242722**, succeeded.
   The only functional difference from the training wheel was the default URI
   fallback in two notebook adapters. This run exercised the branch notebook with
   an existing fitted result (training service stubbed), while downloading and
   displaying the real saved MLflow report. It performed no new training/writes
   to source tables or model versions.

Training experiment: `1593812938705200`.
Run: `0b2c911b3cb1404c80018ea01edbcd79`.

[MLflow run](https://dbc-45604623-c18b.cloud.databricks.com/ml/experiments/1593812938705200/runs/0b2c911b3cb1404c80018ea01edbcd79?o=7474646244882000)

- Model `workspace.skyulf_validation_20260929.shap_0930_r1_voting`, version 1;
  no aliases moved.
- Source `workspace.skyulf_validation_20260929.shap_0930_r1_source`: 120 rows,
  96 training / 24 holdout. Existing workspace resources were preserved.
- Heldout RMSE: `0.03670107790562305`.
- Eight training rows explained; transformed features `x`, `z` only.
- One global chart and three sample waterfalls: **four PNG charts** verified in
  both the saved HTML artifact and Databricks' exported notebook results.
- JSON/HTML, JUnit evidence, registry run linkage and final replay were independently
  reread after completion. The global PNG and a signed waterfall were visually
  inspected. Local receipt: `rehearsals/sm36e_20260930/verified.json`.

Training wheel SHA256:
`9e1600e360e9833fe443c77b5bf21552e965d69e2c6c3e55682035fdf047931d`.
Final replay wheel SHA256:
`b9e419ac16bea4e64804273620d80ee4c434f58e4443785d58751bedf7ef1c9e`.
All 290 installed modules were hash-verified remotely. At replay time, source
matched the final wheel after normalizing Windows line endings.

### Reading-guide follow-up

A short guide explains Base, transformed Value, signed contributions, global
importance, waterfall interpretation and classifier output units. Preview run
`937644155495854`, task `1098896439491618`, succeeded using the saved training
evidence and the updated renderer. Exported notebook output contained the guide
and all four charts. MLflow preview run: `cc8b840e125940eaa03a9b9f71b495ec`.

The user then requested removal of synthetic numerical examples. Those examples
were removed from the renderer; its 17 tests and Ruff/format/Ty passed. This final
wording cleanup was verified locally, not rerun remotely. Previously saved HTML
artifacts retain their original text. No new charts were added.

## Limits

This is bounded training explainability, not per-batch scoring explainability.
The tests do not certify every optional estimator/SHAP combination. Unsupported
models or budgets remain explicitly unavailable. Real compute used serverless;
policy-cluster library declarations were generated and checked, not executed on
a company cluster. The full repository suite and frontend build were not run.
