# Custom feature engineering and multiple models in one Bundle

Date: 2026-09-25. Status: planned only; no new runtime support delivered here.
Update 2026-09-26: SM-33D/E are complete for their documented local/personal
serverless scopes (reports 59/60); SM-34 is next. The company example migration is later,
after the generic Bundle gate, as requested by the user.

Update 2026-09-26 (SM-33F): the user replaced preprocessing questions/presets with
`src/preprocessing.py`. The generated `build_preprocessing()` recipe returns Core
steps and supports self-contained custom Calculator/Applier classes. Source and
learned state are saved with the model for local/MLflow inference; fold-local CV
uses the same path. This brings the single-file part of SM-36a forward. Package
dependency shipping, sibling modules, eligibility/filter/output rules and multiple
model branches remain planned; SM-36a is not complete. See report 61 for evidence.

## Current capability

The local Bundle already accepts `pipeline.preprocessing` and `pipeline.modeling`.
`local_retraining.train_local_candidate` calls `local_batch.fit_local_workflow`,
which fits the existing `SkyulfPipeline` on the training partition and saves the
full fitted pipeline. Registered, compatible preprocessing/model nodes are used
through that config; the empty preprocessing list is an editable starting point.
This does not establish compatibility for every possible custom node/model.
Core has custom Calculator/Applier registration, tuning and CV machinery. Bundle
CV integration is SM-33D; complete tuning/model-search controls and evidence are
SM-36. The current workflow still has one target, one model name and one selected
model version per training/scoring invocation.

## Guided, modular Bundle setup

User refinement: expose the workflow in progressive setup sections, with
optional scenarios. SM-33E delivered the initial sections, a numeric FE preset,
registered model selection, fixed-model CV and offline preview. Advanced search,
broader custom packaging and multiple-model scenarios below remain planned. Retain one editable, validated workflow
configuration as the runtime source of truth; initialization produces it.

| Setup section | User choices | Default / conditional behavior |
| --- | --- | --- |
| 1. Basics and data | Task, engine, sources, keys, target, read limits, split/sample policy | Single model; no invented dates, no implicit sampling. |
| 2. Preprocessing | No preprocessing, a named starter preset, or an ordered configured chain with per-step columns/parameters | Show only compatible nodes. An optional project module covers missing domain transforms. |
| 3. Model | Registered estimator and its parameters | Task-compatible starter; detailed parameters remain editable. |
| 4. Validation and search | Optional CV, strategy/folds; optional hyperparameter search, backend such as Optuna, search spaces and trial/time budgets | Disabled until selected; show only relevant options. |
| 5. Lifecycle and output | Quality metrics/gates, manual/automatic promotion, score selection, append/full refresh, schedules | Preserve existing independent controls. |
| Optional scenario | Several independent targets, or multiple model outputs combined by explicit rules | Add branch/model-set settings only when requested. |

Design and acceptance requirements:

- [ ] Generate available node/model parameters and validation from existing Core
  registry/metadata/config contracts where available. Reuse Core tuning search
  spaces; do not maintain a competing Bundle-only algorithm/parameter catalog.
- [ ] Allow both interactive initialization and an equivalent noninteractive
  config file. Use concise presets/prompts for common setup; advanced ordered
  pipelines/search spaces stay editable in generated config/project code rather
  than requiring a long questionnaire for every parameter.
- [ ] Show a reviewable plan before expensive execution: feature steps in order,
  estimator, final holdout, optional CV/search budget, component models and output.
  Initializing or editing settings must not start training or deployment.
- [ ] Separate final holdout from training CV. When tuning uses CV, reuse one
  coherent fold configuration and refit learned preprocessing inside each fold.
  Do not run duplicate CV or use the final holdout to choose hyperparameters.
- [ ] Present Optuna as a search backend rather than a model or CV strategy.
  Reject unavailable dependencies and unsupported combinations before compute.
- [ ] Presets must not silently select every numeric/category column or introduce
  target leakage. Record actual feature/target bindings and allow user edits.
- [ ] Expose different concepts distinctly: comparing algorithms for one target
  selects a winner; several targets/models used together retain components.
  Do not make a model competition the implicit multi-model scenario.
- [ ] Add scenario-specific project modules only when selected; a single-model
  project stays small. Optional support does not require separate user-facing
  jobs for each model. Model-set semantics remain those in SM-36b/SM-36c below.
- [ ] Verify initialization-to-config-to-runtime parity, parameter roundtrips,
  conditional prompts, invalid combinations and disabled-feature defaults for
  both engines. Examples must distinguish delivered features from planned ones.

Delivery is incremental: SM-33D supplies CV/selection contracts; SM-33E adds the
first guided basics/preprocessing/model/CV setup and live examples. SM-36 adds
search controls; SM-36a adds custom modules; SM-36b/c add optional multi-model
scenarios. SM-42 completes the combined guided setup/examples/operator review,
then SM-43a verifies it. Existing initialization is not claimed to have all these
sections yet. SM-44 remains the later company migration.

## Core CV and tuning routing audit

Source inspected on 2026-09-25. This defines the implementation contract;
the fixed-model CV wiring is now locally verified in report 59; search remains planned.

User decision: offer ordinary model training first, with optional CV. Advanced
search adds strategy, search space and budgets to the selected base model.
Ordinary training permits explicit model parameters; omitting them retains the
selected Core model's defaults. CV evaluates that fixed configuration without
searching for better parameters. Tuning searches configurations and refits the
selected result. Keep one visible CV section, whose settings follow the selected
training path rather than launching duplicate evaluation.

| Selection | Existing Core route | Settings to map |
| --- | --- | --- |
| Ordinary model, CV off | `SkyulfPipeline.fit` | Registered model type and configured parameters |
| Ordinary model, CV on | `StatefulEstimator.cross_validate`, then final pipeline fit on training rows | `n_folds`, `cv_type`, `shuffle`, `random_state`, `time_column`, fold preprocessor |
| Tuning, CV on | `SkyulfPipeline` with `modeling.type=hyperparameter_tuner` and `base_model` | `TuningConfig`: `cv_enabled`, `cv_folds`, `cv_type`, `cv_shuffle`, `cv_random_state`, `cv_time_column` |
| Tuning, CV off | Same tuner, using an internal training-only validation split | `cv_enabled=false` still evaluates candidates; it does not optimize training scores without validation |

Core references relative to `skyulf-core/skyulf/`:

- `modeling/base.py`: `StatefulEstimator.cross_validate` uses `dataset.train`.
- `modeling/cross_validation.py`: splitters, per-fold evaluation and aggregated
  mean/std/min/max metrics.
- `preprocessing/fold_adapter.py`: `FeatureEngineerFoldAdapter` rebuilds learned
  preprocessing per fold; reuse this instead of globally fitting before CV.
- `pipeline/_pipeline.py`: `_fit_tuning_pipeline` already passes raw partitions
  and fold preprocessing to the tuner, then retains final fitted FE for inference.
- `modeling/_tuning/schemas.py`: strategies `grid`, `random`, `optuna`,
  `halving_grid`, `halving_random`; `search_space`, `strategy_params`, `metric`,
  `n_trials`, `timeout` and CV fields. Expose only validated combinations and
  installed optional backends. Budget semantics depend on the strategy; grid
  combinations must not be presented as capped by `n_trials` without verification.

### Existing Canvas Basic/Advanced behavior

The user's Canvas examples were traced through
`frontend/ml-canvas/src/modules/nodes/modeling/trainingSettings/CrossValidationSection.tsx`
and `backend/ml_pipeline/_execution/engine/_node_runners.py`.
The routing table above describes the planned standalone Bundle integration;
it must not be read as a description of current Canvas backend dispatch.

Canvas supervised Basic and Advanced both use `_run_training_tuned` internally.
Basic (`fixed`) creates a grid with one value per configured hyperparameter and
one candidate. Advanced (`tuned`) supplies the search strategy and search space.
Both can then call `_run_tuned_cv` to evaluate the fixed/selected parameters and
report the full metric panel. That evaluation trains fresh fold models; it does
not change the persisted model or select new hyperparameters.

The same `cv_*` settings currently feed search evaluation and control the later
CV report. Consequently the Advanced UI's description of a post-search report
does not mean these settings are independent of candidate evaluation. Explicit
validation data can also override the tuner's splitter as described below.

- [ ] Preserve recognizable Basic versus Advanced choices in Bundle setup:
  Basic uses fixed/default parameters, Advanced adds search space, strategy,
  objective metric, trials/time budget and search seed.
- [ ] Explain search CV versus an optional additional post-selection CV report.
  Reuse fold settings, but do not silently perform both or imply a CV checkbox
  controls reporting only when it also changes candidate selection.
  Post-selection CV is diagnostic, not an untouched final-test estimate.
- [ ] Map folds, stratified method, shuffle and fold seed to existing Core
  fields. Keep the search seed distinct from the fold-split seed.
- [ ] Treat decision-threshold tuning as a separate opt-in capability with
  existing Core binary/probability/validation requirements. Do not imply that
  selecting F1 enables it or that multiclass targets automatically support it.

Implementation requirements from the audit:

- [ ] Map grouped setup choices to these actual APIs. `TuningConfig` uses flat
  CV fields; do not pass an invented nested CV object straight to the tuner.
- [ ] Preserve model defaults and user overrides in ordinary training. For
  tuning, verify how fixed base parameters and searched parameters compose;
  record effective settings, best parameters and trial results in MLflow.
- [ ] Keep the final holdout out of CV, search and learned preprocessing. An
  explicit tuner validation dataset selects `PredefinedSplit` and overrides CV;
  never pass the final evaluation holdout there. Any internal validation split
  must come only from the training partition.
- [ ] Preserve temporal CV ordering metadata through the Bundle split, which
  currently projects features and target only. Keep time metadata out of model
  features. Require an explicit ordering contract and disable temporal shuffle.
- [ ] Reject invalid combinations before compute. Core can fall back from
  stratified regression or unknown splitter names; the Bundle must not silently
  change the user's requested validation method.
- [ ] Keep tuner scoring names (for example `rmse`) distinct from outer MLflow
  comparison names (for example `heldout_rmse`). Reuse Core scoring/direction
  rules instead of passing the outer metric name unchanged.
- [ ] Extend workflow model validation for the tuner wrapper and its compatible
  registered base model. The current validator resolves only ordinary registered
  model calculators, although `SkyulfPipeline` already supports the tuner.
- [ ] Do not advertise true nested hyperparameter search. Core standalone
  `nested_cv` reports inner diagnostic scores for an unchanged configuration;
  the tuning splitter alone does not establish an outer nested-search loop.
  Initial Bundle CV choices remain K-fold, stratified, shuffle and time-series.

Verification: 96 existing Core tests passed across `test_cv_per_fold_refit`,
`test_core_pipeline_tuning_leakage`, `test_cross_validation_time_sort_integrity`
and `test_tuning_time_series_holdout`. This confirms the exercised Core behavior,
including fold refitting and time ordering, not delivered Bundle CV integration
or exhaustive verification of every search backend.

## Reference inspected

Read-only code inspection of
`C:/Users/Murat/Downloads/codes-main (2)/codes-main/mlmodeltesting`:

- `run_monthly_job.py:run`: load companies, score them, save results. No training
  step is present in this production entrypoint.
- `model/scoring.py:score_dataset`: adapts input into Polars and output into rows.
- `model/vendor_leadgen.py:MODELS`: loads four model files for `hazard_A_pct`,
  `mean_age`, `mean_salary`, and `mean_AuM`.
- `compute_leadgen`: cleans/imputes the source, selects eligible prospects, makes
  four predictions from the same eight feature columns, and applies
  `RiskGroup`, `AumGroup` and `LeadGen` rules to produce one color per company.
  The prediction calls are sequential in the code; no parallel execution claim.
- `_clean_and_impute`: computes group/global means and modes from the current
  scoring frame. This is batch-dependent state, not a stored training imputer.
- `XGBRegressorFormula._design_matrix`: creates a Patsy design matrix and reindexes
  to saved feature names. Encoding/category parity needs explicit verification.
- `data/database.py`: reads `CompanyMonthly` and upserts scores into `ProspectScore`
  keyed by company identity.
- `docs/artifacts/reconstructed_training.py` identifies itself as reconstructed
  training code, not the original recovered training source. It must not be used
  as proof of exact retraining parity without independent validation.

No reference code was executed, pickle loaded, database contacted, or artifact
copied. Documented historical row counts/metrics were not reproduced. This is
multiple component models plus business rules, not a competition that discards
all but one winner, and the scoring code does not feed one prediction into the
next model's input.

## SM-36a - Project-owned feature engineering and output rules

Dependency: SM-33D. Execute after SM-36 in the local improvement sequence.

- [ ] Add an opt-in project Python package, with an editable feature-engineering
  module (for example `src/<project_package>/features.py`). Reuse existing
  Calculator/Applier registration; avoid editing Skyulf source per project.
- [ ] Separate stateless expressions, learned transformations and source
  eligibility filters. Preserve record keys and explicit row inclusion/exclusion
  evidence rather than hiding dropped rows inside prediction.
- [ ] Fit learned state only on training rows and inside each CV/tuning fold.
  Reuse the saved state during evaluation, approval and inference; handle unseen
  categories and group-imputation fallbacks explicitly.
- [ ] Package project code, dependencies, state and code identity with the model
  delivery contract. A source file used during train must also be importable when
  MLflow loads the artifact in a fresh scoring/serving process.
- [ ] Support named deterministic post-prediction rules with declared output
  columns/types and versioned parameters. Keep company rules out of generic Core.
- [ ] Verify pandas/Polars parity, clean-process artifact loading, train/inference
  schema and feature order, keys, nulls, unknown groups, and row-scope contracts.
  Expose readable config and an English example before marking the task done.

## SM-36b - Multiple training branches from one source

Dependencies: SM-36, SM-36a.

- [ ] Allow named training branches with their own target, features, preprocessing,
  estimator, optional tuning/CV, quality metric and registered model name.
- [ ] Pin the shared source snapshot and reproducible split/sample evidence.
  Define per-target label availability explicitly; missing labels for one target
  must not silently discard training rows for every other branch.
- [ ] Separate independent targets from a same-target algorithm search. The latter
  is model selection; different-target models may all remain in use.
- [ ] Log a parent workflow run and linked component runs, fitted pipelines,
  metrics, dependencies and immutable model versions. Keep each target's
  champion/challenger comparisons within its own task and metric contract.
- [ ] Support sequential execution first; any concurrent mode must enforce an
  aggregate local memory budget. Three models need not mean three user-facing jobs.
- [ ] Test three branches, distinct label subsets, reproducible retry, a failed
  branch and leakage-free evaluation. Do not publish a complete model set when
  required components failed.

## SM-36c - Use multiple models together and publish coherent output

Dependency: SM-36b.

- [ ] Define a versioned model-set manifest pinning all required component model
  versions, feature code/state, output rules and schemas for one scoring run.
  Resolve mutable aliases once before prediction; never mix versions mid-run.
- [ ] Offer multiple named prediction columns and optional rule-based composition
  into a final result. Join by record identity, not positional zip assumptions.
- [ ] Make model-set activation, validation and rollback coherent. Specify whether
  a component change requires validating/releasing a new complete set, with a
  receipt recording exactly which versions were used.
- [ ] Extend append/full-refresh provenance so a component/rule change is visible
  as a set-version change; the existing single `model_version` must not conceal
  different component combinations.
- [ ] Make partial model failure, row eligibility, output publication and retries
  explicit. Do not write partially composed final scores as successful results.
- [ ] Verify independent outputs and combined results against direct component
  calls, plus key reordering, schema failures, retries and set rollback.
  Package parity for future serving is required; endpoint deployment remains SM-19.

## SM-44 - Later company example migration and acceptance

Dependencies: SM-43a, SM-36c, source/target/environment details confirmed at execution.
Status: LATER; user explicitly deferred this until the generic Bundle is built.

- [ ] Re-audit the supplied model version and authoritative training material.
  Separate loading existing trusted artifacts from retraining new Core models;
  they are different acceptance paths with different parity claims.
- [ ] Map the four targets, shared features, candidate filters and color rules
  into the generic extension/model-set contracts without hardcoding them in Core.
- [ ] Establish baseline predictions and key membership on representative data.
  Decide explicitly between preserving whole-batch imputation semantics and
  retraining with frozen fitted imputation; changing this can change scores.
- [ ] Verify Patsy/category/feature-order parity, model dependencies, all rule
  boundaries and unseen/missing input behavior before replacing the existing job.
- [ ] Choose full-snapshot versus incremental behavior consciously: whole-batch
  statistics cannot be recomputed solely from new rows with identical semantics.
- [ ] Verify source eligibility and final table publication with company keys,
  per-component lineage, metrics, scheduling and rollback in approved resources.
  Live execution requires the applicable environment/resource authorization.

## Verification expectations

Each implementation task needs local integration tests and generated-project
coverage. Include the generic extension/model-set path in SM-42 examples and
SM-43a acceptance. No company production readiness or runtime feature completion
is implied by adding this plan. SM-44 remains a separate later migration.
