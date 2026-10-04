# SM-36a: temporal feature history

Date: 2026-09-28. Status: investigation complete; implementation pending.
Predecessor: `9cc81304` (project packages, inline custom steps and CV test matrix).

## Reproduced gap

Current LagFeatures/RollingAggregate artifacts contain operation settings, not
historical rows. Applying a trained operation to a new increment starts a new
window. Databricks `_incremental_prediction_bridge` supplies only that increment
to `prepared.predict`; it does not retrieve feature history.

Executed both native engines with one entity, four chronological observations
whose values were 10, 20, 30, 40, fitting operation settings on the first three:

| Engine | Feature | Last row alone | Same row with preceding observations |
| --- | --- | --- | --- |
| pandas | lag 1 | null | 30 |
| Polars | lag 1 | null | 30 |
| pandas | rolling mean, window 2 | 40 | 35 |
| Polars | rolling mean, window 2 | 40 | 35 |

The diagnostic asserted all eight outputs; raw results are in the ignored
`rehearsals/sm36a_history_audit.json`. This proves the context gap, not a defect
in the existing within-frame shift/window algorithms. Neither successful
temporal CV nor source-package replay supplies missing observations.

## Relevant implementation boundaries

| Area | Existing code | Required extension |
| --- | --- | --- |
| Temporal operations | `skyulf/preprocessing/time_series/lag.py`, `rolling.py` | Reuse calculations; explicitly validate time, entity and history requirements |
| Fold preprocessing | `skyulf/preprocessing/fold_adapter.py` | Carry permitted historical context independently from rows used to fit learned steps |
| Training source | `integrations/databricks/local_retraining.py` | Pin and read bounded context alongside selected training observations |
| Incremental reads | `integrations/databricks/local_incremental.py` | Fetch required predecessors for new keys without republishing predecessors |
| Period scoring | `integrations/databricks/local_batch.py` | Use the same context contract as incremental scoring |
| Model artifact | `inference/local_pipeline.py` | Persist history requirements and validate the caller's supplied context |
| Project config | `integrations/databricks/project.py`, workflow validation/preview | Expose explicit settings with offline validation and readable requirements |

## Implementation contract

### Backend/artifact review after user feedback

The user requested checking the existing backend before choosing an external
history provider. Saving a bounded history tail in the artifact is a valid
option to assess first; Databricks Feature Engineering or a new source reader
is not a prerequisite for the initial train-to-predict continuity contract.

Verified existing behavior:

- Backend `_run_transformer` saves both processed data and the fitted
  FeatureEngineer. Its composite model bundle retains upstream fitted steps,
  including steps placed before the splitter; it does not join saved data-node
  outputs into a future prediction request.
- `LagFeaturesCalculator.fit` and `RollingAggregateCalculator.fit` save settings
  only. `StatefulTransformer` applies these settings separately to train/test/
  validation frames after a split. DeploymentService passes only the current
  request frame to the saved FeatureEngineer.
- Executed the actual FeatureEngineer and backend artifact-store/deployment
  transform path: pre-split final values were lag=30/rolling=35; separate test
  and reloaded-backend request values were null/40 for a new value of 40.
- 100 existing temporal leakage, backend admission and deployment tests passed.
  An initial pytest temporary-directory permission failure was resolved by using
  a workspace-local basetemp; no production change was required.

Before implementing the contracts below, compare a versioned immutable artifact
tail (per entity, only permitted training history) against a caller-supplied or
Databricks-provided history frame. An artifact tail can seed the first future
batch, but does not acquire subsequent observations automatically. Cross-batch
advancement needs explicit durable state or supplied history; do not mutate the
model artifact implicitly during prediction. A pre-split full-frame tail must
not be copied into every fold because it can contain heldout/future observations.
The initial plan's source-reader choice is therefore provisional.

1. Keep record keys, entity keys, event time and data-availability time separate.
   Source observations and history must be read from a pinned snapshot. Event time
   alone does not establish when a value became available.
2. Start with observed non-target features and a frozen forecast origin. Every
   historical row must be available by that origin and precede the scored
   observation. Reject missing/ambiguous times and duplicate entity/time points
   until an explicit tie policy is provided. Do not silently assume target values
   become observable at event time.
3. Keep the requested prediction rows separate from history rows using immutable
   record keys. Compute temporal features with bounded context, then restore the
   requested keys/order. History rows must never enter prediction publication or
   inflate output counts/watermarks.
4. Resolve required predecessors from explicit lag/window settings. Apply limits
   to both requested and context rows/bytes before local materialization. Missing
   predecessors remain missing feature values under the declared policy; they
   must not trigger an unbounded source read.
5. Learned preprocessing, tuning and thresholds fit only the appropriate training
   partition. Historical context must not enlarge their fit population. Inner,
   outer and final holdout boundaries each supply an independent origin/context.
6. Persist requirements and source/code identities with the model; record actual
   snapshot, origin, context count and requested key identity in run evidence.
   Loading an old model retains its old behavior. History reads belong in the
   Databricks adapter; Core calculations receive explicit bounded frames.
7. The first supported evaluation path is temporal splitting with a fixed origin.
   Walk-forward availability, label-history features, random/group CV combinations
   and serving requests without a history provider require explicit contracts
   before support is claimed. Existing ordinary CV remains unchanged.

## Delivery order and acceptance

- [x] Inspect actual Core and Databricks history boundaries.
- [x] Reproduce the missing-context behavior on pandas and Polars.
- [ ] Implement a validated requirement/context contract and keyed temporal
  feature calculation. Test exact lag/window outputs, multiple entities,
  reordered inputs, gaps, unseen entities, ties, missing times and future rows.
- [ ] Connect training, inner/outer CV and final holdout. Prove future/heldout
  values cannot change training features or learned statistics; preserve labels.
- [ ] Connect pinned bounded history reads to period and incremental scoring.
  Prove full-frame versus chunked prediction parity, two increments, no-op,
  requested-key coverage, row budgets and failure before publication.
- [ ] Persist requirements/evidence and test fresh-process local/MLflow loading.
- [ ] Add generated-project configuration/preview coverage and clear usage docs.
- [ ] Run local gates and a separately authorized cloud acceptance scenario.

Keyed scoring exclusions, post-prediction rules and dependency/asset packaging
remain the other open SM-36a deliverables. SM-36b/c stay dependent on completing
the relevant contracts; this investigation does not mark SM-36a complete.
