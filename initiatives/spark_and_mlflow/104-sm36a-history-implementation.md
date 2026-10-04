# Temporal history implementation plan

Goal: preserve lag/rolling context across training, validation and successful
prediction batches while keeping model artifacts immutable.

Approved scope: report103 and the user's artifact-tail/rolling-buffer design.
Active queue: OPEN_QUEUE_updated.md. Implementation included in the user-requested delivery commit.

- [x] Core: opt-in history_mode=carry; bounded per-entity artifact seed, explicit
  time validation, causal combination, original row order and immutable apply.
  Reuse train-representation hook so training never consumes its own saved tail.
  Explicit prediction sessions produce a proposed next state without committing it.
- [x] Evaluation: fold-local seeds, independent validation/outer evaluation;
  reject incompatible split policies and pre-split learned-history leakage.
- [x] Backend: artifact-backed first prediction plus explicit continuation state;
  preserve existing prediction response for ordinary requests and propagate
  continuation only after successful predictions.
- [x] Databricks: incremental history in the same atomic commit receipt as output
  and source watermark; failed writes/no-op do not advance it; bind to model
  identity. Period scoring uses explicit context without hidden model mutation.
- [x] Document configuration, limits and source/availability requirements;
  expose real template steps and validate artifact/MLflow round trips.
- [x] Verify both engines, multiple entities, batch equivalence, reorder, gaps,
  new entities, missing/tied/late times, budget overflow, CV and retry/failure.
- [x] Run affected tests, Ruff/full Ty/CCN10 and applicable docs/frontend gates.
  Cloud acceptance requires preparing the concrete wheel/notebook payload first.

Evidence and limits: report105. Backend history
is explicitly persisted by its caller; Databricks incremental history is atomic
with publication. This is not a hidden backend history database.

Model artifacts retain a fixed seed. Ongoing history is a separate versioned
state value, never an implicit in-place mutation inside predict/transform.
Only observations available at scoring time belong in prediction inputs;
target-history forecasting is not silently enabled by this feature.
