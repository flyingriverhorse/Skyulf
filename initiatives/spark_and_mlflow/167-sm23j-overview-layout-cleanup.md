# SM-23j — Compact Overview and enrollment placement

Date: 2026-10-05. Base commit: `ebd57f04`.

## User-requested change

Move Current monitoring enrollments from Overview to Drift and data quality,
remove its displayed Monitoring identity column, and remove Latest saved
performance evidence by enrollment from Overview. Retain performance counters
and the existing detailed Model performance page.

The moved table is the full store inventory, independently of the selected chart
context. Its description makes that scope explicit. The underlying monitor ID
remains selected but hidden, preserving distinct enrolled records. All 28 dataset
definitions are unchanged; measurement, policies and retraining are unchanged.

## File ownership and task sequence

`examples/databricks_monitoring` is the standalone shared monitoring Bundle.
Its resource loads `src/monitoring.lvdash.json`. Generated producer projects link
to the shared dashboard and configure their native refresh tasks. Copying this
resource into each model-project template would create multiple dashboards.
The JSON remains the single shared definition; README clarifies this ownership.

SM-56's actual preprocessing safety inventory is complete. SM-56 portable codec
and broader node support remain open. SM-57 depends on that inventory and its own
partition-safety gate, not completion of every portable JSON codec: trusted pickle
and portable `predict_spark` are different routes. SM-57 currently has an inventory
and plan, not a delivered runtime route. See records 165 and 166.

## Verification

- Existing affected dashboard/overview tests: 37 passed. No new mirrored tests
  were added for this reversible widget move.
- Exact equality of all 28 dataset definitions against HEAD passed. Independent
  layout inspection verified no overlapping widgets on any page.
- All 56 default/NULL live SQL cases passed. Same-ID publication/readback matched
  current source, with `embed_credentials=false` and the existing warehouse.
  Revision: `2026-10-05T11:28:37.716Z`; source SHA256:
  `b3841707ad8a8b30891b8044cb6a529b0fd93fa51a96a7b98873e62dbfc10863`.
- Chrome confirmed Overview no longer contains either table or the unused evidence
  heading; its performance counter still shows 5 insufficient-evidence contexts.
  Drift contains all six enrollment rows, with no Monitoring identity header.
  The previously closed CDP session was reopened using the same approved isolated
  Chrome profile; no new login or personal browser profile was needed.
- `git diff --check` passed. No Python/frontend application code changed, so no
  additional static suite, frontend build or MkDocs run was required.
- Applicable pre-commit gates passed, including JSON and whitespace checks.

Evidence: ignored `tmp_repro_artifacts/sm23j/`, including publication/source
readbacks, SQL outcomes, browser assertions and screenshots. No job, data,
training policy or schedule changed. Existing native refresh still addresses
the same dashboard ID `01f1c090238e1b6da5d633032ad9960b`.
