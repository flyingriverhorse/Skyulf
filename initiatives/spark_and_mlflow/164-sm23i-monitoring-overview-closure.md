# SM-23i — Overview performance and batch-monitoring closure

Date: 2026-10-05. Base commit: `7c944603` (SM-23h).

## Approved scope

Finish the current SM-23 batch-monitoring scope before starting SM-57. Overview
must distinguish drift/data-quality health from baseline-based model performance.
Counts represent enrolled monitoring contexts, not necessarily unique models.
An old successful performance check must not hide missing current evidence.

Keep the same published dashboard and producer jobs. No live policy is changed
to retrain, and no training or cluster is created to populate demonstration data.

## Closure boundaries

- SM-23a batch quality/drift, delayed outcomes, shared Delta storage and native
  AI/BI are delivered, with distributed Spark measurement and native job refresh.
- SM-23c functional drift/performance policies and guarded training requests are
  delivered. Monitoring never approves/promotes a trained model automatically.
- SM-23b endpoint inference-table monitoring still depends on SM-19a serving.
- Separate-identity/company acceptance remains in SM-37/43b. This closure does
  not claim production acceptance on behalf of those tasks.
- Optional native Data Profiling and custom slices are future integrations.
  Execution/cost and CPU/RAM presentation are already delivered; real nonempty
  classic-node telemetry remains an explicit SM-23h validation limitation.

## Implementation and verification

Delivered and published. One additional dataset joins current enrollment settings
to concrete-version performance history. Four counters and an evidence table share
the existing catalog/schema/model/version selectors. Disabled enrollments stay
visible but do not enter active performance counts. The former Degraded card is
now Measurement issues, retaining the existing stored status.

Policy reports rank by window end, measurement time and report ID. A later-arriving
old window cannot replace a newer window. Current configuration and model version
must match. Off is authoritative; missing/failed/unknown policy evidence never
becomes healthy. Freshness compares the saved window to the latest mature UTC
policy window, allowing the configured expected monitoring interval. Future or
immature evidence is rejected. These are presentation rules, not new triggers.

- TDD: 20 new executable SQL cases failed against the missing dataset, then
  passed. The deduplicated affected union (`test_monitoring_overview_performance.py`
  and `test_monitoring_dashboard.py`) passed 37 tests. After a test-fixture UTC
  correction and formatting, the changed file's 20 cases passed again and all
  20 collected. No production/dashboard change occurred after this union.
- Independent review found no actionable defect and exercised seven additional
  SQL probes without repeating the suite: missing timestamps/configuration,
  resolved alias, explicit-version precedence, separate enrollments and both
  sides of the freshness boundary.
- Full CI Ruff and Ty scopes passed. Full format scope passed (1,334 files),
  CCN-10 Lizard passed, and `git diff --check` passed. No frontend source changed;
  frontend build and MkDocs were not run.
- Staged pre-commit checks passed, including JSON, Ruff, format and full Ty.
- All 28 datasets passed 56 live default/NULL SQL cases. Same-ID draft update,
  publication and source/API readback matched. Published revision:
  `2026-10-05T11:09:50.074Z`; source SHA256:
  `8ec34e867c4ccf50cd28fe659259bce62bfd40fc3161691baa953769f0dfed7b`.
- Authenticated Chrome showed Within tolerance 0, Performance loss 0,
  Insufficient evidence 5, Policy disabled 1, independently of Healthy 5/Drift 1.
  The single-model selector reduced insufficient evidence to 1 and policy-off to
  0. The evidence table contained all six contexts, with current unavailable
  policy results rather than older loss. Visual inspection confirmed the separate
  sections and readable counters.

The dashboard remains `01f1c090238e1b6da5d633032ad9960b`, warehouse
`d047a4d9aa276958`, `embed_credentials=false`. No job, training, schedule or
retraining policy changed. The existing native refresh task addresses this same
dashboard ID and will refresh the new published definition on its next run.
No extra monitoring run was required for this SQL/presentation-only change.

Ignored evidence: `tmp_repro_artifacts/sm23i/` (SQL results, publication readbacks,
Chrome screenshots, browser counters and selector check).

## Queue result

SM-23 batch scope is closed. SM-23a is batch done and SM-23c functionally done;
SM-23b and the acceptance/enhancement limits above remain explicitly separate.
The next approved work is SM-57, beginning with the SM-56 preprocessing safety
inventory. Old historical queue notes are not current delivery status.
