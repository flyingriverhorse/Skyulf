# SM-23k - Shared monitoring inside the existing template

Date: 2026-10-05. Base commit: `b59b96a4`.

## User-approved scope and final architecture

The user confirmed one shared dashboard: create missing monitoring tables,
otherwise reuse the same tables. The dashboard belongs inside the existing
Databricks project template, not a separate monitoring template or Bundle.

Canonical assets are now:
`skyulf-core/templates/databricks/template/{{.project_name}}/src/monitoring/`.
The old example contains a compatibility README only. Its obsolete standalone
Bundle YAML, notebook and resources are retired. No live resource is deleted by
this source cleanup. The intermediate top-level monitoring directory is gone.

The existing helper supports `--create-dashboard --warehouse-id ...` only for one
owner target. Its resource and URL substitutions bind the same dashboard to the
existing monitoring job. Other projects reuse the published URL. Disabling refresh
preserves dashboard ownership. Current acceptance deployments continue URL reuse;
no second dashboard or ownership transfer was performed.

`ensure_monitoring_store` is called before single/competition scoring/activation
registration, model-set component registration, and fresh scheduled monitoring.
It checks all existing owner markers, table schemas and inventory isolation before
writing. Missing tables/views use conditional creation and are checked again after
creation; compatible objects are reused without persistent DDL. The explicit
initializer remains the owner-run API for view upgrades. Existing SDK enrollment
semantics and development isolation remain unchanged. First use requires creation
permissions; the existing-store no-DDL path is not a cross-principal grant test.

## Overview

- Distinct registered-model, enrollment-context, scoring-input and prediction-table
  counts; one model or table shared by multiple enrollments counts once.
- Compact model/version/context-to-input/prediction/outcome table mapping without
  long monitoring identity hashes. Detailed inventory remains on Drift and data quality.
- Two daily charts count one enrollment per UTC window day, selecting the last
  saved window then deterministic measurement/report ties. Only current config and
  concrete version are eligible. Drift uses actual drift checks, not general health;
  performance uses saved policy outcomes. Failed/missing evidence stays unavailable.
- Days without checks are gaps, not synthetic zero/healthy days. Current data spans
  one UTC day, so the line charts currently render points. More observed days form lines.

## Preprocessing inventory clarification

SM-56 inspected actual apply bodies and separated fitted row-wise operations,
conditional candidates and batch/history/row-count dependent behavior. Small
whole-versus-split probes reproduced Box-Cox fallback, replacement composition,
mixed-value binning and datetime dtype differences. These were inventory probes,
not fixes or certification. SM-57 runtime gating and real Spark UDF delivery remain
open; records 165/166 hold the inventory and implementation plan.

## Verification

- Root dashboard/template affected union: 89 passed, 18 CLI opt-in skips. The exact
  18 skipped CLI generation cases were then run with skyulf: 18 passed. A pytest
  cache permission warning did not affect assertions.
- Final bootstrap/registration/model-set/tasks/review/store/job union after review
  repair: 143 passed, 4 existing Pydantic NumPy-bool deprecation warnings.
- Full CI Ruff, Ruff format (1336 files), Ty scope and Core Lizard CCN 10 passed.
- Six actual CLI strict validations: single/competition/model-set x serverless/
  policy-cluster. Job Compute policy used for lookup only; no cluster launched.
  Existing3 jobs plus 2 refresh tasks retained; no standalone monitoring Bundle.
- Before/after original dashboard-test relocation collection retained 40 node IDs.
  This is collection evidence, not execution; tests above executed final paths.
- Independent Overview read-only review found no concrete issue. Independent runtime
  review reproduced malformed explicit JSON null being treated as a schedule;
  fixed to reject non-object requests. 5 variants failed first, then passed.
- All 31 datasets executed with defaults and NULL parameters: 62 successful warehouse
  queries before same-ID publication. API normalized readback matches final source.
- Published dashboard 01f1c090238e1b6da5d633032ad9960b at 2026-10-05T11:50:22.281Z;
  warehouse d047a4d9aa276958 and embed_credentials=false preserved.
  Source SHA256 620952f9115fc835d35f28a7a59abefb0cb6a98314a5803c5cc4095b1046ff69.
- Authenticated isolated Chrome verified 5 models / 6 contexts / 3 input tables / 5 prediction tables,
  both rendered daily series and all 6 model/table mapping rows. Screenshots inspected.
  Selecting one model reduced all four counters to 1 and filtered mapping to its
  single row; selection was reset afterward.
- Final wheel 9831c4c08801d2e6ef1e2d5708deb53aa9a1df6487ca609ebd8fb5c5aa9524eb deployed
  to the existing three model Bundles. All 9 job IDs/names/schedules/graphs preserved;
  deployment reported 0 created / 0 deleted resources. Existing dashboard URL reused.

## Live verification

Create/reuse isolated Spark run 609921385391965. The first temporary notebook
attempt (368006622960457) failed on UTF-8 BOM before executing test code; its
source was corrected and that attempt stopped. Monitoring runs:
single 6894981083889, competition 705281687719154, model-set 724090769237344.
Isolated Spark create/reuse run 609921385391965 succeeded: 7 conditional DDL
statements created the schema and 6 store objects. A second call ran no persistent
DDL and retained 4 enrollments / 4 immutable reports after replaying each report.
The existing shared store passed the same no-DDL checks (6 enrollments, 55 reports,
0 action rows at readback); concurrent monitoring was allowed to append evidence.
The exact temporary schema was removed after the assertions; no cascade cleanup.
Actual separate-principal grants and concurrent *first* creation are not claimed.

All three final monitoring runs succeeded, including all five tasks per job:
`monitor_model -> monitoring_report -> evaluate_retraining ->
dashboard_refresh_enabled -> refresh_monitoring_dashboard`.
Single and competition each persisted one healthy measurement; model-set persisted
two healthy component measurements in the same configured store. All reported
failed=0. Retraining evaluation stayed disabled; no training was submitted.
Native dashboard refresh succeeded against the existing shared dashboard ID.

Run links:
- [Single](https://dbc-45604623-c18b.cloud.databricks.com/jobs/849383177102322/runs/6894981083889?o=7474646244882000)
- [Competition](https://dbc-45604623-c18b.cloud.databricks.com/jobs/382789798978890/runs/705281687719154?o=7474646244882000)
- [Model-set](https://dbc-45604623-c18b.cloud.databricks.com/jobs/52053718085306/runs/724090769237344?o=7474646244882000)

The temporary bootstrap notebook was removed after successful readback. The run
and its output remain available. Existing production/acceptance resources were
preserved. Final delivery is committed with DCO on branch092; no push requested.

Local reproducibility evidence is ignored under `tmp_repro_artifacts/sm23k/`;
no caches, wheels, screenshots or cloud payloads are committed.

All staged pre-commit checks passed: whitespace/EOF/JSON, Ruff, format, full Ty
and backend/Core complexity. Frontend hooks skipped because no frontend files
changed. No MkDocs build or full local suite was run; affected checks are above.

## Reproduction commands and final source state

Executed against the final four runtime files and canonical template assets,
including the non-object receipt repair (wheel SHA recorded above):

```powershell
.venv/Scripts/python.exe -m pytest skyulf-core/tests/integration/platforms/test_monitoring_dashboard.py skyulf-core/tests/integration/platforms/test_monitoring_compute_dashboard.py skyulf-core/tests/integration/platforms/test_monitoring_overview_performance.py skyulf-core/tests/integration/platforms/test_monitoring_overview_summary.py skyulf-core/tests/integration/platforms/test_monitoring_dashboard_refresh.py skyulf-core/tests/integration/platforms/test_monitoring_dashboard_owner.py skyulf-core/tests/integration/platforms/test_monitoring_template.py --no-cov -q --basetemp=tmp_repro_artifacts/sm23k/pytest-dashboard
# 89 passed, 18 opt-in skips; all 18 subsequently executed below.
$env:SKYULF_BUNDLE_CLI_TEST_PROFILE='skyulf'
.venv/Scripts/python.exe -m pytest skyulf-core/tests/integration/platforms/test_monitoring_template.py::test_generated_projects_bind_independent_monitoring_destination skyulf-core/tests/integration/platforms/test_monitoring_dashboard_refresh.py::test_generated_projects_ship_optional_native_dashboard_refresh --no-cov -q --basetemp=tmp_repro_artifacts/sm23k/pytest-cli
# 18 passed.
.venv/Scripts/python.exe -m pytest skyulf-core/tests/integration/platforms/test_monitoring_bootstrap.py skyulf-core/tests/integration/platforms/test_monitoring_registration.py skyulf-core/tests/integration/platforms/test_monitoring_model_set.py skyulf-core/tests/integration/platforms/test_monitoring_tasks.py skyulf-core/tests/integration/platforms/test_monitoring_review_batch15.py skyulf-core/tests/integration/platforms/test_spark_monitoring_job.py skyulf-core/tests/integration/platforms/test_monitoring_store.py --no-cov -q --basetemp=tmp_repro_artifacts/sm23k/pytest-runtime-final
# 143 passed.
.venv/Scripts/ruff.exe check .
.venv/Scripts/ruff.exe format --check backend skyulf-core tests run_skyulf.py celery_worker.py
.venv/Scripts/ty.exe check backend skyulf-core/skyulf skyulf-core/tests run_skyulf.py celery_worker.py
.venv/Scripts/lizard.exe skyulf-core/skyulf --CCN 10 -w
.venv/Scripts/pre-commit.exe run
```

The independent reviewer re-read the receipt repair and its five no-mutation
regressions, closing P2 without repeating suites. Existing marker/schema checks
are retained; object kind/provider and view-definition attestation are not new
claims in this delivery.
