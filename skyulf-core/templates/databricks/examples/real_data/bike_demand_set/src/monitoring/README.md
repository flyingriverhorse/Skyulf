# Monitoring

Use this project's Spark monitoring job to save drift, performance and serving
observations in the shared store. Configure it before running production jobs.

## Files to edit

| File | Setting |
| --- | --- |
| `deployment/targets.yml` | Target monitoring catalog/schema, enabled flag, labels, thresholds, policies and paused schedule. |
| `deployment/variables.yml` | Shared monitoring variable defaults. |
| `resources/monitoring.job.yml` | Separate measurement, report and retraining job. |
| `src/monitoring/monitoring.lvdash.json` | Shared dashboard definition, managed by its one owning project. |
| `deployment/requirements.txt` | Core/runtime requirement; keep all inventory readers compatible. |

Set the same independent monitoring catalog/schema in every producer. These
settings do not change training, model or prediction destinations. Have the store
owner initialize or upgrade it before producers without creation privileges run.

## Commands

Run from this generated project's root; replace target and profile:

Replace `test` with your target (`test_development` for personal targets) and `YOUR_PROFILE`
with that target's authenticated profile.

```powershell
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run monitoring -t test --profile YOUR_PROFILE --params monitoring_action=prepare_references
databricks bundle run monitoring -t test --profile YOUR_PROFILE
```

To create or connect a dashboard, edit the target's block in
`deployment/dashboards.yml` before validating and deploying; the file explains
both choices and where to find the warehouse ID.

## Rules that matter

- Only shared `mode: production` targets enroll and observe. Development skips
  monitoring. The monitoring schedule starts paused; verify references before
  setting `monitoring_pause_status: "UNPAUSED"` and redeploying.
- Activation/scoring enrolls concrete versions. `bundle deploy` alone does not
  activate a model. `combined_only` model-set outputs cannot supply component
  monitoring; use `all` or `separate_views`.
- Real labels join saved keys; missing outcomes are unavailable, not zero
  accuracy. Label tables do not update training data.
- Retraining is opt-in: `on_drift: retrain` or policy `mode: retrain`, with shared
  freshness, cooldown and request guards. Dashboard refresh only reads results.
- Keep one dashboard owner. Upgrade owned views before new producers/dashboard
  definitions, and never modify Databricks-owned serving log objects.

## Detailed instructions

| Task | Public manual |
| --- | --- |
| Store, grants and upgrades | [Monitoring setup](https://www.skyulf.com/manual/user_guide/databricks_monitoring_setup.html) |
| Enrollment and reference repair | [Enroll and observe](https://www.skyulf.com/manual/user_guide/databricks_monitoring_enrollment.html) |
| Delayed outcomes and matching | [Labels](https://www.skyulf.com/manual/user_guide/databricks_monitoring_labels.html) |
| Dashboard ownership, refresh and lineage | [Shared dashboard](https://www.skyulf.com/manual/user_guide/databricks_monitoring_dashboard.html) |
| Distribution checks and data quality | [Drift](https://www.skyulf.com/manual/user_guide/databricks_drift.html) |
| Tolerance and consecutive windows | [Performance loss](https://www.skyulf.com/manual/user_guide/databricks_performance.html) |
| Endpoint observations and logging errors | [Serving monitoring](https://www.skyulf.com/manual/user_guide/databricks_serving_monitoring.html) |
| Shared automatic training guards | [Retraining](https://www.skyulf.com/manual/user_guide/databricks_retraining.html) |
| Daily traffic increases, hold and rollback | [Daily serving rollout](https://www.skyulf.com/manual/user_guide/databricks_serving_rollout.html) |

Version: these links open the latest public manual. This project's pinned runtime
is in `deployment/requirements.txt`; confirm compatibility before applying newer
instructions. Updating docs does not upgrade artifacts, jobs or shared views.
