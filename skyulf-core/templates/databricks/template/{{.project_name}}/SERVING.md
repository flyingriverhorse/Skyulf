# Serve a registered model

Choose `prediction_delivery: endpoint` or `both` in detailed setup for a certified
pandas single model, competition winner or model set. `endpoint` generates no batch
score job or prediction table; `both` generates both jobs. Quick setup defaults to
`batch`. The monitoring job is available in every mode. Edit `serving_endpoint` in
`config/serving.yml` and set the target's `cost_tag_value` before the first run.

## Commands

From the project root, replace `test` with your target (`test_development` for a
personal target) and `YOUR_PROFILE` with its authenticated profile:

```bash
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
# Resolve champion once, or override with one numeric version:
databricks bundle run serving_endpoint -t test --profile YOUR_PROFILE
databricks bundle run serving_endpoint -t test --profile YOUR_PROFILE --params model_version=7
```

Automatic updates run after completed champion changes in production Bundle
mode. Inspect the child serving run: training only waits for submission. See the
[job settings and operational rules](https://www.skyulf.com/manual/user_guide/databricks_serving.html#configuration-reference)
for permissions, logging, costs, smoke checks, recovery and cleanup.

Set `serving_endpoint.follow_champion` to `true` for automatic updates or `false` for manual runs in
YAML and deploy the updated file. The dispatch tasks already exist; no project
regeneration is needed for this toggle. Personal development targets stay manual.

## Monitor endpoint requests

Set the target's `monitoring_enabled` to `"true"` and
`serving_endpoint.logging_mode` to `telemetry` or `ai_gateway` before creating
the endpoint. Configure `monitoring_catalog` and `monitoring_schema` in
`deployment/variables.yml`. After reference, REST and optional SQL parity passes, the job
automatically enrolls the served concrete version in that store. A model set gets
one entry per component, retaining its parent set/version. Repeated runs reuse
the registration; a replaced version's entries are disabled, preserving history.

Send requests, enable the monitoring schedule for the target, and run `monitoring`
to collect observations. The dashboard's Online serving page shows the saved
observations after logging has delivered the requests; enrollment alone creates
no traffic measurements. Performance additionally needs matched outcome labels.
With `monitoring_enabled: "false"` or an absent flag, the result reports
`monitoring_disabled` and skips all monitoring work, preserving existing registrations.
Endpoint logging and verification still run. When monitoring is enabled,
`logging_mode: disabled` reports `logging_disabled`; enabled logging requires a
monitoring namespace before endpoint changes. `config/inference.yml` retains `score_source_table`
only as the source for the endpoint's bounded smoke sample in endpoint-only mode.

## Choose a caller

- [SQL with ai_query](https://www.skyulf.com/manual/user_guide/databricks_serving.html#sql-with-ai_query): SQL function, group access and a named-argument query.
- [REST applications](https://www.skyulf.com/manual/user_guide/databricks_serving.html#rest-for-applications): application service principal and JSON request.
- [Python notebooks/jobs](https://www.skyulf.com/manual/user_guide/databricks_serving.html#python-sdk-for-notebooks-and-jobs): caller group and SDK request.

## Other optional jobs

Run only the jobs selected and configured in your project:

```bash
databricks bundle run online_feature_lookup_setup -t test --profile YOUR_PROFILE
databricks bundle run online_publication -t test --profile YOUR_PROFILE --params publication_action=start
databricks bundle run online_publication -t test --profile YOUR_PROFILE --params publication_action=status
# Requires an already initialized rollout:
databricks bundle run daily_rollout -t test --profile YOUR_PROFILE
```

- [Explicit create-only notebook API](https://www.skyulf.com/manual/user_guide/databricks_serving.html#the-command)
- [Online feature setup and publication](https://www.skyulf.com/manual/user_guide/databricks_online_features.html)
- [Gradual traffic rollout](https://www.skyulf.com/manual/user_guide/databricks_serving_rollout.html)
- [Serving monitoring](https://www.skyulf.com/manual/user_guide/databricks_serving_monitoring.html)

`deployment/requirements.txt` pins `skyulf-core[mlflow]==0.9.4`. Public links show
the latest guide; check it against your installed version.
