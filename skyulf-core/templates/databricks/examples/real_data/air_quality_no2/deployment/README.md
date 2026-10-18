# Deploy this project

Use a Databricks CLI profile or your existing CI/CD authentication. No GitHub
workflow or company adapter is generated. Keep credentials outside these files.

## Selected layout

- Training engine: `pandas`; inference mode: `spark`.
- Personal development targets: `false`.
- Shared execution identity: `deployer`.
- Managed per-job permissions: `false`.
- Every target grants `CAN_MANAGE` to the deployer and `pipeline_owner`
  (`--var pipeline_owner=<email>` in CI); shared targets may add a `CAN_VIEW` group.

- Spark scoring uses `spark_udf_env_manager=local`
  and 10000-row batches; certified worker runtime requirements still apply.


- Train/score use serverless Python environment 4 by default.

Monitoring always has its own serverless environment and budget-policy settings.

## Files to edit

| File | Purpose |
| --- | --- |
| `deployment/targets.yml` | Literal hosts, workspace roots and target overrides |
| `deployment/variables.yml` | Existing UC namespaces, monitoring store, compute and schedules; service-principal IDs and access when selected |
| `deployment/requirements.txt` | Shared published Core pin and dependencies |
| `deployment/train-requirements.txt` | Training/report dependencies and local tools |
| `resources/*.job.yml` | Generated job definitions; preserve stable job keys |

## Commands

Run from the project root after filling the target's `REPLACE_*` values:

```powershell
uv pip install -r deployment/train-requirements.txt
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run train -t test --profile YOUR_PROFILE
```

Deploy updates definitions; a job run is a separate action. Inspect the resolved
host, schemas, output suffixes, `run_as` and permissions before deployment.

## Rules to keep

- Shared `test`, `syst`, `prod` targets use production mode. There is no `dev`
  target. Personal targets, when generated, start paused with monitoring enrollment
  and automatic retraining disabled; they still require UC grants.
- Deployment credentials and job `run_as` are separate. Configure real application
  IDs and job/data/model grants; different principals alone do not enforce writer separation.
- Preserve bundle identity, roots and `train`/`score`/`monitoring` keys on upgrade.
  Lifecycle retries stay zero. After successful training and failed scoring,
  repair and run score alone; a timeout does not undo committed work.

## Detailed instructions

- [Deployment, identities, ACLs and CI commands](https://www.skyulf.com/manual/user_guide/databricks_deployment.html)
- [Timeouts, finite score retries and notifications](https://www.skyulf.com/manual/user_guide/databricks_deployment.html#time-limits-retries-and-notifications)
- [Local and distributed scoring](https://www.skyulf.com/manual/user_guide/databricks_scoring.html)
- [Existing-project upgrades](https://www.skyulf.com/manual/user_guide/databricks_project_upgrade.html)
- [Monitoring store and access setup](https://www.skyulf.com/manual/user_guide/databricks_monitoring_setup.html)

Version: `deployment/requirements.txt` pins `skyulf-core[mlflow]==0.9.4`.
Public links show the latest documentation. Use the runtime matching the template
and saved model producer; changing a pin does not upgrade existing artifacts.
