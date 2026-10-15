# Preview and local project tools

Run these commands from the project root with the matching Core dependencies
installed. Start with `preview.py` to inspect the configuration before training.

## Preview the selected pipeline

```powershell
python src/tools/preview.py --action train
python src/tools/preview.py --action score
python src/tools/preview.py --list-models
python src/tools/preview.py --list-preprocessors
```

| Option | Meaning |
| --- | --- |
| `--action train\|score` | Validate and display training or scoring settings; default: `score`. |
| `--list-models` | Installed Core model metadata for the project's task; all tasks for a model set. |
| `--list-preprocessors` | Installed Core preprocessing metadata and defaults. |
| `--catalog` | Catalog binding; default is the initialization `catalog` value. |
| `--input-schema` | Input schema binding; default is the initialization `schema` value. |
| `--output-schema` | Output schema binding; default is the initialization `schema` value. |
| `--metadata-schema` | Registry/metadata schema binding; default is the initialization `schema` value. |
| `--resource-suffix` | Target's resource suffix; default is empty. |

Pass bindings matching your chosen Bundle target. Preview does not read
`deployment/targets.yml` or resolve workspace `${...}` substitutions for you.
Unspecified initialization catalog/schema values are editable `REPLACE_TEST_*`
placeholders. `--help` lists the command-line options.

Training preview shows the effective model, inputs, split, CV and policies.
Competition prints one `Candidate:` section per candidate; model sets print
one `Branch:` section per branch. Model/node availability depends on the
locally installed packages and selected options.

Preview resolves trusted custom factories, so keep I/O out of imports and
builders. It starts no cloud job, reads no Delta rows, fits no model and does
not run `preprocessing_probe`. A preview-resolution failure prints
`Configuration error:` and exits with code 2. Fix the stated configuration
before deployment; passing preview does not verify data or cloud permissions.

## Other tools

| Tool and when to use it | Command |
| --- | --- |
| `smoke.py`: check YAML, Python syntax and package snapshots without running editable hooks. | `python src/tools/smoke.py` |
| `smoke.py --probe`: fit on a local sample and run the Spark/endpoint admission lint and probe before a training job. Export one with `SELECT * FROM <training_table> LIMIT 256`; columns must match `input_columns` plus the target. Training on the real holdout decides. | `python src/tools/smoke.py --probe sample.csv` |
| `refresh_feature_graph.py`: after changing upstream feature groups/merges. | `python src/tools/refresh_feature_graph.py` |
| `refresh_training_graph.py`: after changing competition candidates or model-set branch names/count; single model is unchanged. | `python src/tools/refresh_training_graph.py` |
| `configure_monitoring_dashboard.py`: configure refresh of an existing dashboard in a target. | `python src/tools/configure_monitoring_dashboard.py --dashboard-id YOUR_DASHBOARD_ID --target test` |

Smoke prints JSON with `"status": "passed"`, or `"status": "failed"` and exit code 1
for a checked configuration/source failure. It accepts the same five binding
options as preview. It does not test model fit or remote access.

The graph and dashboard tools edit local resource YAML; validate and redeploy
to apply it. Dashboard `--help` also covers explicit ownership and disabling
refresh.

See the [full local tools reference](https://www.skyulf.com/manual/user_guide/databricks_project_setup.html#local-tools-and-preview)
and [dashboard setup](https://www.skyulf.com/manual/user_guide/databricks_monitoring_dashboard.html).
