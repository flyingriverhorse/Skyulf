# Maintaining the Bundle wizard

Edit the topic files here, then regenerate the schema from the repository root:

```sh
python skyulf-core/templates/databricks/build_schema.py
python skyulf-core/templates/databricks/build_schema.py --check
```

The script uses only Python's standard library and also works from another working
directory when invoked by its path. Commit the topic files and the generated
`databricks_template_schema.json` together. Pre-commit and CI reject stale output.
Bundle users continue to run `databricks bundle init` normally; no build step is
required for them. These maintenance files sit outside `template/`, so they are
not copied into generated projects.

`catalog` and `schema` are optional initialization-file inputs, not interactive
questions. Their empty `skip_prompt_if` schema matches every answer set. Without
explicit values, deployment settings are generated with editable `REPLACE_TEST_*`
placeholders, matching the other environments.

## Where to edit

| File | Questions/settings |
| --- | --- |
| `metadata.json` | Root schema fields, including the welcome message |
| `project.json` | Layout, project, engine, task, source, columns and snapshot |
| `training.json` | Split, training windows, sampling and input limits |
| `time_columns.json` | Event-time column parsing and timezone |
| `label_availability.json` | Label availability and its timestamp parsing |
| `models.json` | Classification/regression model selection and parameters |
| `competition.json` | Competition candidate count, models, recipe and budget |
| `competition_ensemble.json` | Shared prototype for single, candidate and branch ensemble questions |
| `branch_count.json` | Number of independent model-set branches |
| `branches.json` | Per-branch targets, models, recipes, search, CV and quality policies |
| `tuning.json` | Search strategy, space, metric, resources and threshold tuning |
| `cross_validation.json` | Ordinary/nested folds, group/time splits and seed |
| `scheduling.json` | Training and scoring schedules |
| `scoring.json` | Single-model prediction source/output, promotion and quality |
| `model_set.json` | Set name, promotion, outputs/views and source-change policy |
| `deployment.json` | Compute, cluster policy and cost tags |
| `identities.json` | Optional personal targets, shared run identities and job ACL ownership |

Each topic file is a JSON object whose keys are the existing wizard field names:

```json
{
  "project_name": {
    "order": 1,
    "type": "string",
    "default": "skyulf_local",
    "description": "Project folder and Bundle name"
  }
}
```

The builder merges every `schema/*.json` topic except `metadata.json`, places its
fields under `properties`, and sorts them by their explicit `order`. Keep orders
unique and keep referenced questions earlier than their dependents. A
`skip_prompt_if` condition can reference a field from another topic; it retains
the same meaning in the merged schema. Duplicate names within or between files
fail instead of overwriting a definition. To add a topic, create another JSON
file here; no list in the builder needs editing.

`competition_ensemble.json` is a repeated question group: `SLOT` marks a
candidate number in field names, descriptions and conditions. The builder expands
it for candidates 1-8 and rebases the same questions for a single ensemble and
branches 1-8. `branches.json` uses zero-based local orders and `branch_SLOT_`
fields, expanded into separate branch groups with their own ensemble questions.
Edit each prototype once. The committed root schema contains only ordinary CLI
properties, with no placeholders. Keep absolute question orders unique.

The same build command also regenerates `library/model_search_space.tmpl` from
Core's hyperparameter registry through `build_model_spaces.py`. Its `--check`
mode checks both artifacts. `library/modeling.tmpl` renders each model's params
and search settings; generated projects have no shared tuning or ensemble hook.
Keep Core as the source of default search ranges instead of editing the generated
catalog by hand.

The root JSON is generated with one compact line per prompt, including its nested
conditions, to keep the expanded artifact manageable. Values and question order
are unchanged by this formatting. Make future edits in the readable topic files
here, then regenerate the root schema.
