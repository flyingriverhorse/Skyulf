# Train and inspect this project

Run from the project root after [START_HERE.md](START_HERE.md). Training saves a
candidate; [SCORING.md](SCORING.md) explains publishing predictions.

## Files to edit

| File | Purpose |
| --- | --- |
| `config/training.yml` | Sources, inputs/target, split/CV, model/tuning, weights and quality policy. |
| `config/pre_split.yml`, `config/preprocessing.yml` | Fixed eligibility and fitted transformations. |
| `src/features/pre_split.py`, `preprocessing.py`, your own modules | Custom factories selected by those recipes. |
| `config/inference.yml` | Saved-model selection, model-set policy and optional scoring handoff. |
| `deployment/targets.yml`, `deployment/variables.yml` | Workspace, UC bindings, identity, compute and schedules. |

Edit `models.main.model` and sibling `tuning`; this layout requires one named model.

Nested model mappings replace shared mappings; they are not deep-merged.
Remove `tuning` to fit fixed parameters. Empty preprocessing recipes do nothing.
Set appropriate positive `max_rows`/`max_input_mb`. A numeric `quality_threshold` is
required for automatic promotion and for approving the first champion.

## Commands and results

```powershell
uv pip install -r deployment/train-requirements.txt
python src/tools/smoke.py
python src/tools/preview.py --action train
python src/tools/refresh_training_graph.py
databricks bundle validate --strict -t test --profile YOUR_PROFILE
databricks bundle deploy -t test --profile YOUR_PROFILE
databricks bundle run train -t test --profile YOUR_PROFILE
```

Refresh the graph after adding/removing/renaming models; parameter-only edits
keep its shape. Preview runs trusted factories but does not read cloud data or fit.
Open **training_report > Output** for metrics, decisions and `next_actions`.
A report can finalize a failed invocation: inspect its reported outcome.

This project has `promotion_policy=manual_approval`,
`score_model_selection=pinned_version`,
`score_handoff=disabled`, and
`auto_rebuild_on_cdf_expiry=false`.
For model sets, inspect their separate saved promotion policy in inference.yml.
Use the serialized train job's Run now parameters for approve/reject/rollback,
copying the exact **Available action** values from `training_report` > Output;
first approval uses expected champion `none`. Do not repair/retry an old lifecycle invocation or edit aliases directly.

## Detailed instructions

- [Training fields and complete layouts](https://www.skyulf.com/manual/user_guide/databricks_training_config.html)
- [Data windows, identity, sampling and splits](https://www.skyulf.com/manual/user_guide/databricks_training_data.html)
- [Model parameters](https://www.skyulf.com/manual/user_guide/databricks_model_catalog.html) and [search/CV](https://www.skyulf.com/manual/user_guide/databricks_tuning.html)
- [Recipes and step parameters](https://www.skyulf.com/manual/user_guide/preprocessing_recipes.html) and [custom factories](https://www.skyulf.com/manual/user_guide/databricks_custom_steps.html)
- [Classification decision thresholds](https://www.skyulf.com/manual/user_guide/databricks_decision_thresholds.html)
- [Reports, graph, charts, SHAP and MLflow parameters](https://www.skyulf.com/manual/user_guide/databricks_training_reports.html)
- [Approval, rejection, rollback and gates](https://www.skyulf.com/manual/user_guide/databricks_lifecycle.html)
- [Saved preprocessing diagnostics](https://www.skyulf.com/manual/user_guide/preprocessing_context.html)

## Version note

Public links show the latest manual; [README.md](README.md) explains how to match
them to this project's pinned Core release.
