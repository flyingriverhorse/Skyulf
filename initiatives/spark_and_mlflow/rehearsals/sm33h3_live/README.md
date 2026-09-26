# SM-33H3 personal serverless rehearsal

Status: explicitly approved and completed SUCCESS on 2026-09-26 as
`848857785722024` in 437.664 seconds. All three tasks passed. This harness
uses the public Bundle action runtime; it is not another production template.

## Exact scope

- Previously selected profile: `skyulf`; existing personal schema
  `workspace.skyulf_lifecycle_test`. No new schema or persistent job.
- New source: `workspace.skyulf_lifecycle_test.sm33h3_20260926_r1_source`.
  Start with 240 synthetic records, CDF enabled; append exactly three later.
- New models: `sm33h3_20260926_r1_pandas_model` and
  `sm33h3_20260926_r1_polars_model` in that schema. One candidate version each,
  then bootstrap champion v1 from a separate task using saved-source approval.
- New prediction outputs: `sm33h3_20260926_r1_pandas_predictions` and
  `sm33h3_20260926_r1_polars_predictions`, also in that schema.
- Wheel, notebook and experiment under
  `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_lifecycle_test/sm33h3/r1`.
- One ephemeral serverless run with sequential `pandas_train`, `polars_train`
  and `score` tasks. Whole-run and task timeouts are 900 seconds; retries and
  serverless automatic optimization are disabled. Existing resources refuse
  overwrite. Existing Bundle jobs, aliases and schedules remain untouched.

## Acceptance

The saved Python recipe combines numeric sentinel replacement, a non-idempotent
feature replacement, target-only cleanup, string-to-number Casting followed by
ManualBounds, TextCleaning of a filter-only flag, an explicitly opted-in custom
eligibility filter, Deduplicate, learned imputation/scaling and three-fold CV.

1. Both engines train/register using Core and log heldout/CV metrics plus cleanup
   evidence. Custom eligibility and dedup must each exclude rows.
2. The training source file is overwritten with an exception after fitting.
   A separate score task loads the saved model/code, registers the saved custom
   filter and approves v1 without reading the current project file.
3. Raw feature replacement applies once: `1 -> 2`, `2 -> 3`; the target and
   filter-only columns are absent from direct local/MLflow prediction inputs.
4. Score writes every source identity, including training-excluded rows: keys
   `0..239`, then `0..242`. Local saved-model and MLflow predictions match.
5. The original 240 key/prediction/model-version triples survive the append.
   Another score is a no-op with the same Delta version and exact output rows.

No optional sentence model downloads, serving endpoints, Spark-native FE,
deletions or production-resource changes are part of this rehearsal.

## Local preparation

`verify_local.py` replaces only Delta reads with the fixture and runs real local
SQLite MLflow training and approval. On 2026-09-26 both engines retained 47 of
240 rows, produced heldout RMSE `6.877804181392917`, and approved champion v1.
The final subprocess exited 0. Evidence is retained in
`.cache/h3-harness-0_0jic8y`; log `.cache/sm33h3-harness-local-final.log`.
This local check does not validate cloud permissions, serverless dependencies
or Delta output writes. Core tests separately exercise fresh-process MLflow
source restoration and mutation guards.

Final scoped review, 289 affected tests, 56 CLI generation tests, Ruff/format,
full ty, strict docs and strict generated dev Bundle validation passed. Current
wheel: `.cache/sm33h3-cli-final/test_cli_emits_independent_pol0/output/`
`sm33_generated/dist/skyulf_core-0.9.0-py3-none-any.whl`. All 248 Core Python
files match source. SHA-256:
`ad08f6d91e3f329eb03294e36ed7d2004019fe14bfcca13cbda51aa196369c24`.
Upload and execution are separate
actions; neither this plan nor `submit.json` runs them automatically. A new
live run creates persistent test resources and consumes workspace compute.

## Approved execution

The user approved this exact scope on 2026-09-26. CLI 1.17.0 and the selected
`skyulf` identity were verified. The wheel digest above matched, including all
248 source files. Wheel and notebook imports refused overwrite. One ephemeral
[run 848857785722024](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/743740415302976/run/848857785722024)
was submitted; no additional run or retry has been submitted. Raw submission
and final task evidence is saved beside this file. Local result verification passed.

## SM-33H3 live acceptance evidence (2026-09-26)

The explicitly approved [run 848857785722024](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/743740415302976/run/848857785722024)
completed **SUCCESS in 437.664 seconds**. Exactly one serverless submission was
made, with no retry, new schema or persistent job. All three tasks succeeded:

| Task | Task run ID | Result |
|---|---|---|
| pandas_train | 1078163679226229 | SUCCESS |
| polars_train | 179579493001815 | SUCCESS |
| score | 679007598316858 | SUCCESS |

Both engines used the existing Core nodes and the custom project filter. Cleanup
retained 47 of 240 rows: 15 missing targets, 10 invalid ages, 12 custom exclusions
and 156 duplicate observations were removed. The final split contained 35
training and 12 heldout rows. The two runs logged the same 30 heldout/CV metrics:

| Metric | pandas | Polars |
|---|---:|---:|
| Heldout MAE | 4.903793942628702 | 4.903793942628702 |
| Heldout RMSE | 6.877804181392917 | 6.877804181392917 |
| Heldout R² | 0.5494839013563249 | 0.5494839013563249 |
| CV RMSE mean | 4.4513347280751 | 4.4513347280751 |

MLflow training runs: pandas `c0b16dd97d4145578773cb65d2602808`, Polars
`f258dfcaf4f446d68d81ad11455790a1`. The source Python file was replaced after
training. A separate task restored the saved code, validated the saved recipe
and evidence, and approved champion v1 for both models. Target/filter columns
were absent from direct saved-model and MLflow prediction inputs; their
predictions matched. Non-idempotent feature replacement was checked explicitly
as `1 -> 2`, `2 -> 3`, with no second application.

Each score output first contained exactly keys 0..239, including rows excluded
from training. After appending three source records, only those three were
scored and each output contained exactly keys 0..242. The original 240
key/prediction/model-version triples remained unchanged. Another score returned
zero input/output rows and `noop=true`; Delta version stayed **2** and exact
published contents remained unchanged. Every prediction was finite and used v1.

Resources retained in `workspace.skyulf_lifecycle_test`:

- `sm33h3_20260926_r1_source`: 243 rows.
- `sm33h3_20260926_r1_pandas_predictions`: 243 predictions.
- `sm33h3_20260926_r1_polars_predictions`: 243 predictions.
- `sm33h3_20260926_r1_pandas_model` and `sm33h3_20260926_r1_polars_model`:
  one version each, champion v1.
- Notebook, wheel and experiment under
  `/Workspace/Users/edwardwolfe99@gmail.com/skyulf_lifecycle_test/sm33h3/r1`.

Wheel SHA-256: `ad08f6d91e3f329eb03294e36ed7d2004019fe14bfcca13cbda51aa196369c24`.
CLI upload used the locally verified wheel containing all 248 current Core
Python files. Raw task outputs and `acceptance-summary.json` are retained in
`rehearsals/sm33h3_live`; `verify_results.py` passed against those actual outputs.

This proves the documented mixed recipe and batch workflow on personal
serverless, not every parameter mode or company production environment.
Optional H3Index/sentence-model execution, temporal history/CV policy and custom
pre-split value normalization remain explicitly tracked under SM-36a and
matrix63. Serving and native Spark expansion are unchanged. **SM-33H3 is DONE
for this scope; SM-34 is READY.** No commit or push was made.
