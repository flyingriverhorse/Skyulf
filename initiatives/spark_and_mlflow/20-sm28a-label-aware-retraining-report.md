# SM-28a label-aware candidate training - 2026-09-24

Status: **PASS** for the standalone, small-data retraining service. The first
local-engine Bundle (SM-20a) is still unstarted; no schedule or promotion was
created.

## Contract

`LocalTrainingSpec` pins one Delta table/version, UTC event interval, label
availability cutoff, temporal holdout boundary, raw input/target columns, row
keys and local resource budget. Spark filters and projects before a bounded
driver read. Unavailable labels are excluded; available targets must be
nonnull. Stable event/key sorting produces disjoint training and holdout rows.
The normal Skyulf pipeline fits on only the earlier rows with pandas or Polars.
The MLflow run logs held-out metrics, config, source and code identity, plus
the fitted model. A new concrete UC model version is registered, then the
candidate and caller-pinned prior version are compared on the same holdout.
The comparison report is logged to `candidate_comparison.json`. Neither path
changes model aliases; staging and promotion are separate operations.

## Local gate

The focused Databricks/MLflow integration gate passed **55 tests, 1 skipped**.
The new retraining tests cover late-label exclusion, cutoff and temporal
membership, deterministic row order, invalid provenance, bounded versioned
reading, failed registration, real pandas/Polars Skyulf fits and SQLite MLflow
model-version comparisons without alias mutation. Ruff and Ty passed; the
documentation built with `mkdocs build --strict`.

## Databricks gate

The existing `skyulf` CLI profile and isolated
`workspace.skyulf_sm24a_20260923` test schema were reused. A one-time
[serverless job](https://dbc-45604623-c18b.cloud.databricks.com/?o=7474646244882000#job/753857281922815/run/487563692360508)
finished **SUCCESS** with no retries and a 900-second task timeout. It ran
the uploaded wheel in the user's `sm28a_r1` folder (SHA-256
`02c63e1b4d24ad2b061d61583debd6b0b227fcfbdf449ddee06d2daba8274c34`).
The source was a read-only copy of the prior real taxi test table into new
managed Delta table `skyulf_sm28a_labels_r1`, pinned at version **0**. The
test derived label availability one hour after pickup and deliberately marked
one label late. Five additional labels crossed the February cutoff naturally.
This timestamp construction tests the API boundary; it is not a claim about
the production arrival time of taxi fare labels.

| Run | UC model version | MLflow run | Train / holdout / unavailable | Holdout MAE / RMSE / R2 |
| --- | ---: | --- | --- | --- |
| Linear baseline | 1 | `b95b219b11994e09bf5a8d8eef7a7cb8` | 1,724 / 1,770 / 6 | 0.612642 / 2.252941 / 0.949224 |
| Imputed, scaled random forest candidate | 2 | `79bcd243aebd4ef1b0655e7e0695f4e7` | 1,724 / 1,770 / 6 | 0.488965 / 2.088749 / 0.956355 |

Both UC versions were independently listed as `READY`. Both MLflow runs were
`FINISHED`, had the same source version and row counts, and independently
listed `model/`, `skyulf_pipeline_config.json` and
`candidate_comparison.json`. The second call pinned version 1 as its comparison
reference. A separate registered-model lookup with aliases included returned
no aliases. The candidate improved RMSE by about 0.1642 on this one holdout;
that does not authorize promotion or imply future performance.

The uploaded wheel preceded only the final public export and stricter
`LocalTrainingSpec` identity/time validation edits. Those final edits passed
the local integration gate; the live run proves the training, publication and
comparison path exercised by that wheel. No inference, schedule or Bundle
was tested by this job. The shared alias-admission rule remains the separate
SM-22b/c promotion contract.
