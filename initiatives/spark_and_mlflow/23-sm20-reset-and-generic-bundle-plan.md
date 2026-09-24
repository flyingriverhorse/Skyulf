# SM-20R: clean workspace and generic first Bundle

Updated: 2026-09-24. This replaces the company-shaped SM-20P template
direction in [the earlier plan](22-sm20p-production-bundle-plan.md). The user
asked for one editable Bundle for any organization, with `dev`, `test`, `syst`
and `prod` targets and a fresh, visible first test. Company-specific catalogs,
hosts, policy and cost tags are examples or later configuration, not defaults
in the reusable Skyulf template.

## Verified reset of the personal workspace

Profile: `skyulf`, host:
`https://dbc-45604623-c18b.cloud.databricks.com`. Before deletion, the CLI
listed ten `skyulf_sm20a_r1` jobs and no active runs. It listed exactly three
Skyulf-owned test schemas under `workspace`. The user explicitly requested
deleting the previously created jobs and schemas. All ten job deletes and the
following three forced schema deletes returned success:

- `workspace.skyulf_sm16_20260922`
- `workspace.skyulf_sm24a_20260923`
- `workspace.skyulf_nyctaxi_e2e_20260923`

Post-deletion `databricks jobs list --profile skyulf` was empty.
`databricks schemas list workspace --profile skyulf` returned only
`workspace.default` and `workspace.information_schema`. Those system schemas
were preserved. The deleted test schemas contained their test Delta tables and
registered models; prior live-test reports remain historical evidence only.
Workspace files and MLflow experiments were not part of this deletion check.

## Minimal resource contract

`bundle deploy` creates jobs and uploads files. It creates no Unity Catalog
source, prediction or control table and registers no model. A first-run setup
step should list its intended UC objects before creating anything.

| Reference | Purpose | New UC object in first run? |
| --- | --- | --- |
| `training_table` | Existing labeled source for fitting | No |
| `score_source_table` | Existing rows to predict, CDF-enabled for later inserts | No; may equal training table |
| `prediction_table` | The single prediction output | Yes, once at explicit setup |
| `score_admission_table` | Internal coordination for safe incremental writes | Yes, once at explicit setup |
| `alias_admission_table` | Internal coordination for alias changes | Only if lifecycle is enabled |
| `model_name` | Registered UC model, not a table | Created by training |

Default to one existing source table for both training and scoring. Users can
split the references when their data lifecycle requires it. Do not create a
second source table just to satisfy the template. Training uses a pinned labeled
snapshot; scoring initially sees the whole source and later only new inserts.
The operator must review whether scoring old labeled rows is intended.

Default jobs: one-time `setup`, candidate `train`, recurring `score`.
`compare`, `stage` and `promote` are generated only when the user opts into
champion/challenger lifecycle. No schedule or endpoint is created by default.
Setup should never overwrite an existing table; it must validate source CDF,
keys, model output and an existing target/control schema before any write.

## Generic target and compute contract

- [x] One generated project exposes `dev`, `test`, `syst`, `prod`. `dev` is the
  default and uses the CLI profile host supplied by the user. Other targets
  each have a separate, conspicuous host/catalog placeholder and must be
  edited before they can deploy. Never use the personal test host as the
  implied company production host.
- [x] Ask only for project name, pandas/Polars engine, compute mode, optional
  lifecycle and dev catalog/schema in the easy path. Serverless is the default
  compute mode. Policy-backed classic compute is optional and configurable;
  no company-specific policy or `PayingRegNo` in the generic defaults.
- [x] Pass active target catalog and schema variables into the Skyulf notebook.
  Validate output/model/control names against the target before any write.
- [x] Render and strictly validate both compute modes and all four target
  shapes. The two generated `dev` shapes passed strict CLI validation. The
  other six shapes passed strict syntax validation only after substituting
  temporary personal-workspace values for the placeholders. Personal `dev` is
  the only live target until the other hosts, catalogs and profiles are supplied.
- [x] Update README, user guide, queue and 0.9.x changelog to state exactly
  which jobs and UC objects are produced at each step.

## First clean live rehearsal

Completed in the isolated `workspace.skyulf_bundle_first_20260924` schema.
The [clean validation report](24-sm20r-clean-generic-bundle-validation-report.md)
records deployment, Polars training, explicit setup, 600 initial predictions,
50 later predictions, no-op replay, and the final 650-row verification. The
final inventory has one source table, one prediction table, one internal
score-control table, one registered model and three persistent Bundle jobs.
Optional lifecycle resources stayed absent. No non-dev target was deployed.
