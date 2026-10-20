# Score saved models in this project

This project uses endpoint-only delivery. It has no batch score job, schedule,
watermark or Delta prediction destination. Use [SERVING.md](SERVING.md) to deploy
the saved model and make SQL, REST or SDK requests. `config/inference.yml` keeps
the source for the endpoint's bounded smoke sample. The eligibility, output and
combined rules below apply to the saved model's endpoint predictions as well.

## Choose the saved scoring policy

Edit `config/scoring.yml` before training a new version. `mode: pre_split`
reuses saved pre-split eligibility; `custom` runs its eligibility and output
lists; `combined` runs pre-split first and keeps its first exclusion reason.
The lists start empty. Set `skip_target_pre_split_steps: true` only when
pre_split/combined should omit filters reading the unavailable target.

Built-in eligibility uses `check: required_values` or `check: value_range`;
outputs use `rule: prediction_band`. Each rule has a name, optional string
version (default `"1"`) and params. Outputs declare columns and dtypes.
`combined` is the separate list of cross-model rules: select `operation: weighted_sum` or `operation: weighted_mean` or `custom: scoring.<function>` and declare
required_components. The top-level `mode: combined` concerns eligibility,
not cross-model arithmetic.

Select optional Python helpers with `custom: scoring.<function>`; never add a
Python scoring builder alongside scoring.yml. Older Python-only projects keep
their builders when no scoring.yml exists. Saved versions replay their captured
rules even after editable YAML or Python changes. Spark/endpoint lint, row and
partition probes still apply to custom functions and reused pre-split filters.
