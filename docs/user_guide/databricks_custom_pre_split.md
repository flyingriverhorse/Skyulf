# Write your own pre-split filter

## When you need it

Use a custom pre-split filter when a fixed eligibility rule must run before the
holdout and CV split, and the
[admitted built-in pre-split steps](preprocessing_recipes.md#pre-split-admission-and-scoring)
cannot express it. A filter keeps or removes whole rows. It learns nothing and
changes no values. Learned transformations belong in `config/preprocessing.yml`;
see [custom steps](databricks_custom_steps.md) for those.

A generated project already ships three filter factories in
`src/features/pre_split.py`: `pre_split.minimum_completeness`,
`pre_split.value_range` and `pre_split.allowed_values`. Their parameters are in
the [ready functions catalog](preprocessing_recipes_custom.md). Select the factory
with `custom:` in `config/pre_split.yml`, not the callback that takes `df`.

## What to edit

### The filter contract

`filter_step` builds a `RowFilterFunction` step:

```text
filter_step(name, function, *, columns, params=None, inference_context=None)
```

| Callback you write | Required return | When Skyulf calls it |
| --- | --- | --- |
| `function(df)`; with nonempty params, `function(df, params)` | A pandas Boolean Series with exactly one non-null True/False per row, aligned to `df.index` | Generated pre-split eligibility before train/holdout splitting; optionally reused by the saved scoring policy before prediction |

`df` is a private pandas copy, also for Polars models. `columns` declares the
columns the filter reads; they are read automatically and need not be model
inputs. Decide missing values explicitly, for example
`(df["x"] > 0).fillna(False)`. Use only data known at the observation cutoff;
eligibility must not depend on future information.

Custom pre-split steps remain declared fixed filters: they preserve survivor
values and order and cannot learn statistics. The generated
`config/scoring.yml` defaults to `mode: pre_split`, which reuses
the saved filters on unlabeled score input. Filters reading the target require
an explicit skip choice, shown below.

The shipped completeness filter treats null/NaN as missing; blank strings and
infinity count as values. With three selected fields and `min_present=2`, rows
with two or three observed values survive without any value or order changes.

### Example: an adult-only filter

For this example, start with a **single regression model**, `engine: pandas`,
with numeric inputs `age` and `income`, target `claim_amount`, and
`record_id` as the record key. Keep the key outside model inputs. The fixed
business rule is age at least 18; missing ages are ineligible.

Create `src/features/eligibility_rules.py`. The callback receives rows; the factory
receives only
the YAML parameters and returns one step.

<!-- example:custom-filter-code -->
```python
# src/features/eligibility_rules.py
from skyulf.preprocessing import filter_step


def adult_rows(df, params):
    """Keep rows meeting a fixed age limit; reject missing ages."""
    return (df[params["column"]] >= params["minimum"]).fillna(False)


def adults_only(column, minimum=18):
    """Build a fixed eligibility filter without learning from the data."""
    return filter_step(
        "adult_only",
        adult_rows,
        columns=[column],
        params={"column": column, "minimum": minimum},
    )


def target_present(df, params):
    """Keep training rows whose actual target is known."""
    return df[params["column"]].notna()


def known_target(column):
    """Declare target usage so scoring can require an explicit skip choice."""
    return filter_step(
        f"known_{column}",
        target_present,
        columns=[column],
        params={"column": column},
    )
```

Replace `config/pre_split.yml` with the following recipes, or merge the two
named recipes into the existing `recipes` mapping. Keep one `default` recipe.

<!-- example:custom-filter-recipes -->
```yaml
version: 1
recipes:
  default: []
  none: []
  adult_custom:
    - custom: eligibility_rules.adults_only
      params: {column: age, minimum: 18}
  adult_labeled:
    - custom: eligibility_rules.adults_only
      params: {column: age, minimum: 18}
    - custom: eligibility_rules.known_target
      params: {column: claim_amount}
```

Here `eligibility_rules.adults_only` resolves to
`src/features/eligibility_rules.py` and calls
`adults_only(column="age", minimum=18)`. Activate `adult_custom` by merging
these fields into the existing `config/training.yml`. Retain its generated
table, model, split, quality and resource settings. The generated
`config/preprocessing.yml` already has an empty `none` recipe.

<!-- example:custom-filter-selection -->
```yaml
defaults:
  input_columns: [age, income]
  target_column: claim_amount
  pre_split_recipe: adult_custom
  preprocessing_recipe: none
```

## The command

Run from the generated project directory:

```powershell
python src/tools/smoke.py
python src/tools/preview.py --action train
python src/tools/preview.py --action score
```

Smoke should return `"status": "passed"` with
`"project_hooks_executed": false`: it checks syntax and declarations.
Training preview resolves `adult_custom` to the `adult_only` step.
Score preview validates the configured scoring workflow; previews make no
remote reads and do not apply filters to rows. The new filter must appear in
the training preview before training a new model.

## What you should see

For example, these training inputs retain record IDs **102, 103, 104, 105**,
in that order, with all survivor values unchanged. These small samples illustrate
the filter; use a source large enough for the configured train/holdout split and
CV when training a Bundle model:

<!-- example:custom-filter-training -->
```csv
record_id,age,income,claim_amount
101,17.0,10.0,1.0
102,18.0,20.0,2.0
103,30.0,30.0,3.0
104,50.0,40.0,4.0
105,70.0,50.0,5.0
106,,60.0,6.0
```

After training, the default `mode: pre_split` uses the saved filter
on unlabeled rows. For these model inputs (supply record keys separately when
running the Bundle batch job):

<!-- example:custom-filter-scoring -->
```csv
age,income
17.0,10.0
18.0,20.0
30.0,30.0
,60.0
```

The four result rows have `scoring_status`
`[excluded, predicted, predicted, excluded]`. Excluded rows remain in the
result with an empty prediction and `exclusion_reason: pre_split:adult_only`;
only the two survivors reach the model.

### Filters that read the target

To also require a known training label, select
`pre_split_recipe: adult_labeled`. For example, if record 104 above has a
missing `claim_amount`, the two filters retain **102, 103, 105**.
Because the target is unavailable at scoring time, the default skip setting
makes training preview stop with:

```text
Pre-split step 'known_claim_amount' reads target 'claim_amount'. Set SKIP_TARGET_PRE_SPLIT_STEPS=True or SCORING_MODE='custom'.
```

If the intended scoring population is adults regardless of label availability,
replace `config/scoring.yml` **before training**:

<!-- example:custom-filter-target-skip -->
```yaml
version: 1
mode: pre_split
skip_target_pre_split_steps: true
eligibility: []
outputs: []
combined: []
```

Rerun smoke and both previews. Training still applies both filters; scoring
skips `known_claim_amount` and keeps `adult_only`, so the unlabeled scoring
example has the same four outcomes. The choice is saved with each new model
version; editing the file does not change an already trained model.

This switch skips an entire target-reading filter. Keep feature eligibility and
target checks in separate steps if the feature rule must remain active.
`mode: custom` instead uses only the separately configured scoring
rules; merely switching modes does not activate any commented examples.
See [scoring eligibility](databricks_scoring.md#2-decide-eligibility-before-training-the-version).

## Common errors and fixes

| Symptom | Fix |
| --- | --- |
| A custom factory cannot be resolved | Use a path relative to `src/features/`, ending in the factory name; retain `__init__.py` in subpackages. |
| A target-reading filter prevents unlabeled scoring | Decide whether it is training-only and set `skip_target_pre_split_steps` in scoring.yml explicitly before training. See [troubleshooting](databricks_troubleshooting.md#a-pre-split-step-reads-the-target). |
| `Filter function must return a boolean Series with one value per row.` | Return a pandas Series aligned to `df.index`, one value per row. |
| `Filter function must return only True/False values, e.g. df['age'] >= 18.` | Fill missing comparisons explicitly, for example with `.fillna(False)`. |
| A filter needs learned statistics | Move the rule to fitted preprocessing; pre-split filters stay fixed. |
| Edits do not alter an existing model | Train a new version; saved source and filter declarations are immutable inputs to scoring. |
