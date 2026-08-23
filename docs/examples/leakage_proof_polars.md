# Proof of Trust: Preventing Data Leakage with Skyulf (polars engine)

This is the [Polars](https://pola.rs/) twin of the [pandas proof](leakage_proof_pandas.md).
The guarantee is engine-agnostic — preprocessing dispatches to whichever
engine the frame uses — so this page runs the exact same experiment on
Polars frames and verifies the same invariants:

1.  **Imputation** on Test data uses the **Train mean**.
2.  **Scaling** on Test data uses the **Train mean/std**.
3.  **Target encoding** uses train-only target statistics and is invariant to test labels.
4.  A **poisoned** test set changes nothing about the fitted artifacts.

All numbers below were produced by actually running these snippets against
the current release. For the full discussion of what is and is not covered
(structure gate, cross-validation caveat), see the
[pandas version's coverage section](leakage_proof_pandas.md#what-is-and-is-not-covered).

## 1. Setup and Data Loading

```python
import numpy as np
import polars as pl
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from skyulf import SkyulfPipeline
from skyulf.data.dataset import SplitDataset

# Load Titanic Dataset (sklearn serves pandas; we convert to Polars)
print("Loading Titanic dataset...")
titanic = fetch_openml("titanic", version=1, as_frame=True)
pdf = titanic.frame[['sex', 'age', 'fare', 'survived']].copy()
pdf['survived'] = pdf['survived'].astype(int)

df = pl.from_pandas(pdf)

print(f"Dataset Shape: {df.shape}")
print(df.head())
print("\nMissing Values:\n", {c: df[c].null_count() for c in df.columns})
```

**Output:**
```text
Loading Titanic dataset...
Dataset Shape: (1309, 4)
shape: (5, 4)
┌────────┬────────┬──────────┬──────────┐
│ sex    ┆ age    ┆ fare     ┆ survived │
│ ---    ┆ ---    ┆ ---      ┆ ---      │
│ cat    ┆ f64    ┆ f64      ┆ i32      │
╞════════╪════════╪══════════╪══════════╡
│ female ┆ 29.0   ┆ 211.3375 ┆ 1        │
│ male   ┆ 0.9167 ┆ 151.55   ┆ 1        │
│ female ┆ 2.0    ┆ 151.55   ┆ 0        │
│ male   ┆ 30.0   ┆ 151.55   ┆ 0        │
│ female ┆ 25.0   ┆ 151.55   ┆ 0        │
└────────┴────────┴──────────┴──────────┘

Missing Values:
 {'sex': 0, 'age': 263, 'fare': 1, 'survived': 0}
```

## 2. Split Data

We split **BEFORE** any processing, then hand the pipeline **Polars** frames.
The engine is picked per-frame: because both halves are `pl.DataFrame`,
every preprocessing node runs its Polars implementation.

```python
X = pdf.drop(columns=['survived'])
y = pdf['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dataset = SplitDataset(
    train=pl.from_pandas(X_train.assign(survived=y_train)),
    test=pl.from_pandas(X_test.assign(survived=y_test)),
)

print(f"Train Shape: {dataset.train.shape}")
print(f"Test Shape: {dataset.test.shape}")
```

**Output:**
```text
Train Shape: (916, 4)
Test Shape: (393, 4)
```

## 3. Define Pipeline

The same leakage-prone recipe as the pandas proof: mean imputation, standard
scaling and target encoding — all fitted through the Polars code paths.

```python
config = {
    "preprocessing": [
        {"name": "impute_age", "transformer": "SimpleImputer",
         "params": {"strategy": "mean", "columns": ["age"]}},
        {"name": "impute_fare", "transformer": "SimpleImputer",
         "params": {"strategy": "mean", "columns": ["fare"]}},
        {"name": "scale_fare", "transformer": "StandardScaler",
         "params": {"columns": ["fare"]}},
        {"name": "encode_sex", "transformer": "TargetEncoder",
         "params": {"columns": ["sex"], "target_column": "survived"}},
    ],
    "modeling": {
        "type": "random_forest_classifier",
        "params": {"n_estimators": 10, "random_state": 42},
    },
}

pipeline = SkyulfPipeline(config)
print("Running pipeline (polars engine)...")
metrics = pipeline.fit(dataset, target_column="survived")
print("Pipeline execution complete.")
```

**Output:**
```text
Running pipeline (polars engine)...
Pipeline execution complete.
```

> Same note as the pandas version: the split is provided *externally* via
> `SplitDataset` (no splitter node in the config), so
> `pipeline.validate_leakage_safety()` reports the advisory no-split
> diagnostic for this config and never raises. The proofs below verify the
> per-step guarantee directly.

## 4. Verification 1: Imputation

```python
imputer_step = pipeline.feature_engineer.fitted_steps[0]
assert imputer_step['name'] == 'impute_age'
fill_values = imputer_step['artifact']['fill_values']

train_age_mean = X_train['age'].mean()
learned_mean = fill_values['age']
print(f"Train Age Mean: {train_age_mean:.4f}")
print(f"Imputer Learned Mean: {learned_mean:.4f}")
np.testing.assert_allclose(train_age_mean, learned_mean)

full_age_mean = pdf['age'].mean()
print(f"Full Dataset Age Mean: {full_age_mean:.4f}")
assert abs(learned_mean - full_age_mean) > 1e-4
print("✅ Imputation proof OK (polars)")
```

**Output:**
```text
Train Age Mean: 29.1023
Imputer Learned Mean: 29.1023
Full Dataset Age Mean: 29.8811
✅ Imputation proof OK (polars)
```

Note the polars-specific detail the node handles for you: pandas' `isna()`
counts `NaN` as missing while polars' `null_count()` does not, so the Polars
missing-count path adds an explicit `is_nan()` term on float columns to keep
both engines in parity.

## 5. Verification 2: Scaling

```python
scaler_step = pipeline.feature_engineer.fitted_steps[2]
assert scaler_step['name'] == 'scale_fare'
artifact = scaler_step['artifact']

train_fare_imputed = X_train["fare"].fillna(X_train["fare"].mean())
train_fare_mean = train_fare_imputed.mean()
train_fare_std = train_fare_imputed.std(ddof=0)

columns = artifact['columns']
fare_idx = columns.index('fare')
print(f"Train Fare Mean: {train_fare_mean:.4f}, Std: {train_fare_std:.4f}")
print(f"Scaler Learned Mean: {artifact['mean'][fare_idx]:.4f}, "
      f"Scale: {artifact['scale'][fare_idx]:.4f}")
np.testing.assert_allclose(train_fare_mean, artifact['mean'][fare_idx])
np.testing.assert_allclose(train_fare_std, artifact['scale'][fare_idx])
print("✅ Scaling proof OK (polars)")
```

**Output:**
```text
Train Fare Mean: 33.7092, Std: 52.7829
Scaler Learned Mean: 33.7092, Scale: 52.7829
✅ Scaling proof OK (polars)
```

## 6. Verification 3: Target Encoding

```python
encoder_step = pipeline.feature_engineer.fitted_steps[3]
assert encoder_step['name'] == 'encode_sex'
encoder = encoder_step['artifact']['encoder_object']

train_sex_means = (X_train.assign(survived=y_train)
                   .groupby('sex', observed=True)['survived'].mean())
categories = encoder.categories_[0]
encodings = encoder.encodings_[0]
print("Train Target Means:\n", train_sex_means)
print("Encoder Learned Means:")
for cat, enc in zip(categories, encodings):
    print(f"  {cat}: {enc:.6f}")

full_sex_means = (pdf.drop(columns=['survived']).assign(survived=pdf['survived'])
                  .groupby('sex', observed=True)['survived'].mean())
male_encoded = encodings[list(categories).index('male')]
assert abs(male_encoded - full_sex_means['male']) > 1e-4, \
    "Leakage detected! Encoded value matches Full Mean."
print("✅ Target encoding proof OK (polars)")
```

**Output:**
```text
Train Target Means:
 sex
female    0.694444
male      0.179054
Name: survived, dtype: float64
Encoder Learned Means:
  female: 0.693502
  male: 0.179250
✅ Target encoding proof OK (polars)
```

The encodings are smoothed/cross-fitted, so they are *near* — not exactly
equal to — the raw train conditional means, and provably different from the
full-dataset means (`male` full mean is 0.190985).

## 7. The Poisoned Dataset (Polars)

Same adversarial experiment: corrupt the test half, refit a fresh pipeline,
compare artifacts. With no leakage, nothing can change.

```python
X_test_poisoned = X_test.copy()
X_test_poisoned['age'] = 10000.0
X_test_poisoned['fare'] = 1000000.0
y_test_poisoned = 1 - y_test

dataset_poisoned = SplitDataset(
    train=pl.from_pandas(X_train.assign(survived=y_train)),
    test=pl.from_pandas(X_test_poisoned.assign(survived=y_test_poisoned)),
)

pipeline_poisoned = SkyulfPipeline(config)
print("Running pipeline on Poisoned Dataset (polars)...")
pipeline_poisoned.fit(dataset_poisoned, target_column="survived")

# Imputation
orig_age = pipeline.feature_engineer.fitted_steps[0]['artifact']['fill_values']['age']
pois_age = pipeline_poisoned.feature_engineer.fitted_steps[0]['artifact']['fill_values']['age']
np.testing.assert_allclose(orig_age, pois_age)
print(f"✅ Imputation invariant: {orig_age:.4f} == {pois_age:.4f}")

# Scaling
o_s = pipeline.feature_engineer.fitted_steps[2]['artifact']
p_s = pipeline_poisoned.feature_engineer.fitted_steps[2]['artifact']
np.testing.assert_allclose(o_s['mean'], p_s['mean'])
np.testing.assert_allclose(o_s['scale'], p_s['scale'])
print(f"✅ Scaling invariant: mean {o_s['mean']}, scale {o_s['scale']}")

# Target encoding
o_e = pipeline.feature_engineer.fitted_steps[3]['artifact']['encoder_object'].encodings_[0]
p_e = pipeline_poisoned.feature_engineer.fitted_steps[3]['artifact']['encoder_object'].encodings_[0]
np.testing.assert_allclose(o_e, p_e)
print(f"✅ Encodings invariant: {o_e}")

```

**Output:**
```text
Running pipeline on Poisoned Dataset (polars)...
✅ Imputation invariant: 29.1023 == 29.1023
✅ Scaling invariant: mean [33.70922076502732], scale [52.7829380540588]
✅ Encodings invariant: [0.69350186 0.17924998]

```

## Conclusion

Every fitted artifact is bit-identical to the pandas proof (same dataset,
same split seed): the Polars engine computes imputation, scaling and target
encoding from the training half only, and is provably invariant under a
poisoned test set. The leakage guarantee is a property of Skyulf's
Calculator/Applier design, not of the execution engine.
