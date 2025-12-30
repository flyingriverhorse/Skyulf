# Proof of Trust: Preventing Data Leakage with Skyulf

One of the biggest risks in Machine Learning is **Data Leakage**: when information from the test set (or the future) accidentally "leaks" into the training process. This creates models that look perfect during training but fail in production.

Common sources of leakage:

1.  **Imputation:** Filling missing values in the Test set using the mean of the *entire* dataset (including Test).
2.  **Scaling:** Normalizing Test data using the min/max of the *entire* dataset.
3.  **Target Encoding:** Encoding categorical features using the target mean of the *entire* dataset.

## The Skyulf Guarantee

Skyulf prevents this by design using the **Calculator / Applier** pattern.

-   **Calculator:** Learns statistics *only* from the Training data.
-   **Applier:** Applies those learned statistics to Test data blindly.

This example proves this behavior using the **Titanic** dataset. We will:

1.  Load the dataset.
2.  Split it into Train/Test.
3.  Run a Skyulf Pipeline.
4.  **Mathematically verify** that the Test data was processed using *only* Training statistics.

## 1. Setup and Data Loading

```python
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from skyulf import SkyulfPipeline
from skyulf.data.dataset import SplitDataset

# Load Titanic Dataset
print("Loading Titanic dataset...")
titanic = fetch_openml("titanic", version=1, as_frame=True)
df = titanic.frame

# Select relevant columns for demonstration
# 'sex': Categorical (needs encoding)
# 'age': Numeric with missing values (needs imputation)
# 'fare': Numeric (needs scaling)
# 'survived': Target
cols = ['sex', 'age', 'fare', 'survived']
df = df[cols].copy()

# Convert target to int
df['survived'] = df['survived'].astype(int)

print(f"Dataset Shape: {df.shape}")
print(df.head())
print("\nMissing Values:\n", df.isnull().sum())
```

**Output:**
```text
Loading Titanic dataset...
Dataset Shape: (1309, 4)
      sex   age      fare  survived
0  female  29.0  211.3375         1
1    male   0.9151.5500         1
2  female   2.0  151.5500         0
3    male  30.0  151.5500         0
4  female  25.0  151.5500         0

Missing Values:
 sex           0
age         263
fare          1
survived      0
dtype: int64
```

## 2. Split Data

We split **BEFORE** any processing to simulate a real-world scenario.

```python
X = df.drop(columns=['survived'])
y = df['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Skyulf Dataset
dataset = SplitDataset(
    train=pd.concat([X_train, y_train], axis=1),
    test=pd.concat([X_test, y_test], axis=1)
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

We intentionally use methods that are prone to leakage if done wrong.

```python
config = {
    "preprocessing": [
        # Imputation 1: Age (Train Mean)
        {
            "name": "impute_age",
            "transformer": "SimpleImputer",
            "params": {"strategy": "mean", "columns": ["age"]}
        },
        # Imputation 2: Fare (Train Mean) - Added for robustness
        {
            "name": "impute_fare",
            "transformer": "SimpleImputer",
            "params": {"strategy": "mean", "columns": ["fare"]}
        },
        # Scaling: Should use Train Mean/Std
        {
            "name": "scale_fare",
            "transformer": "StandardScaler",
            "params": {"columns": ["fare"]}
        },
        # Target Encoding: Should use Train Target Mean
        # This is the most dangerous one! If it sees Test target, it's 100% leakage.
        {
            "name": "encode_sex",
            "transformer": "TargetEncoder",
            "params": {"columns": ["sex"], "target_column": "survived"}
        }
    ],
    "modeling": {
        "type": "random_forest_classifier",
        "params": {"n_estimators": 10, "random_state": 42}
    }
}

pipeline = SkyulfPipeline(config)

# Fit the pipeline
# This runs fit() on Train and transform() on Test
print("Running pipeline...")
metrics = pipeline.fit(dataset, target_column="survived")
print("Pipeline execution complete.")
```

**Output:**
```text
Running pipeline...
Pipeline execution complete.
```

## 4. Verification 1: Imputation Leakage

Did we fill missing 'age' in Test with the Train mean?

```python
# Get the fitted imputer from the pipeline
imputer_step = pipeline.feature_engineer.fitted_steps[0]
assert imputer_step['name'] == 'impute_age'

# The fitted transformer is stored in 'artifact'
artifact = imputer_step['artifact']
fill_values = artifact['fill_values']

# Calculate Train Mean manually
train_age_mean = X_train['age'].mean()
print(f"Train Age Mean: {train_age_mean:.4f}")

# Check what the imputer learned
learned_mean = fill_values['age']
print(f"Imputer Learned Mean: {learned_mean:.4f}")

# Verify
np.testing.assert_allclose(train_age_mean, learned_mean)

# --- NEW: Explicitly check against Full Dataset Mean ---
full_age_mean = df['age'].mean()
print(f"Full Dataset Age Mean: {full_age_mean:.4f}")

# Assert that learned mean is NOT the full mean (additional sanity check)
# Note: In some datasets, train mean could equal full mean by chance.
if abs(learned_mean - full_age_mean) > 1e-4:
    print("✅ Imputation Sanity Check Passed: Learned mean differs from Full Dataset mean.")
else:
    print("⚠️ Warning: Train mean equals Full mean (could be chance, but check split).")

print("✅ Imputation Proof: The imputer learned the mean ONLY from the Training set.")
```

**Output:**
```text
Train Age Mean: 29.1023
Imputer Learned Mean: 29.1023
Full Dataset Age Mean: 29.8811
✅ Imputation Sanity Check Passed: Learned mean differs from Full Dataset mean.
✅ Imputation Proof: The imputer learned the mean ONLY from the Training set.
```

## 5. Verification 2: Scaling Leakage

Did we scale 'fare' using Train Mean/Std?

```python
# Note: Index is 2 because we added impute_fare at index 1
scaler_step = pipeline.feature_engineer.fitted_steps[2]
assert scaler_step['name'] == 'scale_fare'

artifact = scaler_step['artifact']

# Calculate Train Stats manually (Exact)
# We must use the imputed fare for manual calculation to match the pipeline's flow.
train_fare_imputed = X_train["fare"].fillna(X_train["fare"].mean())
train_fare_mean = train_fare_imputed.mean()
train_fare_std  = train_fare_imputed.std(ddof=0) # Sklearn uses ddof=0 for std

print(f"Train Fare Mean: {train_fare_mean:.4f}, Std: {train_fare_std:.4f}")

# Check what the scaler learned
columns = artifact['columns']
fare_idx = columns.index('fare')

learned_mean = artifact['mean'][fare_idx]
learned_scale = artifact['scale'][fare_idx]
print(f"Scaler Learned Mean: {learned_mean:.4f}, Scale: {learned_scale:.4f}")

# Verify (Tight tolerance now possible)
np.testing.assert_allclose(train_fare_mean, learned_mean)
np.testing.assert_allclose(train_fare_std, learned_scale)

# --- NEW: Explicitly check against Full Dataset Stats ---
full_fare_mean = df['fare'].mean()
full_fare_std = df['fare'].std(ddof=0)
print(f"Full Dataset Fare Mean: {full_fare_mean:.4f}, Std: {full_fare_std:.4f}")

# Additional Sanity Checks
if abs(learned_mean - full_fare_mean) > 1e-4:
    print("✅ Scaling Mean Sanity Check Passed: Learned mean differs from Full Dataset mean.")
else:
    print("⚠️ Warning: Train mean equals Full mean.")

if abs(learned_scale - full_fare_std) > 1e-4:
    print("✅ Scaling Std Sanity Check Passed: Learned std differs from Full Dataset std.")
else:
    print("⚠️ Warning: Train std equals Full std.")

print("✅ Scaling Proof: The scaler learned stats ONLY from the Training set.")
```

**Output:**
```text
Train Fare Mean: 33.7092, Std: 52.7829
Scaler Learned Mean: 33.7092, Scale: 52.7829
Full Dataset Fare Mean: 33.2955, Std: 51.7389
✅ Scaling Mean Sanity Check Passed: Learned mean differs from Full Dataset mean.
✅ Scaling Std Sanity Check Passed: Learned std differs from Full Dataset std.
✅ Scaling Proof: The scaler learned stats ONLY from the Training set.
```

## 6. Verification 3: Target Encoding Leakage

Did we encode 'sex' using the Target Mean of the Training set?

```python
# Note: Index is 3
encoder_step = pipeline.feature_engineer.fitted_steps[3]
assert encoder_step['name'] == 'encode_sex'

artifact = encoder_step['artifact']
encoder = artifact['encoder_object']

# Calculate Train Target Mean for 'sex' manually
train_sex_means = pd.concat([X_train, y_train], axis=1).groupby('sex', observed=True)['survived'].mean()
print("Train Target Means:\n", train_sex_means)

# Check what the encoder learned
categories = encoder.categories_[0]
encodings = encoder.encodings_[0]

print("\nEncoder Learned Means:")
for cat, enc in zip(categories, encodings):
    print(f"  {cat}: {enc:.6f}")

# Verify
# Note: Sklearn's TargetEncoder uses cross-fitting and shrinkage (smoothing), 
# so the learned encodings will NOT equal the raw conditional means of the training set.
# However, they must be **invariant** to changes in the Test set.

full_sex_means = pd.concat([X, y], axis=1).groupby('sex', observed=True)['survived'].mean()
print("\nFull Dataset Means (Leakage!):\n", full_sex_means)

# Check 'male'
male_train_mean = train_sex_means['male']
male_full_mean = full_sex_means['male']
male_encoded = encodings[list(categories).index('male')]

print(f"\nComparison for 'male':")
print(f"  Train Mean: {male_train_mean:.6f}")
print(f"  Full Mean:  {male_full_mean:.6f}")
print(f"  Encoded:    {male_encoded:.6f}")

assert abs(male_encoded - male_full_mean) > 1e-4, "Leakage detected! Encoded value matches Full Mean."
print("✅ Target Encoding Proof: The encoder did NOT use the full dataset statistics.")
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

Full Dataset Means (Leakage!):
 sex
female    0.727468
male      0.190985
Name: survived, dtype: float64

Comparison for 'male':
  Train Mean: 0.179054
  Full Mean:  0.190985
  Encoded:    0.179250
✅ Target Encoding Proof: The encoder did NOT use the full dataset statistics.
```

## 7. The Ultimate Test: The "Poisoned" Dataset

To prove beyond doubt that the Test set is ignored during training, we will run an experiment:

1.  Take the original dataset.
2.  **"Poison" the Test set** with extreme outliers and flipped labels.
3.  Train a **new pipeline** on this poisoned dataset.
4.  Compare the learned parameters with the original pipeline.

**Hypothesis:** If there is NO leakage, the learned parameters (Imputation Mean, Scaling Stats, Encodings) must be **identical** to the original run, because the Training set hasn't changed.

```python
# Create a Poisoned Dataset
# We keep Train exactly the same, but corrupt Test.
X_test_poisoned = X_test.copy()
y_test_poisoned = y_test.copy()

# 1. Poison 'age' (Imputation)
# Set all test ages to a massive number. If leakage exists, the mean will skyrocket.
X_test_poisoned['age'] = 10000.0 

# 2. Poison 'fare' (Scaling)
# Set all test fares to a massive number.
X_test_poisoned['fare'] = 1000000.0 

# 3. Poison 'survived' (Target Encoding)
# Flip all labels: 0->1, 1->0. If leakage exists, encodings will flip.
y_test_poisoned = 1 - y_test_poisoned

# Create Skyulf Dataset with Poisoned Test
dataset_poisoned = SplitDataset(
    train=pd.concat([X_train, y_train], axis=1),
    test=pd.concat([X_test_poisoned, y_test_poisoned], axis=1)
)

print("Poisoned Test Stats:")
print(f"  Age Mean: {X_test_poisoned['age'].mean():.2f}")
print(f"  Fare Mean: {X_test_poisoned['fare'].mean():.2f}")
print(f"  Target Mean: {y_test_poisoned.mean():.2f}")

# Run Pipeline on Poisoned Data
pipeline_poisoned = SkyulfPipeline(config) # Same config
print("\nRunning pipeline on Poisoned Dataset...")
pipeline_poisoned.fit(dataset_poisoned, target_column="survived")

# Compare Artifacts

# 1. Imputation (Age)
original_age_mean = pipeline.feature_engineer.fitted_steps[0]['artifact']['fill_values']['age']
poisoned_age_mean = pipeline_poisoned.feature_engineer.fitted_steps[0]['artifact']['fill_values']['age']

print(f"\nImputation Comparison:")
print(f"  Original: {original_age_mean:.4f}")
print(f"  Poisoned: {poisoned_age_mean:.4f}")
np.testing.assert_allclose(original_age_mean, poisoned_age_mean)
print("✅ Imputation is unaffected by Test data.")

# 2. Scaling (Fare)
# Note: Index is 2
original_scaler = pipeline.feature_engineer.fitted_steps[2]['artifact']
poisoned_scaler = pipeline_poisoned.feature_engineer.fitted_steps[2]['artifact']

original_fare_mean = original_scaler['mean'][0] # fare is only col
poisoned_fare_mean = poisoned_scaler['mean'][0]
original_fare_scale = original_scaler['scale'][0]
poisoned_fare_scale = poisoned_scaler['scale'][0]

print(f"\nScaling Comparison:")
print(f"  Original Mean: {original_fare_mean:.4f}, Scale: {original_fare_scale:.4f}")
print(f"  Poisoned Mean: {poisoned_fare_mean:.4f}, Scale: {poisoned_fare_scale:.4f}")

np.testing.assert_allclose(original_fare_mean, poisoned_fare_mean)
np.testing.assert_allclose(original_fare_scale, poisoned_fare_scale)
print("✅ Scaling is unaffected by Test data.")

# 3. Target Encoding
# Note: Index is 3
original_encodings = pipeline.feature_engineer.fitted_steps[3]['artifact']['encoder_object'].encodings_[0]
poisoned_encodings = pipeline_poisoned.feature_engineer.fitted_steps[3]['artifact']['encoder_object'].encodings_[0]

print(f"\nTarget Encoding Comparison (First 5 values):")
print(f"  Original: {original_encodings[:5]}")
print(f"  Poisoned: {poisoned_encodings[:5]}")
np.testing.assert_allclose(original_encodings, poisoned_encodings)
print("✅ Target Encoding is unaffected by Test labels.")

print("\n🎉 FINAL VERDICT: The pipeline is LEAKAGE-RESISTANT by design.")
```

**Output:**
```text
Poisoned Test Stats:
  Age Mean: 10000.00
  Fare Mean: 1000000.00
  Target Mean: 0.57

Running pipeline on Poisoned Dataset...

Imputation Comparison:
  Original: 29.1023
  Poisoned: 29.1023
✅ Imputation is unaffected by Test data.

Scaling Comparison:
  Original Mean: 33.7092, Scale: 52.7829
  Poisoned Mean: 33.7092, Scale: 52.7829
✅ Scaling is unaffected by Test data.

Target Encoding Comparison (First 5 values):
  Original: [0.69350186 0.17924998]
  Poisoned: [0.69350186 0.17924998]
✅ Target Encoding is unaffected by Test labels.

🎉 FINAL VERDICT: The pipeline is LEAKAGE-RESISTANT by design.
```

## Conclusion

We have mathematically verified that:

1.  **Imputation** on Test data used the **Train Mean**.
2.  **Scaling** on Test data used the **Train Mean/Std**.
3.  **Target Encoding** used train-derived target statistics (smoothed/cross-fitted), and is invariant to test labels.

This proves that **Skyulf pipelines are leakage-free by design**. The strict separation of `fit()` (Calculator) and `transform()` (Applier) ensures that no information from the Test set (or future data) can influence the model training. Under adversarial ‘poisoned test’ conditions, the learned preprocessing artifacts are invariant to test data and test labels, providing strong empirical evidence that the pipeline fits strictly on training data for these steps.
