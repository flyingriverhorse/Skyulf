# Validation vs scikit-learn (Proof)

This page gives **reproducible, runnable checks** that:

- Skyulf supports the familiar scikit-learn workflow (build `X`/`y`, run `train_test_split`, fit on train, transform/predict on test).
- Skyulf avoids common forms of data leakage by learning preprocessing parameters from **train only**.

> Goal: show *verifiable behavior*, not claim bit-for-bit identical models.

## 1) scikit-learn-style workflow (X/y + train_test_split)

This mirrors the sklearn pattern:

- sklearn: `fit(X_train, y_train)` then `predict(X_test)`
- Skyulf: pass `SplitDataset(train=(X_train, y_train), test=(X_test, y_test))`

```python
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from skyulf.data.dataset import SplitDataset
from skyulf.pipeline import SkyulfPipeline

# Synthetic classification data
raw = pd.DataFrame(
    {
        "age": [10, 20, None, 40, 50, 60, None, 80],
        "city": ["A", "B", "A", "C", "B", "A", "C", "B"],
        "target": [0, 1, 0, 1, 1, 0, 1, 0],
    }
)

X = raw.drop(columns=["target"])
y = raw["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

dataset = SplitDataset(train=(X_train, y_train), test=(X_test, y_test), validation=None)

# IMPORTANT: because we already split, we do not add TrainTestSplitter here.
config = {
    "preprocessing": [
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"strategy": "mean", "columns": ["age"]},
        },
        {
            "name": "encode",
            "transformer": "OneHotEncoder",
            "params": {"columns": ["city"], "drop_original": True, "handle_unknown": "ignore"},
        },
    ],
    "modeling": {"type": "random_forest_classifier", "params": {"n_estimators": 50, "random_state": 42}},
}

pipeline = SkyulfPipeline(config)
metrics = pipeline.fit(dataset, target_column="target")  # target_column ignored for (X, y) tuples

preds = pipeline.predict(X_test)

# Proof-like checks
assert len(preds) == len(X_test)
assert preds.index.equals(X_test.index)
print("OK: Skyulf fit/predict with sklearn-style train/test split")
print("Metrics keys:", list(metrics.keys()))
```

## 1b) Side-by-side run: sklearn Pipeline vs SkyulfPipeline

This comparison proves both stacks can run the same *shape* of workflow on the same split.
We do **not** assert equality of predictions (different defaults / encodings can legitimately differ).

For a stronger numeric sanity check, we also compute and print test accuracy for both pipelines and the absolute difference.

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer as SkSimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
from sklearn.preprocessing import StandardScaler

from skyulf import SkyulfPipeline
from skyulf.data.dataset import SplitDataset

np.random.seed(42)

# Real dataset + extra categorical feature + missingness
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy().rename(columns={"target": "label"})

df["radius_band"] = pd.cut(
    df["mean radius"],
    bins=[0, 12, 15, 100],
    labels=["small", "medium", "large"],
    include_lowest=True,
)

missing_idx = np.random.choice(df.index, size=25, replace=False)
df.loc[missing_idx, "mean texture"] = np.nan

target_col = "label"
cat_cols = ["radius_band"]
num_cols = [c for c in df.columns if c not in [target_col, *cat_cols]]

X = df[num_cols + cat_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# --- scikit-learn pipeline ---
numeric_features = num_cols
categorical_features = cat_cols

numeric_pipe = SkPipeline(
    steps=[
        ("imputer", SkSimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipe = SkPipeline(
    steps=[
        ("imputer", SkSimpleImputer(strategy="most_frequent")),
        ("onehot", SkOneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_features),
        ("cat", categorical_pipe, categorical_features),
    ],
    remainder="drop",
)

sk_model = LogisticRegression(max_iter=1000, random_state=42)
sk = SkPipeline(steps=[("preprocess", preprocess), ("model", sk_model)])
sk.fit(X_train, y_train)
sk_preds = pd.Series(sk.predict(X_test), index=X_test.index)
sk_acc = accuracy_score(y_test, sk_preds)

# --- Skyulf pipeline ---
train_df = X_train.copy()
train_df[target_col] = y_train
test_df = X_test.copy()
test_df[target_col] = y_test

dataset = SplitDataset(train=train_df, test=test_df, validation=None)
skyulf_config = {
    "preprocessing": [
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"strategy": "mean", "columns": num_cols},
        },
        {
            "name": "impute_cat",
            "transformer": "SimpleImputer",
            "params": {"strategy": "most_frequent", "columns": cat_cols},
        },
        {
            "name": "encode",
            "transformer": "OneHotEncoder",
            "params": {"columns": cat_cols, "drop_original": True, "handle_unknown": "ignore"},
        },
        {
            "name": "scale",
            "transformer": "StandardScaler",
            "params": {"columns": num_cols},
        },
    ],
    "modeling": {"type": "logistic_regression", "params": {"max_iter": 1000, "random_state": 42}},
}

sky = SkyulfPipeline(skyulf_config)
_ = sky.fit(dataset, target_column=target_col)
sky_preds = sky.predict(X_test)
sky_acc = accuracy_score(y_test, sky_preds)
delta = abs(sk_acc - sky_acc)

assert sk_preds.index.equals(X_test.index)
assert sky_preds.index.equals(X_test.index)

# Proof-like checks
assert len(sk_preds) == len(X_test)
assert sk_preds.index.equals(X_test.index)
assert len(sky_preds) == len(X_test)
assert sky_preds.index.equals(X_test.index)

print("OK: sklearn Pipeline and SkyulfPipeline both run")
print(f"sklearn test accuracy: {sk_acc:.4f}")
print(f"skyulf  test accuracy: {sky_acc:.4f}")
print(f"delta accuracy: {delta:.4f}")

# --- Classification metrics (side-by-side) ---
sk_report = classification_report(y_test, sk_preds, output_dict=True, zero_division=0)
sky_report = classification_report(y_test, sky_preds, output_dict=True, zero_division=0)

sk_df = pd.DataFrame(sk_report).T
sky_df = pd.DataFrame(sky_report).T

# Keep a consistent row order: class labels first, then summary rows (if present)
label_rows = [str(v) for v in sorted(pd.unique(y_test))]
summary_rows = [r for r in ["accuracy", "macro avg", "weighted avg"] if r in sk_df.index]
row_order = [r for r in label_rows if r in sk_df.index] + summary_rows

sk_df = sk_df.loc[row_order]
sky_df = sky_df.loc[row_order]

side_by_side = pd.concat(
    {
        "sklearn": sk_df[["precision", "recall", "f1-score", "support"]],
        "skyulf": sky_df[["precision", "recall", "f1-score", "support"]],
    },
    axis=1,
)

print("\nClassification report (side-by-side):")
print(side_by_side.to_string())

labels = sorted(pd.unique(y_test))
cm_sk = confusion_matrix(y_test, sk_preds, labels=labels)
cm_sky = confusion_matrix(y_test, sky_preds, labels=labels)

cm_index = [f"true_{l}" for l in labels]
cm_cols = [f"pred_{l}" for l in labels]

cm_sk_df = pd.DataFrame(cm_sk, index=cm_index, columns=cm_cols)
cm_sky_df = pd.DataFrame(cm_sky, index=cm_index, columns=cm_cols)

print("\nConfusion matrix (sklearn):")
print(cm_sk_df.to_string())
print("\nConfusion matrix (skyulf):")
print(cm_sky_df.to_string())

print("sklearn preds head:")
print(sk_preds.head())
print("skyulf preds head:")
print(sky_preds.head())
```

### Example output (from the notebook)

This is the exact output from one notebook run (same dataset, same random seed/split):

Classification report (side-by-side):

| class/avg | sklearn precision | sklearn recall | sklearn f1-score | sklearn support | skyulf precision | skyulf recall | skyulf f1-score | skyulf support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.962963 | 0.981132 | 0.971963 | 53.000000 | 0.962963 | 0.981132 | 0.971963 | 53.000000 |
| 1 | 0.988764 | 0.977778 | 0.983240 | 90.000000 | 0.988764 | 0.977778 | 0.983240 | 90.000000 |
| accuracy | 0.979021 | 0.979021 | 0.979021 | 0.979021 | 0.979021 | 0.979021 | 0.979021 | 0.979021 |
| macro avg | 0.975864 | 0.979455 | 0.977601 | 143.000000 | 0.975864 | 0.979455 | 0.977601 | 143.000000 |
| weighted avg | 0.979201 | 0.979021 | 0.979060 | 143.000000 | 0.979201 | 0.979021 | 0.979060 | 143.000000 |

Confusion matrix (sklearn):

|  | pred_0 | pred_1 |
|---|---:|---:|
| true_0 | 52 | 1 |
| true_1 | 2 | 88 |

Confusion matrix (skyulf):

|  | pred_0 | pred_1 |
|---|---:|---:|
| true_0 | 52 | 1 |
| true_1 | 2 | 88 |

## 2) Proof of leakage prevention (train-only learned params)

A common leakage bug is fitting preprocessing on the full dataset.

Here we construct a dataset where **train and test have very different distributions**.
If an imputer learns the mean from the full dataset, it will be pulled toward the test distribution.

Skyulf’s pattern learns from train only (Calculator) and applies to test (Applier).

```python
from __future__ import annotations

import pandas as pd

from skyulf.preprocessing.imputation import SimpleImputerCalculator

# Train has small ages; test has huge ages.
X_train = pd.DataFrame({"age": [1.0, 2.0, None, 2.0]})
y_train = pd.Series([0, 1, 0, 1])

X_test = pd.DataFrame({"age": [1000.0, None, 1200.0]})
y_test = pd.Series([0, 1, 1])

cfg = {"strategy": "mean", "columns": ["age"]}

# What train-only mean should be (ignoring NaNs)
expected_train_mean = float(pd.Series([1.0, 2.0, 2.0]).mean())

params = SimpleImputerCalculator().fit((X_train, y_train), cfg)
learned_mean = float(params["fill_values"]["age"])

# Proof: learned mean equals train mean (not influenced by test)
assert abs(learned_mean - expected_train_mean) < 1e-12

# For comparison only: full-data mean would be very different
full_mean = float(pd.concat([X_train["age"], X_test["age"]]).mean())
assert abs(full_mean - expected_train_mean) > 1.0

print("OK: SimpleImputer learns from train only")
print("train_mean:", expected_train_mean)
print("full_mean:", full_mean)
print("learned_mean:", learned_mean)
```

## 3) What this proves (and what it doesn’t)

- Proves the API supports sklearn-style `X/y` workflows and produces aligned predictions.
- Proves at least one common leakage-sensitive node (`SimpleImputer`) learns its statistics from the provided training data.

It does **not** claim Skyulf will produce identical predictions to an arbitrary sklearn pipeline, because:

- different defaults/hyperparameters,
- different encoding conventions,
- and different ordering of operations

can all change results while still being correct.
