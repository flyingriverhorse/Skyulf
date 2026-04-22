# Troubleshooting

Common issues and solutions when using Skyulf.

---

## "Unknown transformer type"

Your step's `transformer` string does not match any registered node in the `NodeRegistry`.

**Fix:** Check spelling and casing. Use the exact key from the [Configuration](configuration.md) page (e.g., `OneHotEncoder`, not `one_hot_encoder`).

---

## "Unknown model type: ..."

The `type` value in your modeling config is not registered.

**Fix:** Ensure you are using one of the 20 supported model keys listed in [Configuration - Modeling config](configuration.md#modeling-config). If using XGBoost, install the extra: `pip install skyulf-core[modeling-xgboost]`.

---

## Resampling errors about non-numeric columns

Oversampling / undersampling (SMOTE, RandomOverSampler, etc.) require all features to be numeric.

**Fix:** Apply an encoder (e.g., `OneHotEncoder` or `OrdinalEncoder`) **before** any resampling step in your preprocessing config.

---

## SMOTE not found / ImportError

The `imbalanced-learn` package is an optional dependency.

**Fix:**

```bash
pip install skyulf-core[preprocessing-imbalanced]
```

---

## XGBoost ImportError

XGBoost estimators are optional and not installed by default.

**Fix:**

```bash
pip install skyulf-core[modeling-xgboost]
```

---

## Optuna / TuningCalculator ImportError

Hyperparameter tuning with Optuna requires the tuning extra.

**Fix:**

```bash
pip install skyulf-core[tuning]
```

---

## "X merges N parallel branches…" (merge advisory banner)

You see an amber banner in the Results panel like:

> *Drop Missing Columns merges 2 parallel branches: TransformationNode + MissingIndicator. 6 overlapping columns take values from MissingIndicator; unique columns from the others are kept as-is. For sequential application, chain them instead.*

### What it means

A node has **two or more incoming edges** that trace back to a common ancestor (sibling fan-in). The engine merges them with two rules:

1. **Union of columns** — any column unique to one branch is kept as-is.
2. **Last-wins on overlap** — for columns present in two or more branches, the **last input** in the node's input list overwrites the earlier ones. Earlier branches' modifications to those columns are silently dropped.

### Why it's usually a bug

If both branches modify the same columns (e.g. `TransformationNode` rescales `SepalLengthCm` and `MissingIndicator` outputs the original `SepalLengthCm` plus `*_missing` flags), only the last branch's values survive. You lose the work the other branch did.

### How to entirely fix it

**Chain the nodes instead of fanning them out.** Instead of:

```
Dataset ──> TransformationNode ──┐
        ──> MissingIndicator ────┴──> DropMissingColumns
```

wire:

```
Dataset ──> TransformationNode ──> MissingIndicator ──> DropMissingColumns
```

Now `MissingIndicator` reads `TransformationNode`'s output, adds `*_missing` flags on top, and passes the combined frame downstream. No fan-in, no warning, all changes preserved.

In the canvas:

1. Delete the `Dataset → MissingIndicator` edge.
2. Drag a new edge from `TransformationNode` → `MissingIndicator`.
3. Re-run preview — the banner will disappear.

### When fan-in is fine

- **Disjoint columns** — branches output non-overlapping columns. The banner says *"No column overlap — all columns from all branches are kept."* You can ignore it.
- **Redundant edge** (e.g. `Splitter → Scaler` plus a direct `Splitter → Encoder` while `Scaler → Encoder` also exists) — the engine detects this and **suppresses the warning entirely** because the descendant supersedes the ancestor under last-wins.

### What the engine guarantees

| Pattern | Behavior |
|---|---|
| Linear chain `A → B → C` | No warning, full data flow |
| Redundant ancestor edge | Suppressed silently |
| Disjoint-column fan-in | Advisory only, all columns kept |
| **Overlapping-column fan-in** | **Last input wins — chain instead** |
| Cycles | Engine rejects |

**Rule of thumb:** if two branches touch the same columns, chain them; if they touch different columns, fanning in is fine.

---

## Target column not found after encoding

If your target column is categorical and you apply `OneHotEncoder` with `drop_original=True`, the target column may be dropped or expanded.

**Fix:** Always run `TrainTestSplitter` (which separates `X` and `y`) **before** encoding. This ensures the target is safely stored in `y` and never touched by the encoder.

---

## Pickle loading errors

If `SkyulfPipeline.load()` fails, the saved model was pickled with a different library version.

**Fix:** Ensure compatible versions of:

- Python (same minor version, e.g., 3.10.x)
- scikit-learn (same minor version)
- pandas (same major version)

Pin versions in `requirements.txt` for reproducibility across environments.

---

## Polars version mismatch

Skyulf requires `polars >= 1.36`. Older versions may cause schema or type errors during data ingestion.

**Fix:**

```bash
pip install --upgrade polars
```

---

## Celery / Redis connection refused (full platform)

The backend's async task queue requires a running Redis instance.

**Fix:**

1. Ensure Redis is running: `redis-cli ping` should return `PONG`.
2. Check `CELERY_BROKER_URL` in your `.env` file (default: `redis://localhost:6379/0`).
3. If using Docker: `docker-compose up redis`.

---

## Feature scaling produces NaN

Standard scaling can produce NaN or Inf if a column has zero variance (all identical values).

**Fix:** Remove constant columns before scaling, or use `auto_detect: True` in the `StandardScaler` params - it automatically skips non-numeric and constant columns.