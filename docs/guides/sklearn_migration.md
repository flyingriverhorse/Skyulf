# Migrating from scikit-learn `Pipeline`

This guide is for developers already comfortable with scikit-learn's
`Pipeline` / `ColumnTransformer` who want to bring that experience to
Skyulf. Each section shows the sklearn code next to the equivalent Skyulf
config, followed by what changes conceptually.

## The core mental-model shift

| scikit-learn | Skyulf |
|---|---|
| Pipeline = Python objects composed in code (`Pipeline([...])`) | Pipeline = a JSON-serializable `dict` (`{"preprocessing": [...], "modeling": {...}}`) |
| `ColumnTransformer` for per-column branching | Each step's `params.columns` list scopes it to specific columns — no separate transformer needed |
| One engine: pandas/NumPy only | Dual-engine: the same config runs on pandas **or** Polars (see [Engine Mechanics](engine_mechanics.md)) |
| `fit`/`transform` return arrays or DataFrames | `fit`/`predict` return dicts of metrics / predictions; the fitted pipeline itself is inspectable (`describe()`, `to_mermaid()`, `fingerprint()`) |
| Persisted via `joblib.dump` | Persisted via `pipeline.save(path)` / `SkyulfPipeline.load(path)` (pickle under the hood) |

Because the config is plain JSON, a Skyulf pipeline can be built, stored,
diffed, and re-run without executing any Python — a canvas UI, a database
row, or a REST payload can all produce the same `config` dict that
`SkyulfPipeline(config)` consumes.

## Side-by-side: impute → encode → scale → train

**scikit-learn**

```python
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_features = ["age", "income"]
categorical_features = ["city"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
        ]), numeric_features),
        ("cat", OneHotEncoder(), categorical_features),
    ]
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=50, max_depth=5)),
])

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=["is_customer"]), data["is_customer"], test_size=0.2
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Skyulf**

```python
from skyulf import SkyulfPipeline

config = {
    "preprocessing": [
        {
            "name": "split_data",
            "transformer": "TrainTestSplitter",
            "params": {"test_size": 0.2, "target_column": "is_customer"},
        },
        {
            "name": "impute_income",
            "transformer": "SimpleImputer",
            "params": {"columns": ["income"], "strategy": "mean"},
        },
        {
            "name": "encode_city",
            "transformer": "OneHotEncoder",
            "params": {"columns": ["city"]},
        },
        {
            "name": "scale_features",
            "transformer": "StandardScaler",
            "params": {"columns": ["age", "income"]},
        },
    ],
    "modeling": {
        "type": "random_forest_classifier",
        "params": {"n_estimators": 50, "max_depth": 5},
    },
}

pipeline = SkyulfPipeline(config)
metrics = pipeline.fit(data, target_column="is_customer")   # split + fit + evaluate
predictions = pipeline.predict(new_data)                    # transform + predict
```

Key differences to notice:

- **The split is a pipeline step, not a separate call.** `TrainTestSplitter`
  runs first in `preprocessing`, so `fit()` handles splitting, transforming,
  training, *and* evaluating test-set metrics in one call — no manual
  `train_test_split` + separate `.score()`.
- **No `ColumnTransformer` nesting.** Every step already scopes itself to
  `params.columns`; there's no need for a wrapper object to route columns to
  sub-pipelines.
- **`metrics` comes back structured**, not just an accuracy float — see
  `docs/examples/quickstart.md` for the full shape (`preprocessing` +
  `modeling` sections, per-metric breakdown).

## Inspecting a fitted pipeline

scikit-learn pipelines are inspected by walking `.named_steps` / `.steps` in
Python. Skyulf gives you the same information without touching internals:

```python
print(pipeline.describe())      # human-readable step-by-step summary
print(pipeline.to_mermaid())    # Mermaid flowchart string (paste into docs/PRs)
print(pipeline.fingerprint())   # deterministic SHA-256 over topology + fitted artifacts
print(pipeline.export_model_card())  # structured dict: lineage, params, metrics, fingerprint
```

`fingerprint()` in particular has no direct sklearn equivalent — it's a
single hash that changes if *either* the config *or* any fitted artifact
changes, so two pipelines with the same fingerprint are guaranteed to
produce the same predictions. Use it to prove which exact pipeline produced
a given prediction in an audit log.

## Column selection: `ColumnTransformer` remainder handling

`ColumnTransformer(..., remainder="passthrough")` is the sklearn way to say
"leave every other column untouched." In Skyulf, any column not named in a
step's `params.columns` is left untouched by that step automatically —
there's no `remainder` argument because steps never touch unlisted columns
in the first place.

## Custom transformers

**scikit-learn** — subclass `BaseEstimator`/`TransformerMixin`:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.medians_ = X.median()
        return self

    def transform(self, X):
        return X.fillna(self.medians_)
```

**Skyulf** — a `Calculator` (fit) / `Applier` (transform) pair registered
under a name your config can reference. See
[Write your own node](../user_guide/extending_custom_nodes.md) for the full
walkthrough (artifact shape, registration, tests). The split mirrors
sklearn's `fit`/`transform` split, but the fitted artifact is an explicit,
serializable object (usually a `TypedDict`) rather than instance attributes
set with a trailing underscore.

## What you get that plain sklearn doesn't have

- **Dual-engine execution** — the exact same config runs against a Polars
  `LazyFrame`/`DataFrame` with no code changes (see
  [Engine Mechanics](engine_mechanics.md)).
- **Schema inference before running** — `Calculator.infer_output_schema()`
  predicts output columns/dtypes from config alone, useful for canvas UIs
  that need to validate a pipeline before any data flows through it.
- **Drift & data-quality primitives** — `skyulf.profiling.drift` (PSI/KS/
  Wasserstein) and `skyulf.profiling.expect` (`expect_no_nulls`,
  `expect_value_range`, ...) ship in the same package, no extra
  Great-Expectations-style dependency.
- **Reproducibility built in** — `fingerprint()` and `export_model_card()`
  (see above) rather than hand-rolled joblib hash tracking.

## What's not (yet) a drop-in replacement

- No `Pipeline.set_params(**kwargs)`-style grid-search integration — use the
  built-in `hyperparameter_tuner` modeling type instead (random / grid /
  Optuna strategies under `modeling._tuning`).
- Custom sklearn transformers aren't auto-wrapped; you write a thin
  Calculator/Applier pair (see above) — usually a few lines more than a bare
  `TransformerMixin` subclass, in exchange for dual-engine support and JSON
  serializability.
