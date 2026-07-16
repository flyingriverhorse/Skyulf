# SHAP Explainability

Skyulf can compute [SHAP](https://shap.readthedocs.io/) (SHapley Additive
exPlanations) values for a trained model, explaining *why* it predicts what
it predicts — both globally (which features matter most overall) and for
individual rows (why this particular prediction).

This is optional and best-effort: if the `shap` package isn't installed, or
the model/data shape isn't supported, the platform simply omits the
explanation rather than failing the training job.

## What you get

`compute_shap_explanation` returns one dict with everything needed to drive
several different visualizations from a single computation:

| Key | Shape | Used for |
|---|---|---|
| `feature_names` | `list[str]` | Column order for the other fields |
| `mean_abs_importance` | `{feature: float}` | Global bar chart — average magnitude of each feature's impact, comparable across runs |
| `samples` | `list[{base_value, feature_values, shap_values}]` | Per-row plots — beeswarm, dependence, and waterfall |

Each entry in `samples` satisfies the SHAP identity:

```
base_value + sum(shap_values.values()) ≈ model's output for that row
```

(for classifiers this reconstructs the predicted class's probability; for
linear models it's in log-odds/margin space, matching `shap`'s own
convention.)

In the web platform, this powers the **SHAP Explainability** tab on the
Experiments page, with four sub-views:

- **Summary** — mean(|SHAP|) bar chart per feature, compared across selected runs (like Feature Importance).
- **Beeswarm** — every sampled row plotted per feature, colored by that row's own feature value (low → high), showing whether high/low values push predictions up or down.
- **Dependence** — a single feature's value vs. its SHAP contribution across rows, revealing linear/non-linear relationships.
- **Waterfall** — one row's prediction broken down feature-by-feature from the base value to the final output.

## Using it from skyulf-core directly

`compute_shap_explanation` lives in the private `_explainability` submodule
(same pattern as `_evaluation`/`_tuning`) — import it directly rather than
through `skyulf.modeling`:

```python
from sklearn.ensemble import RandomForestClassifier
from skyulf.modeling._explainability import compute_shap_explanation

model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)

explanation = compute_shap_explanation(model, X_train)
if explanation is None:
    print("SHAP unavailable (not installed, or unsupported model/data)")
else:
    # Global importance, sorted
    for feature, importance in sorted(
        explanation["mean_abs_importance"].items(), key=lambda kv: -kv[1]
    ):
        print(f"{feature}: {importance:.4f}")

    # Explain a single row (e.g. the first sampled row)
    row = explanation["samples"][0]
    predicted_output = row["base_value"] + sum(row["shap_values"].values())
    print(f"base={row['base_value']:.3f} -> predicted={predicted_output:.3f}")
    for feature, contribution in row["shap_values"].items():
        print(f"  {feature}: {contribution:+.4f}")
```

### Signature

```python
def compute_shap_explanation(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 200,
    max_display_samples: int = 50,
) -> dict[str, Any] | None: ...
```

| Param | Description |
|---|---|
| `model` | Any fitted scikit-learn-compatible estimator. `shap.Explainer` auto-selects Tree/Linear/Kernel algorithms based on the model type. |
| `X` | Feature DataFrame used to compute SHAP values (typically the training or a held-out split). |
| `max_samples` | Rows used to compute a stable `mean_abs_importance` (default 200). Larger datasets are randomly subsampled with a fixed seed. |
| `max_display_samples` | Rows kept in the returned `samples` list for per-row plots (default 50) — independent of `max_samples`, keeping the payload small regardless of computation size. |

Returns `None` (never raises) if `shap` isn't installed, `X` is empty, or
computation fails for any reason — treat a `None` return as "explainability
unavailable," not an error.

### Notes on multi-class models

- For binary classification, SHAP values/base values for the positive class
  (index 1) are used, matching the scikit-learn/SHAP convention.
- For 3+ class models, each row uses **its own predicted class** (via
  `model.predict()`), so the waterfall/dependence values answer "why did the
  model predict what it predicted for this row" rather than an arbitrary
  fixed class.

## Installing the `shap` dependency

`shap` is an optional extra:

```bash
pip install skyulf-core[explainability]
```

The full platform (`pyproject.toml` at the repo root) already pins
`shap>=0.46.0,<1.0.0` as a direct dependency, so no extra install step is
needed when running Skyulf via `run_skyulf.py`.
