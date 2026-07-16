# SHAP Explainability

Skyulf can compute [SHAP](https://shap.readthedocs.io/) values for a trained
model, explaining *why* it predicts what it predicts — both globally (which
features matter most overall) and for individual rows.

It's optional and best-effort: if `shap` isn't installed, or the
model/data shape isn't supported, the platform just omits the explanation
instead of failing the training job.

## Quick start

```bash
pip install skyulf-core[explainability]
```

```python
from sklearn.ensemble import RandomForestClassifier
from skyulf.modeling._explainability import compute_shap_explanation

model = RandomForestClassifier(random_state=42).fit(X_train, y_train)
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

That's it — `explanation` is `None` if SHAP couldn't run (not installed,
empty data, unsupported model), otherwise a plain `dict` you can plug
straight into a chart or print.

## What's in `explanation`

| Key | Shape | Use it for |
|---|---|---|
| `feature_names` | `list[str]` | Column order for the other fields |
| `mean_abs_importance` | `{feature: float}` | Global bar chart of average feature impact |
| `samples` | `list[{base_value, feature_values, shap_values}]` | Per-row plots (beeswarm, dependence, waterfall) |

Each sample follows the SHAP identity: `base_value + sum(shap_values.values())`
≈ the model's output for that row (predicted class's probability for
classifiers; log-odds/margin for linear models — same convention as `shap`
itself).

On the web platform this powers the **SHAP Explainability** tab on the
Experiments page:

- **Summary** — mean(|SHAP|) per feature, compared across runs.
- **Beeswarm** — every sampled row per feature, colored by its value, showing which direction it pushes predictions.
- **Dependence** — one feature's value vs. its SHAP contribution, revealing linear/non-linear relationships.
- **Waterfall** — one row's prediction broken down feature-by-feature.

## Reference

```python
def compute_shap_explanation(
    model: Any,             # any fitted scikit-learn-compatible estimator
    X: pd.DataFrame,        # features to compute SHAP values for
    max_samples: int = 200,        # rows used for mean_abs_importance
    max_display_samples: int = 50, # rows kept in `samples` for per-row plots
) -> dict[str, Any] | None: ...
```

- Multi-class: each row uses **its own predicted class**, so waterfall/dependence explain "why did the model predict what it predicted for this row." Binary classification uses the positive class.
- The full platform (`pyproject.toml` at the repo root) already pins `shap`, so no extra install is needed when running Skyulf via `run_skyulf.py` — the `pip install` above is only for using `skyulf-core` standalone.
