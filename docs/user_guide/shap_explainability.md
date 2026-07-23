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
from skyulf.modeling import compute_shap_explanation

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

    # Feature interactions (tree models only — None otherwise)
    if explanation["interactions"] is not None:
        names = explanation["interactions"]["feature_names"]
        matrix = explanation["interactions"]["matrix"]
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if j > i:
                    print(f"{a} x {b}: {matrix[i][j]:.4f}")
```

That's it — `explanation` is `None` if SHAP couldn't run (not installed,
empty data, unsupported model), otherwise a plain `dict` you can plug
straight into a chart or print.

## What's in `explanation`

| Key | Shape | Use it for |
|---|---|---|
| `feature_names` | `list[str]` | Column order for the other fields |
| `mean_abs_importance` | `{feature: float}` | Global bar chart of average feature impact |
| `samples` | `list[{base_value, feature_values, shap_values}]` | Per-row plots (beeswarm, dependence, waterfall, force) |
| `interactions` | `{feature_names, matrix} \| None` | Feature-pair interaction heatmap — `None` for non-tree models |

Each sample follows the SHAP identity: `base_value + sum(shap_values.values())`
≈ the model's output for that row (predicted class's probability for
classifiers; log-odds/margin for linear models — same convention as `shap`
itself).

`interactions["matrix"]` is a square, symmetric `mean(|SHAP interaction
value|)` matrix over the top-8 features by interaction strength (capped to
keep the payload small); `matrix[i][j]` lines up with `feature_names[i]` /
`feature_names[j]`, and the diagonal is each feature's own main effect.
Only available for tree-based models (Random Forest, Gradient Boosting,
XGBoost, etc.) via `shap.TreeExplainer` — linear/kernel models get `None`.

On the web platform this powers the **SHAP Explainability** tab on the
Experiments page:

- **Summary** — mean(|SHAP|) per feature, compared across runs.
- **Beeswarm** — every sampled row per feature, colored by its value, showing which direction it pushes predictions.
- **Dependence** — one feature's value vs. its SHAP contribution, revealing linear/non-linear relationships.
- **Waterfall** — one row's prediction broken down feature-by-feature.
- **Force Plot** — the same single-row breakdown as Waterfall, compressed into one compact push/pull bar (audit/compliance style).
- **Interaction** — a heatmap of the top feature-pairs' joint influence on predictions (tree models only).

## Plotting the graphs yourself

The web platform renders all six sub-views automatically. If you're using
`skyulf-core` standalone and want the same charts, `explanation` has
everything you need — no extra SHAP computation required. Quick
`matplotlib` recipes for each:

```python
import matplotlib.pyplot as plt
import numpy as np

# Summary — mean(|SHAP|) bar chart
items = sorted(explanation["mean_abs_importance"].items(), key=lambda kv: kv[1])
plt.barh([f for f, _ in items], [v for _, v in items])
plt.xlabel("mean(|SHAP value|)"); plt.show()

# Waterfall — one row, feature-by-feature staircase
row = explanation["samples"][0]
features = sorted(row["shap_values"], key=lambda f: -abs(row["shap_values"][f]))
running = row["base_value"]
for f in features:
    delta = row["shap_values"][f]
    plt.barh(f, delta, left=running if delta > 0 else running + delta,
             color="#ef4444" if delta >= 0 else "#3b82f6")
    running += delta
plt.axvline(row["base_value"], ls="--", color="gray", label="base")
plt.axvline(running, ls="--", color="indigo", label="prediction")
plt.legend(); plt.show()

# Force Plot — same row, compressed into one stacked bar
fig, ax = plt.subplots(figsize=(8, 1.5))
running = row["base_value"]
for f in features:  # order doesn't matter, all bars anchor at base_value
    delta = row["shap_values"][f]
    ax.barh(0, delta, left=running if delta >= 0 else running + delta,
            color="#ef4444" if delta >= 0 else "#3b82f6")
    running += delta
ax.axvline(row["base_value"], ls="--", color="gray")
ax.set_yticks([]); plt.show()

# Dependence — one feature's value vs. its SHAP contribution
feature = explanation["feature_names"][0]
xs = [s["feature_values"][feature] for s in explanation["samples"]]
ys = [s["shap_values"][feature] for s in explanation["samples"]]
plt.scatter(xs, ys); plt.xlabel(feature); plt.ylabel("SHAP value"); plt.show()

# Beeswarm — every feature's SHAP values, colored by that row's feature value
fig, ax = plt.subplots()
for i, feature in enumerate(explanation["feature_names"]):
    shap_vals = [s["shap_values"][feature] for s in explanation["samples"]]
    feat_vals = np.array([s["feature_values"][feature] for s in explanation["samples"]])
    span = feat_vals.max() - feat_vals.min() or 1
    color = (feat_vals - feat_vals.min()) / span
    jitter = i + (np.random.rand(len(shap_vals)) - 0.5) * 0.3
    ax.scatter(shap_vals, jitter, c=color, cmap="coolwarm", s=10)
ax.set_yticks(range(len(explanation["feature_names"])))
ax.set_yticklabels(explanation["feature_names"]); plt.show()

# Interaction — feature-pair heatmap (tree models only; None otherwise)
interactions = explanation["interactions"]
if interactions is not None:
    names, matrix = interactions["feature_names"], np.array(interactions["matrix"])
    plt.imshow(matrix, cmap="Purples")
    plt.xticks(range(len(names)), names, rotation=45, ha="right")
    plt.yticks(range(len(names)), names)
    plt.colorbar(label="mean(|interaction value|)"); plt.show()
```

Want SHAP's own native plots (`shap.plots.waterfall`, `shap.plots.beeswarm`,
`shap.plots.force`, `shap.plots.heatmap`, ...) instead? Those need a live
`shap.Explanation` object, which `compute_shap_explanation` doesn't return
(it only returns a plain, JSON-serializable `dict`). Call `shap.Explainer`
yourself and keep the result if you want that:

```python
import shap

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_train)      # shap.Explanation — full native API
shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)

# Interaction values (tree models only) for shap.plots.heatmap-style use
interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_train)
```

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
