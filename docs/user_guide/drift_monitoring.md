# Drift Monitoring

Data drift occurs when the statistical properties of incoming (production) data diverge from the data the model was trained on. Skyulf provides a `DriftCalculator` to detect this automatically.

## When to use drift detection

- Before running predictions on new data batches.
- As a scheduled health check in production pipelines.
- After a data source change (new CSV, new API feed, schema migration).

## Quick example

```python
import polars as pl
from skyulf.profiling import DriftCalculator

# Reference = your training data
reference = pl.DataFrame({
    "age": [25, 30, 35, 40, 45, 50],
    "income": [30000, 45000, 55000, 65000, 70000, 80000],
})

# Current = new production data
current = pl.DataFrame({
    "age": [60, 65, 70, 75, 80, 85],
    "income": [90000, 95000, 100000, 110000, 120000, 130000],
})

calc = DriftCalculator(reference, current)
report = calc.calculate_drift()

print(f"Drifted columns: {report.drifted_columns_count}")
for col, drift in report.column_drifts.items():
    print(f"  {col}: drift={drift.drift_detected}")
    for m in drift.metrics:
        print(f"    {m.metric}: {m.value:.4f} (threshold={m.threshold}, drifted={m.has_drift})")
```

## Drift metrics

The `DriftCalculator` computes these metrics for each numeric column:

| Metric | What it measures | Default threshold |
|---|---|---|
| **Wasserstein distance** | How much "work" to transform one distribution into the other | 0.1 (normalized) |
| **KS test** (Kolmogorov-Smirnov) | Maximum distance between CDFs; returns a p-value | p < 0.05 |
| **PSI** (Population Stability Index) | Binned distribution shift | 0.2 |
| **KL divergence** | Information-theoretic divergence | 0.1 |

A column is flagged as "drifted" if **any** metric exceeds its threshold.

## Custom thresholds

```python
report = calc.calculate_drift(thresholds={
    "psi": 0.15,
    "ks": 0.01,
    "wasserstein": 0.05,
    "kl_divergence": 0.2,
})
```

## Schema drift

The report also detects structural changes:

- `report.missing_columns` — columns present in reference but absent in current data.
- `report.new_columns` — columns in current data that were not in the reference.

## Report structure

```python
DriftReport(
    reference_rows=1000,
    current_rows=500,
    drifted_columns_count=2,
    column_drifts={
        "age": ColumnDrift(
            column="age",
            metrics=[...],
            drift_detected=True,
            suggestions=["Consider retraining..."],
        ),
    },
    missing_columns=[],
    new_columns=["new_feature"],
)
```

## Data format

`DriftCalculator` works with **Polars DataFrames**. If your data is in Pandas:

```python
import polars as pl

reference_pl = pl.from_pandas(reference_pd)
current_pl = pl.from_pandas(current_pd)
```

## Dependencies

Drift calculation requires `scipy` (installed with `skyulf-core` by default).

## Tips

- Run drift detection **before** prediction to catch issues early.
- Log drift reports over time to track gradual distribution shifts.
- If drift is detected, consider retraining the model on recent data.
- Non-numeric columns are currently skipped — encode them first if you need categorical drift detection.
