# Drift Monitoring

Data drift occurs when the statistical properties of incoming (production) data diverge from the data the model was trained on. Skyulf provides a `DriftCalculator` to detect this automatically.

## When to use drift detection

- Before running predictions on new data batches.
- As a scheduled health check in production pipelines.
- After a data source change (new CSV, new API feed, schema migration).

## Using the Data Drift Analysis page

Upload raw data in the same format as the selected model's source dataset.
Do not manually apply the pipeline's log transforms, scaling or encoding first.
The page compares it with the raw loader snapshot saved during that job, before
preprocessing and train/test splitting. It reads the saved snapshot rather than
reloading a source file that may have changed since training.

The reference contains the rows actually loaded: a sampled loader keeps its
sample, while an unsampled loader keeps the complete loaded dataset. Consequently,
a model trained on 120 of 150 rows can show **Ref: 150 rows** in drift analysis;
the model card's 120 rows still describes its training partition.

The target and columns explicitly named by upstream drop-column configurations
are excluded on both sides. All other raw columns remain eligible, including
categorical inputs that the model later encodes. Missing and new input columns
still produce schema drift. Explicit drops follow the model bundle's exclusion
convention; this is not a complete dependency analysis of generated features.

Existing jobs with a saved execution graph and loader snapshot work without
retraining. If that graph has multiple upstream source datasets or its saved
source is unavailable, the analysis records a failed check rather than comparing
the upload with a transformed reference. Legacy jobs without graph metadata
retain their original saved reference; ensure that reference and upload use the
same preprocessing stage.

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

String, Categorical, Enum and Boolean columns receive categorical drift metrics,
including `psi_categorical`. They do not need encoding first. Rerun saved drift
checks that previously omitted Enum columns to populate those measurements.

The page's **Avg PSI** and **Most Drifted** cards include both numeric `psi`
and categorical `psi_categorical`, with one finite PSI measurement per column.
For example, numeric PSI `0.01` and categorical PSI `5` produce average
`2.5050`, with the categorical feature shown as most drifted. A measured zero
is included; unavailable measurements are excluded. If none are available,
the cards show `—` and **No PSI available** rather than claiming stability.

The `DriftCalculator` computes these metrics for each numeric column:

| Metric | What it measures | Default threshold |
|---|---|---|
| **Wasserstein distance** | How much "work" to transform one distribution into the other, divided by the reference column's standard deviation | 0.1 |
| **KS statistic** (Kolmogorov-Smirnov) | Maximum distance between the two CDFs (0–1) | 0.1 |
| **PSI** (Population Stability Index) | Binned distribution shift | 0.2 |
| **KL divergence** | Information-theoretic divergence | 0.1 |

A column is flagged as "drifted" if **any** metric exceeds its threshold.

Each metric's `value` is reported on the same scale as its `threshold`, so
`value > threshold` reproduces `has_drift`. Two caveats:

- The Wasserstein `value` is the **normalized** distance — the raw distance is
  in the column's own units, so a single threshold could not mean the same
  thing for a column measured in cents and one measured in kilometres. The
  untransformed distance is kept in `raw_value` for display.
- The KS **p-value** is reported alongside the statistic as `ks_test_p_value`
  but never decides drift: it shrinks with sample size, so an identical tiny
  shift looks significant at n=100k and not at n=100. It carries the KS
  statistic's threshold rather than one of its own.

## Custom thresholds

```python
report = calc.calculate_drift(thresholds={
    "psi": 0.15,
    "ks_statistic": 0.01,
    "wasserstein": 0.05,
    "kl_divergence": 0.2,
})
```

The keys are `psi`, `ks_statistic`, `wasserstein` and `kl_divergence`. Anything
else is merged into the dict but never read, so a misspelled key silently leaves
that threshold at its default.

## Schema drift

The report also detects structural changes:

- `report.missing_columns` — columns present in reference but absent in current data.
- `report.new_columns` — columns in current data that were not in the reference.

Both count toward `report.drifted_columns_count`, which therefore covers
distribution drift *and* schema drift. A vanished feature has no values left to
compare, so it never appears in `column_drifts` — the count is the only place it
shows up, and it used to be omitted entirely, leaving a report that claimed zero
drift while the structural change was classified as critical.

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
- Keep categorical labels intact when comparing their distributions.
