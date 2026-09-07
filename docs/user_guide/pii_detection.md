# PII Detection in Dataset Profiling

Skyulf's profiler can flag columns that may contain email addresses or phone
numbers. This is a lightweight, advisory heuristic that helps you review a
dataset before modeling; it is not a complete sensitive-data scanner or a
compliance control.

## Basic example

```python
import polars as pl

from skyulf.profiling.analyzer import EDAAnalyzer

df = pl.DataFrame(
    {
        "name": ["Alice", "Bob", "Carol"],
        "contact": [
            "alice@example.com",
            "+1 (555) 123-4567",
            "bob@example.com",
        ],
        "customer_id": ["10000001", "10000002", "10000003"],
    }
)

profile = EDAAnalyzer(df).analyze()

print("has_pii =", profile.has_pii)
print("pii_columns =", profile.pii_columns)

for alert in profile.pii_alerts:
    print(alert.message)
```

The output is:

```text
has_pii = True
pii_columns = ['contact', 'customer_id']
Column 'contact' may contain PII (Email/Phone).
```

The `contact` result is expected. The `customer_id` values are deliberately
included to show that ordinary numeric identifiers are not flagged as phone
PII after the OC-148 fix.

## Direct profile accessors

The profile exposes PII findings without requiring callers to filter the
generic alert list:

| Accessor | Meaning |
|---|---|
| `profile.has_pii` | `True` when at least one possible PII alert exists |
| `profile.pii_columns` | Flagged column names in alert order |
| `profile.pii_alerts` | The matching `Alert` objects |

The generic `profile.alerts` list remains available for all profiling findings,
including missing values, leakage, outliers, and PII.

## What the detector does and does not do

The detector currently checks Text and Categorical columns for values that
look like email addresses or phone numbers. It samples values from the
column, and a matching value produces a `PII` alert.

It does not currently:

- mask, delete, tokenize, or block the data;
- prevent the data from being used in training or exports;
- identify every kind of personal or regulated data, such as names, addresses,
  national identifiers, payment-card numbers, or health information;
- provide a legal or regulatory classification.

Treat the result as a review signal. Do not display raw values in a user-facing
PII review screen; show the column name, alert category, severity, and
explanation instead.

## Handling false positives

The detector now rejects plain 7-or-more-digit identifiers, ZIP+4 values, and
a single phone-shaped value in an otherwise identifier-like sample. Unusual
formatted values can still produce false positives because this remains a
heuristic; review the alert before taking action.

For comparison with other profiling tools, see
[Skyulf vs. YData vs. Sweetviz](../guides/profiling_comparison.md).
