"""Generate ``tests/data/pipeline_dataset.csv`` — a larger synthetic sample
dataset for pipeline/modeling integration tests.

Unlike ``customers.csv`` (15 rows, used for per-node unit tests), this dataset
has enough rows for a meaningful train/test split and model fitting. It is
generated once with a fixed seed and the output is checked into git — this
script is kept for reproducibility/documentation, but is NOT run as part of
the test suite itself (the CSV is the source of truth, not this script).

Run manually to regenerate: `.\\.venv\\Scripts\\python.exe tests/data/generate_pipeline_dataset.py`
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260705
N_ROWS = 300


def _build_dataset() -> pd.DataFrame:
    """Build the synthetic pipeline dataset with realistic quirks.

    Returns:
        A DataFrame with numeric, categorical, date, and geo columns, a
        binary classification target (``churned``), a continuous regression
        target (``monthly_spend``), and deliberate missing values in several
        feature columns (not in the targets).
    """
    rng = np.random.default_rng(SEED)

    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]
    city_coords = {
        "New York": (40.7128, -74.0060),
        "Los Angeles": (34.0522, -118.2437),
        "Chicago": (41.8781, -87.6298),
        "Houston": (29.7604, -95.3698),
        "Phoenix": (33.4484, -112.0740),
    }
    plan_types = ["basic", "premium", "enterprise"]

    city = rng.choice(cities, size=N_ROWS)
    plan_type = rng.choice(plan_types, size=N_ROWS, p=[0.5, 0.35, 0.15])
    age = rng.normal(40, 12, size=N_ROWS).clip(18, 85).round(0)
    tenure_months = rng.integers(1, 73, size=N_ROWS)

    plan_multiplier = (
        pd.Series(plan_type).map({"basic": 1.0, "premium": 1.8, "enterprise": 3.0}).to_numpy()
    )
    income = (30000 + age * 400 + rng.normal(0, 8000, size=N_ROWS)).clip(15000, 200000).round(0)
    monthly_spend = (
        (20 * plan_multiplier + tenure_months * 0.3 + rng.normal(0, 5, size=N_ROWS))
        .clip(5, None)
        .round(2)
    )

    # Churn probability rises for low-tenure, low-spend, basic-plan customers.
    # Intercept tuned to yield a moderately imbalanced but workable ~25-30%
    # churn rate (not the ~4% an untuned logit would produce) so a train/test
    # split reliably has positive examples in both halves.
    churn_logit = (
        0.4
        - 0.03 * tenure_months
        - 0.01 * monthly_spend
        + (plan_type == "basic") * 1.2
        + rng.normal(0, 0.5, size=N_ROWS)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = (rng.uniform(size=N_ROWS) < churn_prob).astype(int)

    lat = np.array([city_coords[c][0] for c in city]) + rng.normal(0, 0.05, size=N_ROWS)
    lon = np.array([city_coords[c][1] for c in city]) + rng.normal(0, 0.05, size=N_ROWS)

    signup_date = pd.Timestamp("2023-01-01") - pd.to_timedelta(
        rng.integers(0, 900, size=N_ROWS), unit="D"
    )

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, N_ROWS + 1),
            "age": age,
            "income": income,
            "tenure_months": tenure_months,
            "city": city,
            "signup_date": signup_date.strftime("%Y-%m-%d"),
            "lat": lat.round(4),
            "lon": lon.round(4),
            "plan_type": plan_type,
            "monthly_spend": monthly_spend,
            "churned": churned,
        }
    )

    # Deliberate missing values in feature columns only (never in the two
    # targets, `churned`/`monthly_spend`, so downstream fit/evaluate tests
    # don't need to special-case missing labels).
    missing_frac = 0.06
    for col in ("age", "income", "city", "lat", "lon"):
        mask = rng.uniform(size=N_ROWS) < missing_frac
        df.loc[mask, col] = np.nan
    # `lon` is missing wherever `lat` is missing, and vice versa, to keep the
    # lat/lon pair jointly missing (realistic for geo nodes reading a pair).
    joint_geo_missing = df["lat"].isna() | df["lon"].isna()
    df.loc[joint_geo_missing, ["lat", "lon"]] = np.nan

    return df


if __name__ == "__main__":
    output_path = Path(__file__).resolve().parent / "pipeline_dataset.csv"
    _build_dataset().to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
