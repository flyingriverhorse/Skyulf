"""Example: Lightweight EDA utilities available in Skyulf-MLFlow."""

from __future__ import annotations

import pandas as pd

from skyulf_mlflow_library.eda import (
    DomainAnalyzer,
    generate_quality_report,
    get_text_insights,
)


def build_sample_data() -> pd.DataFrame:
    """Return a small marketing-style dataset for demonstration."""

    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "city": ["New York", "Los Angeles", "Chicago", "Boston", "Seattle"],
            "signup_date": pd.date_range("2024-01-01", periods=5, freq="7D"),
            "monthly_spend": [120.5, 95.0, 87.3, 150.2, 102.9],
            "lifetime_value": [860.0, 540.5, 390.0, 1010.0, 620.0],
            "feedback": [
                "Loving the loyalty rewards",
                "App feels slow during checkout",
                "Great support but limited catalog",
                "Promotions keep me engaged",
                "Wish there were more payment options",
            ],
        }
    )


def main() -> None:
    df = build_sample_data()

    print("=== Quality Report Summary ===")
    quality = generate_quality_report(df)
    print(quality.summary)

    print("\n=== Text Insights (feedback column) ===")
    text_report = get_text_insights(df[["feedback"]])
    for column, details in text_report.items():
        print(f"Column: {column}")
        print(f"  Unique values: {details['unique_values']}")
        print(f"  Avg length: {details['avg_length']:.1f}")
        print(f"  Suggested use cases: {', '.join(details['suggested_use_cases'])}")

    print("\n=== Domain Analyzer ===")
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(df.columns, df)
    print("Primary domain:", result.primary_domain)
    print("Recommendations:")
    for item in result.recommendations[:3]:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
