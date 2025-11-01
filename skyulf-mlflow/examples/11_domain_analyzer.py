"""Example: Dataset domain inference with Skyulf-MLFlow."""

from __future__ import annotations

import pandas as pd

from skyulf_mlflow_library.eda import DomainAnalyzer, infer_domain


def build_marketing_dataset() -> pd.DataFrame:
    """Create a synthetic marketing/retail dataset with mixed feature types."""

    return pd.DataFrame(
        {
            "campaign_id": ["CMP-100", "CMP-101", "CMP-102", "CMP-103", "CMP-104"],
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "city": ["New York", "Los Angeles", "Chicago", "New York", "Seattle"],
            "channel": ["Email", "Paid Search", "Social", "Email", "Display"],
            "impressions": [12000, 18500, 9000, 14200, 7400],
            "clicks": [820, 1025, 610, 980, 455],
            "conversions": [120, 160, 90, 130, 60],
            "spend": [4200.50, 6800.00, 3100.75, 5600.20, 2700.40],
            "revenue": [9800.00, 12400.00, 5400.00, 8800.00, 4300.00],
            "campaign_start": pd.date_range("2024-09-01", periods=5, freq="7D"),
        }
    )


def main() -> None:
    data = build_marketing_dataset()
    analyzer = DomainAnalyzer(enable_ml_classifier=False)
    result = analyzer.analyze_dataset_domain(data.columns, data)

    print("Primary domain:", result.primary_domain)
    print("Confidence:", f"{result.primary_confidence:.2f}")
    print("Secondary domains:")
    for domain in result.secondary_domains:
        print(f"  - {domain['domain']} (confidence={domain['confidence']:.2f})")

    print("\nDetected data patterns:")
    for key, value in result.patterns.get("data_patterns", {}).items():
        print(f"  {key}: {value}")

    print("\nTop recommendations:")
    for item in result.recommendations[:4]:
        print(f"  - {item}")

    # Demonstrate helper usage for quick checks
    quick = infer_domain(data)
    print("\nHelper primary domain:", quick.primary_domain)


if __name__ == "__main__":
    main()
