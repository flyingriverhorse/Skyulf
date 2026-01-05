import polars as pl
import numpy as np
from skyulf.profiling.drift import DriftCalculator

def generate_synthetic_data():
    """
    Generates two datasets:
    1. Reference: Normal distribution
    2. Current: Shifted distribution (Drifted)
    """
    np.random.seed(42)
    n_rows = 1000

    # Reference Data (Training)
    ref_data = {
        "age": np.random.normal(30, 5, n_rows),           # Mean 30
        "income": np.random.normal(50000, 10000, n_rows), # Mean 50k
        "score": np.random.beta(2, 5, n_rows),            # Beta dist
        "category": np.random.choice(["A", "B", "C"], n_rows)
    }

    # Current Data (Production) - WITH DRIFT
    curr_data = {
        "age": np.random.normal(35, 5, n_rows),           # DRIFT: Mean shifted to 35
        "income": np.random.normal(50000, 10000, n_rows), # STABLE: Same mean
        "score": np.random.beta(5, 2, n_rows),            # DRIFT: Shape changed
        "category": np.random.choice(["A", "B", "C"], n_rows),
        "new_feature": np.random.random(n_rows)           # SCHEMA DRIFT: New column
    }

    return pl.DataFrame(ref_data), pl.DataFrame(curr_data)

def main():
    print("--- Generating Synthetic Data ---")
    ref_df, curr_df = generate_synthetic_data()
    print(f"Reference Rows: {len(ref_df)}")
    print(f"Current Rows:   {len(curr_df)}")

    print("\n--- Running Drift Analysis ---")
    # Initialize Calculator
    calculator = DriftCalculator(ref_df, curr_df)

    # Calculate Drift
    # We can override thresholds if needed
    report = calculator.calculate_drift(thresholds={
        "psi": 0.1,         # Stricter PSI threshold
        "wasserstein": 0.1
    })

    print(f"\n📊 Drift Report Summary")
    print(f"Drifted Columns: {report.drifted_columns_count}")
    
    # 1. Check Schema Drift
    if report.missing_columns:
        print(f"\n⚠️  Missing Columns (Present in Ref, Missing in Curr):")
        print(f"   {report.missing_columns}")
    
    if report.new_columns:
        print(f"\nℹ️  New Columns (Present in Curr, Missing in Ref):")
        print(f"   {report.new_columns}")

    # 2. Check Statistical Drift
    print("\n🔍 Detailed Column Analysis:")
    for col_name, drift_info in report.column_drifts.items():
        status = "🔴 DRIFT" if drift_info.drift_detected else "🟢 STABLE"
        print(f"\nColumn: '{col_name}' [{status}]")
        
        # Print Metrics
        for metric in drift_info.metrics:
            # Format: Metric Name: Value (Threshold)
            mark = "FAIL" if metric.has_drift else "PASS"
            print(f"  - {metric.metric:<20}: {metric.value:.4f} (Thresh: {metric.threshold}) [{mark}]")
        
        # Print Suggestions
        if drift_info.suggestions:
            print(f"  💡 Suggestion: {drift_info.suggestions[0]}")

if __name__ == "__main__":
    main()
