import polars as pl
import numpy as np
from skyulf.profiling.analyzer import EDAAnalyzer
import sys

def test_eda_features():
    print("Generating synthetic data...")
    # Create synthetic data with structure
    n_rows = 1000
    np.random.seed(42)
    
    # 1. Numeric features with correlation
    x = np.random.normal(0, 1, n_rows)
    y = 2 * x + np.random.normal(0, 0.5, n_rows) # Correlated with x
    z = np.random.normal(5, 2, n_rows) # Independent
    
    # 2. Categorical feature
    categories = ['A', 'B', 'C']
    cat = np.random.choice(categories, n_rows)
    
    # 3. Outliers
    x[0] = 100 # Extreme outlier
    y[0] = 100
    
    # 4. Target variable (Binary classification)
    # Target depends on x and y
    target_prob = 1 / (1 + np.exp(-(x + y)))
    target = (np.random.rand(n_rows) < target_prob).astype(int)
    
    df = pl.DataFrame({
        "feature_x": x,
        "feature_y": y,
        "feature_z": z,
        "category": cat,
        "target": target
    })
    
    print("Initializing Analyzer...")
    analyzer = EDAAnalyzer(df)
    
    print("Running Analysis...")
    profile = analyzer.analyze(target_col="target")
    
    print("\n--- Verification Results ---")
    
    # 1. Basic Stats
    print(f"Row Count: {profile.row_count}")
    assert profile.row_count == 1000
    
    # 2. PCA
    if profile.pca_data:
        print(f"PCA: Generated {len(profile.pca_data)} points")
        # Explained variance is not in the list of points, it might be lost or I need to check where it is stored.
        # Looking at schemas, DatasetProfile doesn't seem to store explained variance separately if pca_data is just a list.
        # Let's check analyzer.py _calculate_pca to see what it returns.
    else:
        print("PCA: Not generated (Check sklearn dependency)")

    # 3. Outliers
    if profile.outliers:
        print(f"Outliers: Detected {profile.outliers.total_outliers} outliers")
        if profile.outliers.top_outliers:
            print(f"Top Anomaly Score: {profile.outliers.top_outliers[0].score}")
            # Check explanation
            if profile.outliers.top_outliers[0].explanation:
                 print(f"Outlier Explanation: {profile.outliers.top_outliers[0].explanation}")
    else:
        print("Outliers: Not generated")

    # 4. Causal Graph
    if profile.causal_graph:
        print(f"Causal Graph: {len(profile.causal_graph.nodes)} nodes, {len(profile.causal_graph.edges)} edges")
        for edge in profile.causal_graph.edges:
            print(f"  Edge: {edge.source} -> {edge.target} ({edge.type})")
    else:
        print("Causal Graph: Not generated (Check causal-learn dependency)")
        
    # 5. Alerts
    print(f"Alerts: {len(profile.alerts)}")
    for alert in profile.alerts:
        print(f"  - [{alert.severity}] {alert.message}")

if __name__ == "__main__":
    try:
        test_eda_features()
        print("\nTest Completed Successfully!")
    except Exception as e:
        print(f"\nTest Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
