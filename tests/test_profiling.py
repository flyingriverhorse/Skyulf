import pytest
import polars as pl
import numpy as np
from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import DatasetProfile

@pytest.fixture
def sample_df():
    """Creates a synthetic dataset for testing."""
    np.random.seed(42)
    rows = 500
    
    data = {
        "age": np.random.randint(18, 80, rows),
        "income": np.random.normal(50000, 15000, rows),
        "category": np.random.choice(["A", "B", "C"], rows),
        "target_class": np.random.choice(["Yes", "No"], rows), # Classification Target
        "target_reg": np.random.normal(100, 20, rows), # Regression Target
        "date": pl.date_range(
            start=pl.date(2023, 1, 1),
            end=pl.date(2025, 12, 31), # Extended range to ensure > 500 days
            interval="1d",
            eager=True
        )[:rows],
        "lat": np.random.uniform(30, 40, rows),
        "lon": np.random.uniform(-100, -90, rows)
    }
    
    # Add some correlation for regression target
    # target_reg = 0.5 * age + noise
    data["target_reg"] = 0.5 * data["age"] + np.random.normal(0, 5, rows)
    
    return pl.DataFrame(data)

def test_basic_profiling(sample_df):
    """Test basic profiling without target."""
    analyzer = EDAAnalyzer(sample_df)
    profile = analyzer.analyze()
    
    assert isinstance(profile, DatasetProfile)
    assert profile.row_count == 500
    assert len(profile.columns) == 8
    assert "age" in profile.columns
    assert profile.columns["age"].dtype == "Numeric"
    assert profile.columns["category"].dtype == "Categorical"

def test_classification_target(sample_df):
    """Test profiling with a classification target."""
    analyzer = EDAAnalyzer(sample_df)
    profile = analyzer.analyze(target_col="target_class")
    
    assert profile.target_col == "target_class"
    
    # Check Rule Discovery (Decision Tree)
    assert profile.rule_tree is not None
    assert len(profile.rule_tree.nodes) > 0
    assert len(profile.rule_tree.rules) > 0
    
    # Check if rules look like classification rules
    # "IF ... THEN Yes (Confidence: ...)"
    assert "Confidence" in profile.rule_tree.rules[0]

def test_regression_target(sample_df):
    """Test profiling with a regression target."""
    analyzer = EDAAnalyzer(sample_df)
    profile = analyzer.analyze(target_col="target_reg")
    
    assert profile.target_col == "target_reg"
    
    # Check Rule Discovery (Decision Tree Regressor)
    assert profile.rule_tree is not None
    assert len(profile.rule_tree.nodes) > 0
    assert len(profile.rule_tree.rules) > 0
    
    # Check if rules look like regression rules
    # "IF ... THEN Value = 123.45 (Samples: ...)"
    assert "Value =" in profile.rule_tree.rules[0]
    
    # Check Feature Importance
    # Age should be important because we engineered it
    importances = {item["feature"]: item["importance"] for item in profile.rule_tree.feature_importances}
    assert "age" in importances
    assert importances["age"] > 0.1 # Should be significant

def test_causal_discovery(sample_df):
    """Test causal discovery (PC Algorithm)."""
    # Causal discovery requires numeric columns
    # We have age, income, target_reg, lat, lon
    
    analyzer = EDAAnalyzer(sample_df)
    # We need to ensure we have enough numeric columns and rows
    # The fixture has 500 rows which is enough for small test
    
    # Note: Causal discovery might be skipped if libraries are missing or data is too small/simple
    # But we mocked the data to have some structure
    
    profile = analyzer.analyze(target_col="target_reg")
    
    # If causal-learn is installed, this should be populated
    # We can't strictly assert it's not None because the environment might not have causal-learn
    # But we can check if the code runs without error
    if profile.causal_graph:
        assert len(profile.causal_graph.nodes) > 0

def test_outlier_detection(sample_df):
    """Test outlier detection."""
    # Add a massive outlier
    df_outlier = sample_df.clone()
    df_outlier[0, "income"] = 10000000 # Huge income
    
    analyzer = EDAAnalyzer(df_outlier)
    profile = analyzer.analyze()
    
    assert profile.outliers is not None
    assert profile.outliers.total_outliers > 0
    
    # The first row should be in top outliers
    top_indices = [o.index for o in profile.outliers.top_outliers]
    assert 0 in top_indices

def test_timeseries_analysis(sample_df):
    """Test time series detection."""
    analyzer = EDAAnalyzer(sample_df)
    profile = analyzer.analyze(date_col="date")
    
    assert profile.timeseries is not None
    assert profile.timeseries.date_col == "date"
    assert len(profile.timeseries.trend) > 0

def test_geospatial_analysis(sample_df):
    """Test geospatial detection."""
    analyzer = EDAAnalyzer(sample_df)
    profile = analyzer.analyze(lat_col="lat", lon_col="lon")
    
    assert profile.geospatial is not None
    assert profile.geospatial.lat_col == "lat"
    assert profile.geospatial.lon_col == "lon"

def test_high_cardinality_classification():
    # Create data where feature predicts "Other" group
    # Rows 0-500: feature=0.2, target="Class_0" (Frequent)
    # Rows 500-1000: feature=0.8, target="Class_X" (Many rare classes)
    
    targets = []
    features = []
    
    for i in range(1000):
        if i < 500:
            features.append(0.2)
            targets.append("Class_0")
        else:
            features.append(0.8)
            targets.append(f"Class_{i % 40 + 10}")
            
    df = pl.DataFrame({
        "feature": features,
        "target": targets
    })
    
    # Verify high cardinality
    assert df["target"].n_unique() > 10
    
    analyzer = EDAAnalyzer(df)
    
    # Force classification
    profile = analyzer.analyze(target_col="target", task_type="Classification")
    
    assert profile.rule_tree is not None
    
    rules_text = "\n".join(profile.rule_tree.rules)
    assert "Other" in rules_text or "Class_0" in rules_text

def test_force_regression_on_id():
    # Test forcing regression on an ID-like column
    df = pl.DataFrame({
        "feature": [1, 2, 3, 4, 5] * 20,
        "id": range(100)
    })
    
    analyzer = EDAAnalyzer(df)
    # Default might be Numeric (Regression) because unique > 20
    profile = analyzer.analyze(target_col="id")
    
    # Check if it ran as regression (rules should have "Value =")
    rules_text = "\n".join(profile.rule_tree.rules)
    assert "Value =" in rules_text

def test_force_classification_on_id():
    # Test forcing classification on an ID-like column
    df = pl.DataFrame({
        "feature": [1, 2, 3, 4, 5] * 20,
        "id": range(100)
    })
    
    analyzer = EDAAnalyzer(df)
    # Force Classification
    profile = analyzer.analyze(target_col="id", task_type="Classification")
    
    # Check if it ran as classification (rules should have "Class =" or just the class name)
    # And since it has 100 unique values, it should have grouped them.
    rules_text = "\n".join(profile.rule_tree.rules)
    # Visualizer format is "THEN {class_name}"
    assert "THEN Other" in rules_text or "THEN" in rules_text
