import pytest
from core.ml_pipeline.recommendations.engine import AdvisorEngine
from core.ml_pipeline.recommendations.schemas import AnalysisProfile, ColumnProfile, ColumnType

@pytest.fixture
def sample_profile():
    return AnalysisProfile(
        row_count=1000,
        column_count=3,
        duplicate_row_count=50,
        columns={
            "age": ColumnProfile(
                name="age",
                dtype="float",
                column_type=ColumnType.NUMERIC,
                missing_count=100,
                missing_ratio=0.1,
                unique_count=50,
                min_value=0,
                max_value=100,
                mean_value=35,
                std_value=15,
                skewness=0.1
            ),
            "salary": ColumnProfile(
                name="salary",
                dtype="float",
                column_type=ColumnType.NUMERIC,
                missing_count=0,
                missing_ratio=0.0,
                unique_count=900,
                min_value=20000,
                max_value=1000000,
                mean_value=60000,
                std_value=50000,
                skewness=5.0 # Highly skewed
            ),
            "city": ColumnProfile(
                name="city",
                dtype="object",
                column_type=ColumnType.CATEGORICAL,
                missing_count=10,
                missing_ratio=0.01,
                unique_count=5,
                top_values={"NY": 400, "LA": 300, "Chicago": 200, "Houston": 90, "Phoenix": 10}
            )
        }
    )

def test_advisor_engine_init():
    engine = AdvisorEngine()
    assert len(engine.plugins) > 0

def test_advisor_analysis(sample_profile):
    engine = AdvisorEngine()
    recommendations = engine.analyze(sample_profile)
    
    assert len(recommendations) > 0
    
    # Check for specific expected recommendations based on the profile
    
    # 1. Imputation for 'age' (10% missing)
    imputation_recs = [r for r in recommendations if r.type == "imputation" and "age" in r.suggested_params.get("columns", [])]
    assert len(imputation_recs) > 0
    
    # 2. Scaling/Transformation for 'salary' (high skewness)
    skew_recs = [r for r in recommendations if r.type == "transformation" and "salary" in r.suggested_params.get("columns", [])]
    assert len(skew_recs) > 0
    
    # 3. Encoding for 'city' (categorical)
    encoding_recs = [r for r in recommendations if r.type == "encoding" and "city" in r.suggested_params.get("columns", [])]
    assert len(encoding_recs) > 0
    
    # 4. Deduplication (duplicate rows > 0)
    dedup_recs = [r for r in recommendations if r.type == "cleaning" and r.rule_id == "duplicate_rows_drop"]
    assert len(dedup_recs) > 0

def test_empty_profile():
    profile = AnalysisProfile(
        row_count=100,
        column_count=0,
        columns={}
    )
    engine = AdvisorEngine()
    recs = engine.analyze(profile)
    # Should handle gracefully, maybe return no recommendations or just generic ones
    assert isinstance(recs, list)
