"""Console summaries distinguish sampled outliers from complete and legacy analyses."""

import pytest

from skyulf.profiling.schemas import DatasetProfile, OutlierAnalysis
from skyulf.profiling.visualizer import EDAVisualizer


@pytest.mark.parametrize(
    ("population", "message"),
    [
        ({"analyzed_rows": 50000, "total_rows": 200000}, "50,000 sampled rows out of 200,000"),
        ({"analyzed_rows": 50000, "total_rows": 50000}, "all 50,000 rows"),
        ({}, "Analyzed row count is unavailable"),
    ],
)
def test_public_console_summary_describes_outlier_population(
    population: dict[str, int], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Printed counts must carry the same measured denominator as the saved profile and UI."""
    profile = DatasetProfile(
        row_count=200000,
        column_count=0,
        columns={},
        duplicate_rows=0,
        missing_cells_percentage=0,
        memory_usage_mb=0,
        outliers=OutlierAnalysis(
            method="IsolationForest",
            total_outliers=2500,
            outlier_percentage=5,
            top_outliers=[],
            analyzed_rows=population.get("analyzed_rows"),
            total_rows=population.get("total_rows"),
        ),
    )

    EDAVisualizer(profile).summary()

    output = " ".join(capsys.readouterr().out.split())
    assert message in output
    assert "2500" in output
    assert "5.00%" in output
