"""Outlier statistics retain their analyzed population through profile serialization."""

import numpy as np
import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import DatasetProfile, OutlierAnalysis


@pytest.mark.parametrize(("row_count", "analyzed_rows"), [(2000, 2000), (200000, 50000)])
def test_real_outlier_analysis_reports_its_denominator(row_count: int, analyzed_rows: int) -> None:
    """A sample percentage must never be presented as a measured full-dataset outlier count."""
    frame = pl.DataFrame({"measurement": np.random.default_rng(302).normal(size=row_count)})

    profile = EDAAnalyzer(frame).analyze()
    restored = DatasetProfile.model_validate_json(profile.model_dump_json())

    assert restored.row_count == row_count
    assert restored.outliers is not None
    result = restored.outliers.model_dump()
    assert result.get("analyzed_rows") == analyzed_rows
    assert result.get("total_rows") == row_count
    assert 0 < result["total_outliers"] <= analyzed_rows // 20
    assert result["outlier_percentage"] == pytest.approx(
        100 * result["total_outliers"] / analyzed_rows
    )
    assert len(result["top_outliers"]) == 20
    assert all(0 <= point["index"] < row_count for point in result["top_outliers"])


def test_legacy_outlier_payload_keeps_unknown_population() -> None:
    """Old persisted profiles remain valid without inventing an analyzed denominator."""
    result = OutlierAnalysis.model_validate(
        {
            "method": "IsolationForest",
            "total_outliers": 2500,
            "outlier_percentage": 5,
            "top_outliers": [],
        }
    ).model_dump()

    assert result.get("analyzed_rows", "absent") is None
    assert result.get("total_rows", "absent") is None
