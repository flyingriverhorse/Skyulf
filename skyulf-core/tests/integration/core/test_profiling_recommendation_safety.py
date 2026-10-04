"""Regression coverage for applicable and consistent EDA transformation advice."""

import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import ColumnProfile, NumericStats


@pytest.mark.parametrize(
    ("values", "expected_method"),
    [
        ([-100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "Yeo-Johnson"),
        ([0.0] * 7 + [100.0], "Yeo-Johnson"),
        ([-100.0] * 7 + [-1.0], "Yeo-Johnson"),
        ([1.0] * 7 + [100.0], "Log or Box-Cox"),
        ([100.0] * 7 + [1.0], "Yeo-Johnson"),
    ],
    ids=["mixed-sign", "zero", "all-negative", "positive-right-skew", "positive-left-skew"],
)
def test_public_profile_recommends_a_compatible_transform(
    values: list[float], expected_method: str
) -> None:
    """The displayed advice must respect the domain and direction of the observed skew."""
    frame = pl.DataFrame({"x": values})
    original = frame.clone()
    profile = EDAAnalyzer(frame).analyze()
    transforms = [rec for rec in profile.recommendations if rec.action == "Transform"]

    assert len(transforms) == 1
    assert transforms[0].column == "x"
    assert expected_method in transforms[0].suggestion
    if expected_method == "Yeo-Johnson":
        assert "Log" not in transforms[0].suggestion
        assert "Box-Cox" not in transforms[0].suggestion
    assert not any(rec.action == "Keep" for rec in profile.recommendations)
    assert frame.equals(original)


@pytest.mark.parametrize(
    "stats",
    [
        NumericStats(skewness=2.3),
        NumericStats(skewness=2.3, min=0.0),
        NumericStats(skewness=2.3, min=-1.0),
        NumericStats(skewness=2.3, min=1.0, zeros_count=1),
        NumericStats(skewness=2.3, min=1.0, negatives_count=1),
    ],
    ids=["unknown-domain", "zero", "negative", "zero-count", "negative-count"],
)
def test_incomplete_or_nonpositive_profile_does_not_recommend_positive_only_transform(
    stats: NumericStats,
) -> None:
    """Missing or conflicting domain evidence must not produce unsafe log advice."""
    profile = ColumnProfile(
        name="x", dtype="Numeric", missing_count=0, missing_percentage=0, numeric_stats=stats
    )
    analyzer = EDAAnalyzer(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
    recommendations = analyzer._generate_recommendations({"x": profile}, [], None)

    assert len(recommendations) == 1
    assert recommendations[0].action == "Transform"
    assert "Yeo-Johnson" in recommendations[0].suggestion


@pytest.mark.parametrize("skewness", [None, 0.0, 1.5, -1.5, float("nan"), float("inf")])
def test_unavailable_or_low_skewness_does_not_request_transformation(
    skewness: float | None,
) -> None:
    """Only a defined skewness beyond the threshold justifies transform advice."""
    profile = ColumnProfile(
        name="x",
        dtype="Numeric",
        missing_count=0,
        missing_percentage=0,
        numeric_stats=NumericStats(skewness=skewness, min=1.0),
    )
    analyzer = EDAAnalyzer(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
    recommendations = analyzer._generate_recommendations({"x": profile}, [], None)

    assert [rec.action for rec in recommendations] == ["Keep"]
    assert "ready for modeling" not in recommendations[0].suggestion.lower()


@pytest.mark.parametrize("scenario", ["transform", "encode", "resample", "impute", "drop"])
def test_public_profile_never_combines_remediation_with_clean_dataset(scenario: str) -> None:
    """Every actionable recommendation must suppress the contradictory clean message."""
    frames = {
        "transform": pl.DataFrame({"x": [1.0] * 7 + [100.0]}),
        "encode": pl.DataFrame(
            {"x": pl.Series([str(i) for i in range(60)] * 2).cast(pl.Categorical)}
        ),
        "resample": pl.DataFrame({"x": ["a"] * 95 + ["b"] * 5}),
        "impute": pl.DataFrame({"x": [1.0, 2.0, 3.0, None]}),
        "drop": pl.DataFrame({"x": [7.0] * 8}),
    }
    profile = EDAAnalyzer(frames[scenario]).analyze(
        target_col="x" if scenario == "resample" else None
    )
    # Exercise the payload that the Insights tab displays, including target advice.
    recommendations = profile.model_dump(mode="json")["recommendations"]
    actions = {rec["action"] for rec in recommendations}

    assert scenario.capitalize() in actions
    assert "Keep" not in actions
    assert all("ready for modeling" not in rec["suggestion"].lower() for rec in recommendations)


def test_balanced_target_can_retain_the_limited_clean_message() -> None:
    """An informational balance result alone does not request remediation."""
    profile = EDAAnalyzer(pl.DataFrame({"target": ["a", "b"] * 50})).analyze(target_col="target")

    assert {rec.action for rec in profile.recommendations} == {"Info", "Keep"}
    assert all(
        "ready for modeling" not in rec.suggestion.lower() for rec in profile.recommendations
    )
