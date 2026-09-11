"""Missing features must not corrupt the association ranking sent to the EDA UI."""

import polars as pl
import pytest

from backend.eda.tasks import _run_eda_analyzer


@pytest.mark.parametrize("target_kind", ["boolean", "categorical"])
def test_eda_report_ranks_observed_associations_before_missingness(target_kind) -> None:
    """The production analyzer path must serialize correct scores and strongest-feature order."""
    target = pl.Series("target", [False] * 2 + [True] * 8)
    if target_kind == "categorical":
        target = pl.Series("target", ["a"] * 2 + ["b"] * 8).cast(pl.Categorical)
    frame = pl.DataFrame(
        {
            "target": target,
            "strong": [0.0] * 2 + [10.0] * 8,
            "sparse": [0.0, 2.0, 2.0, 4.0] + [None] * 6,
        }
    )

    profile = _run_eda_analyzer(frame, {"target_col": "target"})
    payload = profile.model_dump(mode="json")

    assert payload["row_count"] == frame.height
    assert payload["target_col"] == "target"
    assert list(payload["target_correlations"]) == ["strong", "sparse"]
    assert payload["target_correlations"] == pytest.approx({"strong": 1.0, "sparse": 2**-0.5})
    assert payload["columns"]["sparse"]["missing_count"] == 6
