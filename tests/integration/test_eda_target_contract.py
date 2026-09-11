"""EDA configuration and stored report payloads must preserve target semantics."""

import numpy as np
import polars as pl
import pytest

from backend.eda.router import AnalyzeRequest, _build_analysis_config
from backend.eda.tasks import _run_eda_analyzer


@pytest.mark.parametrize("excluded", [False, True])
def test_eda_report_explains_target_omission_and_respects_exclusions(excluded: bool) -> None:
    """Accepted target/exclusion settings must not persist synthetic codes or excluded rules."""
    pytest.importorskip("causallearn")
    rng = np.random.default_rng(239238)
    frame = pl.DataFrame(
        {
            "x": np.r_[rng.normal(4, 0.1, 30), rng.normal(8, 0.1, 30)],
            "z": rng.normal(size=60),
            "species": ["setosa"] * 30 + ["virginica"] * 30,
        }
    )
    request = AnalyzeRequest(target_col="species", exclude_cols=["species"] if excluded else [])

    payload = _run_eda_analyzer(frame, _build_analysis_config(request)).model_dump(mode="json")

    assert payload["target_col"] == "species"
    assert payload["correlations_with_target"] is None
    assert payload["causal_target_exclusion_reason"] == ("excluded" if excluded else "categorical")
    assert payload["causal_graph"]["selection_method"] == "all"
    assert payload["causal_graph"]["nodes"] == [
        {"id": "x", "label": "x"},
        {"id": "z", "label": "z"},
    ]
    if excluded:
        assert payload["rule_tree"] is None
        assert payload["task_type"] is None
        assert payload["excluded_columns"] == ["species"]
        assert "species" not in payload["columns"]
        assert all("species" not in row for row in payload["sample_data"])
    else:
        assert payload["rule_tree"] is not None
        assert payload["task_type"] == "Classification"
        assert set(payload["target_correlations"]) == {"x", "z"}
        assert payload["sample_data"] == frame.to_dicts()
