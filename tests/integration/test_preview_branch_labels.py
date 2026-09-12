"""Preview branch labels stay attached to the measured preprocessing outputs."""

import pandas as pd
import pytest

from tests.integration.test_node_inspection import (
    _node,
    _preview,
    _source,
    preview_client,  # noqa: F401 - reuse the real-router/engine fixture
)


def test_preview_paths_l_and_m_keep_imputation_and_resampling_outputs(preview_client):
    """Canvas's submitted leaf order must label distinct data and inspection receipts alike."""
    pytest.importorskip("imblearn", reason="resampling uses the optional imbalanced-learn extra")
    preview_client.catalog.load.return_value = pd.DataFrame(
        {"value": [1.0, None, 3.0, 4.0, 10.0, 11.0], "target": [0, 0, 0, 0, 1, 1]}
    )
    # The frontend omits its Data Preview sink before submitting this graph,
    # making the imputation node a leaf immediately before resampling.
    nodes = [
        _source(),
        *[
            _node(f"leaf-{index}", "Casting", ["source"], {"_display_name": f"Leaf {index}"})
            for index in range(11)
        ],
        _node(
            "impute",
            "SimpleImputer",
            ["source"],
            {
                "_display_name": "Imputation",
                "columns": ["value"],
                "strategy": "constant",
                "fill_value": -9,
            },
        ),
        _node(
            "resample",
            "Oversampling",
            ["source"],
            {
                "_display_name": "Resampling",
                "method": "random_over",
                "target_column": "target",
                "random_state": 42,
            },
        ),
    ]

    response = _preview(preview_client, nodes, selected=None, inspect_all=True)

    assert response["status"] == "success"
    imputation_label = "Path L · Imputation"
    resampling_label = "Path M · Resampling"
    branches = response["branch_previews"]
    assert len(branches) == 13
    assert list(branches)[-2:] == [imputation_label, resampling_label]
    assert response["branch_node_ids"][imputation_label] == ["source", "impute"]
    assert response["branch_node_ids"][resampling_label] == ["source", "resample"]
    assert response["branch_preview_totals"][imputation_label] == {"_total": 6}
    assert response["branch_preview_totals"][resampling_label] == {"_total": 8}
    assert [row["value"] for row in branches[imputation_label]] == [1, -9, 3, 4, 10, 11]
    assert [row["value"] for row in branches[resampling_label]][:6] == [1, None, 3, 4, 10, 11]
    assert [row["target"] for row in branches[resampling_label]].count(1) == 4
    receipts = {entry["node_id"]: entry for entry in response["node_inspections"]}
    for node_id, label, count in [
        ("impute", imputation_label, 6),
        ("resample", resampling_label, 8),
    ]:
        receipt = receipts[node_id]
        assert receipt["branch_label"] == label
        assert receipt["output"]["status"] == "available"
        assert receipt["output"]["tables"][0]["row_count"] == count
        assert receipt["output"]["tables"][0]["rows"] == branches[label]
