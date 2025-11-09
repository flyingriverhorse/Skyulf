import pandas as pd

from core.feature_engineering.nodes.feature_eng.feature_target_split import apply_feature_target_split


def _build_node(config):
    return {"id": "node-1", "data": {"config": config}}


def test_feature_target_split_with_explicit_columns():
    frame = pd.DataFrame(
        {
            "target": [1, 0, 1],
            "a": [10, 20, 30],
            "b": ["x", "y", "z"],
            "c": [1.1, 2.2, 3.3],
        }
    )
    node = _build_node({"target_column": "target", "feature_columns": ["a", "b"]})

    result, summary, signal = apply_feature_target_split(frame, node)

    assert list(result.columns[:3]) == ["a", "b", "target"]
    assert summary.startswith("Feature/target split; target 'target'")
    assert signal.feature_columns == ["a", "b"]
    assert signal.target_missing_count == 0
    assert not signal.warnings


def test_feature_target_split_auto_detects_features():
    frame = pd.DataFrame(
        {
            "target": ["yes", "no", "yes"],
            "num": [1, 2, 3],
            "cat": ["alpha", "beta", "gamma"],
        }
    )
    node = _build_node({"target_column": "target"})

    _, summary, signal = apply_feature_target_split(frame, node)

    assert "features=2" in summary
    assert set(signal.feature_columns) == {"num", "cat"}
    assert signal.auto_included_columns == ["num", "cat"]
    assert "Auto-selected 2 feature column(s)." in signal.notes


def test_feature_target_split_warns_missing_columns():
    frame = pd.DataFrame({"target": [0, 1], "present": ["a", "b"]})
    node = _build_node({
        "target_column": "target",
        "feature_columns": ["present", "missing_one", "missing_two"],
    })

    _, summary, signal = apply_feature_target_split(frame, node)

    assert "missing=2" in summary
    assert "Configured feature column(s) not found" in " ".join(signal.warnings)
    assert signal.feature_columns == ["present"]
    assert signal.missing_feature_columns == ["missing_one", "missing_two"]
