import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
FEATURE_MODULE_NAME = "core.feature_engineering.nodes.feature_eng.feature_math"


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module specification for {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def feature_math_module():
    """Load feature_math without requiring core package installation."""
    for key in list(sys.modules):
        if key.startswith("core.feature_engineering"):
            sys.modules.pop(key, None)
    sys.modules.pop("core", None)

    core_pkg = types.ModuleType("core")
    core_pkg.__path__ = [str(REPO_ROOT / "core")]  # type: ignore[attr-defined]
    sys.modules["core"] = core_pkg

    feature_eng_pkg = types.ModuleType("core.feature_engineering")
    feature_eng_pkg.__path__ = [str(REPO_ROOT / "core" / "feature_engineering")]  # type: ignore[attr-defined]
    sys.modules["core.feature_engineering"] = feature_eng_pkg
    core_pkg.feature_engineering = feature_eng_pkg

    schemas_module = _load_module_from_path(
        "core.feature_engineering.schemas",
        REPO_ROOT / "core" / "feature_engineering" / "schemas.py",
    )
    feature_eng_pkg.schemas = schemas_module

    nodes_pkg = types.ModuleType("core.feature_engineering.nodes")
    nodes_pkg.__path__ = [str(REPO_ROOT / "core" / "feature_engineering" / "nodes")]  # type: ignore[attr-defined]
    sys.modules["core.feature_engineering.nodes"] = nodes_pkg
    feature_eng_pkg.nodes = nodes_pkg

    feature_eng_subpkg = types.ModuleType("core.feature_engineering.nodes.feature_eng")
    feature_eng_subpkg.__path__ = [
        str(REPO_ROOT / "core" / "feature_engineering" / "nodes" / "feature_eng")
    ]  # type: ignore[attr-defined]
    sys.modules["core.feature_engineering.nodes.feature_eng"] = feature_eng_subpkg
    nodes_pkg.feature_eng = feature_eng_subpkg

    module = _load_module_from_path(
        FEATURE_MODULE_NAME,
        REPO_ROOT / "core" / "feature_engineering" / "nodes" / "feature_eng" / "feature_math.py",
    )

    return module


@pytest.fixture()
def apply_feature_math(feature_math_module):
    return feature_math_module.apply_feature_math


@pytest.fixture()
def feature_math_error(feature_math_module):
    return feature_math_module.FeatureMathOperationError


def _build_node(operations, **config):
    payload = {"operations": operations}
    payload.update(config)
    return {"id": "feature_math_test", "data": {"config": payload}}


def test_arithmetic_operation(apply_feature_math):
    frame = pd.DataFrame({"Elevation": [10, 20], "Slope": [2, 3]})
    node = _build_node([
        {
            "operation_id": "arith",
            "operation_type": "arithmetic",
            "method": "add",
            "input_columns": ["Elevation", "Slope"],
            "output_column": "elevation_slope_sum",
        }
    ])

    result, summary, signal = apply_feature_math(frame.copy(), node)

    assert summary == "Feature math: 1 applied"
    assert signal.operations[0].message.startswith("Applied add")
    assert list(result["elevation_slope_sum"]) == [12, 23]
    assert signal.total_operations == 1
    assert signal.applied_operations == 1
    assert signal.skipped_operations == 0
    assert signal.failed_operations == 0


def test_ratio_operation_with_custom_epsilon(apply_feature_math):
    frame = pd.DataFrame({"num": [10.0, 5.0], "den": [0.0, 2.0]})
    node = _build_node([
        {
            "operation_id": "ratio",
            "operation_type": "ratio",
            "input_columns": ["num"],
            "secondary_columns": ["den"],
            "epsilon": 0.5,
            "output_column": "num_over_den",
        }
    ])

    result, _, signal = apply_feature_math(frame.copy(), node)

    assert pytest.approx(result.loc[0, "num_over_den"], rel=1e-6) == 20.0
    assert pytest.approx(result.loc[1, "num_over_den"], rel=1e-6) == 2.5
    assert signal.applied_operations == 1


def test_stat_operation_mean(apply_feature_math):
    frame = pd.DataFrame(
        {
            "a": [1, 4],
            "b": [3, 5],
            "c": [5, 7],
        }
    )
    node = _build_node([
        {
            "operation_id": "stat",
            "operation_type": "stat",
            "method": "mean",
            "input_columns": ["a", "b", "c"],
            "output_column": "avg_abc",
        }
    ])

    result, _, signal = apply_feature_math(frame.copy(), node)

    assert list(result["avg_abc"]) == [3.0, 5.333333333333333]
    assert signal.applied_operations == 1


def test_similarity_operation_normalized(apply_feature_math):
    frame = pd.DataFrame({"text_a": ["hello", "forest"], "text_b": ["hello", "frost"]})
    node = _build_node([
        {
            "operation_id": "sim",
            "operation_type": "similarity",
            "method": "ratio",
            "input_columns": ["text_a", "text_b"],
            "normalize": True,
            "output_column": "text_similarity",
        }
    ])

    result, _, signal = apply_feature_math(frame.copy(), node)

    assert pytest.approx(result.loc[0, "text_similarity"], rel=1e-6) == 1.0
    assert 0.0 <= result.loc[1, "text_similarity"] <= 1.0
    assert signal.applied_operations == 1


def test_similarity_single_column_self_compare(apply_feature_math):
    frame = pd.DataFrame({"text": ["alpha", "beta"]})
    node = _build_node([
        {
            "operation_id": "sim",
            "operation_type": "similarity",
            "input_columns": ["text"],
            "output_column": "text_similarity",
        }
    ])

    result, summary, signal = apply_feature_math(frame.copy(), node)

    assert summary == "Feature math: 1 applied"
    for value in result["text_similarity"]:
        assert pytest.approx(value, rel=1e-6) == 100.0
    assert signal.applied_operations == 1
    assert signal.skipped_operations == 0


def test_similarity_placeholder_respects_fillna(apply_feature_math):
    frame = pd.DataFrame({"text": ["hello", None]})
    node = _build_node([
        {
            "operation_id": "sim",
            "operation_type": "similarity",
            "input_columns": ["text"],
            "secondary_columns": ["missing"],
            "output_column": "fallback_similarity",
            "fillna": 42,
        }
    ])

    result, summary, signal = apply_feature_math(frame.copy(), node)

    assert summary == "Feature math: 0 applied, 1 skipped"
    assert list(result["fallback_similarity"]) == [42.0, 42.0]
    assert signal.skipped_operations == 1
    assert signal.operations[0].output_columns == ["fallback_similarity"]
    assert any("fallback_similarity" in warning for warning in signal.warnings)


def test_datetime_operation_creates_columns_for_each_input(apply_feature_math):
    frame = pd.DataFrame(
        {
            "ts1": pd.to_datetime(["2024-01-01 08:15:00", "2024-02-02 09:30:00"]),
            "ts2": pd.to_datetime(["2024-03-03 10:45:00", "2024-04-04 23:05:00"]),
        }
    )
    node = _build_node([
        {
            "operation_id": "dt",
            "operation_type": "datetime_extract",
            "input_columns": ["ts1", "ts2"],
            "datetime_features": ["day", "hour"],
            "output_prefix": "feat_",
        }
    ])

    result, _, signal = apply_feature_math(frame.copy(), node)

    expected_columns = {
        "feat_day",
        "feat_hour",
        "feat_ts2_day",
        "feat_ts2_hour",
    }
    assert expected_columns.issubset(result.columns)
    assert list(result["feat_day"]) == [1, 2]
    assert list(result["feat_ts2_hour"]) == [10, 23]
    assert signal.applied_operations == 1


def test_missing_column_is_skipped(apply_feature_math):
    frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    node = _build_node([
        {
            "operation_id": "missing",
            "operation_type": "arithmetic",
            "method": "add",
            "input_columns": ["missing_column"],
            "output_column": "should_not_exist",
        },
        {
            "operation_id": "valid",
            "operation_type": "arithmetic",
            "method": "add",
            "input_columns": ["a", "b"],
            "output_column": "valid_sum",
        },
    ])

    result, _, signal = apply_feature_math(frame.copy(), node)

    assert "should_not_exist" not in result.columns
    assert "valid_sum" in result.columns
    assert signal.total_operations == 2
    assert signal.skipped_operations == 1
    assert signal.applied_operations == 1


def test_output_conflict_raises_when_fail(feature_math_error, apply_feature_math):
    frame = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    node = _build_node(
        [
            {
                "operation_id": "first",
                "operation_type": "arithmetic",
                "method": "add",
                "input_columns": ["x"],
                "output_column": "duplicate_col",
            },
            {
                "operation_id": "second",
                "operation_type": "arithmetic",
                "method": "add",
                "input_columns": ["y"],
                "output_column": "duplicate_col",
            },
        ],
        error_handling="fail",
    )

    with pytest.raises(feature_math_error):
        apply_feature_math(frame.copy(), node)


def test_allow_overwrite_replaces_existing_column(apply_feature_math):
    frame = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
    node = _build_node([
        {
            "operation_id": "first",
            "operation_type": "arithmetic",
            "method": "add",
            "input_columns": ["x"],
            "output_column": "shared",
        },
        {
            "operation_id": "second",
            "operation_type": "arithmetic",
            "method": "add",
            "input_columns": ["y"],
            "output_column": "shared",
            "allow_overwrite": True,
        },
    ])

    result, _, signal = apply_feature_math(frame.copy(), node)

    assert list(result["shared"]) == [10, 20]
    assert signal.applied_operations == 2
