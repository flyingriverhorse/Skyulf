"""Tests for the static column-reference validator (C7 Phase D)."""

import pandas as pd
import pytest

from backend.ml_pipeline._execution._schema_graph import predict_schemas
from backend.ml_pipeline._execution._schema_validator import find_broken_references
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from skyulf.preprocessing import SkyulfSchema


@pytest.mark.parametrize(
    "step,params,field",
    [
        ("GroupImputer", {"group_by": ["a", "ghost"]}, "group_by"),
        ("LagFeatures", {"sort_by": ["ghost"]}, "sort_by"),
        ("DropDuplicates", {"subset": ["ghost"]}, "subset"),
        ("H3Index", {"lat_col": "ghost"}, "lat_col"),
        ("H3Index", {"lon_col": "ghost"}, "lon_col"),
        ("Binning", {"custom_bins": {"ghost": [0, 1]}}, "custom_bins"),
    ],
)
def test_additional_known_column_fields_are_checked(step, params, field):
    """Supported grouping, ordering, geographic and bin mappings must not hide misspelled inputs."""
    config = _config(NodeConfig("step", step, params=params, inputs=["loader"]))
    assert find_broken_references(config, {"loader": _seed()}) == [
        {"node_id": "step", "field": field, "column": "ghost", "upstream_node_id": "loader"}
    ]


@pytest.mark.parametrize("alias", ["FeatureMath", "FeatureGeneration", "FeatureGenerationNode"])
def test_feature_math_validates_ordered_operands_without_flagging_generated_outputs(alias):
    """Earlier generated columns are valid operands while later outputs and misspelled inputs are not."""
    config = _config(
        NodeConfig(
            "math",
            alias,
            inputs=["loader"],
            params={
                "operations": [
                    {
                        "operation_type": "arithmetic",
                        "method": "add",
                        "input_columns": ["a"],
                        "output_column": "total",
                    },
                    {
                        "operation_type": "ratio",
                        "input_columns": ["total"],
                        "secondary_columns": ["b", "ghost"],
                        "output_column": "future",
                    },
                    {
                        "operation_type": "arithmetic",
                        "method": "add",
                        "input_columns": ["a", "next"],
                    },
                    {
                        "operation_type": "arithmetic",
                        "method": "add",
                        "input_columns": ["arithmetic_2"],
                        "output_column": "next",
                    },
                ]
            },
        )
    )
    broken = find_broken_references(config, {"loader": _seed()})
    assert [(item["field"], item["column"]) for item in broken] == [
        ("operations.1.secondary_columns", "ghost"),
        ("operations.2.input_columns", "next"),
    ]


def test_feature_math_generated_calendar_names_and_collision_suffixes_are_valid_inputs():
    """Reference checking must follow the same expanded calendar and collision names as execution."""
    config = _config(
        NodeConfig(
            "math",
            "FeatureMath",
            inputs=["loader"],
            params={
                "operations": [
                    {
                        "operation_type": "datetime_extract",
                        "input_columns": ["c"],
                        "datetime_features": ["year", "month"],
                        "output_column": "calendar",
                    },
                    {
                        "operation_type": "arithmetic",
                        "method": "add",
                        "input_columns": ["calendar_year", "calendar_month"],
                        "output_column": "a",
                    },
                    {
                        "operation_type": "arithmetic",
                        "method": "add",
                        "input_columns": ["a_1"],
                        "output_column": "result",
                    },
                ]
            },
        )
    )
    assert find_broken_references(config, {"loader": _seed()}) == []


def _seed() -> SkyulfSchema:
    return SkyulfSchema.from_dataframe(pd.DataFrame({"a": [1.0], "b": [2.0], "c": ["x"]}))


@pytest.mark.parametrize("operations", [None, 12, "invalid", {"operation_type": "arithmetic"}])
def test_malformed_operation_container_does_not_crash_reference_preview(operations):
    """Imported malformed operation containers must leave other nodes' preview diagnostics usable."""
    config = _config(
        NodeConfig(
            node_id="math",
            step_type="FeatureMath",
            inputs=["loader"],
            params={"operations": operations},
        )
    )
    assert find_broken_references(config, {"loader": _seed()}) == []


@pytest.mark.parametrize("field", ["input_columns", "datetime_features"])
@pytest.mark.parametrize("value", [None, 12, "invalid", [None], [["invalid"]]])
def test_malformed_calendar_options_do_not_crash_reference_preview(field, value):
    """Best-effort input validation must not let invalid calendar shapes crash schema preview."""
    operation = {
        "operation_type": "datetime_extract",
        "input_columns": ["c"],
        "datetime_features": ["year"],
        field: value,
    }
    config = _config(
        NodeConfig(
            node_id="math",
            step_type="FeatureMath",
            inputs=["loader"],
            params={"operations": [operation]},
        )
    )
    assert find_broken_references(config, {"loader": _seed()}) == []


def _config(node: NodeConfig) -> PipelineConfig:
    return PipelineConfig(
        pipeline_id="p",
        nodes=[
            NodeConfig(node_id="loader", step_type="data_loader", params={}, inputs=[]),
            node,
        ],
    )


def test_drops_broken_columns_param() -> None:
    """A `columns` reference to a non-existent column is flagged."""
    config = _config(
        NodeConfig(
            node_id="drop",
            step_type="DropMissingColumns",
            params={"columns": ["a", "ghost"]},
            inputs=["loader"],
        )
    )
    predicted = predict_schemas(config, initial_schemas={"loader": _seed()})
    broken = find_broken_references(config, predicted)

    assert len(broken) == 1
    assert broken[0]["node_id"] == "drop"
    assert broken[0]["column"] == "ghost"
    assert broken[0]["field"] == "columns"
    assert broken[0]["upstream_node_id"] == "loader"


def test_valid_references_emit_nothing() -> None:
    """All known columns -> no flags."""
    config = _config(
        NodeConfig(
            node_id="drop",
            step_type="DropMissingColumns",
            params={"columns": ["a", "b"]},
            inputs=["loader"],
        )
    )
    predicted = predict_schemas(config, initial_schemas={"loader": _seed()})

    assert find_broken_references(config, predicted) == []


def test_unknown_upstream_schema_skips_validation() -> None:
    """When the upstream schema is None we cannot validate -> skip silently."""
    config = _config(
        NodeConfig(
            node_id="drop",
            step_type="DropMissingColumns",
            params={"columns": ["ghost"]},
            inputs=["loader"],
        )
    )
    # No initial seed -> loader schema is None.
    predicted = predict_schemas(config)

    assert find_broken_references(config, predicted) == []


def test_target_field_string_reference() -> None:
    """The `target` (string) field is also validated."""
    config = _config(
        NodeConfig(
            node_id="trainer",
            step_type="training",
            params={"target": "ghost"},
            inputs=["loader"],
        )
    )
    predicted = predict_schemas(config, initial_schemas={"loader": _seed()})
    broken = find_broken_references(config, predicted)

    assert len(broken) == 1
    assert broken[0]["field"] == "target"
    assert broken[0]["column"] == "ghost"


def test_basic_training_target_column_optional_not_flagged() -> None:
    """`training`'s `target_column` is metadata, not a feature reference:
    it should never be flagged even though it's absent from the upstream
    schema (the FeatureTargetSplitter dropped it upstream in real pipelines).
    """
    config = _config(
        NodeConfig(
            node_id="trainer",
            step_type="training",
            params={"target_column": "ghost"},
            inputs=["loader"],
        )
    )
    predicted = predict_schemas(config, initial_schemas={"loader": _seed()})

    assert find_broken_references(config, predicted) == []


def test_training_target_column_optional_not_flagged() -> None:
    """Same as the target_column optional validation but
    for the unified `"training"` step_type (Finding 3): its `target_column`
    must also be treated as optional, not a false-positive amber warning.
    """
    config = _config(
        NodeConfig(
            node_id="trainer",
            step_type="training",
            params={"target_column": "ghost", "run_mode": "fixed"},
            inputs=["loader"],
        )
    )
    predicted = predict_schemas(config, initial_schemas={"loader": _seed()})

    assert find_broken_references(config, predicted) == []


def test_column_types_dict_keys_validated() -> None:
    """`column_types` dict keys are checked too."""
    config = _config(
        NodeConfig(
            node_id="cast",
            step_type="Casting",
            params={"column_types": {"a": "int", "ghost": "float"}},
            inputs=["loader"],
        )
    )
    predicted = predict_schemas(config, initial_schemas={"loader": _seed()})
    broken = find_broken_references(config, predicted)

    columns_flagged = {b["column"] for b in broken}
    assert "ghost" in columns_flagged
    assert "a" not in columns_flagged
