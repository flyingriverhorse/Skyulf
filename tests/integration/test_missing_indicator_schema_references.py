"""Missing-indicator schema predictions must match the columns actually generated."""

import pandas as pd
import polars as pl
import pytest

from backend.ml_pipeline._execution._schema_graph import predict_schemas
from backend.ml_pipeline._execution._schema_validator import find_broken_references
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from skyulf.preprocessing import SkyulfSchema
from skyulf.preprocessing.drop_and_missing.missing_indicator import (
    MissingIndicatorApplier,
    MissingIndicatorCalculator,
)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("suffix", ["_missing", "_flag"])
@pytest.mark.parametrize("selection", ["absent", "mixed", "cascade"])
def test_missing_indicator_schema_matches_runtime_and_downstream_references(
    engine: str, suffix: str, selection: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonexistent source must not make its nonexistent flag look valid downstream."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    data = {"x": [1.0, None]}
    frame = pd.DataFrame(data) if engine == "pandas" else pl.DataFrame(data)
    source = SkyulfSchema.from_dataframe(frame)
    missing_source = f"x{suffix}" if selection == "cascade" else "ghost"
    columns = [missing_source] if selection == "absent" else ["x", missing_source]
    params = {"columns": columns, "flag_suffix": suffix}
    config = PipelineConfig(
        "flags",
        [
            NodeConfig("source", "data_loader"),
            NodeConfig("flags", "MissingIndicator", params, ["source"]),
            NodeConfig(
                "use_flag", "StandardScaler", {"columns": [f"{missing_source}{suffix}"]}, ["flags"]
            ),
        ],
    )
    if selection != "absent":
        config.nodes.append(
            NodeConfig("valid_flag", "StandardScaler", {"columns": [f"x{suffix}"]}, ["flags"])
        )

    predicted = predict_schemas(config, initial_schemas={"source": source})
    fitted = MissingIndicatorCalculator().fit(frame, params)
    actual = MissingIndicatorApplier().apply(frame, fitted)
    broken = find_broken_references(config, predicted)

    assert predicted["flags"] is not None
    assert list(predicted["flags"].columns) == list(actual.columns)
    assert {(ref["node_id"], ref["column"]) for ref in broken} == {
        ("flags", missing_source),
        ("use_flag", f"{missing_source}{suffix}"),
    }
