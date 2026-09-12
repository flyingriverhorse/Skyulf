"""Binning schemas and category keys must survive fitting and engine replay."""

import pickle
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines.registry import EngineRegistry
from skyulf.preprocessing.bucketing import (
    CustomBinningCalculator,
    GeneralBinningApplier,
    GeneralBinningCalculator,
    KBinsDiscretizerCalculator,
)
from skyulf.preprocessing.encoding.dummy import DummyEncoderApplier
from skyulf.preprocessing.pipeline import FeatureEngineer


@pytest.fixture(params=["pandas", "polars", "pandas_wrapped", "polars_wrapped"])
def kind(request: Any) -> str:
    """Exercise native and wrapped callers through the same public contracts."""
    return request.param


def _frame(kind: str, data: dict[str, list[Any]]) -> Any:
    """Build equivalent input frames without hiding their original engine."""
    frame: Any = pd.DataFrame(data)
    if kind.startswith("polars"):
        frame = pl.from_pandas(frame)
    return EngineRegistry.wrap(frame) if kind.endswith("wrapped") else frame


def _pandas(frame: Any) -> pd.DataFrame:
    """Expose output values while using only the public wrapper interface."""
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame


@pytest.mark.parametrize("edges", [[0, 5, 10], [0, 5.0, 10], [0.0, 5.0, 10.0]])
@pytest.mark.parametrize("include_lowest", [True, False])
def test_binning_dummy_pipeline_replays_on_the_other_engine(
    kind: str, edges: list[float], include_lowest: bool
) -> None:
    """Every fitted range category must still activate its dummy on another engine."""
    pipeline = FeatureEngineer(
        steps_config=[
            {
                "name": "bin",
                "transformer": "CustomBinning",
                "params": {
                    "columns": ["v"],
                    "bins": edges,
                    "label_format": "range",
                    "include_lowest": include_lowest,
                    "drop_original": True,
                },
            },
            {"name": "encode", "transformer": "DummyEncoder", "params": {"columns": ["v_binned"]}},
        ]
    )
    values = {"v": [1.0, 5.0, 6.0, 10.0]}
    fitted, _ = pipeline.fit_transform(_frame(kind, values))
    pipeline = pickle.loads(pickle.dumps(pipeline))
    other_kind = (
        kind.replace("pandas", "polars")
        if kind.startswith("pandas")
        else kind.replace("polars", "pandas")
    )
    replayed = pipeline.transform(_frame(other_kind, values))
    expected = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
    np.testing.assert_array_equal(_pandas(fitted).to_numpy(), expected)
    np.testing.assert_array_equal(_pandas(replayed).to_numpy(), expected)
    assert list(_pandas(replayed).columns) == list(_pandas(fitted).columns)


@pytest.mark.parametrize("include_lowest", [True, False])
def test_legacy_range_artifacts_keep_their_fitted_category_keys(
    kind: str, include_lowest: bool
) -> None:
    """Updating binning must not invalidate existing downstream dummy artifacts."""
    frame = _frame(kind, {"v": [1.0, 8.0]})
    if kind.startswith("pandas") and include_lowest:
        labels = ["[0, 5.0]", "[5.0, 10.0]"]
    else:
        bracket = "[" if include_lowest else "("
        labels = [f"{bracket}0, 5]", f"{bracket}5, 10]"]
    artifact = {
        "bin_edges": {"v": [0, 5, 10]},
        "label_format": "range",
        "include_lowest": include_lowest,
        "drop_original": True,
    }
    binned = GeneralBinningApplier().apply(frame, artifact)
    encoded = DummyEncoderApplier().apply(
        binned, {"columns": ["v_binned"], "categories": {"v_binned": labels}}
    )
    assert _pandas(binned)["v_binned"].tolist() == labels
    np.testing.assert_array_equal(_pandas(encoded).to_numpy(), [[1, 0], [0, 1]])


@pytest.mark.parametrize("node", ["general", "custom", "kbins"])
def test_new_range_artifacts_store_engine_independent_keys(kind: str, node: str) -> None:
    """Persisted keys must be deterministic for all calculators sharing the applier."""
    frame = _frame(kind, {"v": [0.0, 2.0, 8.0, 10.0]})
    calculators = {
        "general": GeneralBinningCalculator,
        "custom": CustomBinningCalculator,
        "kbins": KBinsDiscretizerCalculator,
    }
    artifact = calculators[node]().fit(
        frame,
        {
            "columns": ["v"],
            "bins": [0, 5, 10],
            "strategy": "uniform",
            "n_bins": 2,
            "label_format": "range",
        },
    )
    assert artifact["range_labels"]["v"] == ["[0.0, 5.0]", "[5.0, 10.0]"]


@pytest.mark.parametrize("edge_count", [3, 4])
def test_range_precision_cannot_merge_distinct_fitted_bins(kind: str, edge_count: int) -> None:
    """Narrow bins need distinct persisted keys even when display rounding collides."""
    edges = [0, 1e-7, 2e-7, 3e-7][:edge_count]
    values = {"v": [5e-8, 1.5e-7, 2.5e-7][: edge_count - 1]}
    frame = _frame(kind, values)
    artifact = CustomBinningCalculator().fit(
        frame, {"columns": ["v"], "bins": edges, "label_format": "range", "precision": 3}
    )
    binned = _pandas(GeneralBinningApplier().apply(frame, artifact))
    labels = artifact["range_labels"]["v"]
    assert len(set(labels)) == edge_count - 1
    assert binned["v_binned"].tolist() == labels


def test_empty_suffix_can_replace_the_only_column(kind: str) -> None:
    """Dropping every source must keep both replacement values and the row count."""
    frame = _frame(kind, {"v": [1.0, 8.0]})
    artifact = CustomBinningCalculator().fit(
        frame, {"columns": ["v"], "bins": [0, 5, 10], "output_suffix": "", "drop_original": True}
    )
    result = _pandas(GeneralBinningApplier().apply(frame, artifact))
    assert list(result.columns) == ["v"]
    assert result["v"].tolist() == [0, 1]


@pytest.mark.parametrize("node", ["general", "custom", "kbins"])
@pytest.mark.parametrize("inference_only", [False, True])
@pytest.mark.parametrize("drop_original", [False, True])
def test_binning_rejects_retained_output_collisions(
    kind: str, node: str, inference_only: bool, drop_original: bool
) -> None:
    """Selected-column fitting and later inference must both protect retained data."""
    calculators = {
        "general": GeneralBinningCalculator,
        "custom": CustomBinningCalculator,
        "kbins": KBinsDiscretizerCalculator,
    }
    calculator = calculators[node]()
    config = {"columns": ["v"], "bins": [0, 5, 10], "n_bins": 2, "drop_original": drop_original}
    data = {"v": [0.0, 2.0, 8.0, 10.0], "v_binned": [91, 92, 93, 94]}
    frame = _frame(kind, data)
    original = _pandas(frame).copy(deep=True)
    if inference_only:
        artifact = calculator.fit(_frame(kind, {"v": data["v"]}), config)
        with pytest.raises(ValueError, match="collid.*v_binned"):
            GeneralBinningApplier().apply(frame, artifact)
    else:
        with pytest.raises(ValueError, match="collid.*v_binned"):
            calculator.fit(frame, config)
    pd.testing.assert_frame_equal(_pandas(frame), original)


@pytest.mark.parametrize("drop_original", [False, True])
def test_empty_suffix_preserves_the_replacement_or_rejects_retained_source(
    kind: str, drop_original: bool
) -> None:
    """Replacing a dropped source must preserve its bins instead of deleting both."""
    frame = _frame(kind, {"v": [1.0, 8.0], "other": [21, 22]})
    config = {
        "columns": ["v"],
        "bins": [0, 5, 10],
        "output_suffix": "",
        "drop_original": drop_original,
    }
    if not drop_original:
        with pytest.raises(ValueError, match="collid.*v"):
            CustomBinningCalculator().fit(frame, config)
        return
    artifact = CustomBinningCalculator().fit(frame, config)
    result = _pandas(GeneralBinningApplier().apply(frame, artifact))
    assert result["v"].tolist() == [0, 1]
    assert result["other"].tolist() == [21, 22]
    assert _pandas(frame)["v"].tolist() == [1.0, 8.0]


def test_binning_can_reuse_another_dropped_source_name(kind: str) -> None:
    """Each bin must read the original source before any replacement is installed."""
    frame = _frame(kind, {"v": [1.0, 8.0], "v_binned": [9.0, 2.0], "keep": [21, 22]})
    artifact = CustomBinningCalculator().fit(
        frame, {"columns": ["v", "v_binned"], "bins": [0, 5, 10], "drop_original": True}
    )
    result = _pandas(GeneralBinningApplier().apply(frame, artifact))
    assert list(result.columns) == ["keep", "v_binned", "v_binned_binned"]
    assert result["v_binned"].tolist() == [0, 1]
    assert result["v_binned_binned"].tolist() == [1, 0]


@pytest.mark.parametrize("apply_legacy", [False, True])
def test_duplicate_edges_reject_labels_for_a_zero_width_bin(kind: str, apply_legacy: bool) -> None:
    """Ambiguous custom labels must fail visibly before either engine discards them."""
    frame = _frame(kind, {"v": [1.0, 8.0], "other": [21, 22]})
    original = _pandas(frame).copy(deep=True)
    config = {
        "columns": ["v"],
        "strategy": "custom",
        "custom_bins": {"v": [0, 5, 5, 10]},
        "custom_labels": {"v": ["low", "empty", "high"]},
        "drop_original": True,
    }
    with pytest.raises(ValueError, match="v.*custom labels.*unique"):
        if apply_legacy:
            GeneralBinningApplier().apply(frame, {**config, "bin_edges": config["custom_bins"]})
        else:
            GeneralBinningCalculator().fit(frame, config)
    pd.testing.assert_frame_equal(_pandas(frame), original)


@pytest.mark.parametrize("labels", [None, ["low", "high"]])
def test_duplicate_edges_keep_unambiguous_bins_and_custom_labels(kind: str, labels: Any) -> None:
    """Deduplicating edges remains valid when labels describe the distinct intervals."""
    frame = _frame(kind, {"v": [1.0, 8.0]})
    config = {
        "columns": ["v"],
        "strategy": "custom",
        "custom_bins": {"v": [0, 5, 5, 10]},
        "custom_labels": {"v": labels},
    }
    artifact = GeneralBinningCalculator().fit(frame, config)
    result = _pandas(GeneralBinningApplier().apply(frame, artifact))
    assert result["v_binned"].tolist() == (labels or [0, 1])
