"""Missing-label behavior through the public binning calculator/applier API."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from skyulf.engines import SkyulfPolarsWrapper
from skyulf.engines.pandas_engine import SkyulfPandasWrapper
from skyulf.preprocessing.bucketing import GeneralBinningApplier, GeneralBinningCalculator

INPUT_KINDS = ["pandas", "pandas_wrapped", "polars", "polars_wrapped"]


def _frame(kind: str, values: dict[str, list[Any]]) -> Any:
    """Build native and wrapped inputs without converting NaN to Polars null."""
    if kind.startswith("pandas"):
        frame = pd.DataFrame(values)
        return SkyulfPandasWrapper(frame) if kind.endswith("wrapped") else frame
    frame = pl.DataFrame(values)
    return SkyulfPolarsWrapper(frame) if kind.endswith("wrapped") else frame


def _native(frame: Any) -> Any:
    """Inspect public wrapper outputs through their native accessor."""
    return frame.to_native() if hasattr(frame, "to_native") else frame


@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("include_lowest", [True, False])
@pytest.mark.parametrize(
    ("label_format", "custom_labels", "expected_bins"),
    [
        ("ordinal", None, [0, 0, 1, 1]),
        ("bin_index", None, [0, 0, 1, 1]),
        ("range", None, ["[0.0, 5.0]", "[0.0, 5.0]", "[5.0, 10.0]", "[5.0, 10.0]"]),
        ("ordinal", ["low", "high"], ["low", "low", "high", "high"]),
    ],
    ids=["ordinal", "bin_index", "range", "custom"],
)
def test_missing_label_tags_null_nan_and_heldout_values(
    kind: str,
    include_lowest: bool,
    label_format: str,
    custom_labels: list[str] | None,
    expected_bins: list[Any],
) -> None:
    """Heldout gaps must receive the configured label without changing data or selection."""
    training = _frame(kind, {"x": [0.0, 2.0, 8.0, 10.0], "untouched": [20, 30, 40, 50]})
    values = [-1.0, 0.0, 2.0, 5.0, 8.0, 10.0, 11.0, None, float("nan")]
    heldout = _frame(kind, {"x": values, "untouched": list(range(len(values)))})
    original = _native(heldout)
    snapshot = original.copy(deep=True) if kind.startswith("pandas") else original.clone()
    target = np.arange(len(values))
    config: dict[str, Any] = {
        "columns": ["x"],
        "strategy": "equal_width",
        "n_bins": 2,
        "missing_strategy": "label",
        "missing_label": "MISSING_TAG",
        "label_format": label_format,
        "include_lowest": include_lowest,
        "output_suffix": "_tagged",
    }
    if custom_labels is not None:
        config.update(
            strategy="custom",
            custom_bins={"x": [0.0, 5.0, 10.0]},
            custom_labels={"x": custom_labels},
        )

    artifact = GeneralBinningCalculator().fit(training, config)
    output, output_target = GeneralBinningApplier().apply((heldout, target), artifact)
    result = _native(output)

    expected = expected_bins.copy()
    if label_format == "range" and not include_lowest:
        expected = [label.replace("[", "(") for label in expected]
    if kind.startswith("polars"):
        expected = [str(label) for label in expected]
        assert result["x_tagged"].dtype == pl.String
        assert isinstance(output, SkyulfPolarsWrapper) == kind.endswith("wrapped")
        assert_frame_equal(_native(heldout), snapshot)
        assert_frame_equal(result.select("x", "untouched"), snapshot)
    else:
        pd.testing.assert_frame_equal(_native(heldout), snapshot)
        pd.testing.assert_frame_equal(result[["x", "untouched"]], snapshot)
        if custom_labels is None and label_format in ("ordinal", "bin_index"):
            assert result["x_tagged"].dtype == object
            assert isinstance(result["x_tagged"].iloc[2], float)
    first_edge = expected[0] if include_lowest else "MISSING_TAG"
    assert list(result.columns) == ["x", "untouched", "x_tagged"]
    assert output_target is target
    assert result["x_tagged"].to_list() == [
        "MISSING_TAG",
        first_edge,
        *expected,
        "MISSING_TAG",
        "MISSING_TAG",
        "MISSING_TAG",
    ]


@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("label_format", ["ordinal", "bin_index", "range"])
def test_keep_strategy_preserves_missing_values(kind: str, label_format: str) -> None:
    """Enabling label support must leave the existing keep strategy unchanged."""
    training = _frame(kind, {"x": [0.0, 5.0, 10.0]})
    heldout = _frame(kind, {"x": [-1.0, 2.0, 8.0, 11.0, None, float("nan")]})
    config = {
        "columns": ["x"],
        "strategy": "equal_width",
        "n_bins": 2,
        "label_format": label_format,
        "missing_strategy": "keep",
        "missing_label": "UNUSED",
    }
    artifact = GeneralBinningCalculator().fit(training, config)
    result = _native(GeneralBinningApplier().apply(heldout, artifact))
    values = result["x_binned"].to_list()
    assert all(pd.isna(values[i]) for i in (0, 3, 4, 5))
    if label_format == "range":
        assert values[1:3] == ["[0.0, 5.0]", "[5.0, 10.0]"]
    else:
        assert values[1:3] == [0, 1]


@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("missing_label", [None, "", "low"])
def test_default_empty_and_existing_custom_missing_labels(
    kind: str, missing_label: str | None
) -> None:
    """Default, empty, and already-used labels must fill every missing custom bin."""
    training = _frame(kind, {"x": [0.0, 5.0, 10.0]})
    heldout = _frame(kind, {"x": [2.0, 8.0, None, float("nan")]})
    config: dict[str, Any] = {
        "columns": ["x"],
        "strategy": "custom",
        "custom_bins": {"x": [0.0, 5.0, 10.0]},
        "custom_labels": {"x": ["low", "high"]},
        "missing_strategy": "label",
    }
    if missing_label is not None:
        config["missing_label"] = missing_label
    artifact = GeneralBinningCalculator().fit(training, config)
    result = _native(GeneralBinningApplier().apply(heldout, artifact))
    expected_label = "Missing" if missing_label is None else missing_label
    assert result["x_binned"].to_list() == ["low", "high", expected_label, expected_label]


@pytest.mark.parametrize("kind", INPUT_KINDS)
@pytest.mark.parametrize("drop_original", [False, True])
def test_label_mode_preserves_empty_selection_and_only_drops_binned_columns(
    kind: str, drop_original: bool
) -> None:
    """Missing labels must not cause unselected or unfitted columns to be removed."""
    frame = _frame(kind, {"x": [0.0, 5.0, 10.0], "empty": [None] * 3, "other": [1, 2, 3]})
    config = {
        "columns": [],
        "missing_strategy": "label",
        "drop_original": drop_original,
        "n_bins": 2,
    }
    calculator = GeneralBinningCalculator()
    applier = GeneralBinningApplier()
    artifact = calculator.fit(frame, config)
    unchanged = _native(applier.apply(frame, artifact))
    assert list(unchanged.columns) == ["x", "empty", "other"]

    config["columns"] = ["x", "empty", "absent"]
    artifact = calculator.fit(frame, config)
    result = _native(applier.apply(frame, artifact))
    expected_columns = ["empty", "other", "x_binned"]
    if not drop_original:
        expected_columns.insert(0, "x")
    assert result["empty"].to_list() == [None] * 3
    assert result["other"].to_list() == [1, 2, 3]
    assert list(result.columns) == expected_columns


@pytest.mark.parametrize("label_format", ["ordinal", "bin_index", "range"])
def test_polars_label_mode_keeps_string_dtype_across_batches(label_format: str) -> None:
    """A labeled column needs one schema for populated, missing, and empty batches."""
    training = pl.DataFrame({"x": [0.0, 5.0, 10.0]})
    artifact = GeneralBinningCalculator().fit(
        training,
        {
            "columns": ["x"],
            "n_bins": 2,
            "label_format": label_format,
            "missing_strategy": "label",
        },
    )
    batches = []
    for values in ([2.0, 8.0], [None, float("nan")], []):
        frame = pl.DataFrame({"x": pl.Series(values, dtype=pl.Float64)})
        batches.append(GeneralBinningApplier().apply(frame, artifact))
    combined = pl.concat(batches)
    expected_bins = ["[0.0, 5.0]", "[5.0, 10.0]"] if label_format == "range" else ["0", "1"]
    assert all(batch["x_binned"].dtype == pl.String for batch in batches)
    assert combined["x_binned"].to_list() == [*expected_bins, "Missing", "Missing"]
