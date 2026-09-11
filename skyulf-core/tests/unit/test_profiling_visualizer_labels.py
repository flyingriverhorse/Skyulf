"""PCA points stay visible and categorical colors remain reproducible."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from skyulf.profiling.schemas import DatasetProfile, PCAPoint
from skyulf.profiling.visualizer import EDAVisualizer


@pytest.mark.parametrize(
    "labels",
    [
        ["-1", None, "2"],
        ["cat", None, "dog"],
        ["2", None, "group"],
        ["", None, "group"],
        ["nan", None, "1"],
        ["inf", None, "2"],
        ["-inf", None, "2"],
        ["1e309", None, "2"],
        [None, None, None],
        ["-1", "0", "2"],
        ["cat", "dog", "cat"],
    ],
)
def test_pca_renders_every_point_when_labels_are_missing(
    labels: list[str | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Optional target labels must never hide or misalign valid PCA coordinates."""
    profile = DatasetProfile(
        row_count=3,
        column_count=2,
        duplicate_rows=0,
        missing_cells_percentage=0.0,
        memory_usage_mb=0.0,
        columns={},
        pca_data=[PCAPoint(x=i, y=i + 0.5, label=label) for i, label in enumerate(labels)],
    )
    previous_figures = set(plt.get_fignums())
    shown_figures = []

    def capture_plot() -> None:
        """Retain the real figure for assertions after public plot cleanup."""
        shown_figures.append(plt.gcf())

    monkeypatch.setattr(plt, "show", capture_plot)

    try:
        EDAVisualizer(profile).plot()
        assert set(plt.get_fignums()) == previous_figures
        assert len(shown_figures) == 1
        figure = shown_figures[0]
        figure.canvas.draw()
        axes = figure.axes[0]
        assert len(figure.axes) == (1 if all(label is None for label in labels) else 2)
        coordinates = []
        for collection in axes.collections:
            offsets = np.ma.asarray(collection.get_offsets())
            assert not np.ma.getmaskarray(offsets).any()
            alpha = collection.get_alpha()
            assert alpha is None or alpha > 0
            coordinates.extend(offsets.tolist())
        assert sorted(coordinates) == [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]
        if None in labels:
            legend = axes.get_legend()
            assert legend is not None
            assert "Unlabeled" in [text.get_text() for text in legend.get_texts()]
            unlabeled = next(c for c in axes.collections if c.get_label() == "Unlabeled")
            assert np.asarray(unlabeled.get_offsets()).shape[0] == labels.count(None)
            assert unlabeled.get_array() is None
    finally:
        for figure_number in set(plt.get_fignums()) - previous_figures:
            plt.close(figure_number)


@pytest.mark.parametrize("helper_name", ["_pca_color_values", "_geospatial_color_values"])
def test_categorical_colors_are_stable_when_points_are_reordered(helper_name: str) -> None:
    """Shared categorical colors must retain the same label meaning after row reordering."""
    labels = ["zulu", "alpha", "middle", "alpha"]
    helper = getattr(EDAVisualizer, helper_name)

    colors = helper(labels)
    reversed_colors = helper(list(reversed(labels)))

    assert colors == [2, 0, 1, 0]
    assert reversed_colors == [0, 1, 0, 2]


def test_finite_pca_labels_preserve_the_numeric_color_scale() -> None:
    """Numeric targets must retain their values instead of becoming category indices."""
    assert EDAVisualizer._pca_color_values(["-1", "0.5", "2"]) == [-1.0, 0.5, 2.0]


def test_categorical_color_mapping_is_stable_across_python_hash_seeds() -> None:
    """Independent processes must render each category using the same color code."""
    script = """
import json
from skyulf.profiling.visualizer import EDAVisualizer

labels = ["zulu", "alpha", "middle", "alpha", None]
print(json.dumps(EDAVisualizer._label_color_map(labels), sort_keys=True))
"""
    mappings = []
    for seed in ("0", "1"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[2],
            env={**os.environ, "PYTHONHASHSEED": seed},
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        mappings.append(json.loads(result.stdout))

    assert mappings[0] == mappings[1] == {"alpha": 0, "middle": 1, "zulu": 2}
