"""Unit tests for MergeMixin's Polars paths.

The integration suite (``tests/integration/test_merge_everywhere.py``)
exercises merging end-to-end through the engine; these unit tests pin the
engine-boundary behaviour added by the backend Polars migration that is
easy to lose silently:

- ``(X, y)`` tuple coercion and merging keep working when the frames are
  Polars, and the merged result is converted back to Polars so downstream
  nodes keep receiving the configured engine's frame type.
- An empty merged test split defaults to an empty frame of the *configured*
  engine (Polars), not a hard-coded pandas frame.
- Column stripping is a no-op on frames that don't carry the columns, and
  drops in-place on Polars frames.
"""

from types import SimpleNamespace

import pandas as pd
import polars as pl
import pytest

from backend.config import get_settings
from backend.ml_pipeline._execution.engine._merge import MergeMixin
from backend.ml_pipeline._execution.schemas import NodeConfig
from skyulf.data.dataset import SplitDataset


class _Merger(MergeMixin):
    """Minimal harness: no graph, default merge strategy, captured logs."""

    def __init__(self) -> None:
        self._node_configs: dict = {}
        self.logs: list[str] = []

    def log(self, msg: str) -> None:
        self.logs.append(msg)


def test_coerce_tuple_to_frame_polars_reattaches_target() -> None:
    """A Polars ``(X, y)`` payload must coerce to a frame with the target
    reattached — returning None here silently drops training data.
    """
    X = pl.DataFrame({"f": [1, 2]})
    out = _Merger()._coerce_tuple_to_frame((X, [0, 1]), target_col="target")
    assert isinstance(out, pl.DataFrame)
    assert out["f"].to_list() == [1, 2]
    assert out["target"].to_list() == [0, 1]


def test_merge_xy_tuples_polars_returns_polars_frame(monkeypatch) -> None:
    """Merging Polars X parts must yield a Polars merged X (engine round-trip),
    keeping y from the first edge.
    """
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
    y = pl.Series([0, 1])
    artifacts = [
        (pl.DataFrame({"a": [1, 2]}), y),
        (pl.DataFrame({"b": [3.0, 4.0]}), y),
    ]
    node = SimpleNamespace(node_id="m1")

    merged_x, merged_y = _Merger()._merge_xy_tuples(node, artifacts)

    assert isinstance(merged_x, pl.DataFrame)
    assert sorted(merged_x.columns) == ["a", "b"]
    assert merged_x.height == 2
    assert merged_y is y


def test_merge_split_datasets_defaults_empty_test_to_polars_frame(monkeypatch) -> None:
    """When every branch's test split is empty, the merged test must default
    to an empty frame of the configured engine — hard-coding pandas here
    would hand downstream nodes a foreign frame type under SKYULF_ENGINE=polars.
    """
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
    sd1 = SplitDataset(train=pl.DataFrame({"a": [1, 2]}), test=pl.DataFrame(), validation=None)
    sd2 = SplitDataset(train=pl.DataFrame({"b": [3, 4]}), test=pl.DataFrame(), validation=None)
    node = SimpleNamespace(node_id="m1")

    out = _Merger()._merge_split_datasets(node, [sd1, sd2], target_col="")

    assert isinstance(out.train, pl.DataFrame)
    assert sorted(out.train.columns) == ["a", "b"]
    assert isinstance(out.test, pl.DataFrame)
    assert out.test.is_empty()


def test_merge_split_datasets_defaults_empty_test_to_pandas_frame(monkeypatch) -> None:
    """The same default stays pandas when the configured engine is pandas."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas", raising=False)
    sd1 = SplitDataset(train=pd.DataFrame({"a": [1, 2]}), test=pd.DataFrame(), validation=None)
    sd2 = SplitDataset(train=pd.DataFrame({"b": [3, 4]}), test=pd.DataFrame(), validation=None)
    node = SimpleNamespace(node_id="m1")

    out = _Merger()._merge_split_datasets(node, [sd1, sd2], target_col="")

    assert isinstance(out.test, pd.DataFrame)
    assert out.test.empty


def test_strip_columns_polars() -> None:
    """Column stripping must no-op when nothing matches and drop in-place on
    Polars frames (``df.drop(columns=...)`` is pandas-only API).
    """
    merger = _Merger()
    df = pl.DataFrame({"a": [1], "b": [2]})

    assert merger._strip_columns(df, ["nonexistent"]) is df

    stripped = merger._strip_columns(df, ["a"])
    assert isinstance(stripped, pl.DataFrame)
    assert stripped.columns == ["b"]


def test_strip_columns_tuple_payload(monkeypatch) -> None:
    """An ``(X, y)`` tuple payload keeps its shape; only X loses the columns."""
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
    merger = _Merger()
    X = pl.DataFrame({"a": [1], "b": [2]})
    y = pl.Series([0])

    (stripped_x, stripped_y) = merger._strip_columns((X, y), ["a"])

    assert stripped_x.columns == ["b"]
    assert stripped_y is y


def test_merge_split_dataset_xy_part_pandas_with_empty_branch(monkeypatch) -> None:
    """Pandas ``(X, y)`` parts merge column-wise, and a branch whose X is
    empty is skipped rather than failing the whole merge.
    """
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "pandas", raising=False)
    y = pd.Series([0, 1])
    non_empty = [
        (pd.DataFrame({"a": [1, 2]}), y),
        (pd.DataFrame(), y),  # empty X branch must be dropped, not fatal
        (pd.DataFrame({"b": [3, 4]}), y),
    ]
    node = SimpleNamespace(node_id="m1")

    merged_x, merged_y = _Merger()._merge_split_dataset_xy_part(node, "train", non_empty)

    assert isinstance(merged_x, pd.DataFrame)
    assert sorted(merged_x.columns) == ["a", "b"]
    assert merged_y is y


def test_merge_fallback_frames_raises_on_empty_input(monkeypatch) -> None:
    """An empty flattened input must fail loudly with actionable guidance,
    not merge into a silently degraded frame.
    """
    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", "polars", raising=False)
    node = SimpleNamespace(node_id="m1")

    with pytest.raises(ValueError, match="empty DataFrame"):
        _Merger()._merge_fallback_frames(node, [pl.DataFrame()], target_col="")


def _coverage_dataset(engine, *, rows=2, original_rows=3):
    """Represent an already filtered split without sharing caller-owned metadata."""
    frame = pl.DataFrame if engine == "polars" else pd.DataFrame
    return SplitDataset(
        train=frame({"x": [0.0, 1.0, 2.0, 3.0], "target": [0.0, 2.0, 4.0, 6.0]}),
        test=frame(
            {"x": [float(n) for n in range(rows)], "target": [float(2 * n) for n in range(rows)]}
        ),
        evaluation_coverage={
            "test": {
                "input_rows": original_rows,
                "scored_rows": rows,
                "excluded_rows": original_rows - rows,
                "steps": [
                    {
                        "name": "eligible",
                        "input_rows": original_rows,
                        "scored_rows": rows,
                        "excluded_rows": original_rows - rows,
                    }
                ],
            }
        },
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("population", ["matching", "divergent", "rowwise"])
def test_filtered_branch_merge_requires_shared_row_lineage(engine, population):
    """Equal counts cannot certify row identity, and unknown populations cannot be merged safely."""
    first = _coverage_dataset(engine)
    second = _coverage_dataset(
        engine,
        rows=3 if population == "rowwise" else 2,
        original_rows=3 if population == "matching" else 4,
    )
    merger = _Merger()
    merger.merge_warnings = []
    with pytest.raises(ValueError, match="shared row lineage"):
        merger._merge_split_datasets(
            NodeConfig(node_id="merge", step_type="merge", inputs=["left", "right"]),
            [first, second],
            "target",
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_later_filter_preserves_unknown_merge_denominator(engine):
    """A subsequent row filter may measure its own loss but cannot reconstruct original coverage."""
    from skyulf.preprocessing.pipeline import FeatureEngineer

    merged = _coverage_dataset(engine)
    merged.evaluation_coverage["test"].update(
        input_rows=None, excluded_rows=None, reason="Original evaluation population unavailable."
    )
    frame = pl.DataFrame if engine == "polars" else pd.DataFrame
    merged.test = frame({"x": [None, 1.0], "target": [0.0, 2.0]})
    filtered, _ = FeatureEngineer(
        [
            {"name": "null_filter", "transformer": "DropMissingRows", "params": {"subset": ["x"]}},
        ]
    ).fit_transform(merged)
    coverage = filtered.evaluation_coverage["test"]
    assert coverage["input_rows"] is None and coverage["excluded_rows"] is None
    assert coverage["scored_rows"] == 1
    assert coverage["reason"] == merged.evaluation_coverage["test"]["reason"]
    assert coverage["steps"][-1]["excluded_rows"] == 1


def _empty_validation_dataset(engine, paired, population):
    """Keep an explicitly present empty validation payload distinct from an absent split."""
    dataset = _coverage_dataset(engine, rows=0)
    dataset.validation = dataset.test
    dataset.test = dataset.train.clone() if engine == "polars" else dataset.train.copy()
    coverage = dataset.evaluation_coverage.pop("test")
    if population == "unknown":
        coverage.update(input_rows=None, excluded_rows=None, reason="Unknown merged population.")
    if population != "originally_empty":
        dataset.evaluation_coverage["validation"] = coverage
    if paired:
        for name in ("train", "test", "validation"):
            frame = getattr(dataset, name)
            X = frame.drop("target") if engine == "polars" else frame.drop(columns=["target"])
            setattr(dataset, name, (X, frame["target"]))
    return dataset


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("population", ["excluded", "unknown", "originally_empty"])
def test_empty_validation_merge_retains_population_evidence(
    engine, paired, population, monkeypatch
):
    """Fully excluded validation must survive fan-in and remain visible in the public model report."""
    from skyulf.pipeline import SkyulfPipeline

    monkeypatch.setattr(get_settings(), "SKYULF_ENGINE", engine)
    first = _empty_validation_dataset(engine, paired, population)
    merged = _Merger()._merge_split_datasets(
        NodeConfig(node_id="merge", step_type="merge", inputs=["left", "right"]),
        [first, first.copy()],
        "target",
    )
    assert merged.validation is not None
    assert isinstance(merged.validation, tuple) is paired
    frame = merged.validation[0] if paired else merged.validation
    assert isinstance(frame, pl.DataFrame if engine == "polars" else pd.DataFrame)
    assert len(frame) == 0
    report = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "linear_regression"}}).fit(
        merged, "target"
    )["modeling"]
    if population == "originally_empty":
        assert "validation" not in report["splits"]
    else:
        coverage = report["splits"]["validation"].coverage
        assert coverage == first.evaluation_coverage["validation"]
        assert report["splits"]["validation"].metrics == {}
        assert report["raw_data"]["splits"]["validation"]["coverage"] == coverage


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("paired", [False, True])
def test_empty_training_merge_remains_invalid(engine, paired):
    """Retaining empty evaluation payloads must never admit an empty fitting population."""
    first = _empty_validation_dataset(engine, paired, "excluded")
    first.train = first.validation
    with pytest.raises(ValueError, match="empty train splits"):
        _Merger()._merge_split_datasets(
            NodeConfig(node_id="merge", step_type="merge", inputs=["left", "right"]),
            [first, first.copy()],
            "target",
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("split", ["test", "validation"])
def test_different_retained_rows_cannot_merge_with_identical_counts(engine, paired, split):
    """Reset indexes and identical4-to3 counts must not pair unrelated feature and target rows."""
    factory = pd.DataFrame if engine == "pandas" else pl.DataFrame
    first = _coverage_dataset(engine, rows=3, original_rows=4)
    second = first.copy()
    for dataset, column, retained in ((first, "a", [0, 1, 2]), (second, "b", [1, 2, 3])):
        frame = factory({column: retained, "target": [10 * row for row in retained]})
        payload = frame
        if paired:
            X = frame.drop(columns="target") if engine == "pandas" else frame.drop("target")
            payload = (X, frame["target"])
        setattr(dataset, split, payload)
        if split == "validation":
            dataset.evaluation_coverage[split] = dataset.evaluation_coverage.pop("test")
            dataset.test = dataset.train
    assert first.evaluation_coverage == second.evaluation_coverage
    with pytest.raises(ValueError, match=f"filtered {split} branches.*shared row lineage"):
        _Merger()._merge_split_datasets(
            NodeConfig(node_id="merge", step_type="merge", inputs=["left", "right"]),
            [first, second],
            "target",
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("population", ["unknown", "complete", "unrecorded"])
def test_nonempty_branch_merge_requires_known_unfiltered_population(engine, population):
    """Unknown coverage blocks fan-in while unfiltered and legacy unrecorded inputs keep working."""
    first = _coverage_dataset(engine, rows=2, original_rows=2)
    if population == "unknown":
        first.evaluation_coverage["test"].update(input_rows=None, excluded_rows=None)
    elif population == "unrecorded":
        first.evaluation_coverage.clear()
    merger = _Merger()
    node = NodeConfig(node_id="merge", step_type="merge", inputs=["left", "right"])
    if population == "unknown":
        with pytest.raises(ValueError, match="shared row lineage"):
            merger._merge_split_datasets(node, [first, first.copy()], "target")
    else:
        merged = merger._merge_split_datasets(node, [first, first.copy()], "target")
        assert len(merged.test) == 2
        assert merged.evaluation_coverage == first.evaluation_coverage
