"""Constant measurements must not suppress useful numeric EDA diagnostics."""

import numpy as np
import polars as pl
import pytest
from tests.utils.dataset_loader import load_sample_dataset

from skyulf.profiling.analyzer import EDAAnalyzer
from skyulf.profiling.schemas import Alert


@pytest.mark.parametrize("constants_first", [True, False])
def test_causal_cap_keeps_variable_features_before_constants(constants_first: bool) -> None:
    """Schema placement of constants must not consume slots for measurable variables."""
    rng = np.random.default_rng(274)
    variables = {f"feature_{i}": rng.normal(size=200) for i in range(17)}
    variables["outcome"] = rng.normal(size=200)
    constants = {f"constant_{i}": np.full(200, float(i)) for i in range(5)}
    data = constants | variables if constants_first else variables | constants
    analyzer = EDAAnalyzer(pl.DataFrame(data))
    baseline = EDAAnalyzer(pl.DataFrame(variables))

    selected = analyzer._limit_columns_for_pc(analyzer.columns, target_col="outcome")

    assert selected == baseline._limit_columns_for_pc(baseline.columns, target_col="outcome")
    assert len(selected) == 15
    assert "outcome" in selected
    assert all(not column.startswith("constant_") for column in selected)


def test_causal_graph_omits_constants_below_feature_cap() -> None:
    """A small graph still needs valid Fisher-Z inputs and an accurate selection label."""
    pytest.importorskip("causallearn")
    rng = np.random.default_rng(275)
    frame = pl.DataFrame(
        {"x": rng.normal(size=200), "outcome": rng.normal(size=200), "constant": [1.0] * 200}
    )

    graph = EDAAnalyzer(frame)._discover_causal_graph(frame.columns, target_col="outcome")

    assert graph is not None
    assert {node.id for node in graph.nodes} == {"x", "outcome"}
    assert graph.selection_method == "all"


@pytest.mark.parametrize("constant", [[7] * 100, [7] * 90 + [None] * 10, [None] * 100])
@pytest.mark.parametrize("dtype", [pl.Int64, pl.Float64])
def test_public_vif_preserves_variable_diagnostics_with_constant(constant: list, dtype) -> None:
    """Excluded constants must leave real VIF values, warnings, and an omission explanation."""
    x = np.tile([-1.0, -1.0, 1.0, 1.0], 25)
    noise = np.tile([-1.0, 1.0, -1.0, 1.0], 25)
    frame = pl.DataFrame({"x": x, "y": 2 * x + 0.5 * noise}).with_columns(
        pl.Series("constant", constant, dtype=dtype)
    )

    profile = EDAAnalyzer(frame).analyze()

    assert profile.vif == pytest.approx({"x": 17.0, "y": 17.0})
    assert len([alert for alert in profile.alerts if alert.type == "Multicollinearity"]) == 2
    assert any(alert.column == "constant" and "VIF" in alert.message for alert in profile.alerts)


def test_public_vif_explains_insufficient_variable_columns() -> None:
    """A missing VIF result must explain that fewer than two usable features remain."""
    frame = pl.DataFrame({"x": np.arange(30.0), "constant": [1] * 30})

    profile = EDAAnalyzer(frame).analyze()

    assert profile.vif is None
    assert any("VIF" in alert.message and "two" in alert.message for alert in profile.alerts)


def test_vif_with_constant_preserves_customers_sample_diagnostics() -> None:
    """Real missing-value data must retain the same VIF after a constant is appended."""
    frame = load_sample_dataset("customers", engine="polars").select("age", "income")
    baseline = EDAAnalyzer(frame)._calculate_vif(frame.columns)
    frame = frame.with_columns(pl.lit(1).alias("constant"))

    result = EDAAnalyzer(frame)._calculate_vif(frame.columns)

    assert baseline is not None
    assert result == pytest.approx(baseline)


def test_vif_rechecks_variation_after_complete_case_filtering() -> None:
    """A feature made constant by another feature's nulls must not suppress remaining VIF."""
    x = np.tile([-1.0, -1.0, 1.0, 1.0], 25)
    noise = np.tile([-1.0, 1.0, -1.0, 1.0], 25)
    frame = pl.DataFrame(
        {
            "x": x,
            "y": [None] * 20 + (2 * x + 0.5 * noise)[20:].tolist(),
            "becomes_constant": [2.0] * 20 + [1.0] * 80,
        }
    )

    result = EDAAnalyzer(frame)._calculate_vif(frame.columns)

    assert result == pytest.approx({"x": 17.0, "y": 17.0})


def test_public_vif_explains_insufficient_complete_rows() -> None:
    """Usable variable features with too few observations need an explanation for absent VIF."""
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 3.0, 2.0]})

    profile = EDAAnalyzer(frame).analyze()

    assert profile.vif is None
    assert any(
        "VIF" in alert.message and "complete observations" in alert.message
        for alert in profile.alerts
    )


def test_causal_ranking_of_nonfinite_correlations_and_ties_is_deterministic() -> None:
    """Undefined pairwise correlations and equal scores must not depend on schema order."""
    values = np.arange(100.0)
    frame = pl.DataFrame(
        {
            **{f"feature_{i:02}": values for i in range(15)},
            "unpaired": pl.Series([None] * 50 + values[50:].tolist(), dtype=pl.Float64),
            "outcome": pl.Series(values[:50].tolist() + [None] * 50, dtype=pl.Float64),
        }
    )
    forward = EDAAnalyzer(frame)
    reversed_frame = EDAAnalyzer(frame.select(reversed(frame.columns)))

    selected = forward._limit_columns_for_pc(frame.columns, target_col="outcome")

    assert selected == [f"feature_{i:02}" for i in range(14)] + ["outcome"]
    assert reversed_frame._limit_columns_for_pc(reversed_frame.columns, "outcome") == selected


def test_causal_constant_target_uses_variance_selection() -> None:
    """An unusable target must not reserve a graph node or force correlation ranking."""
    pytest.importorskip("causallearn")
    rng = np.random.default_rng(276)
    frame = pl.DataFrame(
        {**{f"x_{i}": rng.normal(size=100) * (i + 1) for i in range(16)}, "target": [1.0] * 100}
    )

    graph = EDAAnalyzer(frame)._discover_causal_graph(frame.columns, target_col="target")

    assert graph is not None
    assert graph.selection_method == "variance"
    assert len(graph.nodes) == 15
    assert "target" not in {node.id for node in graph.nodes}


def test_vif_explains_nonfinite_correlations() -> None:
    """An infinite observation must produce an explicit omission instead of invalid VIF."""
    frame = pl.DataFrame({"x": np.arange(20.0), "y": [float("inf")] + list(range(1, 20))})
    alerts: list[Alert] = []

    with np.errstate(invalid="ignore"):
        result = EDAAnalyzer(frame)._calculate_vif(frame.columns, alerts=alerts)

    assert result is None
    assert len(alerts) == 1
    assert alerts[0].type == "VIF Unavailable"
    assert "do not have finite correlations" in alerts[0].message


def test_vif_inversion_error_retains_residual_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    """An inversion failure must retain accurate VIF without blaming an orthogonal feature."""
    x = np.tile([-1.0, -1.0, 1.0, 1.0], 25)
    noise = np.tile([-1.0, 1.0, -1.0, 1.0], 25)
    frame = pl.DataFrame({"x": x, "y": 2 * x + 0.5 * noise, "independent": x * noise})
    alerts: list[Alert] = []

    def fail_inverse(matrix: np.ndarray) -> np.ndarray:
        """Inject a numerical-library failure while retaining real residual calculations."""
        raise np.linalg.LinAlgError("Numerical inversion failed")

    monkeypatch.setattr(np.linalg, "inv", fail_inverse)

    result = EDAAnalyzer(frame)._calculate_vif(frame.columns, alerts=alerts)

    assert result == pytest.approx({"x": 17.0, "y": 17.0, "independent": 1.0})
    assert alerts == []


def test_causal_empty_candidates_do_not_reintroduce_target() -> None:
    """An empty eligible set must remain empty even when the frame contains a target."""
    analyzer = EDAAnalyzer(pl.DataFrame({"target": np.arange(100.0)}))

    selected = analyzer._limit_columns_for_pc([], target_col="target")

    assert selected == []


@pytest.mark.parametrize("invalid_values", [[1.0] * 100, [None] * 100, [float("inf")] * 100])
def test_causal_selection_omits_undefined_or_zero_variance(invalid_values: list) -> None:
    """Only variable finite measurements may enter causal discovery, even below the cap."""
    frame = pl.DataFrame({"measurement": np.arange(100.0)}).with_columns(
        pl.Series("invalid", invalid_values, dtype=pl.Float64)
    )

    selected = EDAAnalyzer(frame)._limit_columns_for_pc(frame.columns)

    assert selected == ["measurement"]
