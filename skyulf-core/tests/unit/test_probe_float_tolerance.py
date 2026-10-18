"""Probe tolerance applies only to floating outputs, never structure or labels."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.inference._probe_frames import ProbeFailure, assert_same
from skyulf.inference._probe_strategies import run_check


def _frame(engine, values, dtype="float64", name="value"):
    """Build equal typed frames for both supported probe engines."""
    frame = pd.DataFrame({name: pd.Series(values, dtype=dtype)})
    return frame if engine == "pandas" else pl.from_pandas(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "dtype,base,delta", [("float64", 0.5, 2e-16), ("float32", 1e-8, 5e-13), ("Float64", 0.5, 2e-16)]
)
def test_float_noise_is_accepted(engine, dtype, base, delta):
    """Fixed floating tolerances admit numerical noise without changing dtype or null positions."""
    expected = _frame(engine, [base, None], dtype)
    actual = _frame(engine, [base + delta, None], dtype)
    assert assert_same(expected, actual) is None


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_realistic_probability_change_is_rejected(engine):
    """A probability change of one millionth is far outside the approved floating policy."""
    with pytest.raises(ProbeFailure, match="output_mismatch"):
        assert_same(_frame(engine, [0.5]), _frame(engine, [0.500001]))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("name", ["prediction", "threshold_prediction"])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_floating_classification_labels_remain_exact(engine, name, dtype):
    """Floating class labels cannot change merely because they have a numeric dtype."""
    with pytest.raises(ProbeFailure, match="output_mismatch"):
        assert_same(
            _frame(engine, [1.0], dtype, name=name),
            _frame(engine, [1.0 + 1e-13], dtype, name=name),
            exact_columns=(name,),
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("dtype", ["float64", "Float64"])
def test_tiny_input_mutation_remains_exact(engine, dtype):
    """Model and preprocessing input mutation checks must not use output tolerances."""
    with pytest.raises(ProbeFailure, match="input_mutation"):
        assert_same(
            _frame(engine, [0.5], dtype),
            _frame(engine, [0.5 + 1e-13], dtype),
            "input_mutation",
            exact=True,
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "dtype,before,after",
    [
        ("int64", 1_000_000_000, 1_000_000_001),
        ("Int64", 1_000_000_000, 1_000_000_001),
        ("bool", True, False),
        ("string", "normal", "premium"),
    ],
)
def test_nonfloating_values_remain_exact(engine, dtype, before, after):
    """Large integer values must not inherit relative tolerance from dataframe helpers."""
    with pytest.raises(ProbeFailure):
        assert_same(_frame(engine, [before], dtype), _frame(engine, [after], dtype))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("change", ["dtype", "column", "count", "null", "infinity"])
def test_float_tolerance_keeps_schema_and_masks_exact(engine, change):
    """Floating equality cannot conceal changed schemas, row counts, nulls or infinite signs."""
    expected = _frame(engine, [0.5, np.nan, np.inf])
    variants = {
        "dtype": _frame(engine, [0.5, np.nan, np.inf], "float32"),
        "column": _frame(engine, [0.5, np.nan, np.inf], name="other"),
        "count": _frame(engine, [0.5, np.nan]),
        "null": _frame(engine, [np.nan, 0.5, np.inf]),
        "infinity": _frame(engine, [0.5, np.nan, -np.inf]),
    }
    with pytest.raises(ProbeFailure):
        assert_same(expected, variants[change])


def test_polars_null_and_nan_positions_are_distinct():
    """Polars null and IEEE NaN positions must not be conflated by NumPy conversion."""
    with pytest.raises(ProbeFailure):
        assert_same(
            pl.DataFrame({"value": [None, float("nan")]}),
            pl.DataFrame({"value": [float("nan"), None]}),
        )


@pytest.mark.parametrize("change", ["value", "dtype", "name", "order"])
def test_pandas_index_is_exact(change):
    """Even tiny numeric index changes identify different rows and cannot use output tolerance."""
    expected = pd.DataFrame({"value": [0.5, 0.6]}, index=pd.Index([1.0, 2.0], name="row"))
    actual = expected.copy()
    indices = {
        "value": pd.Index([1.0 + 1e-13, 2.0], name="row"),
        "dtype": pd.Index([1, 2], name="row"),
        "name": pd.Index([1.0, 2.0], name="changed"),
        "order": pd.Index([2.0, 1.0], name="row"),
    }
    actual.index = indices[change]
    with pytest.raises(ProbeFailure):
        assert_same(expected, actual)


@pytest.mark.parametrize(
    "check",
    [
        "chunks:1",
        "chunks:2",
        "chunks:7",
        "reverse",
        "index_shift",
        "empty",
        "perturb",
        "fresh_module",
    ],
)
def test_partition_strategies_use_float_output_policy(check):
    """Partitioned and reordered requests retain the approved floating rounding tolerance."""
    frame = pd.DataFrame({"input": np.arange(16, dtype=float)})
    expected = pd.DataFrame({"probability": np.full(16, 0.5)}, index=frame.index)

    def apply(rows):
        """Return an input-independent probability with harmless floating noise."""
        return pd.DataFrame({"probability": np.full(len(rows), 0.5 + 2e-16)}, index=rows.index)

    check_result, _ = run_check(check, apply, frame, expected)
    assert check_result == {"name": check, "status": "passed"}


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("drift_on_call", [1, 2])
def test_repeat_rejects_tiny_floating_drift(engine, drift_on_call):
    """Identical requests must not hide small stateful changes within output tolerance."""
    frame = _frame(engine, [0.5, 0.6])
    calls = 0

    def apply(rows):
        """Expose a tiny output change on either repetition of the same input."""
        nonlocal calls
        calls += 1
        delta = 2e-16 if calls == drift_on_call else 0.0
        return _frame(engine, [0.5 + delta, 0.6])

    result, _ = run_check("repeat", apply, frame, frame)
    assert result == {"name": "repeat", "status": "failed", "reason": "output_mismatch"}


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_repeat_accepts_identical_floating_values(engine):
    """Exact repeated outputs including missing values remain admissible on both engines."""
    frame = _frame(engine, [0.5, np.nan, np.inf])
    result, _ = run_check(
        "repeat", lambda rows: _frame(engine, [0.5, np.nan, np.inf]), frame, frame
    )
    assert result == {"name": "repeat", "status": "passed"}


def test_strategy_keeps_floating_labels_exact():
    """Model label-column policy must reach chunk comparisons, not only the outer API."""
    frame = pd.DataFrame({"input": np.arange(16, dtype=float)})
    expected = pd.DataFrame({"prediction": np.ones(16)}, index=frame.index)

    def apply(rows):
        """Expose a tiny changed floating class label in otherwise identical output."""
        return pd.DataFrame({"prediction": np.full(len(rows), 1.0 + 1e-13)}, index=rows.index)

    result, _ = run_check("chunks:1", apply, frame, expected, exact_columns=("prediction",))
    assert result == {"name": "chunks:1", "status": "failed", "reason": "output_mismatch"}


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("base,delta", [(1.0, 5e-10), (0.0, 5e-13)])
def test_relative_and_absolute_tolerances_are_both_effective(engine, base, delta):
    """The fixed policy admits each stated numeric margin, including near-zero values."""
    assert assert_same(_frame(engine, [base]), _frame(engine, [base + delta])) is None


def test_object_numbers_do_not_receive_float_tolerance():
    """Only floating dtypes qualify; numeric values hidden in object columns stay exact."""
    with pytest.raises(ProbeFailure):
        assert_same(_frame("pandas", [0.5], "object"), _frame("pandas", [0.5 + 1e-13], "object"))


@pytest.mark.parametrize(
    "policy",
    [
        None,
        {},
        {"rtol": 1e-9},
        {"rtol": True, "atol": 1e-12},
        {"rtol": 1e-9, "atol": 1e-12, "extra": 0.0},
        {"rtol": 1e-6, "atol": 1e-12},
    ],
)
def test_diagnostic_policy_cannot_be_resealed_with_missing_or_modified_constants(policy):
    """Training diagnostics must use the same trusted policy as worker evidence."""
    from skyulf.inference.probed_steps import require_comparison_policy

    with pytest.raises(ValueError, match="comparison policy"):
        require_comparison_policy(
            {"report_version": 2, "float_tolerance": policy}, version_key="report_version"
        )
