"""Public EDA regressions for target inference and integer encoding advice."""

import numpy as np
import polars as pl
import pytest

from skyulf.profiling.analyzer import EDAAnalyzer


@pytest.mark.parametrize("row_count", [20, 40, 100])
@pytest.mark.parametrize("dtype", [pl.Boolean, pl.Int64, pl.UInt8])
@pytest.mark.parametrize("balanced", [False, True])
def test_binary_target_advice_does_not_depend_on_frame_size(
    row_count: int, dtype: pl.DataType, balanced: bool
) -> None:
    """Boolean and integer class labels must select a classifier and retain balance advice."""
    positive_count = row_count // 2 if balanced else row_count // 20
    labels = [0] * (row_count - positive_count) + [1] * positive_count
    frame = pl.DataFrame(
        {"target": pl.Series(labels).cast(dtype), "measurement": np.arange(row_count) / 3}
    )

    profile = EDAAnalyzer(frame).analyze(target_col="target")

    assert profile.task_type == "Classification"
    assert profile.rule_tree is not None
    assert len(profile.rule_tree.nodes[0].value) == 2
    assert profile.causal_target_exclusion_reason == "categorical"
    assert profile.correlations_with_target is None
    balance = [rec for rec in profile.recommendations if rec.reason.endswith(" Target")]
    assert len(balance) == 1
    assert balance[0].action == ("Info" if balanced else "Resample")
    assert ("1.00" if balanced else "0.05") in balance[0].suggestion


def test_binary_target_counts_ignore_missing_labels_and_preserve_feature_typing() -> None:
    """Target inference must not reclassify ordinary binary features or count null as a class."""
    frame = pl.DataFrame({"target": [0] * 19 + [1, None], "feature": [0, 1] * 10 + [0]})

    profile = EDAAnalyzer(frame).analyze(target_col="target")

    assert profile.task_type == "Classification"
    assert profile.columns["feature"].dtype == "Numeric"
    balance = [rec for rec in profile.recommendations if rec.action == "Resample"]
    assert len(balance) == 1
    assert "Ratio: 0.05" in balance[0].suggestion


@pytest.mark.parametrize("row_count", [20, 100])
def test_explicit_regression_overrides_binary_target_balance_advice(row_count: int) -> None:
    """An explicitly selected regression task must not recommend class resampling."""
    frame = pl.DataFrame(
        {"target": [0] * (row_count - 1) + [1], "measurement": np.arange(row_count) / 3}
    )

    profile = EDAAnalyzer(frame).analyze(target_col="target", task_type="Regression")

    assert profile.task_type == "Regression"
    assert profile.rule_tree is not None
    assert len(profile.rule_tree.nodes[0].value) == 1
    assert not any(rec.reason.endswith(" Target") for rec in profile.recommendations)
    assert profile.causal_target_exclusion_reason is None


@pytest.mark.parametrize("values", [[0.25, 0.75] * 10, list(range(20))])
def test_numeric_measurement_targets_keep_regression(values: list[float]) -> None:
    """Repeated fractional measurements and nonbinary integer measurements remain regression."""
    frame = pl.DataFrame({"target": values, "measurement": np.arange(len(values)) / 3})

    profile = EDAAnalyzer(frame).analyze(target_col="target")

    assert profile.task_type == "Regression"
    assert not any(rec.reason.endswith(" Target") for rec in profile.recommendations)


def test_repeated_integer_codes_receive_conditional_encoding_advice() -> None:
    """The integer encoding branch must remain reachable without calling measurements categorical."""
    frame = pl.DataFrame({"code": list(range(200)) * 5})

    profile = EDAAnalyzer(frame).analyze()

    assert profile.columns["code"].dtype == "Numeric"
    encodings = [rec for rec in profile.recommendations if rec.action == "Encode"]
    assert len(encodings) == 1
    assert encodings[0].column == "code"
    assert "200" in encodings[0].reason
    assert "if" in encodings[0].suggestion.lower()
    assert "category codes" in encodings[0].suggestion.lower()


@pytest.mark.parametrize("case", ["float", "unique", "few", "regression_target"])
def test_integer_encoding_advice_does_not_overreach(case: str) -> None:
    """Continuous floats, unique IDs, low cardinality and regression targets need no code encoding."""
    values = {
        "float": [value / 2 for value in range(200)] * 5,
        "unique": list(range(1000)),
        "few": list(range(50)) * 20,
        "regression_target": list(range(200)) * 5,
    }[case]
    frame = pl.DataFrame({"measurement": values})

    profile = EDAAnalyzer(frame).analyze(
        target_col="measurement" if case == "regression_target" else None,
        task_type="Regression" if case == "regression_target" else None,
    )

    assert not any(rec.action == "Encode" for rec in profile.recommendations)


@pytest.mark.parametrize("labels", [[1] * 19 + [2], [2] * 19 + [1]])
def test_numeric_class_labels_do_not_receive_numeric_transformation_advice(
    labels: list[int],
) -> None:
    """Resolved class labels need balance advice and categorical analytics without numeric transforms."""
    frame = pl.DataFrame({"target": labels, "measurement": np.arange(20) / 3})

    profile = EDAAnalyzer(frame).analyze(target_col="target")

    assert profile.task_type == "Classification"
    assert profile.rule_tree is not None
    assert len(profile.rule_tree.nodes[0].value) == 2
    assert profile.target_correlations is not None
    assert profile.target_correlations["measurement"] == pytest.approx(np.sqrt(1 / 7))
    target_recs = [rec for rec in profile.recommendations if rec.column == "target"]
    assert {rec.action for rec in target_recs} == {"Resample"}


def test_explicit_regression_target_keeps_applicable_transform_advice() -> None:
    """Numeric measurements still receive the existing domain-aware skewness recommendation."""
    frame = pl.DataFrame({"target": [1] * 19 + [2], "measurement": np.arange(20) / 3})

    profile = EDAAnalyzer(frame).analyze(target_col="target", task_type="Regression")

    target_recs = [rec for rec in profile.recommendations if rec.column == "target"]
    assert {rec.action for rec in target_recs} == {"Transform"}
    assert "Log or Box-Cox" in target_recs[0].suggestion


def test_unsupported_integer_statistics_do_not_emit_unqualified_encoding_advice() -> None:
    """Unknown Int128 statistics must not silently become categorical through code advice."""
    frame = pl.DataFrame({"code": pl.Series(list(range(200)) * 5, dtype=pl.Int128)})

    profile = EDAAnalyzer(frame).analyze()

    assert profile.columns["code"].dtype == "Unknown"
    assert not any(rec.action == "Encode" for rec in profile.recommendations)


def test_unsupported_integer_target_does_not_claim_new_classification_support() -> None:
    """The binary override must stay within the existing supported integer profile dtypes."""
    frame = pl.DataFrame(
        {"target": pl.Series([0, 1] * 10, dtype=pl.Int128), "measurement": np.arange(20) / 3}
    )

    profile = EDAAnalyzer(frame).analyze(target_col="target")

    assert profile.columns["target"].dtype == "Unknown"
    assert profile.task_type is None
    assert not any(rec.reason.endswith(" Target") for rec in profile.recommendations)
