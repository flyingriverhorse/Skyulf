"""Sample-aware distribution evidence, independent of practical effect thresholds."""

import math
from collections import Counter
from typing import Literal

import numpy as np
from pydantic import BaseModel, PrivateAttr

SIGNIFICANCE_LEVEL = 0.05


class DriftEvidence(BaseModel):
    """One feature test with family-wise Bonferroni correction and explicit limitations.

    ``supported`` means statistical support independently of the effect sliders.
    ``not_detected`` means no statistical support, not proof of equivalent
    distributions. Counts describe the observations used by this feature test.
    Tests assume independent observations. Correction covers columns within one
    report, not repeated monitoring windows, and does not establish test power.
    """

    status: Literal["supported", "not_detected", "insufficient_data", "unavailable"] = (
        "not_detected"
    )
    test: str
    p_value: float | None = None
    adjusted_p_value: float | None = None
    significance_level: float = SIGNIFICANCE_LEVEL
    reference_count: int
    current_count: int
    reason: str | None = None
    _minimum_p_value: float = PrivateAttr(default=0.0)


def _sample_resolution(reference_count: int, current_count: int) -> float:
    """Return the exact smallest two-sided separation probability for these sample sizes."""
    total = reference_count + current_count
    log_combinations = (
        math.lgamma(total + 1) - math.lgamma(reference_count + 1) - math.lgamma(current_count + 1)
    )
    return min(1.0, math.exp(math.log(2) - log_combinations))


def numeric_evidence(p_value: float, reference_count: int, current_count: int) -> DriftEvidence:
    """Keep the existing exact-order KS result and its attainable sample-size resolution."""
    evidence = DriftEvidence(
        test="ks_2samp",
        p_value=p_value,
        reference_count=reference_count,
        current_count=current_count,
    )
    evidence._minimum_p_value = _sample_resolution(reference_count, current_count)
    return evidence


def _fisher_evidence(table: np.ndarray) -> tuple[float, float]:
    """Compute a two-sided exact p-value and its attainable fixed-margin endpoint minimum."""
    from scipy.stats import fisher_exact  # noqa: PLC0415 - retain optional SciPy import behavior

    n_reference, n_current = map(int, table.sum(axis=1))
    category_count = int(table[:, 0].sum())
    endpoints = (max(0, category_count - n_current), min(n_reference, category_count))
    minimum = min(
        float(
            fisher_exact(
                [
                    [value, n_reference - value],
                    [category_count - value, n_current - category_count + value],
                ]
            ).pvalue
        )
        for value in endpoints
    )
    return float(fisher_exact(table).pvalue), minimum


def _categorical_test(table: np.ndarray) -> tuple[str, float, float]:
    """Use exact binary/sparse tests and chi-square only for adequately populated cells."""
    from scipy.stats import chi2_contingency  # noqa: PLC0415 - optional SciPy boundary

    reference_count, current_count = map(int, table.sum(axis=1))
    if table.shape[1] == 1:
        return "constant_categories", 1.0, _sample_resolution(reference_count, current_count)
    if table.shape[1] == 2:
        p_value, minimum = _fisher_evidence(table)
        return "fisher_exact", p_value, minimum
    expected = np.outer(
        table.sum(axis=1, dtype=np.float64), table.sum(axis=0, dtype=np.float64)
    ) / table.sum(dtype=np.float64)
    if (table >= 5).all() and (expected >= 5).all():
        return "chi_square", float(chi2_contingency(table, correction=False).pvalue), 0.0
    # Equal count pairs share a test value; calculate each once even if the
    # current data contains many newly observed categories. Correction still
    # counts every category hypothesis, not only distinct count pairs.
    tests = [
        _fisher_evidence(
            np.array(
                [[first, reference_count - first], [second, current_count - second]],
                dtype=np.int64,
            )
        )
        for first, second in set(map(tuple, table.T.tolist()))
    ]
    correction = table.shape[1]
    return (
        "fisher_category_bonferroni",
        min(1.0, correction * min(value[0] for value in tests)),
        min(1.0, correction * min(value[1] for value in tests)),
    )


def categorical_evidence(reference: list[str], current: list[str]) -> DriftEvidence:
    """Test category identity on unsmoothed observed counts, independently of PSI smoothing."""
    reference_counts, current_counts = Counter(reference), Counter(current)
    categories = sorted(reference_counts.keys() | current_counts.keys())
    table = np.array(
        [
            [reference_counts[label] for label in categories],
            [current_counts[label] for label in categories],
        ],
        dtype=np.int64,
    )
    test, p_value, minimum = _categorical_test(table)
    evidence = DriftEvidence(
        test=test,
        p_value=p_value,
        reference_count=len(reference),
        current_count=len(current),
    )
    evidence._minimum_p_value = minimum
    return evidence


def _valid_test(evidence: DriftEvidence) -> bool:
    """Admit finite p-values from replicated observations into the tested feature family."""
    return (
        min(evidence.reference_count, evidence.current_count) >= 2
        and evidence.p_value is not None
        and math.isfinite(evidence.p_value)
        and 0 <= evidence.p_value <= 1
    )


def _correct_one(evidence: DriftEvidence, family_size: int) -> None:
    """Correct one test and distinguish unsupported decisions from unresolvable evidence."""
    if min(evidence.reference_count, evidence.current_count) < 2:
        evidence.status = "insufficient_data"
        evidence.reason = (
            "Insufficient data: at least two observed values per population are required."
        )
        return
    if not _valid_test(evidence):
        evidence.status = "unavailable"
        evidence.reason = "The distribution test did not produce a finite probability."
        return
    assert evidence.p_value is not None
    evidence.adjusted_p_value = min(1.0, evidence.p_value * family_size)
    if evidence._minimum_p_value * family_size > SIGNIFICANCE_LEVEL:
        evidence.status = "insufficient_data"
        evidence.reason = (
            "Insufficient data: this test cannot attain the Bonferroni-corrected "
            f"significance level ({SIGNIFICANCE_LEVEL / family_size:.6g}) with these observations."
        )
    elif evidence.adjusted_p_value <= SIGNIFICANCE_LEVEL:
        evidence.status = "supported"
        evidence.reason = None
    else:
        evidence.status = "not_detected"
        evidence.reason = (
            "No statistical support for a distribution change; equivalence is not established."
        )


def correct_evidence(evidence: list[DriftEvidence]) -> None:
    """Correct finite replicated tests, including conservatively retained infeasible tests.

    The KS feasibility bound assumes no ties; ties can further reduce attainable
    resolution. Neither a feasible test nor a nonsignificant result proves power.
    """
    family_size = max(1, sum(_valid_test(item) for item in evidence))
    for item in evidence:
        _correct_one(item, family_size)
