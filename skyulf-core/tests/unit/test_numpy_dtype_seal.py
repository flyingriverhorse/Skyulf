"""Numeric dtype metadata inside genuine sklearn estimators has a canonical seal."""

import pickle

import numpy as np
import pandas as pd
import pytest

from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.imputation.iterative import IterativeImputerCalculator


def test_real_iterative_imputer_state_has_a_stable_semantic_digest():
    """The saved initial imputer's numeric dtype must support semantic fingerprints and probes."""
    state = IterativeImputerCalculator().fit(
        pd.DataFrame({"a": [0.0, 1.0, None, 3.0], "b": [0.0, None, 4.0, 6.0]}),
        {"columns": ["a", "b"]},
    )
    before = artifact_digest(state)
    restored = pickle.loads(pickle.dumps(state))
    assert artifact_digest(restored) == before
    restored["imputer_object"].initial_imputer_.statistics_[0] += 1.0
    assert artifact_digest(restored) != before


def test_numeric_dtype_encoding_retains_width_kind_endian_and_scalar_framing():
    """Different numeric interpretations must never collapse into one dtype identity."""
    dtypes = [np.dtype(name) for name in ["?", "i1", "u1", "<i4", ">i4", "<f4", "<f8", "<c8"]]
    digests = [artifact_digest(dtype) for dtype in dtypes]
    assert len(set(digests)) == len(dtypes)
    assert artifact_digest(np.dtype(float)) == artifact_digest(np.dtype("float64"))
    assert artifact_digest(np.dtype("float64")) != artifact_digest(np.dtype("float64").str)


@pytest.mark.parametrize(
    "dtype",
    [
        np.dtype("f8", metadata={"unit": "m"}),
        np.dtype("f8", metadata={}),
        np.dtype([("value", "f8")]),
        np.dtype(("f8", (2,))),
        np.dtype("O"),
        np.dtype("U3"),
        np.dtype("datetime64[ns]"),
    ],
)
def test_unsupported_dtype_forms_fail_instead_of_losing_metadata(dtype):
    """The new numeric scalar representation must not silently discard other dtype structure."""
    with pytest.raises(TypeError, match="dtype"):
        artifact_digest(dtype)


def test_existing_numeric_array_encoding_remains_unchanged():
    """Adding first-class dtype support must preserve already recorded array identities."""
    value = np.array([[1.5, -0.0], [np.nan, np.inf]], dtype="<f8")
    assert (
        artifact_digest(value).hex()
        == "4050de9236b84c5f41764e36c8652674e34dd6fc8400fa0a6724697a157d4f52"
    )


def test_actual_longdouble_iterative_fit_keeps_numeric_scalar_variant():
    """Windows longdouble metadata must seal without collapsing into float64 aliases."""
    frame = pd.DataFrame(
        {
            "a": np.array([0, 1, np.nan, 3], dtype=np.longdouble),
            "b": np.array([0, np.nan, 4, 6], dtype=np.longdouble),
        }
    )
    state = IterativeImputerCalculator().fit(frame, {"columns": ["a", "b"]})
    before = artifact_digest(state)
    assert artifact_digest(state) == before
    assert len(artifact_digest(pickle.loads(pickle.dumps(state)))) == len(before)
    original = np.dtype(np.longdouble)
    restored = pickle.loads(pickle.dumps(original))
    # NumPy's own pickle normalizes Windows longdouble to float64; do not hide that change.
    if original.type is restored.type:
        assert artifact_digest(original) == artifact_digest(restored)
    else:
        assert restored.type is np.float64
        assert artifact_digest(original) != artifact_digest(restored)
    assert artifact_digest(np.dtype(np.longdouble)) != artifact_digest(np.dtype(np.float64))
    assert artifact_digest(np.dtype(np.clongdouble)) != artifact_digest(np.dtype(np.complex128))
