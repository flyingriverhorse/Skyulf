"""Integer casts keep exact scalar identities when neighboring rows need float parsing."""

import pickle

import numpy as np
import pandas as pd
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, save_pipeline
from skyulf.inference.preprocessing_probe import probe_fitted_preprocessing
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.casting import CastingApplier, CastingCalculator


@pytest.mark.parametrize("source_dtype", ["object", "string", "string[pyarrow]"])
@pytest.mark.parametrize("coerce", [True, False])
@pytest.mark.parametrize(
    "values,target,expected",
    [
        ([2**53 + 1, "1.0", None], "Int64", [2**53 + 1, 1, None]),
        ([str(2**53 + 1), "1e0", None], "Int64", [2**53 + 1, 1, None]),
        ([str(2**64 - 1), "1.0", None], "UInt64", [2**64 - 1, 1, None]),
    ],
)
def test_decimal_peer_does_not_round_exact_integer_rows(
    source_dtype, coerce, values, target, expected
):
    """An unrelated decimal-form value must not corrupt an exact integer during replay."""
    if "pyarrow" in source_dtype:
        pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"x": pd.Series(values, dtype=source_dtype), "keep": [3, 4, 5]})
    frame.index = [8, 3, 3]
    before = frame.copy(deep=True)
    state = CastingCalculator().fit(
        frame, {"column_types": {"x": target}, "coerce_on_error": coerce}
    )
    saved = pickle.dumps(state)
    expected_frame = frame.copy()
    expected_frame["x"] = pd.array(expected, dtype=target)
    applier = CastingApplier()
    for selection in (slice(None), slice(0, 1), slice(1, 2), slice(2, 3), slice(0, 0)):
        actual = applier.apply(frame.iloc[selection], state)
        pd.testing.assert_frame_equal(actual, expected_frame.iloc[selection])
    pd.testing.assert_frame_equal(frame, before)
    assert pickle.dumps(state) == saved


@pytest.mark.parametrize(
    "values,target,expected",
    [
        ([2**53 + 1, "1.5", None], "Int64", [2**53 + 1, None, None]),
        ([str(2**64 - 1), "-1", "1.0"], "UInt64", [2**64 - 1, None, 1]),
        ([str(2**63), "-1.0", str(2**53 + 1)], "Int64", [None, -1, 2**53 + 1]),
        ([str(2**53 + 1), "bad", "1.0"], "Int64", [2**53 + 1, None, 1]),
        ([str(2**53 + 1), [1, 2], "1.0"], "Int64", [2**53 + 1, None, 1]),
        ([str(2**53 + 1), "inf", "1.0"], "Int64", [2**53 + 1, None, 1]),
        ([str(2**53 + 1), "9.223372036854776e18", "1.0"], "Int64", [2**53 + 1, None, 1]),
        ([str(2**53 + 1), "1.8446744073709552e19", "1.0"], "UInt64", [2**53 + 1, None, 1]),
    ],
)
def test_coercion_preserves_good_integers_beside_invalid_rows(values, target, expected):
    """Fractional, invalid and out-of-range neighbors must become null without damaging valid rows."""
    frame = pd.DataFrame({"x": pd.Series(values, dtype=object)})
    before = frame.copy(deep=True)
    state = CastingCalculator().fit(frame, {"columns": ["x"], "target_type": target})
    result = CastingApplier().apply(frame, state)
    pd.testing.assert_frame_equal(result, pd.DataFrame({"x": pd.array(expected, dtype=target)}))
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize(
    "values,error,match",
    [
        ([2**53 + 1, "1.5"], ValueError, "fractional"),
        ([2**53 + 1, "1.5", "bad"], ValueError, "Unable to parse"),
        ([str(2**63), "1.0"], OverflowError, "out of range"),
    ],
)
def test_strict_mixed_integer_cast_retains_error_priority(values, error, match):
    """Strict casts must retain parse-before-fraction-before-range validation."""
    frame = pd.DataFrame({"x": pd.Series(values, dtype=object)})
    state = CastingCalculator().fit(
        frame, {"columns": ["x"], "target_type": "Int64", "coerce_on_error": False}
    )
    with pytest.raises(error, match=match):
        CastingApplier().apply(frame, state)


@pytest.mark.parametrize("target", ["Int64", "UInt64"])
def test_saved_integer_cast_replays_mixed_requests_without_refitting(target, tmp_path, monkeypatch):
    """Saved models must preserve exact preprocessing through the real context diagnostic."""
    training = pd.DataFrame({"x": [str(i) for i in range(8)], "target": range(8)})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "cast",
                    "transformer": "Casting",
                    "params": {"column_types": {"x": target}},
                }
            ],
            "modeling": {
                "type": "random_forest_regressor",
                "params": {"n_estimators": 2, "max_depth": 2, "random_state": 42, "n_jobs": 1},
            },
        }
    )
    pipeline.fit(
        SplitDataset(train=training.iloc[2:], test=training.iloc[:2]), target_column="target"
    )
    directory = tmp_path / "model"
    save_pipeline(pipeline, directory)

    def forbidden(*args, **kwargs):
        """Loading and scoring saved casting state must never refit it."""
        raise AssertionError("Unexpected fit")

    monkeypatch.setattr(CastingCalculator, "fit", forbidden)
    artifact = load_pipeline(directory)
    request = pd.DataFrame({"x": [str(2**53 + 1), "1.0", "1.5", None]})
    report = probe_fitted_preprocessing(artifact, request, chunk_sizes=(1, 2))
    assert report["status"] == "passed", report
    assert report["steps"][0]["context"] == "row", report
    full = artifact.pipeline.predict(request)
    singletons = np.concatenate(
        [artifact.pipeline.predict(request.iloc[i : i + 1]) for i in range(len(request))]
    )
    assert len(full) == len(request)
    np.testing.assert_array_equal(full, singletons)
