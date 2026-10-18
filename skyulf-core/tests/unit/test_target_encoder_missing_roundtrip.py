"""Target statistics use stable missing keys across persisted native-engine replay."""

import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import TargetEncoder

from skyulf import SkyulfPipeline
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.encoding.target import TargetEncoderApplier, TargetEncoderCalculator


def _training(kind):
    """Make missing rows predictive while literal missing spellings have different labels."""
    missing = {"none": None, "nan": np.nan, "pdna": pd.NA, "polars": None}[kind]
    values = [missing] * 4 + ["a", "nan", "None", "<NA>"] * 2
    labels = [1] * 4 + [0] * 8
    if kind == "polars":
        return pl.DataFrame({"category": values}), pl.Series("target", labels)
    return pd.DataFrame({"category": values}), pd.Series(labels, name="target")


@pytest.mark.parametrize("kind", ["none", "nan", "pdna", "polars"])
def test_target_missing_keys_survive_save_load_and_cross_engine_batches(kind, tmp_path):
    """CSV NaN, JSON null, nullable pandas and Polars null must use the same learned statistic."""
    frame, labels = _training(kind)
    before = frame.clone() if isinstance(frame, pl.DataFrame) else frame.copy(deep=True)
    state = TargetEncoderCalculator().fit(
        (frame, labels), {"columns": ["category"], "smooth": 0.0, "target_type": "binary"}
    )
    path = tmp_path / "target.pkl"
    path.write_bytes(pickle.dumps(state))
    restored = pickle.loads(path.read_bytes())
    TargetEncoderApplier.validate_inference_state(dict(restored))
    digest = artifact_digest(restored)
    samples = [
        pd.DataFrame({"category": [None, np.nan, pd.NA, "a", "nan", "None", "<NA>", "new"]}),
        pl.DataFrame({"category": [None, None, None, "a", "nan", "None", "<NA>", "new"]}),
    ]
    applier = TargetEncoderApplier()
    for sample in samples:
        values = applier.apply(sample, restored)["category"].to_numpy()
        np.testing.assert_allclose(values, [1, 1, 1, 0, 0, 0, 0, 1 / 3])
        chunks = [
            applier.apply(sample[i : i + 1], restored)["category"].to_numpy()
            for i in range(len(sample))
        ]
        np.testing.assert_array_equal(np.concatenate(chunks), values)
        assert applier.apply(sample[:0], restored).shape == (0, 1)
    for missing in (None, np.nan, pd.NA):
        all_missing = pd.DataFrame({"category": [missing, missing]})
        np.testing.assert_allclose(applier.apply(all_missing, restored)["category"], [1, 1])
    assert artifact_digest(restored) == digest
    if isinstance(frame, pl.DataFrame):
        assert frame.equals(before)
    else:
        pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_target_cross_fitting_normalizes_missing_before_fitting_folds(engine):
    """Canonical keys must feed both out-of-fold training and saved full-data statistics."""
    values = [None, np.nan, pd.NA, None, "a", "b", "c", "d", "a", "b", "c", "d"]
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0])
    frame = pd.DataFrame({"category": values})
    y = pd.Series(labels, name="target")
    if engine == "polars":
        frame = pl.from_pandas(frame)
        y = pl.Series("target", labels)
    config = {"columns": ["category"], "smooth": 0.0, "target_type": "binary"}
    state, (encoded, _) = TargetEncoderCalculator().fit_transform_train((frame, y), config)
    expected = TargetEncoder(smooth=0.0, target_type="binary", cv=5, shuffle=True, random_state=42)
    expected_values = expected.fit_transform(
        np.array(["nan"] * 4 + list("abcdabcd"), dtype=object).reshape(-1, 1), labels
    )
    np.testing.assert_allclose(encoded["category"].to_numpy(), expected_values[:, 0])
    result = TargetEncoderApplier().apply(pd.DataFrame({"category": [None, np.nan, pd.NA]}), state)
    np.testing.assert_allclose(result["category"], [0.5, 0.5, 0.5])


def test_legacy_target_state_keeps_raw_sklearn_categories_after_disk_roundtrip(tmp_path):
    """An absent key version must replay the old saved estimator without renaming categories."""
    train = np.array([[None], [None], ["a"], ["a"], ["b"], ["b"]], dtype=object)
    encoder = TargetEncoder(smooth=0.0, target_type="binary").fit(train, [1, 1, 0, 0, 0, 1])
    state = {"type": "target_encoder", "columns": ["category"], "encoder_object": encoder}
    path = tmp_path / "legacy.pkl"
    path.write_bytes(pickle.dumps(state))
    restored = pickle.loads(path.read_bytes())
    TargetEncoderApplier.validate_inference_state(restored)
    sample = pd.DataFrame({"category": [None, "a", "b", "new"]})
    np.testing.assert_allclose(
        TargetEncoderApplier().apply(sample, restored)["category"], [1, 0, 0.5, 0.5]
    )
    assert "category_key_version" not in restored


@pytest.mark.parametrize("version", [True, 0, 2, "1", None])
def test_target_unknown_key_version_fails_inspection_and_apply(version):
    """Unsupported key contracts cannot silently fall back to legacy raw-category replay."""
    frame, labels = _training("none")
    state = dict(TargetEncoderCalculator().fit((frame, labels), {"columns": ["category"]}))
    state["category_key_version"] = version
    with pytest.raises(ValueError, match="category key version"):
        TargetEncoderApplier.validate_inference_state(state)
    with pytest.raises(ValueError, match="category key version"):
        TargetEncoderApplier().apply(frame, state)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_target_pipeline_save_load_preserves_null_predictions(engine, tmp_path, monkeypatch):
    """The real exported pipeline must replay canonical keys after engine and null changes."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    train = pd.DataFrame({"category": [None, "a", "b"] * 8, "target": [9.0, 1.0, 5.0] * 8})
    if engine == "polars":
        train = pl.from_pandas(train)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {
                    "name": "encode",
                    "transformer": "TargetEncoder",
                    "params": {
                        "columns": ["category"],
                        "target_type": "continuous",
                        "smooth": 0.0,
                    },
                }
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(train, target_column="target")
    save_pipeline(pipeline, tmp_path / "pipeline")
    restored = load_pipeline(tmp_path / "pipeline")
    pandas_sample = pd.DataFrame({"category": [None, np.nan, pd.NA, "a", "b", "new"]})
    result = predict_pipeline(pandas_sample, restored)
    polars_sample = pl.DataFrame({"category": [None, None, None, "a", "b", "new"]})
    other = predict_pipeline(polars_sample, restored)
    np.testing.assert_allclose(result.to_numpy(), other.to_numpy())
    np.testing.assert_allclose(result.to_numpy()[:3], np.repeat(result.to_numpy()[[0]], 3, axis=0))
    chunks = [
        predict_pipeline(
            polars_sample[index : index + 1] if engine == "polars" else pandas_sample.iloc[[index]],
            restored,
        ).to_numpy()
        for index in range(len(pandas_sample))
    ]
    np.testing.assert_allclose(np.concatenate(chunks), result.to_numpy())
