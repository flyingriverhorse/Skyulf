"""Target encoder options preserve adaptive cross-fitting and saved full-data replay."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import TargetEncoder

from skyulf.preprocessing._category_keys import category_keys_pandas
from skyulf.preprocessing.encoding.target import TargetEncoderCalculator


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("shuffle,seed", [(True, 17), (False, 42), (True, None)])
def test_target_cv_matches_sklearn_and_caps_small_classes(engine, shuffle, seed):
    """Requested folds and split options must affect actual cross-fitted training values."""
    pdf = pd.DataFrame({"cat": list("aabbccddaabb")})
    labels = [0, 1] * 6
    frame = pl.from_pandas(pdf) if engine == "polars" else pdf
    y = pl.Series("label", labels) if engine == "polars" else pd.Series(labels, name="label")
    config = {
        "columns": ["cat"],
        "cv": 3,
        "shuffle": shuffle,
        "random_state": seed,
        "smooth": 2,
        "target_type": "binary",
    }
    # A supplied seed is intentionally ignored when shuffling is disabled.
    effective_seed = seed if shuffle else None
    if seed is None:
        np.random.seed(123)
    expected = TargetEncoder(
        cv=3, shuffle=shuffle, random_state=effective_seed, smooth=2, target_type="binary"
    ).fit_transform(category_keys_pandas(pdf["cat"]).to_numpy().reshape(-1, 1), labels)
    if seed is None:
        np.random.seed(123)
    state, transformed = TargetEncoderCalculator().fit_transform_train((frame, y), config)
    np.testing.assert_allclose(transformed[0]["cat"].to_numpy(), expected[:, 0])
    assert state["encoder_object"].cv == 3
    assert state["encoder_object"].random_state == effective_seed
    capped, _ = TargetEncoderCalculator().fit_transform_train((frame, y), {**config, "cv": 20})
    assert capped["encoder_object"].cv == 6


@pytest.mark.parametrize("method", ["fit", "fit_transform_train"])
@pytest.mark.parametrize(
    "option,value",
    [
        ("cv", 1),
        ("cv", True),
        ("cv", 2.5),
        ("shuffle", "false"),
        ("shuffle", 0),
        ("random_state", -1),
        ("random_state", True),
        ("random_state", 2**32),
    ],
)
def test_target_rejects_invalid_cv_options(method, option, value):
    """Direct fitting and pipeline cross-fitting must validate the same public controls."""
    frame = pd.DataFrame({"cat": list("aabb")})
    y = pd.Series([0, 1, 0, 1], name="label")
    with pytest.raises(ValueError, match=f"TargetEncoder.*{option}"):
        getattr(TargetEncoderCalculator(), method)((frame, y), {"columns": ["cat"], option: value})
