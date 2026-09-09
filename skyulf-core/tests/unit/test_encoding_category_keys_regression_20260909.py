"""Regression contracts for scalar category identity and artifact compatibility."""

import hashlib
import pickle
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

import skyulf.preprocessing  # noqa: F401 - populate the public preprocessing registry
from skyulf.registry import NodeRegistry


def _frame(values: list[Any], engine: str) -> Any:
    """Preserve scalar types until the requested dataframe engine receives them."""
    frame = pd.DataFrame({"category": pd.Series(values, dtype=object)})
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("node_type", ["LabelEncoder", "OrdinalEncoder", "WOEEncoder"])
@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
def test_numeric_feature_keys_survive_engine_and_artifact_roundtrip(
    node_type, fit_engine, apply_engine
):
    """Serialized learned categories must replay identically across engines and numeric dtypes."""
    train = _frame([1, 1, 2, 2], fit_engine)
    labels: Any = pd.Series([0, 0, 1, 1], name="target")
    if fit_engine == "polars":
        labels = pl.from_pandas(labels)
    artifact = NodeRegistry.get_calculator(node_type)().fit(
        (train, labels), {"columns": ["category"]}
    )
    artifact = pickle.loads(pickle.dumps(artifact))
    applier = NodeRegistry.get_applier(node_type)()
    alone = applier.apply(_frame([1], apply_engine), artifact)
    together = applier.apply(_frame([1.0, 2.5], apply_engine), artifact)

    assert alone["category"].to_list()[0] == together["category"].to_list()[0]


@pytest.mark.parametrize("node_type", ["LabelEncoder", "OrdinalEncoder", "WOEEncoder"])
def test_literal_numeric_strings_do_not_match_learned_numeric_categories(node_type):
    """Normalizing integral floats must not turn a literal string into a known numeric category."""
    train = _frame([1, 1, 2, 2], "pandas")
    labels = pd.Series([0, 0, 1, 1], name="target")
    artifact = NodeRegistry.get_calculator(node_type)().fit(
        (train, labels), {"columns": ["category"]}
    )
    result = (
        NodeRegistry.get_applier(node_type)()
        .apply(_frame([1, 1.0, "1", "1.0", 2.5], "pandas"), artifact)["category"]
        .to_list()
    )
    default = 0.0 if node_type == "WOEEncoder" else -1

    assert result[0] == result[1] != default
    assert result[2:] == [default, default, default]


@pytest.mark.parametrize("node_type", ["LabelEncoder", "OrdinalEncoder", "WOEEncoder"])
def test_mixed_numeric_and_literal_string_categories_remain_distinct(node_type):
    """Fitting mixed object categories must retain distinct numeric and string identities."""
    frame = _frame([1, 1, "1", "1", "1.0", "1.0"], "pandas")
    labels = pd.Series([0, 0, 1, 1, 0, 1], name="target")
    artifact = NodeRegistry.get_calculator(node_type)().fit(
        (frame, labels), {"columns": ["category"]}
    )
    result = NodeRegistry.get_applier(node_type)().apply(frame, artifact)["category"].to_list()

    assert len({result[0], result[2], result[4]}) == 3


@pytest.mark.parametrize(
    "node_type", ["LabelEncoder", "OrdinalEncoder", "WOEEncoder", "HashEncoder"]
)
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_new_artifacts_share_one_missing_key_for_none_nan_and_pd_na(node_type, engine):
    """Missing category identity must not depend on the input null marker or engine."""
    train = _frame([None, "seen", "seen", None], engine)
    labels: Any = pd.Series([0, 1, 1, 0], name="target")
    if engine == "polars":
        labels = pl.from_pandas(labels)
    artifact = NodeRegistry.get_calculator(node_type)().fit(
        (train, labels), {"columns": ["category"], "n_features": 100_003}
    )
    result = (
        NodeRegistry.get_applier(node_type)()
        .apply(_frame([None, np.nan, pd.NA], engine), artifact)["category"]
        .to_list()
    )

    assert len(set(result)) == 1


@pytest.mark.parametrize("marker", [None, np.nan, pd.NA], ids=["none", "nan", "pd_na"])
def test_hash_missing_key_is_independent_of_a_fractional_companion(marker):
    """All supported missing markers must keep their bucket when a batch's dtype changes."""
    artifact = NodeRegistry.get_calculator("HashEncoder")().fit(
        _frame([1, 2], "pandas"), {"columns": ["category"], "n_features": 100_003}
    )
    applier = NodeRegistry.get_applier("HashEncoder")()
    alone = applier.apply(_frame([marker], "pandas"), artifact)
    together = applier.apply(_frame([marker, 2.5], "pandas"), artifact)

    assert alone["category"].iloc[0] == together["category"].iloc[0]


@pytest.mark.parametrize("version", [None, 1])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_existing_hash_artifacts_keep_their_original_buckets(version, engine):
    """An artifact version change must not silently alter hashes used by a fitted model."""
    artifact = {"columns": ["category"], "n_features": 100_003}
    if version is not None:
        artifact["numeric_normalization_version"] = version
    frame: Any = pd.DataFrame({"category": [1.0, 2.5, np.nan]})
    if engine == "polars":
        frame = pl.from_pandas(frame)
    result = NodeRegistry.get_applier("HashEncoder")().apply(frame, artifact)
    keys = ["1" if version == 1 else "1.0", "2.5", "nan"]
    expected = [
        int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "little") % 100_003
        for key in keys
    ]

    assert result["category"].to_list() == expected


@pytest.mark.parametrize("node_type", ["LabelEncoder", "OrdinalEncoder", "WOEEncoder"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_existing_fitted_string_mappings_remain_replayable(node_type, engine):
    """Legacy artifacts without a key version must retain their fitted string lookup."""
    artifact: dict[str, Any] = {"columns": ["category"]}
    if node_type == "LabelEncoder":
        artifact["encoders"] = {"category": LabelEncoder().fit(["1", "2"])}
        expected = [0, -1]
    elif node_type == "OrdinalEncoder":
        artifact["encoder_object"] = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ).fit(np.array([["1"], ["2"]]))
        expected = [0, -1]
    else:
        artifact["mappings"] = {"category": {"1": 1.25, "2": -1.25}}
        expected = [1.25, 0]
    result = NodeRegistry.get_applier(node_type)().apply(_frame(["1", "unseen"], engine), artifact)

    assert result["category"].to_list() == expected


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_explicit_numeric_ordinal_order_survives_float_replay(engine):
    """A user-entered numeric order must remain effective after scalar-key normalization."""
    artifact = NodeRegistry.get_calculator("OrdinalEncoder")().fit(
        _frame([1, 2], engine), {"columns": ["category"], "categories_order": "2,1"}
    )
    result = NodeRegistry.get_applier("OrdinalEncoder")().apply(
        _frame([1.0, 2.0], engine), artifact
    )

    assert result["category"].to_list() == [1.0, 0.0]


@pytest.mark.parametrize("marker", [None, np.nan, pd.NA], ids=["none", "nan", "pd_na"])
@pytest.mark.parametrize("method", ["fit", "fit_transform_train"])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_woe_rejects_missing_targets_before_fit_or_cross_fit(marker, method, engine):
    """Missing labels must never be silently recoded as the negative binary class."""
    frame = _frame(["a", "a", "b", "b"], engine)
    labels: Any = pd.Series([0, 1, marker, 1], name="target")
    if engine == "polars":
        labels = pl.from_pandas(labels)
    calculator = NodeRegistry.get_calculator("WOEEncoder")()

    with pytest.raises(ValueError, match="missing"):
        getattr(calculator, method)((frame, labels), {"columns": ["category"]})


@pytest.mark.parametrize("training", [[1, 2], ["1", "2"]], ids=["numeric", "literal_text"])
def test_polars_casting_does_not_accept_a_different_scalar_category_type(training):
    """Vocabulary matching must not confuse text with numbers while normalizing numeric dtypes."""
    artifact = NodeRegistry.get_calculator("Casting")().fit(
        _frame(training, "polars"), {"columns": ["category"], "target_type": "category"}
    )
    probe = ["1"] if isinstance(training[0], int) else [1.0]
    result = NodeRegistry.get_applier("Casting")().apply(_frame(probe, "polars"), artifact)

    assert result["category"].to_list() == [None]
