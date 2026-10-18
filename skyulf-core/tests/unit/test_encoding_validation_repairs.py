"""Invalid encoding recipes fail at fit while saved valid states keep replaying."""

import pickle
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError, require_capability
from skyulf.preprocessing.encoding.hash import HashEncoderApplier, HashEncoderCalculator
from skyulf.preprocessing.encoding.ordinal import OrdinalEncoderApplier, OrdinalEncoderCalculator
from skyulf.preprocessing.encoding.woe import WOEEncoderCalculator


def _frame(engine, **columns):
    """Exercise the public calculator on each native dataframe engine."""
    return pl.DataFrame(columns) if engine == "polars" else pd.DataFrame(columns)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_hash_default_has_the_same_buckets_as_explicit_eight(engine):
    """Omitting the width must not change a recipe exported with the Canvas default."""
    frame = _frame(engine, category=list("abcdefghijklmnop"))
    calculator = HashEncoderCalculator()
    implicit = calculator.fit(frame, {"columns": ["category"]})
    explicit = calculator.fit(frame, {"columns": ["category"], "n_features": 8})
    applier = HashEncoderApplier()
    np.testing.assert_array_equal(
        applier.apply(frame, implicit).to_numpy(), applier.apply(frame, explicit).to_numpy()
    )
    assert implicit["n_features"] == 8


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("width", [8.0, 8.5, True, np.bool_(True), 0, -1, "8", None, np.nan])
def test_hash_rejects_invalid_bucket_counts_at_fit(engine, width):
    """Fractional modulo and booleans must not silently collapse learned hash buckets."""
    with pytest.raises(ValueError, match="HashEncoder.*n_features.*positive integer"):
        HashEncoderCalculator().fit(
            _frame(engine, category=["a", "b"]),
            {"columns": ["category"], "n_features": width},
        )


@pytest.mark.parametrize("width", [np.int64(8), 10, 257])
def test_hash_saved_width_replays_without_using_the_new_default(width, tmp_path):
    """Serialized explicit and legacy widths must retain their original assignments."""
    frame = pd.DataFrame({"category": list("abcdefgh")})
    state = HashEncoderCalculator().fit(frame, {"columns": ["category"], "n_features": width})
    path = tmp_path / "hash.pkl"
    path.write_bytes(pickle.dumps(state))
    restored = pickle.loads(path.read_bytes())
    HashEncoderApplier.validate_inference_state(dict(restored))
    expected = HashEncoderApplier().apply(frame, state)
    np.testing.assert_array_equal(HashEncoderApplier().apply(frame, restored), expected)
    python_state = HashEncoderCalculator().fit(
        frame, {"columns": ["category"], "n_features": int(width)}
    )
    np.testing.assert_array_equal(expected, HashEncoderApplier().apply(frame, python_state))
    assert restored["n_features"] == width


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("method", ["fit", "fit_transform_train"])
@pytest.mark.parametrize("labels", [[0, 1, 2, 0, 1, 2], [1] * 6, []])
def test_woe_rejects_targets_without_exactly_two_classes(engine, method, labels):
    """An unsupported target must not leave an apparently successful unencoded pipeline."""
    frame = _frame(engine, category=["a", "b"] * (len(labels) // 2))
    y = (
        pl.Series("target", labels, dtype=pl.Int64)
        if engine == "polars"
        else pd.Series(labels, dtype="int64", name="target")
    )
    with pytest.raises(ValueError, match="WOEEncoder.*binary target.*exactly 2 classes"):
        getattr(WOEEncoderCalculator(), method)((frame, y), {"columns": ["category"]})


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("policy", ["ignore", "typo", None, 0])
def test_ordinal_rejects_unknown_policy(engine, policy):
    """A misspelled policy must not silently enable unseen-category replacement."""
    with pytest.raises(
        ValueError, match="OrdinalEncoder.*handle_unknown.*error.*use_encoded_value"
    ):
        OrdinalEncoderCalculator().fit(
            _frame(engine, first=["a", "b"]),
            {"columns": ["first"], "handle_unknown": policy},
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "order",
    [
        "b,a",
        "b,a\ny,x\nq,p",
        [["b", "a"], ["y", "x"]],
        0,
        {"first": ["b", "a"]},
        ["b,a", None],
        ",\ny,x",
    ],
)
def test_ordinal_rejects_malformed_explicit_orders(engine, order):
    """A malformed user order must never silently become alphabetical ordering."""
    with pytest.raises(ValueError, match="OrdinalEncoder.*categories_order"):
        OrdinalEncoderCalculator().fit(
            _frame(engine, first=["a", "b"], second=["x", "y"]),
            {"columns": ["first", "second"], "categories_order": order},
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("explicit_columns", [True, False])
def test_ordinal_valid_order_and_default_unknown_survive_saved_replay(
    engine, explicit_columns, tmp_path
):
    """Strict parsing must preserve explicit orders with selected or detected feature columns."""
    config: dict[str, Any] = {"categories_order": "b,a\ny,x"}
    if explicit_columns:
        config["columns"] = ["first", "second"]
    state = OrdinalEncoderCalculator().fit(
        _frame(engine, first=["a", "b"], second=["x", "y"]), config
    )
    path = tmp_path / "ordinal.pkl"
    path.write_bytes(pickle.dumps(state))
    restored = pickle.loads(path.read_bytes())
    result = OrdinalEncoderApplier().apply(
        _frame(engine, first=["a", "b", "unseen"], second=["x", "y", "unseen"]), restored
    )
    np.testing.assert_array_equal(result.to_numpy(), [[1, 1], [0, 0], [-1, -1]])


def test_onehot_partition_diagnostic_names_the_supported_explicit_settings():
    """The intentional capped-category restriction must tell callers how to select support."""
    with pytest.raises(
        UnsupportedExecutionError, match="max_categories=None.*include_missing=False"
    ):
        require_capability(
            "OneHotEncoder",
            "apply",
            "pandas",
            config={"max_categories": 20, "include_missing": False},
            execution_kind="python_batch",
            context="row",
            row_effect="preserve",
        )
    require_capability(
        "OneHotEncoder",
        "apply",
        "pandas",
        config={"max_categories": None, "include_missing": False},
        execution_kind="python_batch",
        context="row",
        row_effect="preserve",
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("target_first", [True, False])
def test_ordinal_explicit_target_order_uses_paired_target_name(engine, target_first, tmp_path):
    """A named paired target must keep its own order row regardless of selection position."""
    frame = _frame(engine, feature=["a", "b", "a", "b"])
    labels = ["yes", "no", "yes", "no"]
    y = pl.Series("label", labels) if engine == "polars" else pd.Series(labels, name="label")
    config = {
        "columns": ["label", "feature"] if target_first else ["feature", "label"],
        "categories_order": "yes,no\nb,a" if target_first else "b,a\nyes,no",
    }
    state = OrdinalEncoderCalculator().fit((frame, y), config)
    path = tmp_path / "ordinal_target.pkl"
    path.write_bytes(pickle.dumps(state))
    output, target = OrdinalEncoderApplier().apply((frame, y), pickle.loads(path.read_bytes()))
    np.testing.assert_array_equal(output["feature"].to_numpy(), [1, 0, 1, 0])
    np.testing.assert_array_equal(target.to_numpy(), [0, 1, 0, 1])
