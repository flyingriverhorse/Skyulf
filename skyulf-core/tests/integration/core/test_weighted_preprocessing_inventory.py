"""Every remaining registered preprocessing family retains positional weights."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing.pipeline import FeatureEngineer

GENERATION = {
    "operations": [
        {
            "operation_type": "arithmetic",
            "method": "add",
            "input_columns": ["value"],
            "constants": [2],
            "output_column": "added",
        }
    ]
}
CASES = [
    ("CustomBinning", {"columns": ["value"], "bins": [-1, 2, 10]}),
    ("GeneralBinning", {"columns": ["value"], "strategy": "equal_width", "n_bins": 2}),
    ("KBinsDiscretizer", {"columns": ["value"], "n_bins": 2, "strategy": "uniform"}),
    ("DataSnapshot", {"n_rows": 2}),
    ("DatasetProfile", {}),
    ("FeatureGeneration", GENERATION),
    ("FeatureMath", GENERATION),
    ("FeatureGenerationNode", GENERATION),
    ("PolynomialFeaturesNode", {"columns": ["value"], "degree": 2}),
    ("GroupImputer", {"columns": ["missing"], "group_by": "group", "strategy": "mean"}),
    ("GeoDistance", {"lat1_col": "lat", "lon1_col": "lon", "lat2_col": "lat2", "lon2_col": "lon2"}),
    (
        "feature_selection",
        {"method": "variance", "columns": ["value", "constant"], "threshold": 0.0},
    ),
    ("count_vectorizer", {"columns": ["text"], "drop_original": True}),
    ("tfidf_vectorizer", {"columns": ["text"], "drop_original": True}),
    ("hashing_vectorizer", {"columns": ["text"], "drop_original": True, "n_features": 4}),
    ("tokenizer", {"columns": ["text"], "add_token_count": True}),
]


def payload(engine):
    """Return duplicate-index rows with distinguishable targets and weights."""
    X = pd.DataFrame(
        {
            "row": [3, 0, 2, 1],
            "value": [4.0, 1.0, 3.0, 2.0],
            "constant": [1.0] * 4,
            "missing": [4.0, np.nan, 3.0, np.nan],
            "group": ["a", "a", "b", "b"],
            "text": ["four foxes", "one owl", "three cats", "two dogs"],
            "lat": [54.0, 55.0, 56.0, 57.0],
            "lon": [23.0, 24.0, 25.0, 26.0],
            "lat2": [54.1, 55.1, 56.1, 57.1],
            "lon2": [23.1, 24.1, 25.1, 26.1],
        },
        index=[7] * 4,
    )
    y = pd.Series([30, 0, 20, 10], index=X.index, name="target")
    return (pl.from_pandas(X), pl.Series("target", y.to_numpy())) if engine == "polars" else (X, y)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("name,params", CASES)
def test_registered_family_transports_weights(engine, name, params):
    """Weighted output must equal real unweighted behavior, including row ownership."""
    X, y = payload(engine)
    step = {"name": "operation", "transformer": name, "params": params}
    engineer = FeatureEngineer([step])
    (out, target), _ = engineer.fit_transform((X, y), sample_weight=[4.0, 1.0, 3.0, 2.0])
    (expected, expected_y), _ = FeatureEngineer([step]).fit_transform((X, y))
    pd.testing.assert_frame_equal(
        out.to_pandas() if engine == "polars" else out,
        expected.to_pandas() if engine == "polars" else expected,
    )
    np.testing.assert_array_equal(target, expected_y)
    np.testing.assert_array_equal(target, y)
    np.testing.assert_array_equal(engineer.train_sample_weight_, [4.0, 1.0, 3.0, 2.0])
    assert len(out) == 4
    if name in {"FeatureGeneration", "FeatureGenerationNode", "FeatureMath"}:
        np.testing.assert_array_equal(out["added"], [6.0, 3.0, 5.0, 4.0])
    elif name == "GroupImputer":
        np.testing.assert_array_equal(out["missing"], [4.0, 4.0, 3.0, 3.0])
    elif name in {"DataSnapshot", "DatasetProfile"}:
        assert (
            "snapshot" in engineer.fitted_steps[0]["artifact"]
            or "profile" in engineer.fitted_steps[0]["artifact"]
        )
    else:
        assert list(out.columns) != list(X.columns) or not np.array_equal(out["value"], X["value"])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_h3_weights_keep_real_geographic_rows(engine):
    """Actual H3 cells correspond to the same input coordinates and training weights."""
    h3 = pytest.importorskip("h3")
    from skyulf.preprocessing.pipeline import FeatureEngineer

    X, y = payload(engine)
    engineer = FeatureEngineer(
        [
            {
                "name": "h3",
                "transformer": "H3Index",
                "params": {"lat_col": "lat", "lon_col": "lon", "resolution": 5},
            }
        ]
    )
    (out, target), _ = engineer.fit_transform((X, y), sample_weight=[4.0, 1.0, 3.0, 2.0])
    expected = [h3.latlng_to_cell(lat, lon, 5) for lat, lon in zip(X["lat"], X["lon"], strict=True)]
    assert list(out["h3_index"]) == expected
    np.testing.assert_array_equal(target, y)
    np.testing.assert_array_equal(engineer.train_sample_weight_, [4.0, 1.0, 3.0, 2.0])


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_sentence_embedding_weight_transport_without_model_download(monkeypatch, engine):
    """Real node encode/concat behavior preserves row weights around a deterministic encoder."""
    from skyulf.preprocessing.vectorization import sentence_embedder

    class Encoder:
        """Avoid external model downloads while exercising actual text transport."""

        def get_embedding_dimension(self):
            """Expose a fixed embedding width."""
            return 2

        def encode(self, text, **kwargs):
            """Make row identity observable in the embedding values."""
            return np.asarray([[len(value), len(value.split())] for value in text], dtype=float)

    monkeypatch.setattr(sentence_embedder, "_load_model", lambda name: Encoder())
    X, y = payload(engine)
    engineer = FeatureEngineer(
        [
            {
                "name": "embed",
                "transformer": "sentence_embedder",
                "params": {
                    "columns": ["text"],
                    "drop_original": True,
                    "model_name": "test-encoder",
                },
            }
        ]
    )
    (out, target), _ = engineer.fit_transform((X, y), sample_weight=[4.0, 1.0, 3.0, 2.0])
    np.testing.assert_array_equal(out["text__emb__0"], [len(value) for value in X["text"]])
    assert "text" not in out.columns
    np.testing.assert_array_equal(target, y)
    np.testing.assert_array_equal(engineer.train_sample_weight_, [4.0, 1.0, 3.0, 2.0])


def test_alias_identity_does_not_trust_class_substitutions(monkeypatch):
    """Aliases inherit audited identities while replacement implementations still fail closed."""
    from skyulf.modeling._sample_weights import SampleWeightError
    from skyulf.preprocessing.scaling.standard import (
        StandardScalerApplier,
        StandardScalerCalculator,
    )
    from skyulf.registry import NodeRegistry

    monkeypatch.setitem(NodeRegistry._calculators, "ProjectScalerAlias", StandardScalerCalculator)
    monkeypatch.setitem(NodeRegistry._appliers, "ProjectScalerAlias", StandardScalerApplier)
    X, y = payload("pandas")
    engineer = FeatureEngineer(
        [{"name": "scale", "transformer": "ProjectScalerAlias", "params": {"columns": ["value"]}}]
    )
    (out, _), _ = engineer.fit_transform((X, y), sample_weight=[4.0, 1.0, 3.0, 2.0])
    np.testing.assert_array_equal(engineer.train_sample_weight_, [4.0, 1.0, 3.0, 2.0])
    assert abs(out["value"].mean()) < 1e-12

    class Replacement(StandardScalerApplier):
        """A subclass must not inherit an exact-class capability implicitly."""

    monkeypatch.setitem(NodeRegistry._appliers, "ProjectScalerAlias", Replacement)
    with pytest.raises(SampleWeightError, match="unsupported"):
        engineer.fit_transform((X, y), sample_weight=[4.0, 1.0, 3.0, 2.0])


def test_all_registered_builtin_preprocessing_pairs_have_weight_contract():
    """Registry additions must declare a reviewed capability rather than silently omit weights."""
    from skyulf.preprocessing._weight_policy import validate_weighted_steps
    from skyulf.registry import NodeRegistry

    names = [
        name
        for name, calculator in NodeRegistry._calculators.items()
        if calculator.__module__.startswith("skyulf.preprocessing.")
    ]
    for name in names:
        validate_weighted_steps([{"transformer": name}], allow_split=True)
    assert {name for name, _ in CASES} | {"H3Index", "sentence_embedder"} <= set(names)
