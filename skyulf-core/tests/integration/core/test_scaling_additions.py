"""Additional scaling methods preserve sklearn math and fitted inference contracts."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.preprocessing import MinMaxScaler, Normalizer, QuantileTransformer, RobustScaler

from skyulf import SkyulfPipeline
from skyulf.core.schema import SkyulfSchema
from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.pipeline.seal import artifact_digest
from skyulf.preprocessing.inference_context import get_inference_capability
from skyulf.registry import NodeRegistry


def _native(frame, engine):
    """Keep a native frame for each public calculator and applier path."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _training():
    """Include an unselected name matching a statistic to guard against generated collisions."""
    return pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 4.0, 8.0, 16.0],
            "z": [3.0, 5.0, 1.0, 6.0, 2.0, 9.0],
            "quantiles": list("abcdef"),
            "target": [1.0, 2.0, 3.0, 5.0, 9.0, 17.0],
        },
        index=[4, 4, 9, 2, 9, 0],
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("node", "options"),
    [
        ("QuantileTransformer", {"output_distribution": distribution, "n_quantiles": 4})
        for distribution in ("uniform", "normal")
    ]
    + [("Normalizer", {"norm": norm}) for norm in ("l1", "l2", "max")],
)
def test_new_scalers_match_sklearn_and_saved_chunk_replay(engine, node, options, tmp_path):
    """Saved transforms must preserve native schemas, row order and sklearn numerical decisions."""
    train = _training()
    config = {"columns": ["x", "z"], **options}
    calculator = NodeRegistry.get_calculator(node)()
    applier = NodeRegistry.get_applier(node)()
    state = calculator.fit(_native(train, engine), config)
    expected_estimator = (
        QuantileTransformer(random_state=42, **options)
        if node == "QuantileTransformer"
        else Normalizer(**options)
    ).fit(train[["x", "z"]].to_numpy())
    sample = pd.DataFrame(
        {
            "x": [-10.0, 0.0, 100.0],
            "z": [5.0, 0.0, 2.0],
            "quantiles": ["keep", "these", "names"],
            "target": [7.0, 8.0, 9.0],
        },
        index=[3, 3, -1],
    )
    expected = sample.copy()
    expected[["x", "z"]] = expected_estimator.transform(sample[["x", "z"]].to_numpy())
    path = tmp_path / "scaler.pkl"
    path.write_bytes(pickle.dumps(state))
    restored = pickle.loads(path.read_bytes())
    digest = artifact_digest(restored)
    for other_engine in ("pandas", "polars"):
        request = _native(sample, other_engine)
        output = applier.apply(request, restored)
        expected_native = _native(expected, other_engine)
        for size in (3, 1, 0):
            actual = applier.apply(request.head(size), restored)
            if other_engine == "pandas":
                pd.testing.assert_frame_equal(actual, expected_native.head(size))
            else:
                assert actual.equals(expected_native.head(size))
        assert list(output.columns) == list(sample.columns)
        capability = get_inference_capability(node, config, restored, engine=other_engine)
        assert capability is not None and capability.execution_kind == "local"
        assert (capability.context, capability.row_effect) == ("row", "preserve")
    assert artifact_digest(restored) == digest


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["QuantileTransformer", "Normalizer"])
def test_new_scalers_select_features_without_target_and_keep_empty_selection(engine, node):
    """Automatic numeric selection must exclude labels while an explicit empty list remains a no-op."""
    frame = _native(_training(), engine)
    calculator = NodeRegistry.get_calculator(node)()
    state = calculator.fit(frame, {"target_column": "target", "n_quantiles": 4})
    assert state["columns"] == ["x", "z"]
    schema = calculator.infer_output_schema(
        SkyulfSchema.from_dataframe(frame), {"columns": ["x", "z"]}
    )
    assert schema.dtypes["x"] == "float64" and schema.dtypes["z"] == "float64"
    assert calculator.fit(frame, {"columns": []}) == {}
    output = NodeRegistry.get_applier(node)().apply(frame, {})
    assert output is frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("node", "options"),
    [
        ("QuantileTransformer", {"output_distribution": "gamma"}),
        ("QuantileTransformer", {"n_quantiles": 0}),
        ("QuantileTransformer", {"n_quantiles": 2.5}),
        ("QuantileTransformer", {"subsample": 0}),
        ("Normalizer", {"norm": "l3"}),
    ],
)
def test_new_scalers_reject_unsupported_options(engine, node, options):
    """Invalid method options must fail before any fitted artifact can be returned."""
    with pytest.raises(ValueError):
        NodeRegistry.get_calculator(node)().fit(
            _native(_training(), engine), {"columns": ["x", "z"], **options}
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_quantile_missing_values_and_effective_quantile_count(engine):
    """Missing cells stay missing and sklearn's small-sample cap is recorded in saved state."""
    train = _native(pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]}), engine)
    calculator = NodeRegistry.get_calculator("QuantileTransformer")()
    with pytest.warns(UserWarning, match="n_quantiles"):
        state = calculator.fit(train, {"columns": ["x"], "n_quantiles": 1000})
    assert state["transformer_object"].n_quantiles_ == 4
    output = NodeRegistry.get_applier("QuantileTransformer")().apply(train, state)
    np.testing.assert_allclose(output["x"].to_numpy(), [0.0, 4 / 9, np.nan, 1.0], equal_nan=True)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_normalizer_rejects_missing_inputs_without_changing_rows(engine):
    """Row normalization cannot silently drop or impute a missing feature value."""
    with pytest.raises(ValueError, match="NaN"):
        NodeRegistry.get_calculator("Normalizer")().fit(
            _native(pd.DataFrame({"x": [1.0, np.nan], "z": [3.0, 4.0]}), engine),
            {"columns": ["x", "z"]},
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("parameter", ["n_quantiles", "subsample"])
def test_quantile_counts_reject_booleans(engine, parameter):
    """A boolean must not become a count of one and produce an uninspectable fitted state."""
    with pytest.raises(ValueError, match=parameter):
        NodeRegistry.get_calculator("QuantileTransformer")().fit(
            _native(_training(), engine), {"columns": ["x"], parameter: True}
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("clip", [False, True])
def test_minmax_clip_matches_sklearn_and_binds_saved_configuration(engine, clip):
    """Out-of-training-range values use the saved clipping choice, never scoring configuration."""
    train = _native(pd.DataFrame({"x": [1.0, 3.0], "keep": [1, 2]}), engine)
    config = {"columns": ["x"], "feature_range": [-2, 7], "clip": clip}
    calculator = NodeRegistry.get_calculator("MinMaxScaler")()
    applier = NodeRegistry.get_applier("MinMaxScaler")()
    state = calculator.fit(train, config)
    sample = pd.DataFrame({"x": [-1.0, 2.0, 5.0, np.nan], "keep": [1, 2, 3, 4]})
    expected = MinMaxScaler(feature_range=(-2, 7), clip=clip).fit([[1.0], [3.0]])
    output = applier.apply(_native(sample, engine), state)
    np.testing.assert_allclose(
        output["x"].to_numpy(), expected.transform(sample[["x"]].to_numpy())[:, 0]
    )
    assert get_inference_capability("MinMaxScaler", config, state, engine=engine) is not None
    assert (
        get_inference_capability("MinMaxScaler", {**config, "clip": not clip}, state, engine=engine)
        is None
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_robust_unit_variance_matches_sklearn(engine):
    """The learned scale must include sklearn's normal-distribution quantile adjustment."""
    train = _training()
    options = {"quantile_range": (10, 90), "unit_variance": True}
    config = {"columns": ["x", "z"], **options}
    state = NodeRegistry.get_calculator("RobustScaler")().fit(_native(train, engine), config)
    expected = RobustScaler(**options).fit_transform(train[["x", "z"]])
    output = NodeRegistry.get_applier("RobustScaler")().apply(_native(train, engine), state)
    values = output.select(["x", "z"]).to_numpy() if engine == "polars" else output[["x", "z"]]
    np.testing.assert_allclose(values, expected)
    assert get_inference_capability("RobustScaler", config, state, engine=engine) is not None


@pytest.mark.parametrize("node", ["MinMaxScaler", "RobustScaler"])
def test_existing_scaler_default_shape_and_legacy_replay_stay_unchanged(node):
    """New optional flags must not rewrite legacy default artifacts or their numerical results."""
    frame = _training()
    calculator = NodeRegistry.get_calculator(node)()
    applier = NodeRegistry.get_applier(node)()
    state = calculator.fit(frame, {"columns": ["x", "z"]})
    key = "clip" if node == "MinMaxScaler" else "unit_variance"
    assert key not in state
    before = deepcopy(state)
    restored = pickle.loads(pickle.dumps(state))
    pd.testing.assert_frame_equal(applier.apply(frame, restored), applier.apply(frame, before))
    assert state == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", ["QuantileTransformer", "Normalizer"])
@pytest.mark.parametrize("selection", [{}, {"columns": None}])
def test_new_scaler_schema_abstains_for_data_dependent_selection(engine, node, selection):
    """Schema-only previews cannot know which binary or constant columns discovery excludes."""
    frame = _native(
        pd.DataFrame(
            {"x": [1, 2, 3], "binary": [0, 1, 0], "constant": [3, 3, 3], "target": [0, 1, 0]}
        ),
        engine,
    )
    calculator = NodeRegistry.get_calculator(node)()
    config = {"target_column": "target", "n_quantiles": 2, **selection}
    schema = calculator.infer_output_schema(SkyulfSchema.from_dataframe(frame), config)
    state = calculator.fit(frame, config)
    output = NodeRegistry.get_applier(node)().apply(frame, state)
    assert schema is None
    assert state["columns"] == ["x"]
    actual = SkyulfSchema.from_dataframe(output)
    original = SkyulfSchema.from_dataframe(frame)
    assert actual.dtypes["x"].lower() == "float64"
    assert all(
        actual.dtypes[col] == original.dtypes[col] for col in ("binary", "constant", "target")
    )


@pytest.mark.parametrize("node", ["QuantileTransformer", "Normalizer"])
def test_new_scaler_schema_abstains_when_automatic_dtypes_are_unknown(node):
    """An unknown input dtype must keep the preview unknown instead of raising KeyError."""
    calculator = NodeRegistry.get_calculator(node)()
    assert calculator.infer_output_schema(SkyulfSchema.from_columns(["x"]), {}) is None


@pytest.mark.parametrize("node", ["QuantileTransformer", "Normalizer"])
@pytest.mark.parametrize("damage", ["width", "method", "columns"])
def test_new_scaler_state_inspection_rejects_malformed_saved_axes(node, damage):
    """A damaged estimator must not acquire an inference capability through its node name."""
    state = dict(
        NodeRegistry.get_calculator(node)().fit(
            _training(), {"columns": ["x", "z"], "n_quantiles": 4}
        )
    )
    if damage == "width":
        state["transformer_object"].n_features_in_ = 1
    elif damage == "method":
        attribute = "norm" if node == "Normalizer" else "output_distribution"
        setattr(state["transformer_object"], attribute, "unknown")
    else:
        state["columns"] = ["x", "x"]
    with pytest.raises(ValueError):
        NodeRegistry.get_applier(node)().validate_inference_state(state)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("node", "options"),
    [
        ("QuantileTransformer", {"n_quantiles": 4, "output_distribution": "normal"}),
        ("Normalizer", {"norm": "l1"}),
        ("MinMaxScaler", {"clip": True}),
        ("RobustScaler", {"unit_variance": True}),
    ],
)
def test_scaling_pipeline_roundtrip_preserves_cross_engine_predictions(
    engine, node, options, tmp_path, monkeypatch
):
    """Real model export and reload must keep the training transform across inference engines."""
    monkeypatch.setenv("SKYULF_ENGINE", engine)
    train = _training().drop(columns="quantiles")
    frame = _native(train, engine)
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "scale", "transformer": node, "params": {"columns": ["x", "z"], **options}}
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=frame, test=frame.head(0)), target_column="target")
    sample = train.drop(columns="target")
    expected = pipeline.predict(_native(sample, engine))
    save_pipeline(pipeline, tmp_path / "pipeline")
    restored = load_pipeline(tmp_path / "pipeline")
    for other_engine in ("pandas", "polars"):
        result = predict_pipeline(_native(sample, other_engine), restored)
        np.testing.assert_allclose(result.prediction, expected)
