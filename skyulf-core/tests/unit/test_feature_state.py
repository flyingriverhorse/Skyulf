"""Portable FE state must be complete, bounded and safe to load without a runtime."""

import json
import pickle

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.core.execution import ExecutionOptions, FrameSpec
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.pipeline import FeatureEngineer


def _engineer(engine="pandas", **kwargs):
    """Fit a real two-step pipeline whose held-out values expose accidental refitting."""
    fe = FeatureEngineer(
        [
            {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}},
            {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
        ],
        **kwargs,
    )
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    fe.fit_transform(pl.from_pandas(frame) if engine == "polars" else frame)
    return fe


@pytest.mark.parametrize("fit_engine", ["pandas", "polars"])
@pytest.mark.parametrize("apply_engine", ["pandas", "polars"])
def test_local_pipeline_round_trip_does_not_refit(fit_engine, apply_engine):
    """Loading learned FE must preserve step order and train-only statistics on either engine."""
    fe = _engineer(fit_engine)
    payload = fe.export_state()
    restored = FeatureEngineer.from_state(payload, execution_options=ExecutionOptions(apply_engine))
    local = pd.DataFrame({"id": [2, 1], "x": [100.0, np.nan]})
    data = pl.from_pandas(local) if apply_engine == "polars" else local
    out = restored.transform(data)
    np.testing.assert_allclose(
        out["x"].to_numpy(), [120.02499739637573, 0.0], rtol=1e-10, atol=1e-12
    )
    assert out["id"].to_list() == [2, 1]
    assert restored.export_state() == payload
    assert [step["name"] for step in restored.steps_config] == ["fill", "scale"]


def test_unfitted_and_failed_refit_cannot_export_partial_state():
    """An incomplete fit must never become a seemingly valid portable inference pipeline."""
    fe = _engineer()
    with pytest.raises(ValueError, match="fitted"):
        FeatureEngineer([]).export_state()
    with pytest.raises(ValueError):
        fe.fit_transform(pd.DataFrame({"x": ["invalid", "text"]}))
    with pytest.raises(ValueError, match="fitted"):
        fe.export_state()


def test_fitted_empty_and_noop_pipelines_round_trip():
    """An intentionally empty pipeline is distinguishable from an unfitted object."""
    for steps in (
        [],
        [{"name": "skip", "transformer": "StandardScaler", "params": {"columns": []}}],
    ):
        fe = FeatureEngineer(steps)
        data = pd.DataFrame({"x": [1.0, np.nan]})
        fe.fit_transform(data)
        restored = FeatureEngineer.from_state(fe.export_state())
        pd.testing.assert_frame_equal(restored.transform(data), data)


def test_total_wire_budget_applies_on_export_and_before_parse(monkeypatch):
    """A pipeline of small artifacts must not bypass the total payload byte limit."""
    fe = _engineer()
    payload = fe.export_state()
    fe.execution_options = ExecutionOptions("pandas", state_max_bytes=len(payload) - 1)
    with pytest.raises(ValueError, match="max_bytes"):
        fe.export_state()
    fe.execution_options = ExecutionOptions("pandas", state_max_bytes=len(payload))
    assert fe.export_state() == payload

    def forbidden(*args, **kwargs):
        """An oversized input must be rejected before attempting JSON parsing."""
        pytest.fail("Parsed oversized state")

    monkeypatch.setattr(json, "loads", forbidden)
    with pytest.raises(ValueError, match="max_bytes"):
        FeatureEngineer.from_state(
            payload, execution_options=ExecutionOptions("pandas", state_max_bytes=len(payload) - 1)
        )


@pytest.mark.parametrize(
    "payload", [b"[]", b"{}", b"null", b'{"format_version":1,"format_version":1}', b"\xff", b"NaN"]
)
def test_malformed_payload_is_rejected(payload):
    """Malformed input cannot produce a runnable object or invoke a pickle fallback."""
    with pytest.raises(ValueError):
        FeatureEngineer.from_state(payload)


def test_step_reordering_and_config_corruption_are_detected():
    """The envelope checksum must cover ordered steps and configuration as well as node state."""
    payload = _engineer().export_state()
    document = json.loads(payload)
    document["steps"].reverse()
    with pytest.raises(ValueError, match="digest"):
        FeatureEngineer.from_state(json.dumps(document).encode())
    document = json.loads(payload)
    document["steps"][0]["name"] = "tampered"
    with pytest.raises(ValueError, match="digest"):
        FeatureEngineer.from_state(json.dumps(document).encode())


def test_unknown_version_and_pickle_payload_rejected():
    """New versions and Python object serialization require explicit APIs, never guessing."""
    fe = _engineer()
    document = json.loads(fe.export_state())
    document["format_version"] = 2
    with pytest.raises(ValueError, match="version"):
        FeatureEngineer.from_state(json.dumps(document).encode())
    with pytest.raises(ValueError):
        FeatureEngineer.from_state(pickle.dumps(fe))


def test_unsupported_nodes_and_unserializable_configuration_rejected():
    """A local-only or custom object pipeline must not acquire a misleading portable label."""
    fe = FeatureEngineer(
        [{"name": "range", "transformer": "MinMaxScaler", "params": {"columns": ["x"]}}]
    )
    fe.fit_transform(pd.DataFrame({"x": [1.0, 3.0]}))
    with pytest.raises(ValueError, match="[Uu]nsupported"):
        fe.export_state()
    portable = _engineer()
    portable.fitted_steps[0]["params"]["custom"] = object()
    with pytest.raises(TypeError, match="[Uu]nsupported"):
        portable.export_state()


def test_spark_context_is_rebound_and_protected_features_rejected():
    """Runtime identity comes from the caller and must not overlap learned feature columns."""
    payload = _engineer().export_state()
    with pytest.raises(ValueError, match="frame_spec"):
        FeatureEngineer.from_state(payload, execution_options=ExecutionOptions("spark"))
    with pytest.raises(ValueError, match="row_keys|target|protected"):
        FeatureEngineer.from_state(
            payload, frame_spec=FrameSpec(("x",)), execution_options=ExecutionOptions("spark")
        )
    restored = FeatureEngineer.from_state(
        payload, frame_spec=FrameSpec(("new_id",)), execution_options=ExecutionOptions("spark")
    )
    assert restored.frame_spec == FrameSpec(("new_id",))
    assert restored.export_state() == payload


def test_disabled_scaler_and_nonfinite_constant_configuration_round_trip():
    """Tagged configuration must retain NaN scalars and disabled optional statistics."""
    fe = FeatureEngineer(
        [
            {
                "name": "fill",
                "transformer": "SimpleImputer",
                "params": {"columns": ["x"], "strategy": "constant", "fill_value": float("nan")},
            },
            {
                "name": "scale",
                "transformer": "StandardScaler",
                "params": {"columns": ["x"], "with_mean": False, "with_std": False},
            },
        ]
    )
    data = pl.DataFrame({"x": [1.0, None]})
    fe.fit_transform(data)
    restored = FeatureEngineer.from_state(fe.export_state())
    assert np.isnan(restored.steps_config[0]["params"]["fill_value"])
    assert restored.fitted_steps[1]["artifact"]["mean"] is None
    assert restored.export_state() == fe.export_state()


def test_compact_unicode_payload_obeys_received_wire_budget():
    """Canonical escaping of nested state must not impose a second artificial input limit."""
    column = "ölçüm" * 50
    fe = FeatureEngineer(
        [{"name": "fill", "transformer": "SimpleImputer", "params": {"columns": [column]}}]
    )
    fe.fit_transform(pd.DataFrame({column: [1.0, 3.0]}))
    payload = json.dumps(
        json.loads(fe.export_state()), ensure_ascii=False, separators=(",", ":")
    ).encode()
    restored = FeatureEngineer.from_state(
        payload, execution_options=ExecutionOptions("pandas", state_max_bytes=len(payload))
    )
    assert restored.transform(pd.DataFrame({column: [np.nan]})).iloc[0, 0] == 2.0


def test_escaped_surrogate_names_keep_previous_json_compatibility():
    """Compact UTF-8 encoding must still preserve names previously represented with escapes."""
    column = "x\ud800"
    fe = FeatureEngineer(
        [{"name": "fill", "transformer": "SimpleImputer", "params": {"columns": [column]}}]
    )
    fe.fit_transform(pd.DataFrame({column: [1.0, 3.0]}))
    restored = FeatureEngineer.from_state(fe.export_state())
    assert restored.transform(pd.DataFrame({column: [np.nan]})).iloc[0, 0] == 2.0


@pytest.mark.parametrize("tuning", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_model_pipeline_exports_only_successful_preprocessing(tuning, empty):
    """Tuning must publish exportable FE, and failed model refits must invalidate it."""
    modeling = {"type": "ridge_regression"}
    if tuning:
        modeling = {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "ridge_regression"},
            "strategy": "grid",
            "metric": "r2",
            "search_space": {"alpha": [1.0]},
            "cv_folds": 2,
            "n_jobs": 1,
        }
    steps = (
        []
        if empty
        else [{"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}}]
    )
    pipeline = SkyulfPipeline({"preprocessing": steps, "modeling": modeling})
    data = pd.DataFrame({"x": np.arange(12, dtype=float), "target": np.arange(12) * 2.0})
    pipeline.fit(data, target_column="target")
    original = pipeline.feature_engineer
    restored = FeatureEngineer.from_state(original.export_state())
    query = pd.DataFrame({"x": [100.0]})
    pd.testing.assert_frame_equal(restored.transform(query), original.transform(query))
    with pytest.raises(ValueError):
        pipeline.fit(data.assign(target=np.nan), target_column="target")
    with pytest.raises(ValueError, match="fitted"):
        pipeline.feature_engineer.export_state()
