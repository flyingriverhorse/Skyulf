"""Model sets preserve keyed component outcomes and independent rule eligibility."""

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.inference._manifest import ColumnSpec


def _api():
    """Make the initial missing scoring implementation a clear behavioral failure."""
    assert importlib.util.find_spec("skyulf.inference.model_set_scoring") is not None
    return importlib.import_module("skyulf.inference.model_set_scoring")


def test_keyed_assembly_preserves_input_order():
    """Reordered component results must attach to the original business records."""
    keys = pd.DataFrame({"id": [11, 7, 3]})
    result = _api().assemble_component_outputs(
        keys, pd.DataFrame({"id": [3, 11, 7], "a__prediction": [30, 110, 70]}), ["id"]
    )
    assert result["a__prediction"].tolist() == [110, 70, 30]


@pytest.mark.parametrize("ids", [[11, 11, 3], [11, 7], [11, 7, 4], [11, 7, None]])
def test_keyed_assembly_rejects_changed_key_set(ids):
    """Duplicate, missing, foreign and null keys must never yield a partial result."""
    with pytest.raises(ValueError, match="key"):
        _api().assemble_component_outputs(
            pd.DataFrame({"id": [11, 7, 3]}), pd.DataFrame({"id": ids}), ["id"]
        )


def _components():
    """Declare two direct estimator branches without synthetic outcome columns."""
    return tuple(
        SimpleNamespace(
            branch=name, output_schema=(ColumnSpec(name="prediction", dtype="float64"),)
        )
        for name in ("a", "b")
    )


SOURCE = """
import pandas as pd
def output(inputs, predictions, params):
    return pd.DataFrame({params['column']: predictions[params['branch'] + '__prediction'] * 2}, index=inputs.index)
"""


def _rules():
    """Give each rule an independent dependency and an explicit scalar output."""
    return {
        "outputs": [
            {
                "name": branch + "_rule",
                "version": "1",
                "function": "output",
                "params": {"column": branch + "_value", "branch": branch},
                "columns": [{"name": branch + "_value", "dtype": "float64"}],
                "required_components": [branch],
            }
            for branch in ("a", "b")
        ]
    }


def test_composition_excludes_only_the_dependent_rule():
    """An unavailable branch must not suppress independently computable business outputs."""
    predictions = pd.DataFrame(
        {
            "id": [1, 2],
            "a__prediction": [None, 4.0],
            "b__prediction": [5.0, 6.0],
            "a__scoring_status": ["excluded", "predicted"],
            "a__exclusion_reason": ["negative", None],
            "b__scoring_status": ["predicted", "predicted"],
            "b__exclusion_reason": [None, None],
        }
    )
    result = _api().compose_model_set_outputs(
        pd.DataFrame({"id": [1, 2]}), predictions, _rules(), SOURCE
    )
    assert result["b_value"].tolist() == [10.0, 12.0]
    assert result["a_value"].isna().tolist() == [True, False]
    assert result["a_rule__scoring_status"].tolist() == ["excluded", "predicted"]
    assert "negative" in result.loc[0, "a_rule__exclusion_reason"]


@pytest.mark.parametrize("dependencies", [[], ["unknown"], ["a", "a"], ["b_rule"]])
def test_composition_requires_direct_unique_dependencies(dependencies):
    """Invalid dependencies must fail during package validation before scoring."""
    config = _rules()
    config["outputs"][0]["required_components"] = dependencies
    with pytest.raises(ValueError, match="required_components"):
        _api().validate_model_set_composition(
            config, SOURCE, _components(), (ColumnSpec(name="id", dtype="int64"),)
        )


@pytest.fixture
def fitted_set(tmp_path):
    """Fit independent feature contracts using both supported local engines."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
    from skyulf.inference.model_set import ComponentReference, save_model_set
    from skyulf.pipeline import SkyulfPipeline

    components = {}
    for branch, engine, feature in (("a", "pandas", "x"), ("b", "polars", "y")):
        train = pd.DataFrame(
            {feature: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "target": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]}
        )
        config = {"preprocessing": [], "modeling": {"type": "linear_regression"}}
        if branch == "a":
            config["project_python_source"] = (
                "import pandas as pd\ndef eligibility(frame, params):\n    return pd.Series(['negative' if x < 0 else None for x in frame.x], index=frame.index)\n"
            )
            config["project_scoring"] = {
                "eligibility": [
                    {"name": "nonnegative", "version": "1", "function": "eligibility", "params": {}}
                ],
                "outputs": [],
            }
        pipeline = SkyulfPipeline(config)
        if engine == "polars":
            train = pl.from_pandas(train)
            split = SplitDataset(train=train.head(5), test=train.tail(1))
        else:
            split = SplitDataset(train=train.iloc[:5], test=train.iloc[5:])
        pipeline.fit(split, target_column="target")
        path = tmp_path / branch
        save_local_pipeline(pipeline, path)
        digest = load_local_pipeline(path).manifest.pipeline_sha256
        components[branch] = (
            ComponentReference(name="model_" + branch, version="1", digest=digest),
            path,
        )
    return save_model_set(
        tmp_path / "set",
        components,
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        composition_source=SOURCE,
        composition_config=_rules(),
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fitted_mixed_engine_set_preserves_independent_outcomes(fitted_set, engine):
    """Real fitted engines must score only their own inputs and retain excluded records."""
    frame = pd.DataFrame(
        {"id": [9, 3, 7], "x": [-1.0, 8.0, 9.0], "y": [10.0, 11.0, 12.0]}, index=[8, 8, 2]
    )
    native = pl.from_pandas(frame) if engine == "polars" else frame
    scored = _api().score_model_set(native, fitted_set)
    assert scored.frame.id.tolist() == [9, 3, 7]
    np.testing.assert_allclose(scored.frame["b_value"].astype(float), [40.0, 44.0, 48.0])
    np.testing.assert_allclose(scored.frame["a_value"].dropna().astype(float), [32.0, 36.0])
    assert scored.frame["a__scoring_status"].tolist() == ["excluded", "predicted", "predicted"]
    assert scored.frame.index.tolist() == ([0, 1, 2] if engine == "polars" else [8, 8, 2])
    assert list(scored.frame) == [c.name for c in fitted_set.manifest.output_schema]
    assert scored.history == {}


@pytest.mark.parametrize("dtype", [pl.Int8, pl.Int32, pl.Int64, pl.UInt64])
def test_nullable_polars_integer_input_matches_component_prediction(tmp_path, dtype):
    """The model-set pandas boundary must preserve a nullable fitted integer contract."""
    from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
    from skyulf.inference.local_scoring import score_local_pipeline
    from skyulf.inference.model_set import ComponentReference, save_model_set
    from skyulf.pipeline import SkyulfPipeline

    train = pl.DataFrame(
        {"x": pl.Series([1, 2, 3, 4, 5, 6], dtype=dtype), "target": [2, 4, 6, 8, 10, 12]}
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}}
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(train, target_column="target")
    path = tmp_path / "component"
    save_local_pipeline(pipeline, path)
    local = load_local_pipeline(path)
    artifact = save_model_set(
        tmp_path / "set",
        {
            "a": (
                ComponentReference(
                    name="model_a", version="1", digest=local.manifest.pipeline_sha256
                ),
                path,
            )
        },
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    frame = pl.DataFrame({"id": [3, 1, 2], "x": pl.Series([None, 7, 8], dtype=dtype)})
    expected = score_local_pipeline(frame.select("x"), local)

    actual = _api().score_model_set(frame, artifact).frame

    assert actual["id"].tolist() == [3, 1, 2]
    np.testing.assert_allclose(actual["a__prediction"].astype(float), expected["prediction"])


@pytest.mark.parametrize("dtype,value", [(pl.Int64, 2**53 + 1), (pl.UInt64, 2**64 - 1)])
def test_model_set_input_keeps_large_nullable_integers_exact(dtype, value):
    """Restoring an integer dtype after float conversion must not retain rounded values."""
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(
            record_key_columns=("id",),
            record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
            input_schema=(
                ColumnSpec(name="id", dtype="int64"),
                ColumnSpec(name="x", dtype="int64"),
            ),
        )
    )
    frame = pl.DataFrame({"id": [1, 2], "x": pl.Series([None, value], dtype=dtype)})

    raw = _api()._raw_input(frame, artifact, max_rows=10, max_bytes=10000)
    restored = pl.from_pandas(raw)

    assert restored["x"].to_list() == [None, value]
    assert restored.schema["x"] == dtype


@pytest.mark.parametrize(
    "column", ["ID", "A__prediction", "a_rule__scoring_status", "b_rule__exclusion_reason"]
)
def test_rule_columns_cannot_overwrite_keys_components_or_outcomes(column):
    """Case-insensitive collisions must fail before any executable callbacks run."""
    config = _rules()
    config["outputs"][0]["columns"][0]["name"] = column
    with pytest.raises(ValueError, match="collid"):
        _api().validate_model_set_composition(
            config, SOURCE, _components(), (ColumnSpec(name="id", dtype="int64"),)
        )


@pytest.mark.parametrize("expression", ["'wrong'", "None", "float('nan')"])
def test_rule_rejects_missing_or_incompatible_values(expression):
    """Eligible business outputs cannot silently become null or change their declared type."""
    source = SOURCE.replace("predictions[params['branch'] + '__prediction'] * 2", expression)
    predictions = pd.DataFrame(
        {"a__prediction": [2.0], "a__scoring_status": ["predicted"], "a__exclusion_reason": [None]}
    )
    config = _rules()
    config["outputs"] = config["outputs"][:1]
    with pytest.raises(ValueError, match="incompatible|missing"):
        _api().compose_model_set_outputs(pd.DataFrame({"id": [1]}), predictions, config, source)


def test_saved_source_tampering_after_load_fails(fitted_set):
    """A loaded artifact must not execute source replaced after package validation."""
    (fitted_set.directory / "composition.py").write_text(
        SOURCE + "\n# replaced\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="checksum|digest|changed|disagree"):
        _api().predict_model_set(pd.DataFrame({"id": [1], "x": [1.0], "y": [2.0]}), fitted_set)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_composition_cannot_depend_on_undeclared_raw_inputs(fitted_set, tmp_path, engine):
    """Local validation cannot certify inputs missing from the batch and PyFunc schemas."""
    from skyulf.inference.model_set import save_model_set

    artifact = save_model_set(
        tmp_path / "hidden_dependency",
        {
            component.branch: (
                component.reference,
                fitted_set.directory / "components" / component.branch,
            )
            for component in fitted_set.manifest.components
        },
        record_key_schema=fitted_set.manifest.record_key_schema,
        composition_source=SOURCE.replace("* 2", "* inputs['multiplier']"),
        composition_config=_rules(),
    )
    frame = pd.DataFrame({"id": [1], "x": [2.0], "y": [3.0], "multiplier": [2.0]})
    native = pl.from_pandas(frame) if engine == "polars" else frame
    with pytest.raises(KeyError, match="multiplier"):
        _api().predict_model_set(native, artifact)


@pytest.mark.parametrize("kwargs", [{"max_rows": 1}, {"max_bytes": 1}, {"max_rows": True}])
def test_scoring_bounds_native_input_before_work(fitted_set, kwargs):
    """Caller budgets must reject oversized requests rather than unbounded conversion."""
    with pytest.raises(ValueError, match="max_rows|max_bytes"):
        _api().predict_model_set(
            pd.DataFrame({"id": [1, 2], "x": [1.0, 2.0], "y": [2.0, 3.0]}), fitted_set, **kwargs
        )


def test_saved_set_replays_in_isolated_process_without_original_models(fitted_set):
    """A shipped set must carry everything needed after original component sources disappear."""
    for name in ("a", "b"):
        shutil.rmtree(fitted_set.directory.parent / name)
    code = """
import json, sys
import pandas as pd
from skyulf.inference.model_set import load_model_set
from skyulf.inference.model_set_scoring import predict_model_set
artifact = load_model_set(sys.argv[1])
frame = pd.DataFrame({'id':[1], 'x':[2.], 'y':[3.]})
result = predict_model_set(frame, artifact)
print(json.dumps({'a':float(result.a_value.iloc[0]), 'b':float(result.b_value.iloc[0])}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code, str(fitted_set.directory)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == pytest.approx({"a": 8.0, "b": 12.0})


@pytest.fixture
def temporal_set(tmp_path):
    """Package separate temporal models so each branch owns its continuation."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
    from skyulf.inference.model_set import ComponentReference, save_model_set
    from skyulf.pipeline import SkyulfPipeline

    components = {}
    for branch, engine in (("a", "pandas"), ("b", "polars")):
        frame = pd.DataFrame({"t": np.arange(20, dtype=np.int64), "v": np.arange(20, dtype=float)})
        frame["target"] = frame.v.rolling(3, min_periods=1).mean()
        native = pl.from_pandas(frame) if engine == "polars" else frame
        pipeline = SkyulfPipeline(
            {
                "preprocessing": [
                    {
                        "name": "rolling",
                        "transformer": "RollingAggregate",
                        "params": {
                            "columns": ["v"],
                            "window": 3,
                            "sort_by": "t",
                            "history_mode": "carry",
                        },
                    },
                    {
                        "name": "drop_clock",
                        "transformer": "DropMissingColumns",
                        "params": {"columns": ["t", "v"], "missing_threshold": None},
                    },
                ],
                "modeling": {"type": "linear_regression"},
            }
        )
        pipeline.fit(SplitDataset(train=native[:16], test=native[16:]), target_column="target")
        path = tmp_path / branch
        save_local_pipeline(pipeline, path)
        components[branch] = (
            ComponentReference(
                name="temporal_" + branch,
                version="1",
                digest=load_local_pipeline(path).manifest.pipeline_sha256,
            ),
            path,
        )
    return save_model_set(
        tmp_path / "temporal_set",
        components,
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )


def test_temporal_set_continuation_matches_uninterrupted_prediction(temporal_set):
    """One receipt must advance both engines causally without mutating the caller's state."""
    api = _api()
    frame = pd.DataFrame(
        {"id": [20, 21, 22, 23], "t": [20, 21, 22, 23], "v": [20.0, 21.0, 22.0, 23.0]}
    )
    expected = api.score_model_set(frame, temporal_set)
    first = api.score_model_set(frame.iloc[:2], temporal_set)
    before = deepcopy(first.history)
    second = api.score_model_set(frame.iloc[2:], temporal_set, history_state=first.history)
    result = pd.concat([first.frame, second.frame])
    for branch in ("a", "b"):
        np.testing.assert_allclose(
            result[f"{branch}__prediction"].astype(float),
            expected.frame[f"{branch}__prediction"].astype(float),
        )
    assert set(second.history["components"]) == {"a", "b"}
    assert first.history == before


def test_temporal_set_bootstrap_rebuilds_from_complete_snapshot(temporal_set):
    """Full-refresh history must start empty instead of replaying a future training seed."""
    frame = pd.DataFrame({"id": [0, 1, 2], "t": [0, 1, 2], "v": [0.0, 1.0, 2.0]})
    result = _api().score_model_set(frame, temporal_set, bootstrap_history=True)
    np.testing.assert_allclose(
        result.frame["a__prediction"].astype(float), [0.0, 0.5, 1.0], atol=1e-12
    )
    assert set(result.history["components"]) == {"a", "b"}


@pytest.mark.parametrize("change", ["missing", "different_set", "unknown", "static_state"])
def test_temporal_set_rejects_incompatible_history(temporal_set, change):
    """Continuation cannot silently reset one branch or mix complete set versions."""
    frame = pd.DataFrame({"id": [20], "t": [20], "v": [20.0]})
    history = _api().score_model_set(frame, temporal_set).history
    if change == "missing":
        history["components"].pop("a")
    elif change == "different_set":
        history["model_set_sha256"] = "0" * 64
    elif change == "unknown":
        history["components"]["other"] = {}
    else:
        history["components"]["a"] = {"version": 1, "model_id": "other", "steps": {}}
    with pytest.raises(ValueError, match="history|History"):
        _api().score_model_set(
            pd.DataFrame({"id": [21], "t": [21], "v": [21.0]}), temporal_set, history_state=history
        )


def test_component_failure_never_returns_partial_set(fitted_set, monkeypatch):
    """Later component failures must prevent exposure of earlier successful outcomes."""
    api = _api()
    original = api.score_local_pipeline
    calls = []

    def failing(frame, artifact):
        """Allow the first branch to succeed and fail the second estimator."""
        calls.append(artifact.manifest.fitted_engine)
        if len(calls) == 2:
            raise RuntimeError("failed component")
        return original(frame, artifact)

    monkeypatch.setattr(api, "score_local_pipeline", failing)
    with pytest.raises(RuntimeError, match="failed component"):
        api.score_model_set(pd.DataFrame({"id": [1], "x": [2.0], "y": [3.0]}), fitted_set)
    assert calls == ["pandas", "polars"]


def test_empty_set_prediction_keeps_declared_schema(fitted_set):
    """An empty bounded request must not invoke estimators that require sample rows."""
    frame = pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
            "x": pd.Series(dtype="float64"),
            "y": pd.Series(dtype="float64"),
        }
    )
    result = _api().score_model_set(frame, fitted_set)
    assert result.frame.empty
    assert list(result.frame) == [column.name for column in fitted_set.manifest.output_schema]


def test_empty_mixed_engine_string_inputs_keep_declared_output(tmp_path):
    """Empty string columns cannot block a complete refresh through Polars null inference."""
    from tests.integration.platforms.test_local_pipeline_artifact import _fitted_pipeline

    from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
    from skyulf.inference.model_set import ComponentReference, save_model_set

    components = {}
    for engine in ("pandas", "polars"):
        pipeline, _ = _fitted_pipeline(engine)
        path = tmp_path / engine
        save_local_pipeline(pipeline, path)
        components[engine] = (
            ComponentReference(
                name=engine,
                version="1",
                digest=load_local_pipeline(path).manifest.pipeline_sha256,
            ),
            path,
        )
    artifact = save_model_set(
        tmp_path / "set",
        components,
        record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
    )
    frame = pd.DataFrame(
        {
            "id": pd.Series(dtype="int64"),
            "city": pd.Series(dtype="object"),
            "amount": pd.Series(dtype="float64"),
        }
    )
    result = _api().score_model_set(frame, artifact)
    assert result.frame.empty
    assert list(result.frame) == [column.name for column in artifact.manifest.output_schema]


def test_null_component_history_cannot_silently_reset_carry(temporal_set):
    """A present but null history entry must not masquerade as valid continuation."""
    frame = pd.DataFrame({"id": [20], "t": [20], "v": [20.0]})
    history = _api().score_model_set(frame, temporal_set).history
    history["components"]["a"] = None
    with pytest.raises(ValueError, match="history|History"):
        _api().score_model_set(
            pd.DataFrame({"id": [21], "t": [21], "v": [21.0]}), temporal_set, history_state=history
        )


def test_empty_temporal_batch_preserves_continuation(temporal_set):
    """No rows must preserve every previous temporal tail rather than clear it."""
    frame = pd.DataFrame({"id": [20], "t": [20], "v": [20.0]})
    first = _api().score_model_set(frame, temporal_set)
    result = _api().score_model_set(frame.iloc[:0], temporal_set, history_state=first.history)
    assert result.history == first.history
    assert result.frame.empty
