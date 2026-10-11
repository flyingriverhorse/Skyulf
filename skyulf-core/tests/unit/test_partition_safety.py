"""Partition preflight must reject batch-dependent inference before worker execution."""

from dataclasses import FrozenInstanceError, replace
from importlib import import_module, util

import numpy as np
import pandas as pd
import pytest

from skyulf.core.capabilities import UnsupportedExecutionError
from skyulf.data.dataset import SplitDataset
from skyulf.inference.fitted_pipeline import load_pipeline, predict_pipeline, save_pipeline
from skyulf.pipeline import SkyulfPipeline


def _gate(artifact):
    """The public admission API is required before distributed publication."""
    name = "skyulf.inference.partition_safety"
    assert util.find_spec(name) is not None, "Missing partition-safety preflight"
    return import_module(name).require_partition_safe_pipeline(artifact)


@pytest.fixture
def artifact(tmp_path):
    """Real fitted and reloaded state catches configuration and transport drift."""
    train = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "y": [3.0, 5.0, 7.0, 9.0, 11.0, 13.0]}
    )
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}},
                {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=train.iloc[:4], test=train.iloc[4:]), target_column="y")
    save_pipeline(pipeline, tmp_path / "artifact")
    return load_pipeline(tmp_path / "artifact")


def test_admitted_composition_matches_split_batches(artifact):
    """Missing rows and uneven batches must produce the same keyed model values."""
    evidence = _gate(artifact)
    query = pd.DataFrame({"x": [np.nan, 20.0, -1.0, np.nan, 7.0]}, index=[7, 3, 99, 1, 4])
    whole = predict_pipeline(query, artifact)
    split = pd.concat(
        [predict_pipeline(query.iloc[s], artifact) for s in [slice(0, 1), slice(1, 3), slice(3, 5)]]
    )
    pd.testing.assert_frame_equal(whole, split, atol=1e-12, rtol=1e-12)
    assert evidence.pipeline_sha256 == artifact.manifest.pipeline_sha256
    assert evidence.output_schema == (("prediction", "float64"),)
    assert _gate(artifact) == evidence
    with pytest.raises(FrozenInstanceError):
        evidence.fitted_engine = "polars"


@pytest.mark.parametrize(
    "node",
    [
        "Unknown",
        "PowerTransformer",
        "GeneralTransformation",
        "ValueReplacement",
        "Binning",
        "IQR",
        "LagFeatures",
        "RollingAggregate",
        "ColumnFunction",
        "FittedFunction",
    ],
)
def test_unsafe_nodes_rejected_with_step_identity(artifact, node):
    """Unknown, row-changing, history and batch-fallback bodies cannot gain admission."""
    artifact.pipeline.feature_engineer.fitted_steps[0]["type"] = node
    with pytest.raises(UnsupportedExecutionError) as error:
        _gate(artifact)
    assert node in str(error.value)
    assert "fill" in str(error.value)


def test_wrong_engine_rejected(artifact):
    """A fitted Polars artifact must never silently switch worker engines."""
    changed = replace(
        artifact, manifest=artifact.manifest.model_copy(update={"fitted_engine": "polars"})
    )
    with pytest.raises(UnsupportedExecutionError, match="pandas"):
        _gate(changed)


@pytest.mark.parametrize(
    "change", ["strategy", "flags", "columns", "state", "applier", "history", "recipe"]
)
def test_state_config_and_implementation_mismatch_rejected(artifact, change):
    """A stale recipe or substituted implementation invalidates the safety promise."""
    steps = artifact.pipeline.feature_engineer.fitted_steps
    if change == "strategy":
        steps[0]["params"]["strategy"] = "constant"
    elif change == "flags":
        steps[1]["artifact"]["with_mean"] = False
    elif change == "columns":
        steps[0]["params"]["columns"] = ["other"]
    elif change == "state":
        steps[1]["artifact"]["mean"] = []
    elif change == "applier":
        steps[0]["applier"] = object()
    elif change == "history":
        steps[0]["artifact"]["history_mode"] = "carry"
    else:
        artifact.pipeline.feature_engineer.steps_config[0]["params"]["strategy"] = "constant"
    with pytest.raises(UnsupportedExecutionError):
        _gate(artifact)


@pytest.mark.parametrize("scoring", [{"eligibility": []}, {"pre_split": {"rules": ["custom"]}}])
def test_scoring_and_reused_pre_split_rejected_without_callbacks(artifact, scoring):
    """Whole-frame scoring policies require a separate proof even with safe FE."""
    artifact.pipeline.config["project_scoring"] = scoring
    with pytest.raises(UnsupportedExecutionError, match="project_scoring"):
        _gate(artifact)


def test_custom_model_rejected_without_predict(artifact):
    """Safe preprocessing cannot certify a batch-global estimator."""

    class BatchModel:
        """Would yield population-dependent predictions if accidentally invoked."""

        def predict(self, values):
            """A gate must never execute estimator code."""
            raise AssertionError("Prediction ran during preflight")

    artifact.pipeline.model_estimator.model = BatchModel()
    with pytest.raises(UnsupportedExecutionError, match="model"):
        _gate(artifact)


@pytest.mark.parametrize("strategy", ["mean", "constant"])
@pytest.mark.parametrize("flags", [(True, True), (True, False), (False, True), (False, False)])
def test_node_options_and_nullable_composition_parity(tmp_path, strategy, flags):
    """All admitted scaler flags and imputer strategies retain per-row semantics."""
    train = pd.DataFrame(
        {"x": pd.Series([1, 2, 3, 4, 5, 6], dtype="Int64"), "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    )
    params = {"columns": ["x"], "strategy": strategy}
    if strategy == "constant":
        params["fill_value"] = 0
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "fill", "transformer": "SimpleImputer", "params": params},
                {
                    "name": "scale",
                    "transformer": "StandardScaler",
                    "params": {"columns": ["x"], "with_mean": flags[0], "with_std": flags[1]},
                },
            ],
            "modeling": {"type": "linear_regression"},
        }
    )
    pipeline.fit(SplitDataset(train=train.iloc[:4], test=train.iloc[4:]), target_column="y")
    save_pipeline(pipeline, tmp_path / "options")
    loaded = load_pipeline(tmp_path / "options")
    certificate = _gate(loaded)
    query = pd.DataFrame({"x": pd.Series([None, 2**53 + 1, 2, None], dtype="Int64")})
    whole = predict_pipeline(query, loaded)
    split = pd.concat([predict_pipeline(query.iloc[[i]], loaded) for i in range(len(query))])
    pd.testing.assert_frame_equal(whole, split)
    assert _gate(loaded) == certificate


@pytest.mark.parametrize("use_thresholds", [False, True])
def test_logistic_probability_and_threshold_parity(tmp_path, use_thresholds):
    """Classification probabilities and saved thresholds must be partition invariant."""
    train = pd.DataFrame(
        {"x": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, -4.0, 4.0], "y": [0, 0, 0, 1, 1, 1, 0, 1]}
    )
    pipeline = SkyulfPipeline({"preprocessing": [], "modeling": {"type": "logistic_regression"}})
    pipeline.fit(SplitDataset(train=train.iloc[:6], test=train.iloc[6:]), target_column="y")
    if use_thresholds:
        pipeline._tuned_thresholds = {0: 0.4, 1: 0.6}
    save_pipeline(pipeline, tmp_path / "classification", use_tuned_thresholds=use_thresholds)
    loaded = load_pipeline(tmp_path / "classification")
    certificate = _gate(loaded)
    query = pd.DataFrame({"x": [-8.0, 0.0, 0.001, 9.0]})
    whole = predict_pipeline(query, loaded)
    split = pd.concat([predict_pipeline(query.iloc[[i]], loaded) for i in range(len(query))])
    pd.testing.assert_frame_equal(whole, split, atol=1e-14, rtol=1e-14)
    assert certificate.output_schema == (
        ("prediction", "int64"),
        ("probability_0", "float64"),
        ("probability_1", "float64"),
    )


@pytest.mark.parametrize("node", ["Deduplicate", "DropMissingRows", "RowFilterFunction"])
def test_reviewed_preserve_rows_skips_never_execute(artifact, node):
    """Train-only filtering callbacks must remain skipped even for unsafe query rows."""
    from skyulf.registry import NodeRegistry

    engineer = artifact.pipeline.feature_engineer
    recipe = {"name": "filter", "transformer": node, "params": {}}
    record = {
        "name": "filter",
        "type": node,
        "applier": NodeRegistry.get_applier(node)(),
        "params": {},
        "artifact": {"function": "would_fail_if_executed"},
    }
    engineer.steps_config = [recipe, *engineer.steps_config]
    artifact.pipeline.config["preprocessing"] = engineer.steps_config
    engineer.fitted_steps.insert(0, record)
    certificate = _gate(artifact)
    query = pd.DataFrame({"x": [np.nan, 5.0, 5.0]})
    expected = predict_pipeline(query, artifact)
    actual = pd.concat([predict_pipeline(query.iloc[[i]], artifact) for i in range(3)])
    pd.testing.assert_frame_equal(expected, actual)
    assert certificate.steps[0].action == "skip_preserve_rows"
    assert len(actual) == 3


@pytest.mark.parametrize(
    "context,kind,effect",
    [
        ("global", "python_batch", "preserve"),
        ("row", "native", "preserve"),
        ("row", "python_batch", "filter"),
    ],
)
def test_capability_must_describe_worker_semantics(artifact, monkeypatch, context, kind, effect):
    """Engine matching alone cannot admit native, population or filtering behavior."""
    from skyulf.core.capabilities import ExecutionCapability
    from skyulf.preprocessing.imputation.simple import SimpleImputerCalculator

    monkeypatch.setattr(
        SimpleImputerCalculator,
        "__execution_capabilities__",
        (ExecutionCapability("pandas", "apply", kind, effect, context),),
    )
    with pytest.raises(UnsupportedExecutionError, match="No declared support"):
        _gate(artifact)


def test_evidence_detaches_fitted_values(artifact):
    """The worker certificate must change when learned state changes after preflight."""
    evidence = _gate(artifact)
    artifact.pipeline.feature_engineer.fitted_steps[0]["artifact"]["fill_values"]["x"] = 100.0
    changed = _gate(artifact)
    assert changed.state_sha256 != evidence.state_sha256
    assert changed.steps[0].state_sha256 != evidence.steps[0].state_sha256


def test_captured_project_source_is_not_executed(artifact):
    """Dormant project helpers do not make the known effective inference chain unsafe."""
    artifact.pipeline.config["project_python_source"] = (
        "raise AssertionError('gate executed project code')"
    )
    assert _gate(artifact).fitted_engine == "pandas"


@pytest.mark.parametrize(
    "change", ["empty_columns", "skip_contract", "active_contract", "source_recipe"]
)
def test_effective_chain_and_explicit_selection_cannot_drift(artifact, change):
    """Admission must use the real preserve-rows chain and all retained recipes."""
    engineer = artifact.pipeline.feature_engineer
    if change == "empty_columns":
        engineer.fitted_steps[0]["params"]["columns"] = []
    elif change == "active_contract":
        engineer._RESAMPLING_TYPES = {"SimpleImputer"}
    elif change == "source_recipe":
        artifact.pipeline.config["preprocessing"] = []
    else:
        from skyulf.registry import NodeRegistry

        engineer.steps_config = [
            {"name": "drop", "transformer": "Deduplicate", "params": {}},
            *engineer.steps_config,
        ]
        engineer.fitted_steps.insert(
            0,
            {
                "name": "drop",
                "type": "Deduplicate",
                "applier": NodeRegistry.get_applier("Deduplicate")(),
                "artifact": {},
                "params": {},
            },
        )
        engineer._ROW_DROPPING_TYPES = set()
    with pytest.raises(UnsupportedExecutionError):
        _gate(artifact)


@pytest.mark.parametrize("engine", ["polars", "spark"])
def test_fitted_engine_options_must_match_worker(artifact, engine):
    """Explicit retained engine overrides must not force workers into another runtime."""
    from skyulf.core.execution import ExecutionOptions

    artifact.pipeline.feature_engineer.execution_options = ExecutionOptions(engine=engine)
    with pytest.raises(UnsupportedExecutionError, match="engine"):
        _gate(artifact)


def test_empty_and_all_null_feature_batches_preserve_schema(artifact):
    """Empty Arrow partitions and all-null batches retain the admitted feature schema."""
    certificate = _gate(artifact)
    engineer = artifact.pipeline.feature_engineer
    empty = pd.DataFrame({"x": pd.Series([], dtype="float64")})
    all_null = pd.DataFrame({"x": [np.nan, np.nan]})
    empty_result = engineer.transform(empty, preserve_rows=True)
    all_null_result = engineer.transform(all_null, preserve_rows=True)
    assert empty_result.shape == (0, 1)
    assert empty_result.dtypes.to_dict() == all_null_result.dtypes.to_dict()
    assert all_null_result["x"].notna().all()
    assert _gate(artifact) == certificate


def test_private_estimator_callback_is_rejected(artifact):
    """GI-1: sklearn predict delegates to private methods that can be batch dependent."""
    artifact.pipeline.model_estimator.model._decision_function = lambda x: np.full(
        len(x), np.asarray(x).mean()
    )
    with pytest.raises(UnsupportedExecutionError, match="Overridden"):
        _gate(artifact)


@pytest.fixture(params=["linear_regression", "logistic_regression"])
def tuned_artifact(tmp_path, request):
    """Fit the actual tuple/wrapper representation generated bundles use."""
    classifier = request.param == "logistic_regression"
    x = [-4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, -5.0, 5.0]
    train = pd.DataFrame({"x": x, "y": [int(v > 0) if classifier else 2 * v + 1 for v in x]})
    pipeline = SkyulfPipeline(
        {
            "preprocessing": [
                {"name": "fill", "transformer": "SimpleImputer", "params": {"columns": ["x"]}},
                {"name": "scale", "transformer": "StandardScaler", "params": {"columns": ["x"]}},
            ],
            "modeling": {
                "type": "hyperparameter_tuner",
                "base_model": {"type": request.param},
                "strategy": "grid",
                "metric": "accuracy" if classifier else "mse",
                "search_space": {},
                "cv_folds": 2,
                "n_trials": 1,
                "n_jobs": 1,
            },
        }
    )
    pipeline.fit(SplitDataset(train=train.iloc[:8], test=train.iloc[8:]), target_column="y")
    save_pipeline(pipeline, tmp_path / "tuned")
    return load_pipeline(tmp_path / "tuned")


def test_real_tuner_partition_and_composition_parity(tuned_artifact):
    """Generated tuned models retain regression or probability parity across batches."""
    evidence = _gate(tuned_artifact)
    query = pd.DataFrame({"x": [np.nan, -2.0, 3.0, 5.0]}, index=[9, 4, 6, 1])
    whole = predict_pipeline(query, tuned_artifact)
    split = pd.concat(
        [predict_pipeline(query.iloc[[i]], tuned_artifact) for i in range(len(query))]
    )
    pd.testing.assert_frame_equal(whole, split, rtol=1e-12, atol=1e-12)
    assert _gate(tuned_artifact) == evidence


def test_tuning_result_is_bound_in_certificate(tuned_artifact):
    """The certificate must cover the whole tuning tuple rather than only its model."""
    evidence = _gate(tuned_artifact)
    tuned_artifact.pipeline.model_estimator.model[1].best_score += 0.25
    assert _gate(tuned_artifact).state_sha256 != evidence.state_sha256


@pytest.mark.parametrize(
    "change",
    [
        "tuple_length",
        "result_type",
        "wrapper_extra",
        "base_extra",
        "base_type",
        "result_callback",
        "model_callback",
        "excluded",
        "bad_threshold",
    ],
)
def test_tuning_wrapper_changes_are_rejected(tuned_artifact, change):
    """A reviewed tuning facade must not authorize custom or malformed serving behavior."""
    estimator = tuned_artifact.pipeline.model_estimator
    if change == "tuple_length":
        estimator.model = (*estimator.model, None)
    elif change == "result_type":
        estimator.model = (estimator.model[0], object())
    elif change == "wrapper_extra":
        estimator.applier.custom = True
    elif change == "base_extra":
        estimator.applier.base_applier.predict = lambda *args: None
    elif change == "base_type":
        estimator.applier.base_applier = object()
    elif change == "result_callback":
        estimator.model[1].custom = lambda *args: None
    elif change == "model_callback":
        estimator.model[0]._decision_function = lambda x: np.zeros(len(x))
    elif change == "excluded":
        estimator.model[1].excluded_feature_columns = ["x"]
    else:
        estimator.model[1].decision_thresholds = {0: -1.0, 1: 1.0}
    with pytest.raises(UnsupportedExecutionError):
        _gate(tuned_artifact)


@pytest.mark.parametrize("tuned_artifact", ["logistic_regression"], indirect=True)
@pytest.mark.parametrize("pipeline_thresholds", [False, True])
def test_tuner_threshold_modes_preserve_batch_parity(tuned_artifact, tmp_path, pipeline_thresholds):
    """Both wrapper-owned and explicitly selected pipeline thresholds stay row-local."""
    before = _gate(tuned_artifact)
    pipeline = tuned_artifact.pipeline
    pipeline.model_estimator.model[1].decision_thresholds = {0: 0.9, 1: 0.1}
    assert _gate(tuned_artifact).state_sha256 != before.state_sha256
    if pipeline_thresholds:
        pipeline._tuned_thresholds = {0: 0.3, 1: 0.7}
    save_pipeline(pipeline, tmp_path / "thresholds", use_tuned_thresholds=pipeline_thresholds)
    loaded = load_pipeline(tmp_path / "thresholds")
    evidence = _gate(loaded)
    query = pd.DataFrame({"x": [-0.9, 0.0, 0.5, np.nan]})
    whole = predict_pipeline(query, loaded)
    split = pd.concat([predict_pipeline(query.iloc[[i]], loaded) for i in range(len(query))])
    pd.testing.assert_frame_equal(whole, split, rtol=1e-12, atol=1e-12)
    assert _gate(loaded) == evidence
