"""Exercise backend fold reconstruction, repeated fitting, and actual row membership."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from backend.data.catalog import FileSystemCatalog
from backend.ml_pipeline._execution.engine import PipelineEngine
from backend.ml_pipeline._execution.schemas import NodeConfig, PipelineConfig
from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from skyulf.modeling.base import extract_xy
from skyulf.preprocessing.fold_adapter import FeatureEngineerFoldAdapter
from skyulf.preprocessing.pipeline import FeatureEngineer

_OPERATIONS = [
    ("StandardScaler", {"columns": ["x"]}),
    ("MinMaxScaler", {"columns": ["x"]}),
    ("MaxAbsScaler", {"columns": ["x"]}),
    ("RobustScaler", {"columns": ["x"]}),
    ("PowerTransformer", {"columns": ["x"], "method": "yeo-johnson"}),
    ("SimpleImputer", {"columns": ["x", "z"], "strategy": "mean"}),
    ("KNNImputer", {"columns": ["x", "z"], "n_neighbors": 2}),
    (
        "IterativeImputer",
        {"columns": ["x", "z"], "max_iter": 3, "random_state": 0, "estimator": "BayesianRidge"},
    ),
    ("MissingIndicator", {"columns": ["x"]}),
    ("DropMissingColumns", {"missing_threshold": 25}),
    ("KBinsDiscretizer", {"columns": ["x"], "strategy": "uniform", "n_bins": 3}),
    ("GeneralBinning", {"columns": ["x"], "strategy": "equal_width", "n_bins": 3}),
    ("CustomBinning", {"columns": ["x"], "bins": [-1000, 0, 10, 1000]}),
    ("VarianceThreshold", {"columns": ["x", "z"], "threshold": 0.0}),
    (
        "CorrelationThreshold",
        {"columns": ["x", "z"], "correlation_method": "pearson", "threshold": 0.7},
    ),
    (
        "UnivariateSelection",
        {"columns": ["x", "z"], "method": "select_k_best", "score_func": "f_classif", "k": 1},
    ),
    (
        "ModelBasedSelection",
        {
            "columns": ["x", "z"],
            "estimator": "logistic_regression",
            "method": "select_from_model",
            "threshold": "mean",
        },
    ),
    ("DummyEncoder", {"columns": ["city"]}),
    ("OneHotEncoder", {"columns": ["city"]}),
    ("LabelEncoder", {"columns": ["city"]}),
    ("OrdinalEncoder", {"columns": ["city"]}),
    ("HashEncoder", {"columns": ["city"]}),
    ("TargetEncoder", {"columns": ["city"], "target_column": "target"}),
    ("WOEEncoder", {"columns": ["city"], "regularization": 0.5}),
    ("count_vectorizer", {"columns": ["text"]}),
    ("tfidf_vectorizer", {"columns": ["text"]}),
    ("hashing_vectorizer", {"columns": ["text"], "n_features": 8}),
    ("Casting", {"column_types": {"city": "category"}}),
    (
        "GeneralTransformation",
        {"transformations": [{"column": "x", "method": "yeo-johnson"}]},
    ),
] + [
    (
        transformer,
        {
            "operations": [
                {
                    "operation_type": "group_agg",
                    "method": "mean",
                    "input_columns": ["city"],
                    "secondary_columns": ["x"],
                    "output_column": "city_x",
                }
            ]
        },
    )
    for transformer in ["FeatureGeneration", "FeatureGenerationNode", "FeatureMath"]
]


def _frame():
    """Provide persistent row identities and both categorical and numerical signal."""
    rng = np.random.default_rng(101)
    target = np.arange(96) % 2
    return pd.DataFrame(
        {
            "row_id": np.arange(96),
            "x": target + rng.normal(size=96),
            "z": rng.normal(size=96),
            "city": [f"city_{index % 4}" for index in range(96)],
            "text": [f"sample token{index % 5} word{index % 3}" for index in range(96)],
            "target": target,
        }
    )


def _native_frame(frame):
    """Compare real frame contents across the backend's supported dataframe wrappers."""
    if hasattr(frame, "to_native"):
        frame = frame.to_native()
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)


def _assert_payload_equal(actual, expected):
    """Check transformed values, feature names, and target alignment together."""
    pd.testing.assert_frame_equal(
        _native_frame(actual[0]), _native_frame(expected[0]), check_dtype=False
    )
    np.testing.assert_array_equal(np.asarray(actual[1]), np.asarray(expected[1]))


@pytest.mark.parametrize("transformer,params", _OPERATIONS, ids=[case[0] for case in _OPERATIONS])
@pytest.mark.parametrize("composite", [False, True], ids=["separate", "composite"])
def test_backend_replay_fits_each_fold_fresh_and_preserves_saved_state(
    tmp_path, transformer, params, composite
):
    """Backend replay must match a fresh fold fit without mutating the saved full-train pipeline."""
    frame = _frame()
    if transformer in {
        "SimpleImputer",
        "KNNImputer",
        "IterativeImputer",
        "MissingIndicator",
        "DropMissingColumns",
    }:
        frame.loc[frame.index % 3 == 0, "x"] = np.nan
    operation = {"name": "candidate", "transformer": transformer, "params": deepcopy(params)}
    split_params = {"target_column": "target", "test_size": 0.2, "random_state": 42}
    split_step = {"name": "split", "transformer": "TrainTestSplitter", "params": split_params}
    loader = NodeConfig("load", "data_loader")
    split = NodeConfig("split", "TrainTestSplitter", inputs=["load"], params=split_params)
    features = NodeConfig(
        "features",
        "feature_engineering",
        inputs=["load" if composite else "split"],
        params={"steps": [split_step, operation] if composite else [operation]},
    )
    model = NodeConfig("model", "training", inputs=["features"], params={"target_column": "target"})
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, FileSystemCatalog())
    nodes = [loader, features, model] if composite else [loader, split, features, model]
    engine._node_configs = {node.node_id: node for node in nodes}
    store.save("load", frame)
    if not composite:
        engine._run_transformer(split)
    engine._run_feature_engineering(features)
    resolved, fallback = engine._resolve_fold_preprocessing(model, "target")
    assert fallback is None
    assert resolved is not None
    adapter, (raw_x, raw_y), _validation = resolved
    raw_x = _native_frame(raw_x)
    np.testing.assert_array_equal(raw_x["row_id"], frame.loc[raw_x.index, "row_id"])
    np.testing.assert_array_equal(raw_y, frame.loc[raw_x.index, "target"])
    saved = store.load("features_pipeline")
    saved_before = saved.transform((raw_x.copy(), raw_y.copy()))
    halfway = len(raw_x) // 2
    for fold_slice in [slice(None, halfway), slice(halfway, None), slice(None, halfway)]:
        fold_x = raw_x.iloc[fold_slice].copy()
        fold_y = raw_y.iloc[fold_slice].copy()
        fresh = FeatureEngineer([deepcopy(operation)])
        expected, _metrics = fresh.fit_transform(
            (fold_x.copy(), fold_y.copy()), target_column="target"
        )
        actual = adapter.fit_transform(fold_x.copy(), fold_y.copy())
        _assert_payload_equal(actual, expected)
        _assert_payload_equal(
            adapter.transform(fold_x.copy(), fold_y.copy()),
            fresh.transform((fold_x.copy(), fold_y.copy())),
        )
    saved_after = saved.transform((raw_x.copy(), raw_y.copy()))
    _assert_payload_equal(saved_after, saved_before)


@pytest.mark.parametrize(
    "strategy,cv_enabled",
    [
        ("fixed", True),
        ("grid", True),
        ("random", True),
        ("halving_grid", True),
        ("grid", False),
        ("random", False),
    ],
    ids=["fixed-cv", "grid-cv", "random-cv", "halving-cv", "grid-holdout", "random-holdout"],
)
def test_real_tuning_fits_only_original_training_row_ids(
    tmp_path, monkeypatch, strategy, cv_enabled
):
    """Equal row counts cannot hide test or validation rows entering a preprocessing fit."""
    frame = _frame().drop(columns=["city", "text"])
    path = tmp_path / "rows.csv"
    frame.to_csv(path, index=False)
    fit_row_ids = []
    original_fit = FeatureEngineerFoldAdapter.fit_transform

    def record_fit(adapter, X, y):
        """Record the actual incoming identities while executing the real fold transformation."""
        fit_row_ids.append(set(_native_frame(X)["row_id"]))
        return original_fit(adapter, X, y)

    monkeypatch.setattr(FeatureEngineerFoldAdapter, "fit_transform", record_fit)
    store = LocalArtifactStore(str(tmp_path / "artifacts"))
    engine = PipelineEngine(store, FileSystemCatalog())
    params = {
        "target_column": "target",
        "algorithm": "logistic_regression",
        "evaluate": False,
        "cv_enabled": cv_enabled,
        "cv_folds": 2,
    }
    if strategy != "fixed":
        params.update(
            run_mode="tuned",
            tuning_config={
                "strategy": strategy,
                "search_space": {"C": [0.5, 2.0]},
                "n_trials": 2,
                "metric": "accuracy",
                "cv_enabled": cv_enabled,
                "cv_folds": 2,
                "random_state": 42,
            },
        )
    result = engine.run(
        PipelineConfig(
            "membership-audit",
            [
                NodeConfig("load", "data_loader", params={"path": str(path)}),
                NodeConfig(
                    "split",
                    "TrainTestSplitter",
                    inputs=["load"],
                    params={
                        "target_column": "target",
                        "test_size": 0.2,
                        "validation_size": 0.2,
                        "random_state": 42,
                    },
                ),
                NodeConfig("scale", "StandardScaler", inputs=["split"], params={"columns": ["x"]}),
                NodeConfig("model", "training", inputs=["scale"], params=params),
            ],
        )
    )
    assert result.status == "success", result.node_results["model"].error
    original = store.load("split")
    train_ids = set(_native_frame(extract_xy(original.train, "target")[0])["row_id"])
    test_ids = set(_native_frame(extract_xy(original.test, "target")[0])["row_id"])
    validation_ids = set(_native_frame(extract_xy(original.validation, "target")[0])["row_id"])
    assert fit_row_ids
    assert all(ids <= train_ids for ids in fit_row_ids)
    assert all(ids.isdisjoint(test_ids | validation_ids) for ids in fit_row_ids)
    assert any(len(ids) < len(train_ids) for ids in fit_row_ids) is cv_enabled
