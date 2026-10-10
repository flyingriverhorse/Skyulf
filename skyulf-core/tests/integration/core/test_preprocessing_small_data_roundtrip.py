"""Every registered preprocessor executes real small-data saved-state replay."""

import pickle
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.data.dataset import SplitDataset
from skyulf.preprocessing.function_steps import function_ref
from skyulf.registry import NodeRegistry


def _ratio(frame):
    """Compute a real custom feature without learning request statistics."""
    return frame["x"] / frame["b"]


def _learn_mean(frame, target):
    """Capture a training mean in the custom fitted-function state."""
    return {"mean": float(frame["missing"].mean())}


def _fill_mean(frame, state):
    """Use the saved mean when applying the fitted function."""
    return frame["missing"].fillna(state["mean"])


def _keep(frame):
    """Select identifiable rows through the real custom filter wrapper."""
    return frame["row"] % 2 == 0


_RECIPES: dict[str, dict[str, Any]] = {
    **{
        name: {"columns": ["x"]}
        for name in (
            "MinMaxScaler",
            "MaxAbsScaler",
            "StandardScaler",
            "RobustScaler",
            "PowerTransformer",
            "IQR",
            "ZScore",
            "Winsorize",
        )
    },
    **{
        name: {"columns": ["cat"]}
        for name in (
            "DummyEncoder",
            "OneHotEncoder",
            "OrdinalEncoder",
            "LabelEncoder",
            "HashEncoder",
            "TargetEncoder",
            "WOEEncoder",
        )
    },
    **{
        name: {"columns": ["missing", "b"]}
        for name in (
            "SimpleImputer",
            "KNNImputer",
            "IterativeImputer",
        )
    },
    **{
        name: {"columns": ["text"], "drop_original": True}
        for name in (
            "count_vectorizer",
            "tfidf_vectorizer",
            "hashing_vectorizer",
            "tokenizer",
        )
    },
    **{
        name: {"columns": ["x", "b"], "degree": 2}
        for name in (
            "FeatureInteraction",
            "PolynomialFeaturesNode",
            "PolynomialFeatures",
        )
    },
    **{
        name: {"columns": ["x", "constant"]}
        for name in (
            "VarianceThreshold",
            "ModelBasedSelection",
            "UnivariateSelection",
            "feature_selection",
        )
    },
    **{
        name: {"transformations": [{"column": "x", "method": "square"}]}
        for name in (
            "SimpleTransformation",
            "GeneralTransformation",
        )
    },
    **{
        name: {
            "operations": [{"method": "add", "input_columns": ["x", "b"], "output_column": "sum"}]
        }
        for name in (
            "FeatureGenerationNode",
            "FeatureGeneration",
            "FeatureMath",
        )
    },
    **{
        name: {"test_size": 0.25, "validation_size": 0.25, "random_state": 42}
        for name in ("TrainTestSplitter", "Split")
    },
    "AliasReplacement": {"columns": ["cat"], "alias_type": "custom", "custom_map": {"red": "r"}},
    "Casting": {"columns": ["digits"], "target_type": "int64"},
    "ClipValues": {"bounds": {"x": {"lower": 2.0, "upper": 20.0}}},
    "CorrelationThreshold": {"columns": ["x", "twice"], "threshold": 0.9},
    "CustomBinning": {"columns": ["x"], "bins": [-1, 8, 16, 128]},
    "GeneralBinning": {"columns": ["x"], "strategy": "equal_width", "n_bins": 3},
    "KBinsDiscretizer": {"columns": ["x"], "strategy": "uniform", "n_bins": 3},
    "DataSnapshot": {"n_rows": 3},
    "DatasetProfile": {},
    "DateFeatures": {"columns": ["date"], "features": ["year", "month", "day"]},
    "Deduplicate": {"subset": ["cat"], "keep": "first"},
    "DropMissingColumns": {"columns": ["all_missing"]},
    "DropMissingRows": {"subset": ["missing"]},
    "MissingIndicator": {"columns": ["missing"]},
    "EllipticEnvelope": {"columns": ["x", "b"], "contamination": 0.125, "random_state": 42},
    "GeoDistance": {"lat1_col": "lat", "lon1_col": "lon", "lat2_col": "lat2", "lon2_col": "lon2"},
    "H3Index": {"lat_col": "lat", "lon_col": "lon", "resolution": 5},
    "GroupImputer": {"columns": ["missing"], "group_by": "cat", "strategy": "mean"},
    "InvalidValueReplacement": {"columns": ["signed"], "rule": "negative", "replacement": 0.0},
    "LagFeatures": {"columns": ["x"], "lags": [1], "sort_by": "row", "group_by": ["cat"]},
    "RollingAggregate": {
        "columns": ["x"],
        "window": 2,
        "aggregations": ["mean"],
        "sort_by": "row",
        "group_by": ["cat"],
    },
    "ManualBounds": {"bounds": {"x": {"upper": 20.0}}},
    "Oversampling": {"method": "random_over", "random_state": 42},
    "Undersampling": {"method": "random_under_sampling", "random_state": 42},
    "TextCleaning": {"columns": ["text"], "operations": [{"op": "trim"}]},
    "ValueReplacement": {"columns": ["cat"], "mapping": {"red": "scarlet"}},
    "feature_target_split": {"target_column": "target"},
    "sentence_embedder": {"columns": ["text"], "drop_original": True},
    "ColumnFunction": {"function": function_ref(_ratio), "output": ["ratio"]},
    "FittedFunction": {
        "learn": function_ref(_learn_mean),
        "apply": function_ref(_fill_mean),
        "output": ["missing"],
        "replace": True,
    },
    "RowFilterFunction": {"function": function_ref(_keep), "columns": ["row"]},
}
_RECIPES["HashEncoder"]["n_features"] = 4
_RECIPES["hashing_vectorizer"]["n_features"] = 4
_RECIPES["UnivariateSelection"].update(method="select_k_best", k=1, score_func="f_classif")
_RECIPES["ModelBasedSelection"].update(estimator="linear_regression", problem_type="regression")


def _data(node, engine):
    """Keep the 32-row fixture small while making every chosen operation observable."""
    rows = np.arange(32)
    x = rows.astype(float)
    x[-1] = 96.0
    missing = x.copy()
    missing[[1, 4, 7]] = np.nan
    frame = pd.DataFrame(
        {
            "row": rows,
            "x": x,
            "b": (rows * 7 % 19 + 2).astype(float),
            "twice": x * 2,
            "constant": 1.0,
            "missing": missing,
            "all_missing": np.nan,
            "signed": x - 5,
            "cat": ["red", "blue", "green", "blue"] * 8,
            "digits": rows.astype(str),
            "text": [" hello world ", "hello", "world", "test"] * 8,
            "date": pd.date_range("2024-01-01", periods=32).strftime("%Y-%m-%d"),
            "lat": 54.0 + rows / 100,
            "lon": 23.0 + rows / 100,
            "lat2": 54.1 + rows / 100,
            "lon2": 23.1 + rows / 100,
        },
        index=rows + 100,
    )
    target = (frame["cat"] == "red").astype(int).rename("target")
    if node in {"Oversampling", "Undersampling"}:
        frame = frame[["row", "x", "b"]]
    if node == "feature_target_split":
        frame["target"] = target
    if engine == "polars":
        return pl.from_pandas(frame), pl.Series("target", target.to_numpy())
    return frame, target


@pytest.fixture
def native_encoder(tmp_path):
    """Build actual weighted embeddings locally; only sentence cases need the optional extra."""
    sentence_transformers = pytest.importorskip("sentence_transformers")
    modules = pytest.importorskip("sentence_transformers.sentence_transformer.modules")
    tokenizers = pytest.importorskip("sentence_transformers.sentence_transformer.modules.tokenizer")
    torch = pytest.importorskip("torch")
    tokenizer = tokenizers.WhitespaceTokenizer(
        vocab=["[PAD]", "hello", "world", "test"], stop_words=[], do_lower_case=True
    )
    encoder = sentence_transformers.SentenceTransformer(
        modules=[
            modules.WordEmbeddings(tokenizer, torch.eye(4, dtype=torch.float32)),
            modules.Pooling(4, pooling_mode="mean"),
        ],
        device="cpu",
    )
    path = tmp_path / "native-encoder"
    encoder.save_pretrained(str(path), create_model_card=False)
    return str(path)


def _assert_equal(actual, expected):
    """Compare saved replay exactly, including structured partitions and target order."""
    assert type(actual) is type(expected)
    if isinstance(actual, SplitDataset):
        for slot in ("train", "test", "validation", "train_sample_weight", "evaluation_coverage"):
            _assert_equal(getattr(actual, slot), getattr(expected, slot))
    elif isinstance(actual, tuple):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_equal(left, right)
    elif isinstance(actual, (pd.DataFrame, pl.DataFrame)):
        left = actual.to_pandas() if isinstance(actual, pl.DataFrame) else actual
        right = expected.to_pandas() if isinstance(expected, pl.DataFrame) else expected
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    elif isinstance(actual, (pd.Series, pl.Series)):
        left = actual.to_pandas() if isinstance(actual, pl.Series) else actual
        right = expected.to_pandas() if isinstance(expected, pl.Series) else expected
        pd.testing.assert_series_equal(left, right, check_exact=True)
    else:
        assert actual == expected


def _assert_effect(node, result, frame, target, state):
    """Require real transformation or the documented inspection/split result."""
    if isinstance(result, SplitDataset):
        assert sum(len(slot[0]) for slot in (result.train, result.test, result.validation)) == len(
            frame
        )
        rows = []
        for out, out_y in (result.train, result.test, result.validation):
            assert len(out) > 0
            rows.extend(out["row"])
            np.testing.assert_array_equal(
                out_y, np.asarray(target)[np.asarray(out["row"], dtype=int)]
            )
        assert sorted(rows) == list(range(len(frame)))
        return
    out, out_y = result
    assert len(out) == len(out_y)
    np.testing.assert_array_equal(out_y, np.asarray(target)[np.asarray(out["row"], dtype=int)])
    if node in {"DataSnapshot", "DatasetProfile"}:
        assert state["snapshot" if node == "DataSnapshot" else "profile"]
        _assert_equal(out, frame)
    else:
        left = out.to_pandas() if isinstance(out, pl.DataFrame) else out
        right = frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame
        assert not left.equals(right), f"{node} recipe silently did nothing"
        _assert_selected_values(node, left)


def _assert_selected_values(node, output):
    """Pin group isolation and row selection beyond equality of two replay calls."""
    indexed = output.set_index("row")
    if node == "GroupImputer":
        np.testing.assert_array_equal(
            indexed.loc[[1, 4, 7], "missing"], [313 / 14, 108 / 7, 313 / 14]
        )
        observed = indexed.drop(index=[1, 4, 7])
        np.testing.assert_array_equal(observed["missing"], observed["x"])
    elif node == "LagFeatures":
        np.testing.assert_array_equal(indexed.loc[:4, "x_lag_1"], [np.nan, np.nan, np.nan, 1, 0])
        assert indexed.loc[6, "x_lag_1"] == 2
    elif node == "RollingAggregate":
        np.testing.assert_array_equal(indexed.loc[:4, "x_roll_mean_2"], [0, 1, 2, 2, 2])
        assert indexed.loc[6, "x_roll_mean_2"] == 4
    elif node == "Deduplicate":
        assert list(indexed.index) == [0, 1, 2]


def test_small_data_recipes_cover_the_complete_registry():
    """New registrations cannot escape actual fit/apply coverage behind a static count."""
    assert set(_RECIPES) == set(NodeRegistry.list_transformers())
    assert len(_RECIPES) == 67
    assert len({NodeRegistry.get_applier(node) for node in _RECIPES}) == 63


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", sorted(_RECIPES))
def test_registered_preprocessor_saved_state_roundtrip(
    node, engine, request, tmp_path, monkeypatch
):
    """A real fitted state survives disk reload and applies without calculator refitting."""
    config = deepcopy(NodeRegistry.get_all_metadata()[node]["params"])
    config.update(deepcopy(_RECIPES[node]))
    if node == "sentence_embedder":
        config["model_name"] = request.getfixturevalue("native_encoder")
    frame, target = _data(node, engine)
    payload = frame if node == "feature_target_split" else (frame, target)
    calculator = NodeRegistry.get_calculator(node)
    applier = NodeRegistry.get_applier(node)()
    state = calculator().fit(payload, config)
    assert state, f"{node} did not fit an active state"
    state_path = tmp_path / "fitted.pkl"
    state_path.write_bytes(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))

    def forbidden_fit(*args, **kwargs):
        """Any request-time calculator learning invalidates the replay contract."""
        raise AssertionError("saved-state apply attempted calculator.fit")

    monkeypatch.setattr(calculator, "fit", forbidden_fit)
    expected = applier.apply(deepcopy(payload), state)
    restored = pickle.loads(state_path.read_bytes())
    actual = applier.apply(deepcopy(payload), restored)
    _assert_equal(actual, expected)
    _assert_effect(node, actual, frame, target, restored)
