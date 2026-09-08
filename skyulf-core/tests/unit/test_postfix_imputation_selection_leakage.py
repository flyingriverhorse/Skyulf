"""Counterfactual leakage probes after the polynomial and split-boundary fixes."""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.leakage import step_learns_from_data, validate_leakage_safety
from skyulf.preprocessing.pipeline import FeatureEngineer
from skyulf.registry import NodeRegistry


def _convert(frame: Any, engine: str) -> Any:
    """Keep identical typed values while exercising each supported frame engine."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _to_pandas(frame: Any) -> pd.DataFrame:
    """Compare training features without depending on an engine's frame equality API."""
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("selection", [{}, {"columns": ["x"]}], ids=["auto", "explicit"])
@pytest.mark.parametrize("fill", [0.0, -1.0, None], ids=["zero", "negative", "default"])
@pytest.mark.parametrize("before", [True, False], ids=["before-split", "after-split"])
@pytest.mark.parametrize(
    "training_values",
    [[np.nan] * 4, [np.nan, 2.0, np.nan, 4.0]],
    ids=["all-missing", "mixed"],
)
def test_constant_imputation_training_values_ignore_heldout_availability(
    engine: str,
    selection: dict[str, Any],
    fill: float | None,
    before: bool,
    training_values: list[float],
) -> None:
    """An admitted fixed constant must not learn whether heldout rows make a column nonempty."""
    params = {"strategy": "constant", **selection}
    if fill is not None:
        params["fill_value"] = fill
    impute = {
        "name": "impute",
        "transformer": "SimpleImputer",
        "params": params,
    }
    split = {
        "name": "split",
        "transformer": "TrainTestSplitter",
        "params": {
            "test_size": 1 / 3,
            "shuffle": False,
            "random_state": 42,
            "target_column": "target",
        },
    }
    steps = [impute, split] if before else [split, impute]
    assert validate_leakage_safety({"preprocessing": steps, "modeling": {}}) == []
    training_outputs = []
    fitted_fills = []
    for heldout in ([np.nan, np.nan], [5.0, 9.0]):
        frame = pd.DataFrame(
            {
                "row_id": range(6),
                "x": training_values + heldout,
                "target": [0, 1, 0, 1, 0, 1],
            }
        )
        original = frame.copy(deep=True)
        native_frame = _convert(frame, engine)
        engineer = FeatureEngineer(steps)
        output, _ = engineer.fit_transform(native_frame, target_column="target")
        training = _to_pandas(output.train[0])
        testing = _to_pandas(output.test[0])
        assert training["row_id"].to_list() == [0, 1, 2, 3]
        assert testing["row_id"].to_list() == [4, 5]
        assert output.train[1].to_list() == [0, 1, 0, 1]
        assert output.test[1].to_list() == [0, 1]
        assert _to_pandas(native_frame).equals(original)
        expected_fill = 0.0 if fill is None else fill
        assert training["x"].to_list() == [
            expected_fill if np.isnan(value) else value for value in training_values
        ]
        assert testing["x"].to_list() == [
            expected_fill if np.isnan(value) else value for value in heldout
        ]
        training_outputs.append(training)
        fitted_fills.append(engineer.fitted_steps[0]["artifact"]["fill_values"]["x"])
        assert fitted_fills[-1] == expected_fill

    assert training_outputs[0].equals(training_outputs[1]), {
        "fitted_fills": fitted_fills,
        "train_with_missing_holdout": training_outputs[0]["x"].to_list(),
        "train_with_observed_holdout": training_outputs[1]["x"].to_list(),
    }


_FILTERED_SELECTIONS = [
    ("PolynomialFeatures", {"columns": ["absent"], "auto_detect": True}),
    ("PolynomialFeatures", {"columns": ["target"], "auto_detect": True}),
    ("PolynomialFeaturesNode", {"columns": ["absent"], "auto_detect": True}),
    ("PolynomialFeaturesNode", {"columns": ["target"], "auto_detect": True}),
    ("MissingIndicator", {"columns": ["absent"]}),
    ("HashEncoder", {"columns": ["target"]}),
    ("CustomBinning", {"columns": ["absent"], "bins": [0.0, 5.0, 10.0]}),
    ("count_vectorizer", {"columns": ["target"]}),
    ("tfidf_vectorizer", {"columns": ["target"]}),
]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("node", _FILTERED_SELECTIONS)
def test_filtered_explicit_selection_does_not_restart_automatic_discovery(
    engine: str, node: tuple[str, dict[str, Any]]
) -> None:
    """Removing unavailable or target columns must not turn an exempt selection into discovery."""
    node_id, selection = node
    config = {**selection, "target_column": "target"}
    assert not step_learns_from_data(node_id, config, target_column="target")
    training = pd.DataFrame({"x": [0.0, 1.0, 0.0, 1.0], "text": ["aa", "bb", "aa", "bb"]})
    train_target = pd.Series([0, 1, 0, 1], name="target")
    outputs = []
    for heldout in (
        {"x": [0.0, 1.0], "text": ["aa", "bb"]},
        {"x": [np.nan, 5.0], "text": ["novel", "rare"]},
    ):
        pooled = pd.concat([training, pd.DataFrame(heldout)], ignore_index=True)
        target = pd.Series([0, 1, 0, 1, 0, 1], name="target")
        calculator = NodeRegistry.get_calculator(node_id)()
        applier = NodeRegistry.get_applier(node_id)()
        artifact = calculator.fit((_convert(pooled, engine), _convert(target, engine)), config)
        transformed, _ = applier.apply(
            (_convert(training, engine), _convert(train_target, engine)), artifact
        )
        outputs.append(_to_pandas(transformed))

    assert all(output.equals(training) for output in outputs)


def _default_frame(kind: str, all_missing: bool) -> pd.DataFrame:
    """Construct stable dtypes so missingness probes do not accidentally change the schema."""
    numeric = [np.nan] * 4 if all_missing else [np.nan, 2.0, np.nan, 4.0]
    text = [np.nan] * 4 if all_missing else [np.nan, "a", np.nan, "b"]
    dtype = {
        "nullable-int": "Int64",
        "nullable-float": "Float64",
        "numeric-object": "object",
    }.get(kind, "float64")
    columns: dict[str, Any] = {}
    if kind != "text":
        columns["x"] = pd.Series(numeric, dtype=dtype)
    if kind in {"text", "mixed"}:
        columns["text"] = pd.Series(text, dtype="object")
    frame = pd.DataFrame(columns)
    frame.index = pd.Index([8, 3, 8, 1], name="row")
    return frame


def _typed_default_frame(frame: pd.DataFrame, engine: str) -> Any:
    """Keep declared text and numeric-object control dtypes stable in Polars."""
    if engine == "pandas":
        return frame
    overrides: dict[str, Any] = {"text": pl.String} if "text" in frame.columns else {}
    if "x" in frame.columns and frame["x"].dtype == object:
        overrides["x"] = pl.Float64
    return pl.from_pandas(frame, schema_overrides=overrides)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "kind", ["float", "nullable-int", "nullable-float", "numeric-object", "text", "mixed"]
)
@pytest.mark.parametrize("explicit", [True, False], ids=["explicit", "auto"])
@pytest.mark.parametrize("all_missing", [True, False], ids=["all-missing", "mixed"])
def test_default_constant_preserves_dtype_policy_and_heldout_isolation(
    engine: str, kind: str, explicit: bool, all_missing: bool
) -> None:
    """Default fills must retain established dtype semantics without learning heldout availability."""
    training = _default_frame(kind, all_missing)
    target = pd.Series([0, 1, 0, 1], index=training.index, name="target")
    native_training = _typed_default_frame(training, engine)
    native_target = _convert(target, engine)
    original = _to_pandas(native_training).copy(deep=True)
    config: dict[str, Any] = {
        "strategy": "constant",
        "fill_value": None,
        "target_column": "target",
    }
    if explicit:
        config["columns"] = list(training.columns)
    default: Any = (
        "missing_value" if engine == "pandas" and kind in {"numeric-object", "text", "mixed"} else 0
    )
    outputs = []
    for heldout_missing in (True, False):
        pooled = pd.concat([training, _default_frame(kind, heldout_missing)])
        pooled_target = pd.Series([0, 1, 0, 1] * 2, index=pooled.index, name="target")
        artifact = NodeRegistry.get_calculator("SimpleImputer")().fit(
            (_typed_default_frame(pooled, engine), _convert(pooled_target, engine)), config
        )
        transformed, transformed_target = NodeRegistry.get_applier("SimpleImputer")().apply(
            (native_training, native_target), artifact
        )
        assert artifact["fill_values"] == dict.fromkeys(training.columns, default)
        result = _to_pandas(transformed)
        for column in training.columns:
            expected_fill = str(default) if engine == "polars" and column == "text" else default
            assert result[column].to_list() == [
                expected_fill if pd.isna(value) else value for value in original[column].to_list()
            ]
            if engine == "pandas":
                expected_dtype = (
                    "object" if isinstance(default, str) else str(training[column].dtype)
                )
                assert str(transformed[column].dtype) == expected_dtype
        assert transformed_target.to_list() == [0, 1, 0, 1]
        if engine == "pandas":
            assert transformed.index.equals(training.index)
            assert transformed_target.index.equals(target.index)
        else:
            assert transformed.schema == native_training.schema
        outputs.append(result)

    assert outputs[0].equals(outputs[1])
    assert _to_pandas(native_training).equals(original)
    assert native_target.to_list() == target.to_list()


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("all_missing", [True, False], ids=["all-missing", "mixed"])
def test_explicit_text_constant_fills_without_replacing_observed_values(
    engine: str, all_missing: bool
) -> None:
    """The numeric leakage fix must also preserve configured text constants on text columns."""
    frame = _default_frame("text", all_missing)
    native = _typed_default_frame(frame, engine)
    config = {"strategy": "constant", "fill_value": "replacement", "columns": ["text"]}
    artifact = NodeRegistry.get_calculator("SimpleImputer")().fit(native, config)
    output = NodeRegistry.get_applier("SimpleImputer")().apply(native, artifact)

    assert artifact["fill_values"] == {"text": "replacement"}
    assert output["text"].to_list() == [
        "replacement" if pd.isna(value) else value for value in frame["text"].to_list()
    ]


@pytest.mark.parametrize("fill", [0.0, -1.0])
def test_explicit_constant_does_not_read_fitted_statistics(
    monkeypatch: pytest.MonkeyPatch, fill: float
) -> None:
    """Explicit fills must come from configuration even when validation is delegated to sklearn."""
    import skyulf.preprocessing.imputation.simple as module

    class ValidationOnlyImputer:
        """Allow validation while rejecting access to learned replacement values."""

        def __init__(self, **kwargs: Any) -> None:
            """Require the constant-only empty-column safeguard at construction."""
            assert kwargs["keep_empty_features"] is True

        def fit(self, frame: pd.DataFrame) -> "ValidationOnlyImputer":
            """Accept the selected frame without constructing learned statistics."""
            assert list(frame.columns) == ["x"]
            return self

        @property
        def statistics_(self) -> Any:
            """Fail if configured replacement values are obtained from fitted statistics."""
            raise AssertionError("Explicit constants must not read statistics_")

    monkeypatch.setattr(module, "SimpleImputer", ValidationOnlyImputer)
    artifact = module.SimpleImputerCalculator().fit(
        pd.DataFrame({"x": [np.nan, np.nan]}),
        {"strategy": "constant", "fill_value": fill, "columns": ["x"]},
    )

    assert artifact["fill_values"] == {"x": fill}
