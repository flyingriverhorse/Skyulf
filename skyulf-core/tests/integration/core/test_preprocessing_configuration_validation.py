"""Fit rejects misspelled choices while saved preprocessing remains replayable."""

import pickle
from copy import deepcopy

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.registry import NodeRegistry


def _frame(engine):
    """Keep enough observed neighbors for every supported iterative estimator."""
    frame = pd.DataFrame(
        {
            "x": [1.0, 4.0, np.nan, 16.0, 25.0, 36.0, 49.0, 64.0],
            "z": [2.0, 8.0, 18.0, 32.0, 50.0, np.nan, 98.0, 128.0],
            "date": ["2026-10-01", "2026-10-02"] * 4,
        }
    )
    return pl.from_pandas(frame) if engine == "polars" else frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("node", "config", "bad_choice", "valid_choice"),
    [
        (
            "SimpleTransformation",
            {"transformations": [{"column": "x", "method": "logg"}]},
            "logg",
            "square_root",
        ),
        (
            "SimpleTransformation",
            {
                "transformations": [
                    {"column": "x", "method": "log"},
                    {"column": "z", "method": "logg"},
                ]
            },
            "logg",
            "exponential",
        ),
        ("DateFeatures", {"columns": ["date"], "features": ["hours"]}, "hours", "hour"),
        (
            "DateFeatures",
            {"columns": ["date"], "features": ["year", "hours"]},
            "hours",
            "is_month_end",
        ),
        (
            "IterativeImputer",
            {"columns": ["x", "z"], "estimator": "BayesianRidg"},
            "BayesianRidg",
            "BayesianRidge",
        ),
        (
            "RollingAggregate",
            {"columns": ["x"], "aggregations": "average"},
            "average",
            "mean",
        ),
        (
            "RollingAggregate",
            {"columns": ["x"], "aggregations": ["mean", "average"]},
            "average",
            "median",
        ),
    ],
)
def test_unknown_fit_choices_fail_visibly(engine, node, config, bad_choice, valid_choice):
    """A typo must not save a partially configured or substituted pipeline."""
    frame = _frame(engine)
    original, original_config = deepcopy(frame), deepcopy(config)
    with pytest.raises(ValueError) as error:
        NodeRegistry.get_calculator(node)().fit(frame, config)
    assert bad_choice in str(error.value) and valid_choice in str(error.value)
    assert config == original_config
    if engine == "pandas":
        pd.testing.assert_frame_equal(frame, original)
    else:
        assert frame.equals(original)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("columns", [[], ["absent"]])
def test_invalid_estimator_is_not_hidden_by_empty_selection(engine, columns):
    """A data-dependent no-op must not hide an invalid estimator setting."""
    with pytest.raises(ValueError, match="BayesianRidg.*BayesianRidge"):
        NodeRegistry.get_calculator("IterativeImputer")().fit(
            _frame(engine), {"columns": columns, "estimator": "BayesianRidg"}
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("node", "config"),
    [
        ("SimpleTransformation", {"transformations": [{"column": "x", "method": None}]}),
        ("DateFeatures", {"columns": ["date"], "features": ["year", None]}),
        ("IterativeImputer", {"columns": ["x", "z"], "estimator": None}),
        ("RollingAggregate", {"columns": ["x"], "aggregations": ["mean", None]}),
    ],
)
def test_non_string_choice_has_actionable_fit_error(engine, node, config):
    """Malformed option values must report supported choices rather than disappear."""
    with pytest.raises(ValueError, match="Supported"):
        NodeRegistry.get_calculator(node)().fit(_frame(engine), config)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("node", "field", "column", "choice", "output", "expected"),
    [
        ("DateFeatures", "features", "date", "year", "date_year", [2026, 2026]),
        ("RollingAggregate", "aggregations", "x", "sum", "x_roll_sum_3", [1.0, 5.0]),
    ],
)
def test_valid_choice_iterators_are_consumed_once(
    engine, node, field, column, choice, output, expected
):
    """Validation must not exhaust valid Python choices before saving their concrete values."""
    state = NodeRegistry.get_calculator(node)().fit(
        _frame(engine), {"columns": [column], field: iter([choice])}
    )
    assert state[field] == [choice]
    result = NodeRegistry.get_applier(node)().apply(_frame(engine), state)
    np.testing.assert_allclose(result[output].to_numpy()[:2], expected)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("node", "config", "output_column", "expected"),
    [
        ("SimpleTransformation", {}, "x", [1.0, 4.0]),
        (
            "SimpleTransformation",
            {"transformations": [{"column": "x", "method": "square_root"}]},
            "x",
            [1.0, 2.0],
        ),
        (
            "SimpleTransformation",
            {"transformations": [{"column": "x", "method": "exponential"}]},
            "x",
            [np.e, np.exp(4.0)],
        ),
        ("DateFeatures", {"columns": ["date"]}, "date_month", [10, 10]),
        ("DateFeatures", {"columns": ["date"], "features": []}, "date_year", [2026, 2026]),
        ("DateFeatures", {"columns": ["date"], "features": None}, "date_day", [1, 2]),
        ("RollingAggregate", {"columns": ["x"]}, "x_roll_mean_3", [1.0, 2.5]),
        (
            "RollingAggregate",
            {"columns": ["x"], "aggregations": "sum"},
            "x_roll_sum_3",
            [1.0, 5.0],
        ),
        ("RollingAggregate", {"columns": ["x"], "aggregations": []}, "x", [1.0, 4.0]),
        ("RollingAggregate", {"columns": ["x"], "aggregations": None}, "x", [1.0, 4.0]),
    ],
)
def test_valid_defaults_and_aliases_survive_disk_replay(
    engine, node, config, output_column, expected, tmp_path, monkeypatch
):
    """Fit-time rejection must not alter valid saved defaults or cross-engine application."""
    calculator = NodeRegistry.get_calculator(node)
    state = calculator().fit(_frame(engine), config)
    path = tmp_path / "fitted.pkl"
    path.write_bytes(pickle.dumps(state))

    def forbidden_fit(*args, **kwargs):
        """Applying a saved artifact must not refit with the new configuration guards."""
        pytest.fail("saved application refitted the calculator")

    monkeypatch.setattr(calculator, "fit", forbidden_fit)
    restored = pickle.loads(path.read_bytes())
    before = deepcopy(restored)
    for apply_engine in ("pandas", "polars"):
        result = NodeRegistry.get_applier(node)().apply(_frame(apply_engine), restored)
        np.testing.assert_allclose(result[output_column].to_numpy()[:2], expected)
    assert restored == before


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("alias", "estimator_class"),
    [
        (None, "BayesianRidge"),
        ("bayesian_ridge", "BayesianRidge"),
        ("DecisionTree", "DecisionTreeRegressor"),
        ("decision_tree", "DecisionTreeRegressor"),
        ("ExtraTrees", "ExtraTreesRegressor"),
        ("extra_trees", "ExtraTreesRegressor"),
        ("KNeighbors", "KNeighborsRegressor"),
        ("knn", "KNeighborsRegressor"),
        (" DECISION-TREE ", "DecisionTreeRegressor"),
    ],
)
def test_iterative_estimator_aliases_fit_save_apply(engine, alias, estimator_class, tmp_path):
    """Documented and Canvas aliases must retain their fitted predictor across reload."""
    config = {"columns": ["x", "z"], "max_iter": 2, "random_state": 0}
    if alias is not None:
        config["estimator"] = alias
    state = NodeRegistry.get_calculator("IterativeImputer")().fit(_frame(engine), config)
    assert type(state["imputer_object"].estimator).__name__ == estimator_class
    path = tmp_path / "iterative.pkl"
    path.write_bytes(pickle.dumps(state))
    restored = pickle.loads(path.read_bytes())
    outputs = []
    for apply_engine in ("pandas", "polars"):
        result = NodeRegistry.get_applier("IterativeImputer")().apply(
            _frame(apply_engine), restored
        )
        outputs.append(
            result.select(["x", "z"]).to_numpy()
            if apply_engine == "polars"
            else result[["x", "z"]].to_numpy()
        )
    assert np.isfinite(outputs).all()
    np.testing.assert_allclose(outputs[0], outputs[1])
    np.testing.assert_allclose(outputs[0][0], [1.0, 2.0])
