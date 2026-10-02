"""Function steps written by a beginner must match the built-in Core nodes exactly.

Each helper below is what a project author would write in a Bundle instead of a
Calculator/Applier pair. The tests compare them with SimpleImputer,
StandardScaler, MinMaxScaler, DropMissingRows and ManualBounds on the same data,
then pin the edge cases where plain functions are most likely to go wrong.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal

from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing import FeatureEngineer, column_step, filter_step, fitted_step

NUMERIC = ["income", "age"]


# ---- user functions: the code a project author writes ----------------------


def learn_means(df, y, params):
    """Learn one training mean per configured column."""
    return {c: df[c].mean() for c in params["columns"]}


def learn_medians(df, y, params):
    """Learn one training median per configured column."""
    return {c: df[c].median() for c in params["columns"]}


def learn_most_frequent(df, y, params):
    """Learn the most frequent value; ties pick the smallest like scikit-learn."""
    return {c: df[c].mode().iloc[0] for c in params["columns"]}


def fill_learned(df, state, params):
    """Fill missing values with the saved per-column values."""
    return df[params["columns"]].fillna(state)


def learn_standard(df, y, params):
    """Learn population mean and std; a constant column keeps scale 1 like scikit-learn."""
    return {c: [df[c].mean(), df[c].std(ddof=0) or 1.0] for c in params["columns"]}


def apply_standard(df, state, params):
    """Standardize with the saved training statistics."""
    return pd.DataFrame({c: (df[c] - m) / s for c, (m, s) in state.items()})


def learn_minmax(df, y, params):
    """Learn training min and range per column."""
    return {c: [df[c].min(), (df[c].max() - df[c].min()) or 1.0] for c in params["columns"]}


def apply_minmax(df, state, params):
    """Scale to [0, 1] with the saved training min and range."""
    return pd.DataFrame({c: (df[c] - lo) / span for c, (lo, span) in state.items()})


def age_present(df):
    """Keep rows that have an age, like DropMissingRows(subset=["age"])."""
    return df["age"].notna()


def age_in_range(df, params):
    """Keep rows inside bounds; missing values stay, like ManualBounds."""
    age = df["age"]
    return age.between(params["lower"], params["upper"]) | age.isna()


def learn_count(df, y):
    """Return a NumPy integer, as pandas reductions often do."""
    return {"n": df["age"].count()}


def apply_count(df, state):
    """Broadcast the learned count."""
    return pd.Series(state["n"], index=df.index, dtype="float64")


def learn_code_means(df, y):
    """Group by an integer code; the keys are integers, not strings."""
    return {"means": df.groupby("code")["income"].mean().to_dict()}


def apply_code_means(df, state):
    """Map codes to learned means."""
    return df["code"].map(state["means"])


def learn_city_medians(df, y):
    """Group medians that are NaN for a city with no income values."""
    return {"medians": df.groupby("city")["income"].median().to_dict()}


def mutate_input(df):
    """A careless function that writes into the frame it was given."""
    df["age"] = -1
    df["leaked"] = 1
    return df["income"] * 2


def nullable_mask(df):
    """A nullable-boolean mask that still contains missing values."""
    return df["age"].astype("Int64") >= 18


def reset_index_mask(df):
    """A mask that lost the original row labels."""
    return (df["age"] >= 18).reset_index(drop=True)


def double_income(df):
    """Simple derived column."""
    return df["income"] * 2


# ---- data -------------------------------------------------------------------


def _train() -> pd.DataFrame:
    """Training rows with missing values, a constant column and a mode tie."""
    return pd.DataFrame(
        {
            "income": [100.0, np.nan, 300.0, 400.0, 250.0, np.nan],
            "age": [20.0, 35.0, np.nan, 50.0, 20.0, 35.0],
            "flat": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
            "city": ["B", "A", "B", "A", None, "C"],
        }
    )


def _batch() -> pd.DataFrame:
    """Unseen rows with values outside the training range and new missing values."""
    return pd.DataFrame(
        {
            "income": [np.nan, 1000.0, -50.0],
            "age": [np.nan, 99.0, 1.0],
            "flat": [5.0, 7.0, np.nan],
            "city": [None, "Z", "A"],
        }
    )


def _engine(frame: pd.DataFrame, engine: str):
    """Convert a pandas fixture to the requested engine."""
    return pl.from_pandas(frame) if engine == "polars" else frame


def _pd(frame) -> pd.DataFrame:
    """Compare results independently of engine."""
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


def _both(core_steps, function_steps, engine):
    """Fit Core and function pipelines and return train and batch outputs of each."""
    results = []
    for steps in (core_steps, function_steps):
        engineer = FeatureEngineer(steps)
        train_out, _ = engineer.fit_transform(_engine(_train(), engine))
        batch_out = engineer.transform(_engine(_batch(), engine))
        results.append((_pd(train_out), _pd(batch_out)))
    return results


def _assert_same(core, mine, columns):
    """Training and inference outputs agree on the compared columns."""
    for core_frame, my_frame in zip(core, mine, strict=True):
        assert_frame_equal(
            my_frame[columns].reset_index(drop=True),
            core_frame[columns].reset_index(drop=True),
            check_dtype=False,
            atol=1e-12,
        )


# ---- preprocessing parity -----------------------------------------------------


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(("strategy", "learn"), [("mean", learn_means), ("median", learn_medians)])
def test_fitted_imputer_matches_simple_imputer(engine, strategy, learn):
    """Training statistics fill both splits; batch NaNs use training values, not their own."""
    core = [
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"strategy": strategy, "columns": NUMERIC},
        }
    ]
    mine = [
        fitted_step(
            "impute", learn, fill_learned, output=NUMERIC, replace=True, params={"columns": NUMERIC}
        )
    ]
    core_out, my_out = _both(core, mine, engine)
    _assert_same(core_out, my_out, NUMERIC)
    assert my_out[1]["income"].iloc[0] == core_out[1]["income"].iloc[0]


def test_fitted_most_frequent_matches_simple_imputer_on_ties():
    """A two-way tie must resolve to the same category as scikit-learn."""
    core = [
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"strategy": "most_frequent", "columns": ["city"]},
        }
    ]
    mine = [
        fitted_step(
            "impute",
            learn_most_frequent,
            fill_learned,
            output=["city"],
            replace=True,
            params={"columns": ["city"]},
        )
    ]
    core_out, my_out = _both(core, mine, "pandas")
    _assert_same(core_out, my_out, ["city"])
    assert my_out[1]["city"].tolist() == ["A", "Z", "A"]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("node", "learn", "apply"),
    [
        ("StandardScaler", learn_standard, apply_standard),
        ("MinMaxScaler", learn_minmax, apply_minmax),
    ],
)
def test_fitted_scalers_match_core_including_a_constant_column(engine, node, learn, apply):
    """A constant training column must not divide by zero, exactly like Core."""
    columns = [*NUMERIC, "flat"]
    impute = {
        "name": "impute",
        "transformer": "SimpleImputer",
        "params": {"strategy": "mean", "columns": columns},
    }
    core = [impute, {"name": "scale", "transformer": node, "params": {"columns": columns}}]
    mine = [
        impute,
        fitted_step(
            "scale", learn, apply, output=columns, replace=True, params={"columns": columns}
        ),
    ]
    core_out, my_out = _both(core, mine, engine)
    _assert_same(core_out, my_out, columns)
    assert np.isfinite(my_out[1]["flat"]).all()


@pytest.mark.parametrize("tuned", [False, True], ids=["model", "cv_tuner"])
def test_function_recipe_predicts_like_the_core_recipe(tuned):
    """Swapping Core nodes for functions must not change a trained model's predictions."""
    columns = NUMERIC
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"income": rng.normal(300, 50, 60), "age": rng.normal(40, 9, 60)})
    frame.loc[::7, "income"] = np.nan
    frame.loc[::5, "age"] = np.nan
    frame["target"] = 0.02 * frame["income"].fillna(300) + frame["age"].fillna(40)
    modeling = {"type": "ridge_regression", "params": {"alpha": 0.5}}
    if tuned:
        modeling = {
            "type": "hyperparameter_tuner",
            "base_model": {"type": "ridge_regression"},
            "strategy": "grid",
            "metric": "r2",
            "search_space": {"alpha": [0.01, 10.0]},
            "cv_folds": 3,
            "n_jobs": 1,
        }
    core = [
        {
            "name": "impute",
            "transformer": "SimpleImputer",
            "params": {"strategy": "median", "columns": columns},
        },
        {"name": "scale", "transformer": "StandardScaler", "params": {"columns": columns}},
    ]
    mine = [
        fitted_step(
            "impute",
            learn_medians,
            fill_learned,
            output=columns,
            replace=True,
            params={"columns": columns},
        ),
        fitted_step(
            "scale",
            learn_standard,
            apply_standard,
            output=columns,
            replace=True,
            params={"columns": columns},
        ),
    ]
    new = pd.DataFrame({"income": [np.nan, 500.0], "age": [30.0, np.nan]})
    predictions = []
    for steps in (core, mine):
        pipeline = SkyulfPipeline({"preprocessing": steps, "modeling": modeling})
        pipeline.fit(frame, target_column="target")
        predictions.append(np.asarray(pipeline.predict(new), dtype=float))
    np.testing.assert_allclose(predictions[1], predictions[0], rtol=1e-10)


# ---- filter parity ------------------------------------------------------------


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_filter_matches_drop_missing_rows_and_keeps_target_aligned(engine):
    """The same rows and targets survive as with DropMissingRows."""
    y = pd.Series([1, 2, 3, 4, 5, 6], name="target")
    results = []
    for step in (
        {"name": "drop", "transformer": "DropMissingRows", "params": {"subset": ["age"]}},
        filter_step("drop", age_present, columns=["age"]),
    ):
        data = (_engine(_train(), engine), pl.Series("target", y) if engine == "polars" else y)
        (out_X, out_y), _ = FeatureEngineer([step]).fit_transform(data)
        results.append((_pd(out_X).reset_index(drop=True), list(out_y)))
    assert_frame_equal(results[1][0], results[0][0])
    assert results[1][1] == results[0][1] == [1, 2, 4, 5, 6]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_filter_matches_manual_bounds_and_keeps_missing_values(engine):
    """Inclusive bounds and missing-value retention match ManualBounds."""
    bounds = {"lower": 20.0, "upper": 35.0}
    results = []
    for step in (
        {"name": "bounds", "transformer": "ManualBounds", "params": {"bounds": {"age": bounds}}},
        filter_step("bounds", age_in_range, columns=["age"], params=bounds),
    ):
        out, _ = FeatureEngineer([step]).fit_transform(_engine(_train(), engine))
        results.append(_pd(out).reset_index(drop=True))
    assert_frame_equal(results[1], results[0])
    assert len(results[1]) == 5


# ---- edge cases: preprocessing ---------------------------------------------------


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_user_function_cannot_mutate_the_callers_frame(engine):
    """Writing into ``df`` inside a function must not leak into the pipeline data."""
    data = _engine(_train(), engine)
    out, _ = FeatureEngineer([column_step("x", mutate_input, output="x")]).fit_transform(data)
    result = _pd(out)
    assert "leaked" not in result.columns
    assert result["age"].equals(_train()["age"])
    assert _pd(data)["age"].equals(_train()["age"])


def test_polars_untouched_columns_keep_their_dtypes():
    """Pandas round trips must not turn nullable ints into floats or change datetimes."""
    data = pl.DataFrame(
        {
            "income": [1.0, 2.0, 3.0],
            "count": pl.Series([1, None, 3], dtype=pl.Int64),
            "flag": pl.Series([True, None, False], dtype=pl.Boolean),
            "when": pl.Series([datetime(2024, 1, d) for d in (1, 2, 3)], dtype=pl.Datetime("us")),
            "kind": pl.Series(["a", "b", "a"], dtype=pl.Categorical),
        }
    )
    out, _ = FeatureEngineer([column_step("x", double_income, output="x")]).fit_transform(data)
    assert out.select(data.columns).schema == data.schema
    assert out["count"].to_list() == [1, None, 3]
    assert out["x"].to_list() == [2.0, 4.0, 6.0]


def test_numpy_scalars_in_learned_state_are_saved_as_plain_json():
    """``Series.count()`` returns a NumPy integer; beginners must not have to call int()."""
    engineer = FeatureEngineer([fitted_step("n", learn_count, apply_count, output="n")])
    engineer.fit_transform(_train())
    state = engineer.fitted_steps[0]["artifact"]["state"]
    assert state == {"n": 5} and type(state["n"]) is int


def test_non_string_state_keys_are_rejected_with_guidance():
    """Integer keys would silently become strings when saved and stop matching."""
    frame = pd.DataFrame({"code": [1, 1, 2], "income": [1.0, 3.0, 5.0]})
    step = fitted_step("codes", learn_code_means, apply_code_means, output="code_mean")
    with pytest.raises(ValueError, match=r"keys must be strings.*astype\(str\)"):
        FeatureEngineer([step]).fit_transform(frame)


def test_nan_in_learned_state_is_rejected_with_guidance():
    """A group with no values yields NaN; the error must say so, not just 'JSON'."""
    frame = pd.DataFrame({"city": ["A", "B"], "income": [1.0, np.nan]})
    step = fitted_step("med", learn_city_medians, apply_code_means, output="m")
    with pytest.raises(ValueError, match="NaN or infinity"):
        FeatureEngineer([step]).fit_transform(frame)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_empty_and_single_row_batches_score(engine):
    """Inference on zero or one row must work with the saved state."""
    step = fitted_step(
        "impute",
        learn_means,
        fill_learned,
        output=NUMERIC,
        replace=True,
        params={"columns": NUMERIC},
    )
    engineer = FeatureEngineer([step])
    engineer.fit_transform(_engine(_train(), engine))
    assert len(engineer.transform(_engine(_batch().iloc[:0], engine))) == 0
    one = _pd(engineer.transform(_engine(_batch().iloc[:1], engine)))
    assert one["income"].tolist() == [262.5]


def test_reordered_columns_at_inference_use_names_not_positions():
    """Inference batches may arrive with columns in a different order."""
    step = fitted_step(
        "scale",
        learn_standard,
        apply_standard,
        output=NUMERIC,
        replace=True,
        params={"columns": NUMERIC},
    )
    frame = _train().fillna(0.0)
    engineer = FeatureEngineer([step])
    expected, _ = engineer.fit_transform(frame)
    shuffled = engineer.transform(frame[["city", "age", "flat", "income"]])
    assert_frame_equal(shuffled[NUMERIC], expected[NUMERIC])


def test_missing_input_column_names_the_failing_step():
    """A batch without a required column must point at the step, not a bare KeyError."""
    engineer = FeatureEngineer([column_step("double", double_income, output="x")])
    engineer.fit_transform(_train())
    with pytest.raises(ValueError, match="double_income failed: KeyError") as info:
        engineer.transform(_train().drop(columns="income"))
    assert "income" in str(info.value)


# ---- edge cases: filters ----------------------------------------------------------


def test_nullable_mask_with_missing_values_explains_fillna():
    """``Int64 >= 18`` yields <NA>; the user must be told how to decide those rows."""
    frame = pd.DataFrame({"age": [10.0, np.nan, 30.0]})
    with pytest.raises(ValueError, match=r"fillna\(False\)"):
        FeatureEngineer([filter_step("adult", nullable_mask, columns=["age"])]).fit_transform(frame)


def test_mask_with_lost_row_labels_is_rejected():
    """A mask aligned by position instead of row label could keep the wrong rows."""
    frame = pd.DataFrame({"age": [10, 30, 40]}, index=[5, 6, 7])
    with pytest.raises(ValueError, match="one value per row"):
        FeatureEngineer([filter_step("adult", reset_index_mask, columns=["age"])]).fit_transform(
            frame
        )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_filter_that_removes_every_row_returns_an_empty_frame(engine):
    """An all-False mask is a valid result; the caller decides whether empty is an error."""
    frame = _engine(pd.DataFrame({"age": [1.0, 2.0]}), engine)
    out, _ = FeatureEngineer(
        [filter_step("adult", age_present_over_100, columns=["age"])]
    ).fit_transform(frame)
    assert len(out) == 0 and list(out.columns) == ["age"]


def test_filter_keeps_duplicate_index_labels_aligned_with_target():
    """Duplicate pandas labels must not duplicate or misalign targets."""
    frame = pd.DataFrame({"age": [10, 30, 40, 50]}, index=[0, 0, 1, 1])
    y = pd.Series([1, 2, 3, 4], index=frame.index)
    step = filter_step("adult", age_present_over_18, columns=["age"])
    (out_X, out_y), _ = FeatureEngineer([step]).fit_transform((frame, y))
    assert out_X["age"].tolist() == [30, 40, 50]
    assert out_y.tolist() == [2, 3, 4]


def age_present_over_100(df):
    """Reject every row in the fixtures."""
    return df["age"] > 100


def age_present_over_18(df):
    """Keep adults."""
    return df["age"] >= 18
