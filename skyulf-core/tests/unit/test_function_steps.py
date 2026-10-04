"""Plain-function steps must be simple to write and as safe as Calculator/Applier pairs."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.preprocessing import FeatureEngineer, column_step, filter_step, fitted_step


def income_per_age(df):
    """Example column: income divided by age."""
    return df["income"] / df["age"]


def scaled(df, params):
    """Example column using user params."""
    return df["income"] * params["factor"]


def two_columns(df):
    """Example returning two columns at once."""
    return pd.DataFrame({"a": df["age"] + 1, "b": df["age"] - 1})


def drop_first_row(df):
    """Invalid example: changes the number of rows."""
    return df["age"].iloc[1:]


def adult_flag_array(df):
    """Example returning a plain NumPy array, as np.where does."""
    return np.where(df["age"] >= 18, 1, 0)


def sorted_ages(df):
    """Invalid example: same rows, but in a different order."""
    return df["age"].sort_values(ascending=False)


def learn_city_median(df, y):
    """Learn the training median income per city."""
    return {"medians": df.groupby("city")["income"].median().to_dict()}


def apply_city_median(df, state):
    """Map each city to its saved median; unseen cities become NaN."""
    return df["city"].map(state["medians"]).astype(float)


def learn_target_mean(df, y):
    """Learn the training target mean per city (target encoding)."""
    return {"means": y.groupby(df["city"].to_numpy()).mean().to_dict()}


def apply_target_mean(df, state):
    """Apply the saved target means without seeing the target."""
    return df["city"].map(state["means"]).astype(float)


def learn_not_json(df, y):
    """Invalid example: returns a non-JSON state."""
    return {"frame": df}


def adult(df):
    """Keep rows with age at least 18."""
    return df["age"] >= 18


def not_boolean(df):
    """Invalid example: returns numbers instead of True/False."""
    return df["age"]


def _frame(engine):
    """Small mixed frame in the requested engine."""
    frame = pd.DataFrame(
        {
            "income": [100.0, 200.0, 300.0, 400.0],
            "age": [10, 20, 30, 40],
            "city": ["A", "A", "B", "B"],
        }
    )
    return pl.from_pandas(frame) if engine == "polars" else frame


def _pandas(frame):
    """Compare results independently of engine."""
    return frame.to_pandas() if isinstance(frame, pl.DataFrame) else frame


def _fit(steps, data):
    """Fit a FeatureEngineer and return it with the transformed training data."""
    engineer = FeatureEngineer(steps)
    out, _ = engineer.fit_transform(data)
    return engineer, out


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_column_step_adds_a_column_in_the_callers_engine(engine):
    """A four-line function must become a saved feature without changing other columns."""
    data = _frame(engine)
    _, out = _fit([column_step("ratio", income_per_age, output="income_per_age")], data)
    assert isinstance(out, type(data))
    result = _pandas(out)
    assert result["income_per_age"].tolist() == [10.0, 10.0, 10.0, 10.0]
    assert list(result.columns) == ["income", "age", "city", "income_per_age"]


def test_column_step_keeps_pandas_index_and_passes_params():
    """User params reach the function and original row labels are preserved."""
    data = _frame("pandas").set_index(pd.Index([7, 3, 9, 1]))
    _, out = _fit([column_step("scaled", scaled, output="scaled", params={"factor": 2})], data)
    assert out.index.tolist() == [7, 3, 9, 1]
    assert out["scaled"].tolist() == [200.0, 400.0, 600.0, 800.0]


def test_column_step_supports_several_outputs():
    """A DataFrame result maps to the listed output names in order."""
    _, out = _fit([column_step("two", two_columns, output=["plus", "minus"])], _frame("pandas"))
    assert out["plus"].tolist() == [11, 21, 31, 41]
    assert out["minus"].tolist() == [9, 19, 29, 39]


def test_lambda_and_nested_functions_are_rejected_with_guidance():
    """Unsaveable functions must fail when the recipe is built, not at inference."""
    with pytest.raises(ValueError, match="top-level def"):
        column_step("bad", lambda df: df["age"], output="x")

    def nested(df):
        """Nested function; cannot be found by name later."""
        return df["age"]

    with pytest.raises(ValueError, match="top-level def"):
        column_step("bad", nested, output="x")


def test_column_step_rejects_row_changes():
    """A column step must not silently drop or reorder rows."""
    with pytest.raises(ValueError, match="keep every row"):
        _fit([column_step("bad", drop_first_row, output="x")], _frame("pandas"))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_column_step_accepts_numpy_arrays_in_row_order(engine):
    """np.where results carry no index; they must follow df row order, not be rejected."""
    data = _frame(engine)
    if engine == "pandas":
        data = data.set_index(pd.Index([7, 3, 9, 1]))
    _, out = _fit([column_step("adult", adult_flag_array, output="is_adult")], data)
    assert _pandas(out)["is_adult"].tolist() == [0, 1, 1, 1]
    if engine == "pandas":
        assert out.index.tolist() == [7, 3, 9, 1]


def test_column_step_reports_reordered_rows_clearly():
    """Sorting inside a step must name the order problem, not claim rows went missing."""
    with pytest.raises(ValueError, match="different index or order"):
        _fit([column_step("bad", sorted_ages, output="x")], _frame("pandas"))


def test_existing_column_requires_explicit_replace():
    """Overwriting an input column is allowed only when requested."""
    with pytest.raises(ValueError, match="replace=True"):
        _fit([column_step("bad", income_per_age, output="income")], _frame("pandas"))
    _, out = _fit(
        [column_step("ok", income_per_age, output="income", replace=True)], _frame("pandas")
    )
    assert out["income"].tolist() == [10.0, 10.0, 10.0, 10.0]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fitted_step_learns_once_and_reuses_state(engine):
    """New batches must use the training medians, never their own statistics."""
    step = fitted_step("city_median", learn_city_median, apply_city_median, output="city_median")
    engineer, out = _fit([step], _frame(engine))
    assert _pandas(out)["city_median"].tolist() == [150.0, 150.0, 350.0, 350.0]
    assert engineer.fitted_steps[0]["artifact"]["state"] == {"medians": {"A": 150.0, "B": 350.0}}
    new = pd.DataFrame({"income": [9999.0, 1.0], "age": [50, 60], "city": ["B", "NEW"]})
    if engine == "polars":
        new = pl.from_pandas(new)
    result = _pandas(engineer.transform(new))["city_median"].tolist()
    assert result[0] == 350.0 and np.isnan(result[1])


def test_fitted_step_can_learn_from_the_target_but_apply_cannot_see_it():
    """Target encoding learns from y on training rows; inference needs no target."""
    data = _frame("pandas")
    X, y = data, pd.Series([1.0, 3.0, 10.0, 20.0], name="target")
    step = fitted_step("city_te", learn_target_mean, apply_target_mean, output="city_te")
    engineer, (out_X, _) = _fit([step], (X, y))
    assert out_X["city_te"].tolist() == [2.0, 2.0, 15.0, 15.0]
    assert engineer.transform(X.iloc[:1])["city_te"].tolist() == [2.0]


def learn_target_mean_no_leak(df, y):
    """Learn per-city target means and prove the target column itself is hidden."""
    assert "target" not in df.columns
    return {"means": y.groupby(df["city"].to_numpy()).mean().to_dict()}


def apply_target_mean_no_leak(df, state):
    """Apply saved means and prove the target column is hidden here too."""
    assert "target" not in df.columns
    return df["city"].map(state["means"]).astype(float)


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fitted_step_reads_target_column_from_pipeline_and_hides_it(engine):
    """Real pipelines keep y inside X; learn must still receive it and never see it in df."""
    frame = _pandas(_frame("pandas")).assign(target=[1.0, 3.0, 10.0, 20.0])
    data = pl.from_pandas(frame) if engine == "polars" else frame
    step = fitted_step(
        "city_te", learn_target_mean_no_leak, apply_target_mean_no_leak, output="city_te"
    )
    engineer = FeatureEngineer([step])
    out, _ = engineer.fit_transform(data, target_column="target")
    out = _pandas(out)
    assert out["city_te"].tolist() == [2.0, 2.0, 15.0, 15.0]
    assert out["target"].tolist() == [1.0, 3.0, 10.0, 20.0]
    scored = _pandas(engineer.transform(_frame(engine)))
    assert scored["city_te"].tolist() == [2.0, 2.0, 15.0, 15.0]


def test_fitted_step_rejects_unsaveable_state():
    """Learned state must be plain JSON so it can be saved and digested."""
    step = fitted_step("bad", learn_not_json, apply_city_median, output="x")
    with pytest.raises(ValueError, match="JSON"):
        _fit([step], _frame("pandas"))


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_filter_step_keeps_rows_values_and_target_aligned(engine):
    """Filtering selects rows only; survivors and their targets stay unchanged."""
    data = _frame(engine)
    y = pd.Series([0, 1, 0, 1]) if engine == "pandas" else pl.Series("y", [0, 1, 0, 1])
    step = filter_step("adult_only", adult, columns=["age"])
    assert step["pre_split"] == {
        "effect": "filter",
        "required_columns": ["age"],
        "learns_from_data": False,
    }
    _, (out_X, out_y) = _fit([step], (data, y))
    assert _pandas(out_X)["age"].tolist() == [20, 30, 40]
    assert list(out_y) == [1, 0, 1]


def test_filter_step_is_skipped_at_inference_like_other_row_droppers():
    """Inference must never silently drop requested rows."""
    engineer, _ = _fit([filter_step("adult_only", adult, columns=["age"])], _frame("pandas"))
    assert len(engineer.transform(_frame("pandas"), preserve_rows=True)) == 4


def test_filter_step_requires_boolean_mask():
    """Numeric masks are ambiguous and must be rejected."""
    with pytest.raises(ValueError, match="True/False"):
        _fit([filter_step("bad", not_boolean, columns=["age"])], _frame("pandas"))
