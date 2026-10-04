"""Scoring rules preserve every requested row and reject ambiguous outputs."""

import importlib
import importlib.util
from copy import deepcopy

import pandas as pd
import polars as pl
import pytest

SOURCE = """
import pandas as pd
def eligibility(frame, params):
    return pd.Series(["negative" if x < params["minimum"] else None for x in frame["x"]], index=frame.index)
def outputs(frame, predictions, params):
    return pd.DataFrame({"adjusted": predictions["prediction"] + params["offset"]}, index=frame.index)
"""


def _api():
    """Report missing implementation as the initial behavioral failure."""
    spec = importlib.util.find_spec("skyulf.inference.project_scoring")
    assert spec is not None, "Project scoring contract is not implemented"
    return importlib.import_module("skyulf.inference.project_scoring")


def _config():
    """Use explicit source-owned function names and immutable rule versions."""
    return {
        "eligibility": [
            {"name": "valid", "version": "1", "function": "eligibility", "params": {"minimum": 0}}
        ],
        "outputs": [
            {
                "name": "adjust",
                "version": "1",
                "function": "outputs",
                "params": {"offset": 3},
                "columns": [{"name": "adjusted", "dtype": "float64"}],
            }
        ],
    }


def _predict(frame):
    """Stand in for a model with hand-checkable continuous predictions."""
    values = frame["x"].to_list()
    return pd.DataFrame({"prediction": [float(x * 2) for x in values]})


def _run(frame, *, source=SOURCE, config=None, predict=_predict):
    """Invoke the public scoring boundary with a recorded prediction schema."""
    return _api().run_project_scoring(
        frame,
        predict,
        source=source,
        config=_config() if config is None else config,
        row_keys=["id"],
        prediction_dtypes={"prediction": "float64"},
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_exclusions_preserve_keys_and_apply_outputs_only_to_predictions(engine):
    """Exclusions must not shift predictions onto another row's business key."""
    frame = pd.DataFrame({"id": [11, 12, 13], "x": [2, -1, 4]}, index=[8, 8, 2])
    if engine == "polars":
        frame = pl.from_pandas(frame)
    result = _run(frame)
    assert result["id"].tolist() == [11, 12, 13]
    assert result["prediction"].dropna().tolist() == [4, 8]
    assert result["adjusted"].dropna().tolist() == [7, 11]
    assert result["scoring_status"].tolist() == ["predicted", "excluded", "predicted"]
    assert result["exclusion_reason"].fillna("").tolist() == ["", "negative", ""]
    assert result.index.tolist() == ([8, 8, 2] if engine == "pandas" else [0, 1, 2])


def test_all_excluded_never_calls_model_or_output_hook():
    """An all-excluded batch still produces typed publishable coverage rows."""

    def forbidden(frame):
        """Fail immediately if inference runs without any eligible row."""
        pytest.fail("model invoked for excluded rows")

    source = SOURCE.replace(
        'return pd.DataFrame({"adjusted": predictions["prediction"] + params["offset"]}, index=frame.index)',
        'raise AssertionError("output hook invoked")',
    )
    result = _run(pd.DataFrame({"id": [1], "x": [-1]}), source=source, predict=forbidden)
    assert result["prediction"].isna().all()
    assert str(result["adjusted"].dtype) == "Float64"
    assert result["scoring_status"].tolist() == ["excluded"]


@pytest.mark.parametrize("expression", ["frame.index[::-1]", "pd.RangeIndex(len(frame) - 1)"])
def test_eligibility_rejects_reordered_or_dropped_rows(expression):
    """Callback row mutations must fail before model predictions can be associated."""
    source = (
        f"import pandas as pd\ndef eligibility(frame, params):\n    return pd.Series(None, index={expression}, dtype='string')\n"
        + SOURCE[SOURCE.index("def outputs") :]
    )
    with pytest.raises(ValueError, match="index|rows"):
        _run(pd.DataFrame({"id": [1, 2], "x": [1, 2]}), source=source)


@pytest.mark.parametrize("value", ["7", "True", "''", "[]"])
def test_eligibility_requires_nonempty_reason_or_null(value):
    """Reason values carry audit meaning and cannot be coerced from arbitrary objects."""
    source = SOURCE.replace(
        '["negative" if x < params["minimum"] else None for x in frame["x"]]',
        f'[{value} for x in frame["x"]]',
    )
    with pytest.raises(ValueError, match="reason"):
        _run(pd.DataFrame({"id": [1], "x": [1]}), source=source)


@pytest.mark.parametrize("name", ["prediction", "id", "run_id", "scoring_status", "probability_4"])
def test_output_cannot_overwrite_inputs_predictions_or_metadata(name):
    """Business rules cannot silently replace model results or publication identity."""
    config = _config()
    config["outputs"][0]["columns"][0]["name"] = name
    with pytest.raises(ValueError, match="column|reserved|overwrite"):
        _run(pd.DataFrame({"id": [1], "x": [1]}), config=config)


@pytest.mark.parametrize(
    "dtype,value", [("float64", "'bad'"), ("int64", "1.2"), ("string", "1"), ("bool", "1")]
)
def test_output_declared_type_rejects_lossy_coercion(dtype, value):
    """Explicit types must reject semantic changes such as stringifying a number."""
    config = _config()
    config["outputs"][0]["columns"][0]["dtype"] = dtype
    source = SOURCE.replace(
        'predictions["prediction"] + params["offset"]', f"[{value}] * len(frame)"
    )
    with pytest.raises(ValueError, match="dtype|type"):
        _run(pd.DataFrame({"id": [1], "x": [1]}), config=config, source=source)


@pytest.mark.parametrize("change", ["version", "unknown", "nan", "foreign"])
def test_contract_rejects_unversioned_nonjson_or_foreign_callbacks(change):
    """Saved configuration must be reproducible using only its trusted project module."""
    config = _config()
    rule = config["eligibility"][0]
    if change == "version":
        rule.pop("version")
    elif change == "unknown":
        rule["extra"] = True
    elif change == "nan":
        rule["params"]["minimum"] = float("nan")
    else:
        rule["function"] = "pd.isna"
    with pytest.raises(ValueError):
        _api().validate_scoring_config(config, SOURCE)


def test_callback_mutations_do_not_escape_into_prediction_inputs_or_config():
    """Trusted callbacks still receive defensive inputs to keep inference repeatable."""
    source = SOURCE.replace(
        "    return pd.Series(",
        '    frame["x"] = 99\n    params["minimum"] = -1\n    return pd.Series(',
    )
    config = _config()
    original = deepcopy(config)
    frame = pd.DataFrame({"id": [1], "x": [2]})
    result = _run(frame, source=source, config=config)
    assert result["prediction"].tolist() == [4]
    assert frame["x"].tolist() == [2]
    assert config == original


def test_prediction_index_must_match_normalized_eligible_rows():
    """A model wrapper returning permuted rows must not be accepted positionally."""

    def reversed_prediction(frame):
        """Return valid numbers attached to the wrong row order."""
        return _predict(frame).iloc[::-1]

    with pytest.raises(ValueError, match="index|rows"):
        _run(pd.DataFrame({"id": [1, 2], "x": [1, 2]}), predict=reversed_prediction)


def test_saved_package_callback_resolves_without_preimporting_submodule():
    """Artifact scoring must load dotted callbacks directly from the saved package."""
    files = {"__init__.py": "", "rules.py": SOURCE}
    source = (
        "from skyulf.inference.project_package import install_project_package\n"
        f"install_project_package(__name__, {files!r})\n"
    )
    config = _config()
    config["eligibility"][0]["function"] = "rules.eligibility"
    config["outputs"][0]["function"] = "rules.outputs"
    result = _run(pd.DataFrame({"id": [1], "x": [2]}), source=source, config=config)
    assert result["adjusted"].tolist() == [7]


@pytest.mark.parametrize(
    "expression",
    ["values.iloc[::-1]", "values.iloc[:0]", "values.rename(columns={'adjusted': 'wrong'})"],
)
def test_output_rule_rejects_changed_index_length_or_columns(expression):
    """Same-length business output cannot silently attach values to different rows."""
    source = (
        SOURCE.replace("    return pd.DataFrame(", "    values = pd.DataFrame(")
        + f"    return {expression}\n"
    )
    with pytest.raises(ValueError, match="rows|index|columns"):
        _run(pd.DataFrame({"id": [1, 2], "x": [1, 2]}), source=source)


def test_first_exclusion_wins_and_output_rules_follow_declared_order():
    """Ordering is persisted behavior for conflicting reasons and derived fields."""
    source = (
        SOURCE
        + """
def second_eligibility(frame, params):
    return pd.Series(["second" if x < 1 else None for x in frame["x"]], index=frame.index)
def second_output(frame, predictions, params):
    return pd.DataFrame({"decision": predictions["adjusted"] > 6}, index=frame.index)
"""
    )
    config = _config()
    config["eligibility"].append(
        {"name": "second", "version": "1", "function": "second_eligibility", "params": {}}
    )
    config["outputs"].append(
        {
            "name": "decision",
            "version": "1",
            "function": "second_output",
            "params": {},
            "columns": [{"name": "decision", "dtype": "bool"}],
        }
    )
    result = _run(pd.DataFrame({"id": [1, 2, 3], "x": [-1, 0, 2]}), source=source, config=config)
    assert result["exclusion_reason"].fillna("").tolist() == ["negative", "second", ""]
    assert result["decision"].dropna().tolist() == [True]


@pytest.mark.parametrize(
    "dtype,value,expected",
    [
        ("string", "'hold'", "hold"),
        ("int64", "7", 7),
        ("bool", "True", True),
        ("float64", "7", 7.0),
    ],
)
def test_declared_outputs_accept_scalar_types_and_nulls(dtype, value, expected):
    """Nullable output conversion preserves declared scalar meaning in mixed batches."""
    config = _config()
    config["outputs"][0]["columns"][0]["dtype"] = dtype
    source = SOURCE.replace(
        'predictions["prediction"] + params["offset"]', f"pd.Series([{value}, None], dtype=object)"
    )
    result = _run(pd.DataFrame({"id": [1, 2], "x": [1, 2]}), source=source, config=config)
    assert result["adjusted"].iloc[0] == expected
    assert pd.isna(result["adjusted"].iloc[1])


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", ""),
        ("function", "../outside"),
        ("params", []),
        ("columns", []),
        ("columns", [{"name": "bad", "dtype": "object"}]),
    ],
)
def test_rule_validation_rejects_invalid_declarations(field, value):
    """Invalid saved rule declarations must fail before invoking project callbacks."""
    config = _config()
    config["outputs"][0][field] = value
    with pytest.raises(ValueError):
        _api().validate_scoring_config(config, SOURCE)


def test_validation_copies_config_and_rejects_duplicate_output_columns():
    """Duplicated declared columns would silently overwrite earlier rule outputs."""
    config = _config()
    validated = _api().validate_scoring_config(config, SOURCE)
    validated["outputs"][0]["params"]["offset"] = 900
    assert config["outputs"][0]["params"]["offset"] == 3
    second = deepcopy(config["outputs"][0])
    second["name"] = "second"
    config["outputs"].append(second)
    with pytest.raises(ValueError, match="unique"):
        _api().validate_scoring_config(config, SOURCE)


def test_empty_batch_returns_typed_schema_without_model_call():
    """Empty incremental input must remain a valid no-op scoring batch."""

    def forbidden(frame):
        """Catch accidental prediction on an empty batch."""
        pytest.fail("empty model call")

    result = _run(
        pd.DataFrame({"id": pd.Series(dtype="int64"), "x": pd.Series(dtype="float64")}),
        predict=forbidden,
    )
    assert result.empty
    assert result.columns.tolist() == [
        "id",
        "prediction",
        "adjusted",
        "scoring_status",
        "exclusion_reason",
    ]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("column", ["prediction", "probability_0"])
@pytest.mark.parametrize("invalid", [None, pd.NA, float("nan"), float("inf"), -float("inf")])
def test_estimator_missing_and_nonfinite_outputs_cannot_be_published(engine, column, invalid):
    """Eligible rows cannot become successful outcomes with missing estimator values."""
    frame = pd.DataFrame({"id": [1, 2, 3], "x": [2, -1, 4]})
    if engine == "polars":
        frame = pl.from_pandas(frame)

    def invalid_estimator(selected):
        """Return a malformed estimator value alongside valid eligible predictions."""
        predictions = pd.DataFrame(
            {"prediction": [4.0, 8.0], "probability_0": [0.2, 0.8]}, dtype=object
        )
        predictions.loc[1, column] = invalid
        return predictions

    config = _config()
    config["outputs"] = []
    with pytest.raises(ValueError, match="missing|nonfinite|incompatible"):
        _api().run_project_scoring(
            frame,
            invalid_estimator,
            source=SOURCE,
            config=config,
            row_keys=["id"],
            prediction_dtypes={"prediction": "float64", "probability_0": "float64"},
        )
