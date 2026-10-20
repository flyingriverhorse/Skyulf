"""Spark Decimal model values must enter pandas as floats without changing identities."""

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from skyulf.engines.spark_engine import normalize_spark_decimals, restore_spark_decimal_dtypes
from skyulf.integrations.databricks.jobs.serving_endpoint_job import sample_serving_rows
from skyulf.integrations.databricks.scoring.batch.frame_batch import SourceSpec, read_source
from skyulf.integrations.databricks.scoring.incremental.incremental_batch import bounded_frame
from skyulf.integrations.databricks.serving.endpoints import validate_json_rows
from skyulf.integrations.databricks.training.fitting.candidate import (
    TrainingSpec,
    _materialize_training_rows,
    apply_pre_split_step,
)


def _source(records, types):
    """Expose Spark schema and bounded rows without requiring a local Spark runtime."""
    source = MagicMock()
    source.columns = list(types)
    source.schema.fields = [
        SimpleNamespace(
            name=name,
            dataType=SimpleNamespace(
                typeName=lambda dtype=dtype: dtype.split("(")[0],
                simpleString=lambda dtype=dtype: dtype,
            ),
        )
        for name, dtype in types.items()
    ]
    for method in ("where", "select", "orderBy", "limit"):
        getattr(source, method).return_value = source

    def select_expr(*expressions):
        """Execute only the Decimal cast projection under test on these bounded records."""
        converted = dict(types)
        names = []
        for expression in expressions:
            if expression.startswith("CAST("):
                name = expression.split(" AS ")[-1].strip("`").replace("``", "`")
                converted[name] = "double"
                names.append(name)
        if not names:
            return source
        rows = [
            {
                name: float(value) if name in names and value is not None else value
                for name, value in record.items()
            }
            for record in records
        ]
        return _source(rows, converted)

    source.selectExpr.side_effect = select_expr
    rows = [
        SimpleNamespace(asDict=lambda recursive=True, record=record: dict(record))
        for record in records
    ]
    source.toLocalIterator.side_effect = lambda: iter(rows)
    source.collect.return_value = rows
    return source


def _spark(source):
    """Retain the pinned Delta-reader chain and serving table access."""
    spark = MagicMock()
    spark.read.format.return_value = spark.read
    spark.read.option.return_value = spark.read
    spark.read.table.return_value = source
    spark.table.return_value = source
    return spark


@pytest.mark.parametrize("values", [[Decimal("1.25"), None], [None, None]])
def test_batch_decimal_float_dtype_including_all_null(values):
    """All-null Decimal columns must retain numeric dtype instead of becoming categorical."""
    rows = [{"id": index, "amount": value} for index, value in enumerate(values)]
    source = _source(rows, {"id": "bigint", "amount": "decimal(17,6)"})
    spec = SourceSpec(
        table="main.data.amounts",
        version=2,
        period_start=datetime(2026, 1, 1, tzinfo=UTC),
        period_end=datetime(2026, 2, 1, tzinfo=UTC),
        record_key_columns=("id",),
        input_columns=("amount",),
        max_rows=5,
        max_bytes=10000,
    )
    frame = read_source(_spark(source), spec)
    assert str(frame.amount.dtype) == "float64"
    assert frame.id.tolist() == [0, 1]
    assert frame.amount.isna().tolist() == [value is None for value in values]


def test_training_decimal_features_targets_weights_preserve_group_identity():
    """Numeric ML values normalize while precision-sensitive group identifiers remain exact."""
    spec = TrainingSpec(
        table="main.data.amounts",
        version=2,
        record_key_columns=("id",),
        input_columns=("amount",),
        target_column="target",
        weight_column="weight",
        group_column="group",
        max_rows=5,
        max_bytes=10000,
    )
    identity = Decimal("9007199254740993")
    source = _source(
        [
            {
                "id": "a",
                "amount": Decimal("1.25"),
                "target": Decimal("2.5"),
                "weight": Decimal("0.75"),
                "group": identity,
            }
        ],
        {
            "id": "string",
            "amount": "decimal(17,6)",
            "target": "decimal(17,6)",
            "weight": "decimal(10,4)",
            "group": "decimal(20,0)",
        },
    )
    frame = _materialize_training_rows(source, spec, spec.source_columns)
    assert [str(frame[name].dtype) for name in ("amount", "target", "weight")] == ["float64"] * 3
    assert frame.group.iloc[0] == identity
    assert frame.attrs["source_column_conversions"] == (
        {"column": "amount", "source_dtype": "decimal(17,6)", "target_dtype": "float64"},
        {"column": "target", "source_dtype": "decimal(17,6)", "target_dtype": "float64"},
        {"column": "weight", "source_dtype": "decimal(10,4)", "target_dtype": "float64"},
    )


@pytest.mark.parametrize("identity_is_input", [False, True])
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_training_decimal_deduplicate_identity_keeps_distinct_rows(identity_is_input, engine):
    """Pre-split duplicate filtering must compare exact identities before numeric modeling."""
    import polars as pl

    identities = [Decimal("9007199254740992"), Decimal("9007199254740993")]
    step = {
        "name": "dedup",
        "transformer": "Deduplicate",
        "params": {"subset": ["business_id"]},
    }
    spec = TrainingSpec(
        table="main.data.amounts",
        version=0,
        record_key_columns=("id",),
        input_columns=("amount", "business_id") if identity_is_input else ("amount",),
        target_column="target",
        max_rows=5,
        max_bytes=10000,
        pre_split_steps=(step,),
    )
    source = _source(
        [
            {"id": str(index), "amount": Decimal("1.25"), "target": 2.0, "business_id": key}
            for index, key in enumerate(identities)
        ],
        {
            "id": "string",
            "amount": "decimal(17,6)",
            "target": "double",
            "business_id": "decimal(20,0)",
        },
    )
    frame = _materialize_training_rows(source, spec, spec.source_columns)
    native = pl.from_pandas(frame) if engine == "polars" else frame
    filtered = apply_pre_split_step(native, step, keys=["id"], target_column="target")

    assert filtered["business_id"].to_list() == identities
    assert filtered["id"].to_list() == ["0", "1"]
    assert frame.attrs["source_column_conversions"] == (
        {"column": "amount", "source_dtype": "decimal(17,6)", "target_dtype": "float64"},
    )


@pytest.mark.parametrize("identity_is_input", [False, True])
def test_native_training_decimal_deduplicate_keeps_exact_identity(delta_spark, identity_is_input):
    """Native Spark projection must not collapse duplicate keys before local eligibility."""
    identities = [Decimal("9007199254740992"), Decimal("9007199254740993")]
    step = {"name": "dedup", "transformer": "Deduplicate", "params": {"subset": ["business_id"]}}
    spec = TrainingSpec(
        table="main.data.amounts",
        version=0,
        record_key_columns=("id",),
        input_columns=("amount", "business_id") if identity_is_input else ("amount",),
        target_column="target",
        max_rows=5,
        max_bytes=10000,
        pre_split_steps=(step,),
    )
    source = delta_spark.createDataFrame(
        [(str(index), Decimal("1.25"), 2.0, identity) for index, identity in enumerate(identities)],
        "id string, amount decimal(17,6), target double, business_id decimal(20,0)",
    )
    frame = _materialize_training_rows(source, spec, spec.source_columns)
    filtered = apply_pre_split_step(frame, step, keys=["id"], target_column="target")

    assert filtered["business_id"].to_list() == identities
    assert filtered["id"].to_list() == ["0", "1"]
    assert str(frame.amount.dtype) == "float64"


def test_incremental_decimal_features_preserve_exact_keys():
    """Rounding a Decimal record key must never silently merge different identities."""
    identities = [Decimal("9007199254740992"), Decimal("9007199254740993")]
    source = _source(
        [{"id": key, "amount": Decimal("1.25")} for key in identities],
        {"id": "decimal(20,0)", "amount": "decimal(17,6)"},
    )
    frame = bounded_frame(source, ("id", "amount"), ("id",), 5, 10000)
    assert str(frame.amount.dtype) == "float64"
    assert frame.id.tolist() == identities


def test_serving_sample_decimal_values_are_json_ready():
    """REST records and SQL DOUBLE arguments must share normalized sampled values."""
    source = _source([{"amount": Decimal("1.25")}], {"amount": "decimal(17,6)"})
    rows = sample_serving_rows(
        _spark(source), "main.data.amounts", (("amount", "float64"),), rows=1
    )
    assert rows == [{"amount": 1.25}]
    assert type(rows[0]["amount"]) is float


def test_serving_sample_preserves_decimal_record_keys():
    """Smoke sampling must not round model-set identities above float64 precision."""
    identities = [Decimal("9007199254740992"), Decimal("9007199254740993")]
    source = _source(
        [{"id": key, "amount": Decimal("1.25")} for key in identities],
        {"id": "decimal(20,0)", "amount": "decimal(17,6)"},
    )
    rows = sample_serving_rows(
        _spark(source),
        "main.data.amounts",
        (("id", "object"), ("amount", "float64")),
        rows=2,
        record_key_columns=("id",),
    )
    assert [row["id"] for row in rows] == identities
    assert all(type(row["amount"]) is float for row in rows)
    with pytest.raises(ValueError, match="non-JSON"):
        validate_json_rows(rows)


def test_json_type_error_is_distinct_from_nonfinite_values():
    """Operators need the real transport failure instead of an incorrect finite-number diagnosis."""
    with pytest.raises(ValueError, match="non-JSON"):
        validate_json_rows([{"amount": Decimal("1.25")}])
    with pytest.raises(ValueError, match="finite"):
        validate_json_rows([{"amount": float("nan")}])


def test_unrelated_pandas_decimal_objects_are_not_coerced():
    """A plain pandas categorical/object input has no Spark provenance authorizing conversion."""
    frame = pd.DataFrame({"amount": [Decimal("1.25")]})
    with pytest.raises(ValueError, match="non-JSON"):
        validate_json_rows([{"amount": frame.amount.iloc[0]}])


def test_decimal_projection_quotes_names_and_preserves_other_types():
    """A numeric conversion must not reinterpret punctuation or parse categorical strings."""
    source = _source(
        [{"a`b.c": Decimal("1.25"), "category": "1.25", "id": 9007199254740993}],
        {"a`b.c": "decimal(17,6)", "category": "string", "id": "bigint"},
    )
    selected, conversions = normalize_spark_decimals(source, columns=("a`b.c", "category"))
    source.selectExpr.assert_called_once_with(
        "CAST(`a``b.c` AS DOUBLE) AS `a``b.c`", "`category`", "`id`"
    )
    row = selected.collect()[0].asDict()
    assert row == {"a`b.c": 1.25, "category": "1.25", "id": 9007199254740993}
    assert conversions == (
        {"column": "a`b.c", "source_dtype": "decimal(17,6)", "target_dtype": "float64"},
    )


def test_empty_decimal_frame_keeps_float_dtype_and_detached_evidence():
    """Empty reads still carry observed numeric schema without sharing mutable metadata."""
    conversions = (
        {"column": "amount", "source_dtype": "decimal(17,6)", "target_dtype": "float64"},
    )
    frame = restore_spark_decimal_dtypes(pd.DataFrame(columns=["amount"]), conversions)
    conversions[0]["source_dtype"] = "changed"
    assert str(frame.amount.dtype) == "float64"
    assert frame.attrs["source_column_conversions"][0]["source_dtype"] == "decimal(17,6)"


def test_spark_pyfunc_normalizes_only_values_not_lookup_keys():
    """Worker Arrow conversion must not alter the identifiers used by native feature joins."""
    from skyulf.integrations.mlflow.spark.spark_model import _normalize_model_values

    identity = Decimal("9007199254740993")
    source = _source(
        [{"id": "a", "entity": identity, "amount": Decimal("1.25")}],
        {"id": "string", "entity": "decimal(20,0)", "amount": "decimal(17,6)"},
    )
    binding = {
        "lookup_spec": {
            "lookups": [
                {
                    "table_name": "main.data.features",
                    "lookup_key": ["entity"],
                    "feature_names": ["feature"],
                }
            ]
        }
    }
    selected = _normalize_model_values(
        source, [("entity", "object"), ("amount", "float64")], ("id",), binding
    )
    assert selected.collect()[0].asDict() == {"id": "a", "entity": identity, "amount": 1.25}
    assert type(selected.collect()[0].asDict()["amount"]) is float


@pytest.mark.parametrize("path", ["python_pipeline", "native_features", "pyfunc"])
def test_native_worker_arrow_inputs_are_float64(delta_spark, path):
    """Actual Spark Arrow batches must retain numeric dtype and exact large integer keys."""
    from skyulf.core.execution import FrameSpec
    from skyulf.inference._manifest import ColumnSpec
    from skyulf.inference.spark import _select_input
    from skyulf.integrations.mlflow.spark.spark_model import _named_inputs, _normalize_model_values

    source = delta_spark.createDataFrame(
        [(9007199254740993, Decimal("1.25"), None), (9007199254740994, None, None)],
        "id long, amount decimal(17,6), missing decimal(17,6)",
    )
    inputs = [("amount", "float64"), ("missing", "float64")]
    if path == "pyfunc":
        normalized = _normalize_model_values(source, inputs, ("id",), None)
        selected = normalized.select("id", _named_inputs(normalized, inputs).alias("inputs"))
        selected = selected.select("id", "inputs.*")
    else:
        manifest = MagicMock()
        manifest.input_schema = tuple(ColumnSpec(name=name, dtype=dtype) for name, dtype in inputs)
        selected = _select_input(source, manifest, FrameSpec(record_key_columns=("id",)), path)

    def check_batches(batches):
        """Assert the worker receives floats before any model or preprocessor runs."""
        for batch in batches:
            assert str(batch.amount.dtype) == "float64"
            assert str(batch.missing.dtype) == "float64"
            yield batch

    rows = (
        selected.mapInPandas(check_batches, "id long, amount double, missing double")
        .orderBy("id")
        .collect()
    )
    assert [row.id for row in rows] == [9007199254740993, 9007199254740994]
    assert [row.amount for row in rows] == [1.25, None]
    assert [row.missing for row in rows] == [None, None]
