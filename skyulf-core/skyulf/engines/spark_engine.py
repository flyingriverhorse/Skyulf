"""Optional Spark adapter with no session ownership or driver materialization."""

import importlib
from collections.abc import Iterable
from typing import Any, NoReturn

import pandas as pd

from .registry import BaseEngine, EngineName, EngineRegistry


class DistributedMaterializationError(TypeError):
    """A local-only operation was requested for distributed Spark data."""


def _reject_local(operation: str) -> NoReturn:
    """Fail before collecting rows or counting a distributed frame."""
    raise DistributedMaterializationError(
        f"Spark data is distributed; {operation} is a local-only operation. "
        "Use native Spark transformations or an explicitly supported worker inference path. "
        "Skyulf will not collect the dataset to the driver."
    )


def is_spark_input(data: Any) -> bool:
    """Recognize Spark namespaces and subclasses without importing optional dependencies."""
    if isinstance(data, SkyulfSparkWrapper):
        return True
    return any(base.__module__.split(".", 1)[0] == "pyspark" for base in type(data).__mro__)


def normalize_spark_decimals(
    frame: Any, *, columns: Iterable[str] | None = None
) -> tuple[Any, tuple[dict[str, str], ...]]:
    """Project actual Spark Decimal model columns to double before pandas transport.

    Callers exclude identity, grouping and time controls. Precision beyond
    float64's approximately 15–17 significant digits can be lost. No rows are read.
    """
    names = None if columns is None else set(columns)
    fields = getattr(getattr(frame, "schema", None), "fields", ())
    conversions = tuple(
        {
            "column": field.name,
            "source_dtype": field.dataType.simpleString(),
            "target_dtype": "float64",
        }
        for field in fields
        if field.dataType.typeName() == "decimal" and (names is None or field.name in names)
    )
    if not conversions:
        return frame, conversions
    converted = {item["column"] for item in conversions}
    expressions = []
    for name in frame.columns:
        quoted = "`" + name.replace("`", "``") + "`"
        expressions.append(f"CAST({quoted} AS DOUBLE) AS {quoted}" if name in converted else quoted)
    return frame.selectExpr(*expressions), conversions


def restore_spark_decimal_dtypes(
    frame: pd.DataFrame, conversions: tuple[dict[str, str], ...]
) -> pd.DataFrame:
    """Retain float64 for empty/all-null converted columns and original schema evidence."""
    result = frame.astype({item["column"]: item["target_dtype"] for item in conversions})
    result.attrs["source_column_conversions"] = tuple(dict(item) for item in conversions)
    return result


class SkyulfSparkWrapper:
    """Expose lazy projection and metadata without pretending to be a local frame."""

    def __init__(self, data: Any) -> None:
        """Wrap a real Spark DataFrame without starting or storing a session."""
        if not SparkEngine.is_compatible(data):
            raise TypeError("Spark engine requires a pyspark.sql.DataFrame.")
        self._df = data.to_native() if isinstance(data, SkyulfSparkWrapper) else data

    @property
    def columns(self) -> list[str]:
        """Return column names without collecting rows."""
        return self._df.columns

    @property
    def schema(self) -> Any:
        """Return the native Spark schema without collecting rows."""
        return self._df.schema

    def select(self, columns: list[str]) -> "SkyulfSparkWrapper":
        """Project literal names, escaping Spark SQL identifiers including backticks."""
        quoted = ["`" + name.replace("`", "``") + "`" for name in columns]
        return SkyulfSparkWrapper(self._df.select(*quoted))

    def to_native(self) -> Any:
        """Return the same distributed frame for explicit native Spark operations."""
        return self._df

    def __len__(self) -> NoReturn:
        """Reject implicit row counting; native count is an explicit Spark action."""
        _reject_local("len")

    @property
    def shape(self) -> NoReturn:
        """Reject local shape semantics that would hide a Spark count action."""
        _reject_local("shape")

    def to_pandas(self) -> NoReturn:
        """Reject whole-frame collection into pandas on the driver."""
        _reject_local("to_pandas")

    def to_numpy(self) -> NoReturn:
        """Reject whole-frame collection into NumPy on the driver."""
        _reject_local("to_numpy")

    def to_arrow(self) -> NoReturn:
        """Reject whole-frame collection into Arrow on the driver."""
        _reject_local("to_arrow")


class SparkEngine(BaseEngine):
    """Resolve real Spark frames lazily; leave session creation to the caller."""

    name = EngineName.SPARK

    @classmethod
    def ensure_available(cls) -> None:
        """Load the optional runtime only when Spark is explicitly requested."""
        try:
            importlib.import_module("pyspark.sql")
        except ImportError as exc:
            raise ImportError(
                "Spark engine requires the optional skyulf-core[spark] dependency."
            ) from exc

    @classmethod
    def is_compatible(cls, data: Any) -> bool:
        """Accept actual classic/Connect DataFrame classes, not namespace lookalikes."""
        if not is_spark_input(data):
            return False
        cls.ensure_available()
        native = data.to_native() if isinstance(data, SkyulfSparkWrapper) else data
        return isinstance(native, importlib.import_module("pyspark.sql").DataFrame)

    @classmethod
    def wrap(cls, data: Any) -> SkyulfSparkWrapper:
        """Wrap a distributed dataframe without executing its query."""
        return data if isinstance(data, SkyulfSparkWrapper) else SkyulfSparkWrapper(data)

    @classmethod
    def to_numpy(cls, df: Any) -> NoReturn:
        """Reject implicit conversion of distributed inputs for local sklearn."""
        _reject_local("to_numpy")

    @classmethod
    def from_pandas(cls, df: Any) -> NoReturn:
        """Require explicit creation through a caller-owned SparkSession."""
        raise TypeError("Use your SparkSession.createDataFrame explicitly; Skyulf owns no session.")

    @classmethod
    def create_dataframe(cls, data: Any) -> NoReturn:
        """Require explicit creation through a caller-owned SparkSession."""
        raise TypeError("Use your SparkSession.createDataFrame explicitly; Skyulf owns no session.")


EngineRegistry.register("spark", SparkEngine)
