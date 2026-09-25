"""Strict source date rules shared by distributed reads and local training splits."""

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from importlib import import_module
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

_TOKENS = {
    "%Y": r"[0-9]{4}",
    "%m": r"[0-9]{2}",
    "%d": r"[0-9]{2}",
    "%H": r"[0-9]{2}",
    "%M": r"[0-9]{2}",
    "%S": r"[0-9]{2}",
    "%f": r"[0-9]{1,6}",
    "%z": r"(?:Z|[+-][0-9]{2}:?[0-9]{2})",
}
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True, kw_only=True)
class TrainingDateSpec:
    """Declare numeric string formats, source timezone and date-only calendar semantics.

    Strings require a format using %Y, %m, %d and optionally %H, %M, %S,
    %f and %z. Local times require an IANA timezone. Date-only values require
    date_only='midnight' and a timezone. Native aware datetimes retain their
    instant, including already-normalized values returned by the Spark reader.
    """

    format: str | None = None
    timezone: str | None = None
    date_only: Literal["reject", "midnight"] = "reject"

    def __post_init__(self) -> None:
        """Fail offline for incomplete formats and contradictory calendar policies."""
        if self.date_only not in ("reject", "midnight"):
            raise ValueError("date_only must be reject or midnight.")
        if self.timezone is not None:
            if not isinstance(self.timezone, str) or not self.timezone:
                raise ValueError("timezone must be an IANA timezone name.")
            try:
                ZoneInfo(self.timezone)
            except (ValueError, ZoneInfoNotFoundError) as exc:
                raise ValueError("timezone must be an available IANA timezone name.") from exc
        if self.date_only == "midnight" and self.timezone is None:
            raise ValueError("date_only=midnight requires a timezone.")
        if self.format is not None:
            self._validate_format()

    def _validate_format(self) -> None:
        """Limit parsing to full numeric calendar dates and explicit clock/offset fields."""
        if not isinstance(self.format, str) or not self.format:
            raise ValueError("format must be a nonempty explicit numeric date format.")
        tokens = re.findall(r"%.", self.format)
        literals = re.sub(r"%.", "", self.format)
        if (
            not {"%Y", "%m", "%d"}.issubset(tokens)
            or len(tokens) != len(set(tokens))
            or any(token not in _TOKENS for token in tokens)
            or re.search(r"[^ /:.,T+\-]", literals)
        ):
            raise ValueError("format requires %Y, %m, %d and supported numeric directives.")
        timed = "%H" in tokens
        if (
            ("%M" in tokens) != timed
            or any(token in tokens and not timed for token in ("%S", "%f", "%z"))
            or ("%f" in tokens and "%S" not in tokens)
        ):
            raise ValueError("format time requires %H and %M; fractions require %S.")
        if not timed and self.date_only != "midnight":
            raise ValueError("Date-only format requires date_only=midnight.")
        if "%z" in tokens and self.timezone is not None:
            raise ValueError("Offset format must not also declare a source timezone.")
        if "%z" not in tokens and self.timezone is None:
            raise ValueError("Local format requires a source timezone.")


def training_date_spec(value: Any) -> TrainingDateSpec:
    """Decode a strict configuration object without silently accepting unknown fields."""
    if not isinstance(value, dict):
        raise ValueError("Date parsing settings must be an object.")
    try:
        return TrainingDateSpec(**value)
    except TypeError as exc:
        raise ValueError("Unknown date parsing settings.") from exc


def _local_instant(value: datetime, timezone: str) -> datetime:
    """Reject DST gaps and overlaps rather than choosing a fold implicitly."""
    zone = ZoneInfo(timezone)
    candidates = {
        candidate.astimezone(UTC)
        for fold in (0, 1)
        if (candidate := value.replace(tzinfo=zone, fold=fold))
        .astimezone(UTC)
        .astimezone(zone)
        .replace(tzinfo=None)
        == value
    }
    if len(candidates) != 1:
        raise ValueError("Source local time is ambiguous or nonexistent in its timezone.")
    return candidates.pop()


def parse_training_date(
    value: Any, spec: TrainingDateSpec, *, allow_null: bool = False
) -> datetime | None:
    """Normalize one declared source value to UTC without inferring a format or zone."""
    if (
        value is None
        or value is pd.NaT
        or value is pd.NA
        or (isinstance(value, float) and pd.isna(value))
    ):
        if allow_null:
            return None
        raise ValueError("Training event time must not be null.")
    if isinstance(value, str):
        if spec.format is None:
            raise ValueError("String training dates require an explicit format.")
        pattern = re.escape(spec.format)
        for token, expression in _TOKENS.items():
            pattern = pattern.replace(re.escape(token), expression)
        if not re.fullmatch(pattern, value):
            raise ValueError("Training date does not match its declared format.")
        value = datetime.strptime(value, spec.format)
    elif isinstance(value, date) and not isinstance(value, datetime):
        if spec.date_only != "midnight":
            raise ValueError("Date-only source requires date_only=midnight and timezone.")
        value = datetime.combine(value, time.min)
    if not isinstance(value, datetime):
        raise ValueError("Training dates must be datetime, date or explicitly formatted strings.")
    if isinstance(value, pd.Timestamp):
        if value.nanosecond:
            raise ValueError("Training dates support microsecond precision, not nanoseconds.")
        value = value.to_pydatetime()
    if value.tzinfo is not None and value.utcoffset() is not None:
        normalized = value.astimezone(UTC)
        if normalized.astimezone(value.tzinfo).replace(tzinfo=None) != value.replace(tzinfo=None):
            raise ValueError("Training date is not a valid local instant.")
        return normalized
    if spec.timezone is None:
        raise ValueError("Naive training dates require an explicit source timezone.")
    return _local_instant(value, spec.timezone)


def instant_microseconds(value: datetime) -> int:
    """Encode UTC transport with integer arithmetic, including pre-epoch instants."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Transport requires a timezone-aware instant.")
    delta = value.astimezone(UTC) - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def instant_from_microseconds(value: int | None) -> datetime | None:
    """Restore an instant without Spark/Python process timezone conversion."""
    return None if value is None else _EPOCH + timedelta(microseconds=value)


def normalize_training_dates(
    source: Any,
    *,
    event_column: str | None,
    result_column: str | None,
    event_spec: TrainingDateSpec,
    result_spec: TrainingDateSpec,
) -> Any:
    """Validate the pinned source before filtering, and return integer UTC date columns.

    Validation scans distributed rows, returning at most one bounded error row.
    Native Spark timestamps use unix_micros so Python's process timezone never
    interprets a naive timestamp received from Spark. Other types run the same
    strict scalar parser on workers, independent of Spark's date parser policy.
    """
    if event_column is None and result_column is None:
        return source
    F = import_module("pyspark.sql.functions")
    types = import_module("pyspark.sql.types")

    expressions = {}
    for name, spec, nullable in (
        (event_column, event_spec, False),
        (result_column, result_spec, True),
    ):
        if name is None:
            continue
        dtype = source.schema[name].dataType
        value = F.col(name)
        if isinstance(dtype, types.TimestampType):
            if spec != TrainingDateSpec():
                raise ValueError(f"{name}: native Spark timestamps require default parsing rules.")
            micros = F.expr(f"unix_micros(`{name}`)")
            error = F.lit(None).cast("string")
            if not nullable:
                error = F.when(value.isNull(), F.lit("Training event time must not be null."))
            expressions[name] = F.struct(micros.alias("micros"), error.alias("error"))
            continue
        if isinstance(dtype, types.StringType):
            if spec.format is None:
                raise ValueError(f"{name}: string training dates require an explicit format.")
        elif isinstance(dtype, types.DateType):
            if spec.date_only != "midnight" or spec.timezone is None or spec.format is not None:
                raise ValueError(
                    f"{name}: native dates require date_only=midnight, timezone and no format."
                )
        elif isinstance(dtype, types.TimestampNTZType):
            if spec.timezone is None or spec.format is not None or spec.date_only != "reject":
                raise ValueError(f"{name}: native local timestamps require timezone only.")
        else:
            raise ValueError(f"{name}: unsupported training date source type.")
        expressions[name] = F.udf(
            _date_transport_parser(spec, nullable), "struct<micros:long,error:string>"
        )(value)
    normalized = source.select(
        *(
            expressions[name].alias(name) if name in expressions else F.col(name)
            for name in source.columns
        )
    )
    errors = [F.col(name).getField("error") for name in expressions]
    invalid = (
        normalized.select(F.coalesce(*errors).alias("error"))
        .where(F.col("error").isNotNull())
        .limit(1)
    )
    for row in invalid.toLocalIterator():
        raise ValueError(f"Invalid training dates in {', '.join(expressions)}: {row['error']}")
    return normalized.select(
        *(
            F.col(name).getField("micros").alias(name) if name in expressions else F.col(name)
            for name in source.columns
        )
    )


def _date_transport_parser(spec: TrainingDateSpec, nullable: bool) -> Any:
    """Bind worker parsing rules without capturing Spark objects in a UDF closure."""

    def parse(value: Any) -> tuple[int | None, str | None]:
        """Return a bounded error instead of letting invalid rows disappear in filters."""
        try:
            instant = parse_training_date(value, spec, allow_null=nullable)
            return (None if instant is None else instant_microseconds(instant), None)
        except (ValueError, TypeError, OverflowError) as exc:
            return None, str(exc)

    return parse
