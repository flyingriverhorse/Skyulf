"""Only confirmed CDF history loss may authorize a pinned full replacement."""

from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.data.admission import BatchConflictError
from skyulf.integrations.databricks.scoring.incremental.incremental_batch import (
    select_incremental_rows,
)


class StructuredError(RuntimeError):
    """Represent Spark's public structured error interface without a JVM."""

    def __init__(self, condition):
        """Keep free-form text deliberately unrelated to the structured condition."""
        super().__init__("remote read failed")
        self.condition = condition

    def getCondition(self):
        """Expose the machine-readable Spark error condition."""
        return self.condition


class SparkConnectGrpcException(StructuredError):
    """Mirror Connect's discarded Java causes and separately retained server stacktrace."""

    __module__ = "pyspark.errors.exceptions.connect"

    def __init__(self, condition, stacktrace):
        """Retain the same public structured error and stacktrace interfaces as Connect."""
        super().__init__(condition)
        self._stacktrace = stacktrace

    def getStackTrace(self):
        """Expose only the server stacktrace, independently of the displayed message."""
        return self._stacktrace


def request_fields():
    """Pin one lost interval and the last trusted publication state."""
    return {
        "version": 1,
        "layout": "single_model",
        "source_table": "db.source",
        "source_table_id": "source-id",
        "target_table": "db.target",
        "target_table_id": "target-id",
        "target_version": 4,
        "source_start_version": 7,
        "source_end_version": 10,
        "model_name": "db.model",
        "model_version": "2",
        "model_digest": "a" * 64,
    }


@pytest.mark.parametrize(
    "condition,recoverable",
    [
        ("DELTA_CHANGE_DATA_FILE_NOT_FOUND", True),
        ("DELTA_TRUNCATED_TRANSACTION_LOG", True),
        ("DELTA_MISSING_FILES_UNEXPECTED_VERSION", True),
        ("DELTA_UNSUPPORTED_TIME_TRAVEL_BEYOND_DELETED_FILE_RETENTION_DURATION", True),
        ("DELTA_MISSING_CHANGE_DATA", False),
        ("DELTA_FILE_NOT_FOUND", False),
        ("FAILED_READ_FILE.DBR_FILE_NOT_EXIST", False),
        ("INSUFFICIENT_PERMISSIONS", False),
        ("CONNECTION_FAILED", False),
    ],
)
@pytest.mark.parametrize("stage", ["read", "action"])
def test_only_structured_history_errors_normalize_during_cdf(condition, recoverable, stage):
    """Permission, corruption and disabled CDF failures cannot trigger destructive recovery."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import CdfHistoryExpired

    spark = Mock()
    reader = spark.read.format.return_value
    reader.option.return_value = reader
    selected = reader.table.return_value
    error = StructuredError(condition)
    boundary = (
        reader.table if stage == "read" else selected.where.return_value.limit.return_value.count
    )
    boundary.side_effect = error
    expected = CdfHistoryExpired if recoverable else StructuredError
    with pytest.raises(expected) as failure:
        select_incremental_rows(spark, "db.source", 7, 10, None, Mock())
    assert failure.value.__cause__ is error if recoverable else failure.value is error


@pytest.mark.parametrize(
    "condition",
    [
        "DELTA_TRUNCATED_TRANSACTION_LOG",
        "DELTA_UNSUPPORTED_TIME_TRAVEL_BEYOND_DELETED_FILE_RETENTION_DURATION",
    ],
)
def test_snapshot_failure_is_never_reclassified_as_cdf_expiry(condition):
    """A broken bootstrap snapshot must fail rather than manufacture incremental recovery."""
    spark = Mock()
    error = StructuredError(condition)
    spark.read.format.return_value.option.return_value.table.side_effect = error
    with pytest.raises(StructuredError) as failure:
        select_incremental_rows(spark, "db.source", None, 10, None, Mock())
    assert failure.value is error


def test_nested_structured_cause_is_recognized_without_message_matching():
    """Spark wrappers may carry the precise Delta condition only on their nested cause."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import (
        CdfHistoryExpired,
        normalize_cdf_error,
    )

    outer = StructuredError("FAILED_READ_FILE.DBR_FILE_NOT_EXIST")
    outer.__cause__ = StructuredError("DELTA_CHANGE_DATA_FILE_NOT_FOUND")
    with pytest.raises(CdfHistoryExpired), normalize_cdf_error():
        raise outer
    fake = RuntimeError("[DELTA_CHANGE_DATA_FILE_NOT_FOUND]")
    with pytest.raises(RuntimeError) as failure, normalize_cdf_error():
        raise fake
    assert failure.value is fake


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"source_start_version": None},
        {"source_end_version": 7},
        {"target_version": -1},
        {"source_start_version": True},
        {"model_version": "champion"},
        {"model_digest": "bad"},
        {"request_digest": "untrusted"},
        {"layout": "other"},
        {"source_table_id": "target-id"},
        {"source_table": "db.target"},
        {"model_name": "x" * 50000},
    ],
)
def test_recovery_request_rejects_unpinned_or_unbounded_input(changes):
    """Notebook task values cannot weaken pinned model and source identity checks."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import validate_recovery_request

    with pytest.raises(ValueError):
        validate_recovery_request(request_fields() | changes)


def test_request_error_copies_input_and_hashes_canonical_content():
    """The routed request must survive serialization without mutable caller state."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import (
        CdfRecoveryRequired,
        recovery_request_digest,
    )

    original = request_fields()
    error = CdfRecoveryRequired(original)
    original["source_end_version"] = 11
    assert error.request["source_end_version"] == 10
    assert "CDF" in str(error) and "db.source" in str(error)
    reordered = dict(reversed(list(error.request.items())))
    assert recovery_request_digest(reordered) == recovery_request_digest(error.request)
    assert recovery_request_digest(original) != recovery_request_digest(error.request)


@pytest.mark.parametrize("layout", ["single_model", "model_set"])
def test_recovery_state_accepts_only_pinned_base_or_exact_committed_request(layout):
    """A foreign publication cannot be overwritten or mistaken for this recovery retry."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import (
        check_recovery_state,
        recovery_receipt_fields,
    )

    request = request_fields() | {"layout": layout}
    prefix = "model_set" if layout == "model_set" else "model"
    previous = {
        "source_table_id": "source-id",
        "target_table_id": "target-id",
        "source_end_version": 7,
        f"{prefix}_name": "db.model",
        f"{prefix}_version": "2",
        f"{prefix}_digest": "a" * 64,
    }
    assert check_recovery_state(request, previous, 4) is False
    with pytest.raises(BatchConflictError):
        check_recovery_state(request, previous, 5)
    committed = (
        previous
        | recovery_receipt_fields(request)
        | {
            "source_end_version": 10,
            "expected_target_version": 4,
        }
    )
    assert check_recovery_state(request, committed, 5) is True
    for change in (
        {"source_table_id": "replaced"},
        {f"{prefix}_digest": "b" * 64},
        {"cdf_recovery_request_digest": "wrong"},
        {"source_end_version": 11},
    ):
        with pytest.raises(BatchConflictError):
            check_recovery_state(request, committed | change, 5)
    with pytest.raises(BatchConflictError):
        check_recovery_state(request, committed, 6)


@pytest.mark.parametrize("bad_previous", [None, {"source_end_version": 6}])
def test_recovery_state_requires_trusted_prior_watermark(bad_previous):
    """An absent or unrelated receipt cannot authorize an overwrite of existing predictions."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import check_recovery_state

    with pytest.raises(BatchConflictError):
        check_recovery_state(request_fields(), bad_previous, 4)


def test_new_selected_model_may_recover_after_older_model_receipt():
    """Incremental model selection may advance independently of a source-history outage."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import check_recovery_state

    previous = {
        "source_table_id": "source-id",
        "target_table_id": "target-id",
        "source_end_version": 7,
        "model_name": "db.model",
        "model_version": "1",
        "model_digest": "b" * 64,
    }
    assert check_recovery_state(request_fields(), previous, 4) is False


@pytest.mark.parametrize(
    "namespace", ["com.databricks.sql.transaction.tahoe", "org.apache.spark.sql.delta"]
)
def test_connect_read_wrapper_recovers_structured_java_history_cause(namespace):
    """Serverless transports the exact Delta condition in a Java cause header, not Python causes."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import (
        CdfHistoryExpired,
        normalize_cdf_error,
    )

    error = SparkConnectGrpcException(
        "FAILED_READ_FILE.DBR_FILE_NOT_EXIST",
        "org.apache.spark.SparkException\n\tat reader.scan(Reader.scala:4)\n"
        f"Caused by: {namespace}.DeltaFileNotFoundException: "
        "[DELTA_CHANGE_DATA_FILE_NOT_FOUND] Missing retained CDF file.\n"
        "\tat cdf.reader.scan(Reader.scala:8)",
    )
    with pytest.raises(CdfHistoryExpired) as failure, normalize_cdf_error():
        raise error
    assert failure.value.__cause__ is error


@pytest.mark.parametrize(
    "header",
    [
        "java.io.FileNotFoundException: [DELTA_CHANGE_DATA_FILE_NOT_FOUND] file",
        "com.databricks.sql.transaction.tahoe.DeltaFileNotFoundException: [DELTA_FILE_NOT_FOUND] file",
        "com.databricks.sql.transaction.tahoe.DeltaFileNotFoundException: input mentions [DELTA_CHANGE_DATA_FILE_NOT_FOUND]",
        "org.apache.spark.sql.delta.DeltaFileNotFoundException: [DELTA_CHANGE_DATA_FILE_NOT_FOUND_EXTRA] file",
    ],
)
def test_connect_generic_or_freeform_missing_file_causes_still_fail(header):
    """A file path or free-form message containing an error token cannot trigger replacement."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import normalize_cdf_error

    error = SparkConnectGrpcException("FAILED_READ_FILE.DBR_FILE_NOT_EXIST", "Caused by: " + header)
    with pytest.raises(SparkConnectGrpcException) as failure, normalize_cdf_error():
        raise error
    assert failure.value is error


def test_nonconnect_stacktrace_and_connect_permission_wrapper_are_not_recoverable():
    """A known cause token cannot override a different transport's error or permissions."""
    from skyulf.integrations.databricks.data.delta_io.cdf_recovery import normalize_cdf_error

    trace = "Caused by: org.apache.spark.sql.delta.DeltaFileNotFoundException: [DELTA_CHANGE_DATA_FILE_NOT_FOUND] missing"
    permission = SparkConnectGrpcException("INSUFFICIENT_PERMISSIONS", trace)

    class OtherStacktraceError(StructuredError):
        """Expose a similar method on an unrelated remote transport."""

        def getStackTrace(self):
            """Return the text without claiming the Spark Connect error contract."""
            return trace

    generic = OtherStacktraceError("FAILED_READ_FILE.DBR_FILE_NOT_EXIST")
    for error in (permission, generic):
        with pytest.raises(StructuredError) as failure, normalize_cdf_error():
            raise error
        assert failure.value is error
