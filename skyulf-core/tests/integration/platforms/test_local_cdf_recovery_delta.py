"""Verify recovery table identity, history and views with the optional real Delta lane."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from tests.integration.platforms.test_local_delta_publish import (
    local_delta_case,  # noqa: F401 - shared fixture
)

from skyulf.integrations.databricks import local_incremental as batch
from skyulf.integrations.databricks.cdf_recovery import CdfHistoryExpired, CdfRecoveryRequired
from skyulf.integrations.databricks.delta import table_identity
from skyulf.integrations.databricks.local_sdk import InputSource


@pytest.mark.parametrize("empty", [False, True])
def test_real_delta_recovery_preserves_identity_view_and_retained_history(
    local_delta_case,
    monkeypatch,
    empty,
):
    """A full replacement must remain the same Delta table and expose only the pinned snapshot."""
    spark, source, target, prepared, _, _, admission = local_delta_case
    spark.sql(f"DELETE FROM {target}")
    spark.sql(f"ALTER TABLE {source} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
    config = prepared.config.model_copy(
        update={
            "source": InputSource(
                kind="uc_table",
                table=source,
                read_mode="incremental",
                max_rows=10,
                max_bytes=10000,
            )
        }
    )
    prepared = replace(prepared, config=config)

    def run(request=None):
        """Exercise the public entry against the same admitted physical target."""
        return batch.run_incremental_local_batch(
            spark,
            prepared,
            record_key_columns=("id",),
            period_column="event_time",
            admission=admission,
            recovery_request=request,
        )

    first = run()
    original_id = table_identity(spark, target)
    view = target + "_recovery_view"
    spark.sql(f"CREATE VIEW {view} AS SELECT * FROM {target}")
    try:
        if empty:
            spark.sql(f"DELETE FROM {source}")
        else:
            spark.createDataFrame(
                [(3, datetime(2026, 1, 8, tzinfo=UTC), 6.0)],
                "id long, event_time timestamp, x double",
            ).write.format("delta").mode("append").saveAsTable(source)
        original_select = batch.select_incremental_rows

        def expired_read(spark, source, prior, upper, period, functions):
            """Simulate only removed CDF history; retain all real snapshot and write operations."""
            if prior is not None:
                raise CdfHistoryExpired("retained CDF interval expired")
            return original_select(spark, source, prior, upper, period, functions)

        with monkeypatch.context() as patch:
            patch.setattr(batch, "select_incremental_rows", expired_read)
            with pytest.raises(CdfRecoveryRequired) as failure:
                run()
        request = failure.value.request
        spark.createDataFrame(
            [(4, datetime(2026, 1, 9, tzinfo=UTC), 8.0)],
            "id long, event_time timestamp, x double",
        ).write.format("delta").mode("append").saveAsTable(source)
        rebuilt = run(request)
        assert not rebuilt.noop and rebuilt.output_count == (0 if empty else 3)
        assert table_identity(spark, target) == original_id
        assert spark.table(view).count() == rebuilt.output_count
        old = spark.read.format("delta").option("versionAsOf", first.commit_version).table(target)
        assert old.count() == 2
        replay = run(request)
        assert replay.noop and replay.commit_version == rebuilt.commit_version
        continued = run()
        assert continued.input_count == 1
        assert sorted(row.id for row in spark.table(view).collect()) == (
            [4] if empty else [1, 2, 3, 4]
        )
    finally:
        spark.sql(f"DROP VIEW IF EXISTS {view}")
