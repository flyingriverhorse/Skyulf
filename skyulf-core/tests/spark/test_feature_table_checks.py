"""Exercise native feature table admission against real distributed Spark rows."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.feature_store.config import (
    FeatureLookupSpec,
    FeatureTrainingSpec,
)
from skyulf.integrations.databricks.feature_store.table_checks import validate_feature_tables


def _lookup():
    """Use distinct source and feature key names in their declared primary-key order."""
    return FeatureTrainingSpec(
        lookups=(
            FeatureLookupSpec(
                table_name="workspace.features.history",
                lookup_key=("entity",),
                feature_names=("amount",),
                timestamp_lookup_key="observed_at",
            ),
        ),
        label="target",
    )


def _validate(
    spark, rows, *, primary=("company", "feature_time"), times=("feature_time",), source_type="long"
):
    """Keep reads pinned while substituting only the workspace catalog and SDK metadata."""
    features = spark.createDataFrame(rows, "company long, feature_time timestamp, amount double")
    source = spark.createDataFrame(
        [(1, datetime(2026, 1, 2), 0.0)],
        f"entity {source_type}, observed_at timestamp, target double",
    )
    proxy = Mock()
    proxy.read.format.return_value.option.return_value.table.return_value = features
    client = Mock()
    client.get_table.return_value = SimpleNamespace(primary_keys=primary, timestamp_keys=times)
    validate_feature_tables(
        proxy,
        source,
        _lookup(),
        client,
        [
            {
                "table_name": "workspace.features.history",
                "table_id": "known",
                "version": 7,
            }
        ],
    )
    proxy.read.format.return_value.option.assert_called_once_with("versionAsOf", 7)
    return features


def test_native_feature_keys_accept_multiple_historical_rows(spark):
    """Uniqueness is entity plus time, so valid history must remain available to as-of lookup."""
    rows = [(1, datetime(2026, 1, 1), 2.0), (1, datetime(2026, 1, 3), 4.0)]
    assert _validate(spark, rows).count() == 2


@pytest.mark.parametrize("bad", ["duplicate", "null_entity", "null_time"])
def test_native_feature_keys_reject_ambiguous_or_null_rows(spark, bad):
    """UC primary keys are not enforced, so actual violations must fail before SDK joins."""
    rows = [(1, datetime(2026, 1, 1), 2.0)]
    if bad == "duplicate":
        rows *= 2
    elif bad == "null_entity":
        rows = [(None, datetime(2026, 1, 1), 2.0)]
    else:
        rows = [(1, None, 2.0)]
    with pytest.raises(ValueError, match="primary keys"):
        _validate(spark, rows)


def test_native_feature_timestamp_requires_catalog_timeseries_metadata(spark):
    """A timestamp column alone must not imply an as-of feature table."""
    with pytest.raises(ValueError, match="TIMESERIES"):
        _validate(spark, [(1, datetime(2026, 1, 1), 2.0)], primary=("company",), times=())


def test_native_feature_key_types_do_not_implicitly_cast(spark):
    """A changed lookup key representation must fail rather than silently lose matches."""
    with pytest.raises(ValueError, match="key type"):
        _validate(spark, [(1, datetime(2026, 1, 1), 2.0)], source_type="int")
