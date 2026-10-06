"""Real Delta commits preserve complete model-set publication and retry boundaries."""

import os
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from tests.integration.platforms.test_model_set_batch import _saved_set
from tests.integration.platforms.test_model_set_scoring import temporal_set as temporal_set

from skyulf.inference.model_set_scoring import score_model_set
from skyulf.integrations.databricks.data.admission import BatchConflictError, SingleWriterAdmission
from skyulf.integrations.databricks.model_sets.model_set_batch import run_model_set_batch
from skyulf.integrations.databricks.scoring.incremental.local_incremental import (
    SourceChangeRequiresRebuild,
)
from skyulf.integrations.mlflow.registration.registry import ResolvedModel


@pytest.fixture
def tables(delta_spark):
    """Own unique disposable source/output tables without altering existing user data."""
    schema = os.environ.get("SKYULF_MODEL_SET_TEST_SCHEMA", "default")
    prefix = f"{schema}.sm36c_{uuid4().hex[:12]}"
    source, target = prefix + "_source", prefix + "_predictions"
    rows = [(3, "Vilnius", 7.0), (1, "Riga", 8.0), (2, "Tallinn", 9.0)]
    delta_spark.createDataFrame(rows, "id long, city string, amount double").write.format(
        "delta"
    ).option("delta.enableChangeDataFeed", "true").saveAsTable(source)
    try:
        yield delta_spark, source, target
    finally:
        delta_spark.sql(f"DROP TABLE IF EXISTS {target}").collect()
        delta_spark.sql(f"DROP TABLE IF EXISTS {source}").collect()


def _score(
    tables,
    model,
    artifact,
    mode="incremental_append",
    publication=None,
    source_change_policy="reject",
):
    """Exercise the public scorer using one explicitly serialized test writer."""
    spark, source, target = tables
    return run_model_set_batch(
        spark,
        model,
        artifact,
        source_table=source,
        prediction_table=target,
        admission=SingleWriterAdmission(),
        model_change_mode=mode,
        max_rows=100,
        max_bytes=4 * 1024 * 1024,
        publication=publication,
        source_change_policy=source_change_policy,
    )


def test_delta_combined_only_omits_intermediate_predictions(tmp_path, tables):
    """The physical Delta schema must contain only requested business values and provenance."""
    artifact, model, _ = _saved_set(tmp_path, _SOURCE, _rule(10))
    result = _score(tables, model, artifact, publication={"mode": "combined_only"})
    spark, _, target = tables
    frame = spark.table(target).orderBy("id").toPandas()
    assert "amount__prediction" not in frame.columns
    assert frame.adjusted_value.tolist() == pytest.approx([28, 51, 39])
    assert frame.model_set_digest.unique().tolist() == [artifact.manifest.set_sha256]
    assert result.output_count == 3
    assert _score(tables, model, artifact, publication={"mode": "combined_only"}).noop
    with pytest.raises(ValueError, match="schema"):
        _score(tables, model, artifact)


def test_delta_named_views_share_one_commit_and_preserve_results_on_rule_failure(tmp_path, tables):
    """Views must reflect the common commit and stay at its last successful results on failure."""
    artifact, model, _ = _saved_set(tmp_path / "good", _SOURCE, _rule(10))
    spark, _, target = tables
    model_view, combined_view = target + "_amount", target + "_business"
    policy = {
        "mode": "separate_views",
        "model_views": {"amount": model_view},
        "combined_view": combined_view,
    }
    try:
        result = _score(tables, model, artifact, "full_rebuild", publication=policy)
        assert spark.catalog.getTable(model_view).tableType.upper() == "VIEW"
        model_rows = spark.table(model_view).orderBy("id").toPandas()
        business_rows = spark.table(combined_view).orderBy("id").toPandas()
        assert "adjusted_value" not in model_rows.columns
        assert "amount__prediction" not in business_rows.columns
        np.testing.assert_allclose(business_rows.adjusted_value, model_rows.amount__prediction + 10)
        assert _score(tables, model, artifact, "full_rebuild", publication=policy).noop
        bad, bad_model, _ = _saved_set(
            tmp_path / "bad",
            "def adjust(inputs, predictions, params):\n    raise RuntimeError('rule failed')\n",
            _rule(20),
        )
        with pytest.raises(RuntimeError, match="rule failed"):
            _score(tables, bad_model, bad, "full_rebuild", publication=policy)
        assert spark.sql(f"DESCRIBE HISTORY {target}").first()["version"] == result.commit_version
        assert spark.table(model_view).count() == spark.table(combined_view).count() == 3
        assert spark.table(combined_view).select("model_set_digest").first()[0] == model.digest
    finally:
        spark.sql(f"DROP VIEW IF EXISTS {model_view}").collect()
        spark.sql(f"DROP VIEW IF EXISTS {combined_view}").collect()


def _rule(offset):
    """Keep an identical output schema while changing the recorded business rule."""
    return {
        "outputs": [
            {
                "name": "adjusted",
                "version": str(offset),
                "function": "adjust",
                "params": {"offset": offset},
                "columns": [{"name": "adjusted_value", "dtype": "float64"}],
                "required_components": ["amount"],
            }
        ]
    }


_SOURCE = (
    "import pandas as pd\n"
    "def adjust(inputs, predictions, params):\n"
    "    return pd.DataFrame({'adjusted_value': predictions['amount__prediction'] + params['offset']}, index=inputs.index)\n"
)


def test_delta_append_retry_and_complete_set_provenance(tmp_path, tables):
    """Actual CDF appends must add each key once with the selected complete set identity."""
    artifact, model, _ = _saved_set(tmp_path)
    first = _score(tables, model, artifact)
    second = _score(tables, model, artifact)
    spark, source, target = tables
    assert first.output_count == 3 and second.noop
    spark.createDataFrame([(4, "Riga", 10.0)], "id long, city string, amount double").write.format(
        "delta"
    ).mode("append").saveAsTable(source)
    third = _score(tables, model, artifact)
    rows = spark.table(target).orderBy("id").toPandas()
    assert third.input_count == third.output_count == 1
    assert rows.id.tolist() == [1, 2, 3, 4]
    assert rows.model_set_digest.unique().tolist() == [artifact.manifest.set_sha256]
    assert rows.model_set_version.unique().tolist() == [model.version]
    assert third.manifest["source_start_version"] == first.source_end_version + 1


def test_delta_full_rebuild_and_rollback_recompute_same_snapshot(tmp_path, tables):
    """Set changes and reversals must publish fresh complete generations even without new input."""
    first_set, first_model, _ = _saved_set(tmp_path / "first", _SOURCE, _rule(0))
    second_set, second_model, _ = _saved_set(tmp_path / "second", _SOURCE, _rule(100))
    _score(tables, first_model, first_set, "full_rebuild")
    spark, _, target = tables
    before = spark.table(target).orderBy("id").toPandas()
    changed = _score(tables, second_model, second_set, "full_rebuild")
    current = spark.table(target).orderBy("id").toPandas()
    np.testing.assert_allclose(current.adjusted_value, before.adjusted_value + 100)
    assert not changed.noop and changed.output_count == 3
    restored = _score(tables, first_model, first_set, "full_rebuild")
    after = spark.table(target).orderBy("id").toPandas()
    np.testing.assert_allclose(after.adjusted_value, before.adjusted_value)
    assert restored.commit_version > changed.commit_version
    assert _score(tables, first_model, first_set, "full_rebuild").noop


def test_delta_rule_failure_preserves_prior_output_and_receipt(tmp_path, tables):
    """A later rule failure must not overwrite any earlier committed prediction."""
    good, good_model, _ = _saved_set(tmp_path / "good", _SOURCE, _rule(0))
    bad, bad_model, _ = _saved_set(
        tmp_path / "bad",
        "def adjust(inputs, predictions, params):\n    raise RuntimeError('controlled rule failure')\n",
        _rule(1),
    )
    first = _score(tables, good_model, good, "full_rebuild")
    spark, _, target = tables
    with pytest.raises(RuntimeError, match="controlled rule failure"):
        _score(tables, bad_model, bad, "full_rebuild")
    assert spark.sql(f"DESCRIBE HISTORY {target}").first()["version"] == first.commit_version
    assert (
        spark.table(target).select("model_set_digest").distinct().first()[0]
        == good.manifest.set_sha256
    )
    assert spark.table(target).count() == 3


def test_delta_duplicate_insert_key_is_rejected_before_publication(tmp_path, tables):
    """A source append cannot silently duplicate an already scored record identity."""
    artifact, model, _ = _saved_set(tmp_path)
    first = _score(tables, model, artifact)
    spark, source, target = tables
    spark.createDataFrame([(1, "Riga", 10.0)], "id long, city string, amount double").write.format(
        "delta"
    ).mode("append").saveAsTable(source)
    with pytest.raises(BatchConflictError, match="already has"):
        _score(tables, model, artifact)
    assert spark.sql(f"DESCRIBE HISTORY {target}").first()["version"] == first.commit_version
    assert spark.table(target).count() == 3


def test_delta_empty_snapshot_rebuild_removes_prior_rows(tmp_path, tables):
    """Empty Spark snapshots must keep declared output types and clear stale generations."""
    first_set, first_model, _ = _saved_set(tmp_path / "first", _SOURCE, _rule(0))
    second_set, second_model, _ = _saved_set(tmp_path / "second", _SOURCE, _rule(1))
    first = _score(tables, first_model, first_set, "full_rebuild")
    spark, source, target = tables
    spark.sql(f"DELETE FROM {source}").collect()
    empty = _score(tables, second_model, second_set, "full_rebuild")
    assert not empty.noop and empty.output_count == 0
    assert empty.commit_version > first.commit_version
    assert spark.table(target).count() == 0
    assert _score(tables, second_model, second_set, "full_rebuild").noop


def test_delta_integer_keys_widen_without_changing_record_identity(tmp_path, tables):
    """Source INT keys must publish losslessly into the model-set BIGINT key contract."""
    artifact, model, _ = _saved_set(tmp_path)
    spark, source, target = tables
    spark.sql(f"DROP TABLE {source}").collect()
    spark.createDataFrame(
        [(3, "Vilnius", 7.0), (1, "Riga", 8.0)], "id int, city string, amount double"
    ).write.format("delta").option("delta.enableChangeDataFeed", "true").saveAsTable(source)
    result = _score(tables, model, artifact)
    assert result.output_count == 2
    assert [row.id for row in spark.table(target).orderBy("id").select("id").collect()] == [1, 3]


@pytest.mark.parametrize("output_mode", ["all", "combined_only", "separate_views"])
def test_delta_source_corrections_rebuild_same_set_and_resume_append(tmp_path, tables, output_mode):
    """Update/delete recovery must replace one generation in every output mode then resume CDF."""
    artifact, model, _ = _saved_set(tmp_path, _SOURCE, _rule(10))
    spark, source, target = tables
    policy = {"mode": output_mode}
    views = []
    if output_mode == "separate_views":
        views = [target + "_amount", target + "_business"]
        policy.update(model_views={"amount": views[0]}, combined_view=views[1])
    try:
        first = _score(tables, model, artifact, publication=policy)
        spark.sql(f"UPDATE {source} SET amount = 100.0 WHERE id = 3").collect()
        spark.sql(f"DELETE FROM {source} WHERE id = 2").collect()
        spark.sql(f"INSERT INTO {source} VALUES (4, 'Riga', 10.0)").collect()
        with pytest.raises(SourceChangeRequiresRebuild):
            _score(tables, model, artifact, publication=policy)
        assert spark.sql(f"DESCRIBE HISTORY {target}").first()["version"] == first.commit_version
        rebuilt = _score(
            tables, model, artifact, publication=policy, source_change_policy="rebuild_on_change"
        )
        rows = spark.table(target).orderBy("id").toPandas()
        assert rows.id.tolist() == [1, 3, 4]
        assert rows.adjusted_value.tolist() == pytest.approx([28, 132, 30])
        assert rebuilt.manifest["source_rebuilt"] is True
        assert rebuilt.manifest["write_mode"] == "overwrite"
        assert rebuilt.commit_version == first.commit_version + 1
        assert _score(
            tables, model, artifact, publication=policy, source_change_policy="rebuild_on_change"
        ).noop
        spark.sql(f"INSERT INTO {source} VALUES (5, 'Riga', 11.0)").collect()
        appended = _score(
            tables, model, artifact, publication=policy, source_change_policy="rebuild_on_change"
        )
        assert appended.input_count == appended.output_count == 1
        assert appended.manifest["source_rebuilt"] is False
        assert spark.table(target).count() == 4
        for view in views:
            assert spark.table(view).count() == 4
        spark.sql(f"DELETE FROM {source}").collect()
        empty = _score(
            tables, model, artifact, publication=policy, source_change_policy="rebuild_on_change"
        )
        assert not empty.noop and empty.output_count == spark.table(target).count() == 0
        for view in views:
            assert spark.table(view).count() == 0
        assert _score(
            tables, model, artifact, publication=policy, source_change_policy="rebuild_on_change"
        ).noop
    finally:
        for view in reversed(views):
            spark.sql(f"DROP VIEW IF EXISTS {view}").collect()


def test_delta_failed_correction_preserves_output_and_can_retry(tmp_path, tables):
    """A corrected row that breaks a business rule must not erase the last valid generation."""
    source_code = _SOURCE.replace(
        "    return pd.DataFrame",
        "    if (inputs['amount'] < 0).any():\n        raise ValueError('invalid business input')\n"
        "    return pd.DataFrame",
    )
    artifact, model, _ = _saved_set(tmp_path, source_code, _rule(10))
    first = _score(tables, model, artifact)
    spark, source, target = tables
    before = spark.table(target).orderBy("id").toPandas()
    spark.sql(f"UPDATE {source} SET amount = -1 WHERE id = 3").collect()
    with pytest.raises(ValueError, match="invalid business input"):
        _score(tables, model, artifact, source_change_policy="rebuild_on_change")
    pd.testing.assert_frame_equal(before, spark.table(target).orderBy("id").toPandas())
    assert spark.sql(f"DESCRIBE HISTORY {target}").first()["version"] == first.commit_version
    spark.sql(f"UPDATE {source} SET amount = 100 WHERE id = 3").collect()
    fixed = _score(tables, model, artifact, source_change_policy="rebuild_on_change")
    assert fixed.manifest["source_rebuilt"] is True
    assert spark.table(target).where("id = 3").first()["adjusted_value"] == pytest.approx(132)


def test_delta_temporal_correction_resets_both_engines_then_continues(temporal_set, tables):
    """Persisted rolling history must be rebuilt from corrections before accepting new rows."""
    spark, source, target = tables
    spark.sql(f"DROP TABLE {source}").collect()
    spark.createDataFrame(
        [(0, 0, 0.0), (1, 1, 1.0), (2, 2, 2.0)], "id long, t long, v double"
    ).write.format("delta").option("delta.enableChangeDataFeed", "true").saveAsTable(source)
    model = ResolvedModel(
        "temporal", "1", "models:/temporal/1", None, temporal_set.manifest.set_sha256
    )
    _score(tables, model, temporal_set)
    spark.sql(f"UPDATE {source} SET v = 12.0 WHERE id = 0").collect()
    rebuilt = _score(tables, model, temporal_set, source_change_policy="rebuild_on_change")
    spark.sql(f"INSERT INTO {source} VALUES (3, 3, 3.0)").collect()
    appended = _score(tables, model, temporal_set, source_change_policy="rebuild_on_change")
    expected = score_model_set(
        spark.table(source).orderBy("id").toPandas(), temporal_set, bootstrap_history=True
    )
    actual = spark.table(target).orderBy("id").toPandas()
    for branch in ("a", "b"):
        np.testing.assert_allclose(
            actual[branch + "__prediction"].astype(float),
            expected.frame[branch + "__prediction"].astype(float),
        )
    assert appended.manifest["set_history"] == expected.history
    assert rebuilt.manifest["source_rebuilt"] is True
    assert appended.input_count == 1
