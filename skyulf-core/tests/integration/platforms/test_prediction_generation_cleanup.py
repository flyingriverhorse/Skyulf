"""Old full-rebuild prediction generations are dropped automatically, keeping recent ones."""

import pytest

from skyulf.integrations.databricks.scoring.shared import prediction_output


class _Rows:
    """A SQL result exposing collect()."""

    def __init__(self, rows):
        """Keep the rows."""
        self.rows = rows

    def collect(self):
        """Return the rows."""
        return self.rows


class _Spark:
    """List generation tables with their properties and record drops."""

    def __init__(self, tables):
        """``tables`` maps table name to its properties."""
        self.tables, self.dropped = dict(tables), []

    def sql(self, statement):
        """Answer SHOW TABLES, SHOW TBLPROPERTIES and DROP TABLE."""
        text = statement.replace("`", "")
        if text.startswith("SHOW TABLES IN"):
            return _Rows([{"tableName": name.split(".")[-1]} for name in self.tables])
        if text.startswith("SHOW TBLPROPERTIES"):
            name = text.split()[2]
            return _Rows([{"key": k, "value": v} for k, v in self.tables[name].items()])
        if text.startswith("DROP TABLE IF EXISTS"):
            self.dropped.append(text.split()[-1])
            return _Rows([])
        raise AssertionError(statement)


OWNED = {"skyulf.mode": "full_rebuild", "skyulf.model_name": "c.ml.model"}


def _tables(*versions, props=OWNED):
    """Generation tables of c.out.preds for the given versions."""
    return {f"c.out.preds_v{v}": dict(props) for v in versions}


def test_keeps_the_active_and_previous_generation_by_default():
    """Each rescore_all keeps a full copy; only the newest two are needed for rollback."""
    spark = _Spark(_tables(1, 2, 3, 4))
    dropped = prediction_output.prune_prediction_generations(
        spark, "c.out.preds", "c.out.preds_v4", model_name="c.ml.model"
    )
    assert sorted(dropped) == ["c.out.preds_v1", "c.out.preds_v2"]


def test_the_active_generation_is_never_dropped_even_after_a_rollback():
    """After rolling back to v1, the active v1 stays even though it is the oldest."""
    spark = _Spark(_tables(1, 2, 3, 4))
    dropped = prediction_output.prune_prediction_generations(
        spark, "c.out.preds", "c.out.preds_v1", model_name="c.ml.model", keep=2
    )
    assert "c.out.preds_v1" not in dropped
    assert sorted(dropped) == ["c.out.preds_v2"]


def test_null_keeps_every_generation():
    """keep_prediction_generations: null restores the previous keep-all behavior."""
    spark = _Spark(_tables(1, 2, 3))
    assert (
        prediction_output.prune_prediction_generations(
            spark, "c.out.preds", "c.out.preds_v3", model_name="c.ml.model", keep=None
        )
        == []
    )


@pytest.mark.parametrize(
    "props",
    [{"skyulf.mode": "full_rebuild", "skyulf.model_name": "c.ml.other"}, {"owner": "someone"}],
)
def test_tables_of_other_models_or_owners_are_left_alone(props):
    """A same-named table that another model or person created is never dropped."""
    spark = _Spark(_tables(1, props=props) | _tables(2, 3))
    dropped = prediction_output.prune_prediction_generations(
        spark, "c.out.preds", "c.out.preds_v3", model_name="c.ml.model"
    )
    assert dropped == []


@pytest.mark.parametrize("keep", [0, -1, "2", True])
def test_keep_must_be_a_positive_integer(keep):
    """A typo must not drop every generation."""
    with pytest.raises(ValueError, match="keep_prediction_generations"):
        prediction_output.prune_prediction_generations(
            _Spark({}), "c.out.preds", "c.out.preds_v1", model_name="c.ml.model", keep=keep
        )
