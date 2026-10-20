"""Spark training reads record DECIMAL conversions; replay checks must accept that evidence."""

import pytest
from test_databricks_lifecycle_tasks import _call, staged  # noqa: F401 - pytest fixture

_DECIMAL_X = {"column": "x", "source_dtype": "decimal(10,2)", "target_dtype": "float64"}
_DECIMAL_KEY = {"column": "id", "source_dtype": "decimal(18,0)", "target_dtype": "float64"}


@pytest.mark.parametrize("conversions", [(), (_DECIMAL_X, _DECIMAL_KEY)])
def test_single_model_registers_with_observed_source_conversions(staged, conversions):  # noqa: F811
    """Every Spark read records conversions, even none; validate_model used to reject them all."""
    _, client, config, _, frame = staged
    frame["x"] = frame["x"].astype("float64")  # what the Spark read's DOUBLE cast produces
    frame.attrs["source_column_conversions"] = conversions
    prepared = _call(staged, "prepare", config=config, action="train", experiment_name="staged")
    trained = _call(staged, "train", prepared.reference)
    registered = _call(staged, "evaluate_register", trained.reference)
    assert registered.output["candidate_version"] == "1"
    version = client.get_model_version(config["model_name"], "1")
    assert version.run_id == prepared.reference["run_id"]


def test_tampered_conversion_evidence_still_blocks_registration(staged, monkeypatch):  # noqa: F811
    """Accepting observed conversions must not accept a fitted config that differs from them."""
    from skyulf.integrations.databricks.training.fitting import candidate

    _, _, config, _, frame = staged
    frame["x"] = frame["x"].astype("float64")
    frame.attrs["source_column_conversions"] = (_DECIMAL_X,)
    original = candidate._fit_data_config

    def forged(*args, **kwargs):
        """Seal a conversion the prepared data never observed."""
        result = original(*args, **kwargs)
        return {**result, "source_column_conversions": []}

    monkeypatch.setattr(candidate, "_fit_data_config", forged)
    prepared = _call(staged, "prepare", config=config, action="train", experiment_name="staged")
    trained = _call(staged, "train", prepared.reference)
    with pytest.raises(ValueError, match="differs from invocation"):
        _call(staged, "evaluate_register", trained.reference)


def test_competition_winner_registers_with_observed_source_conversions(staged):  # noqa: F811
    """Separate candidate tasks (the cloud path) must adopt a winner with sealed conversions."""
    from test_competition_lifecycle import _competition

    from skyulf.integrations.databricks.jobs.training.training_nodes import run_competition_training

    _, client, config, context, frame = staged
    frame["x"] = frame["x"].astype("float64")
    frame.attrs["source_column_conversions"] = (_DECIMAL_X,)
    _competition(config)
    prepared = _call(staged, "initialize", config=config, action="train", experiment_name="staged")
    loaded = _call(staged, "load_data", prepared.reference)
    split = _call(staged, "prepare_dataset", loaded.reference)
    for name in ("strong", "weak"):
        run_competition_training(
            None,
            name=name,
            context=context,
            tracking_uri=config["tracking_uri"],
            reference=split.reference,
        )
    selected = _call(staged, "select_best_model", split.reference)
    registered = _call(staged, "evaluate_register", selected.reference)
    assert registered.output["candidate_version"] == "1"
    assert (
        client.get_model_version(config["model_name"], "1").run_id == prepared.reference["run_id"]
    )
