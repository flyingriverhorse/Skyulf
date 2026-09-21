"""Reject invalid numeric node payloads before jobs are queued."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.ml_pipeline._internal._schemas import PipelineConfigModel


@pytest.fixture
def client():
    """Exercise the actual pipeline request model through FastAPI parsing."""
    app = FastAPI()

    @app.post("/run")
    def run(config: PipelineConfigModel):
        """Return a marker only after request validation succeeds."""
        return {"accepted": config.pipeline_id}

    return TestClient(app)


@pytest.mark.parametrize("value", ["", None, "1e400", "NaN", "Infinity", -1])
@pytest.mark.parametrize(
    "step,params,field",
    [
        ("TrainTestSplitter", {}, "test_size"),
        ("IQR", {}, "multiplier"),
        ("feature_selection", {"method": "variance_threshold"}, "threshold"),
        ("training", {"run_mode": "fixed", "cv_enabled": True}, "cv_folds"),
    ],
)
def test_invalid_node_numbers_return_422(client, value, step, params, field):
    """Direct clients receive a field-specific error instead of a queued broken job."""
    response = client.post(
        "/run",
        json={
            "pipeline_id": "p",
            "nodes": [
                {"node_id": "n", "step_type": step, "params": {**params, field: value}},
            ],
        },
    )
    assert response.status_code == 422
    assert field in response.text


@pytest.mark.parametrize("field,value", [("n_trials", 0), ("n_trials", 1.5), ("cv_folds", 1)])
def test_invalid_tuning_numbers_return_422(client, field, value):
    """Advanced budgets and folds cannot bypass validation through nested payloads."""
    response = client.post(
        "/run",
        json={
            "pipeline_id": "p",
            "nodes": [
                {
                    "node_id": "n",
                    "step_type": "training",
                    "params": {
                        "run_mode": "tuned",
                        "tuning_config": {"search_space": {"max_depth": [None, 2]}, field: value},
                    },
                },
            ],
        },
    )
    assert response.status_code == 422
    assert field in response.text


def test_supported_sentinels_and_server_override_remain_valid(client):
    """Seeds, optional candidates and server-controlled workers are not false failures."""
    response = client.post(
        "/run",
        json={
            "pipeline_id": "p",
            "nodes": [
                {
                    "node_id": "split",
                    "step_type": "TrainTestSplitter",
                    "params": {"random_state": 0},
                },
                {
                    "node_id": "select",
                    "step_type": "feature_selection",
                    "params": {"method": "select_from_model", "threshold": "1.25*mean"},
                },
                {
                    "node_id": "tune",
                    "step_type": "training",
                    "params": {
                        "run_mode": "tuned",
                        "tuning_config": {
                            "n_trials": 1,
                            "random_state": 0,
                            "n_jobs": 0,
                            "search_space": {"max_depth": [None, 2]},
                        },
                    },
                },
            ],
        },
    )
    assert response.status_code == 200


@pytest.mark.parametrize("literal", ["1e400", "NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_number_returns_serializable_422(client, literal):
    """Overflowing JSON numbers must not make validation-error serialization fail."""
    payload = (
        '{"pipeline_id":"p","nodes":[{"node_id":"n","step_type":"IQR","params":{"multiplier":'
        + literal
        + "}}]}"
    )
    response = client.post("/run", content=payload, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "multiplier" in response.text


@pytest.mark.parametrize(
    "params",
    [
        {"method": "select_from_model", "threshold": "0.5", "max_features": None},
        {"method": "select_from_model", "threshold": None},
        {"method": "select_k_best", "k": "all"},
        {"method": "rfe", "k": None, "step": 0.5},
    ],
)
def test_core_supported_selector_options_are_not_rejected(client, params):
    """The API guard must preserve existing Core selector sentinels and fractions."""
    response = client.post(
        "/run",
        json={
            "pipeline_id": "p",
            "nodes": [
                {"node_id": "n", "step_type": "feature_selection", "params": params},
            ],
        },
    )
    assert response.status_code == 200


def test_tuned_model_type_alias_still_checks_stacking_cv(client):
    """The model_type spelling accepted by execution must not bypass CV validation."""
    response = client.post(
        "/run",
        json={
            "pipeline_id": "p",
            "nodes": [
                {
                    "node_id": "n",
                    "step_type": "training",
                    "params": {
                        "run_mode": "tuned",
                        "model_type": "stacking_classifier",
                        "tuning_config": {"cv": 1},
                    },
                },
            ],
        },
    )
    assert response.status_code == 422


def test_oversized_integer_returns_422(client):
    """Oversized JSON integers must not raise an uncaught float-conversion overflow."""
    response = client.post(
        "/run",
        json={
            "pipeline_id": "p",
            "nodes": [
                {"node_id": "n", "step_type": "IQR", "params": {"multiplier": 10**400}},
            ],
        },
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "params",
    [
        {"method": "variance", "threshold": -1},
        {"method": "generic_univariate_select", "mode": "percentile", "percentile": 150},
        {"method": "generic_univariate_select", "mode": "k_best", "k": 0},
        {"method": "generic_univariate_select", "mode": "fpr", "alpha": 2},
        {"method": "rfe", "n_features_to_select": -1, "k": 2},
    ],
)
def test_consumed_selector_alias_fields_are_validated(client, params):
    """Validation follows the same aliases and precedence as the Core selector."""
    response = client.post(
        "/run",
        json={
            "pipeline_id": "p",
            "nodes": [
                {"node_id": "n", "step_type": "feature_selection", "params": params},
            ],
        },
    )
    assert response.status_code == 422


def test_inactive_cv_and_optional_fixed_workers_are_preserved(client):
    """Unused CV settings and sklearn's default worker sentinel must remain valid."""
    response = client.post(
        "/run",
        json={
            "pipeline_id": "p",
            "nodes": [
                {
                    "node_id": "n",
                    "step_type": "training",
                    "params": {
                        "run_mode": "fixed",
                        "cv_enabled": False,
                        "cv_folds": 1,
                        "hyperparameters": {"n_jobs": None},
                    },
                },
            ],
        },
    )
    assert response.status_code == 200


def test_actual_run_route_rejects_invalid_config_before_job_creation(monkeypatch):
    """The production submission route must reject malformed numbers before queuing work."""
    import importlib
    from unittest.mock import AsyncMock

    from backend.database.engine import get_async_session

    routes = importlib.import_module("backend.ml_pipeline._internal._routers.run_pipeline")
    create_job = AsyncMock()
    monkeypatch.setattr(routes.JobManager, "create_job", create_job)
    monkeypatch.setattr(routes.limiter, "enabled", False)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[get_async_session] = lambda: None
    with TestClient(app) as http:
        response = http.post(
            "/run",
            json={
                "pipeline_id": "p",
                "nodes": [
                    {
                        "node_id": "n",
                        "step_type": "training",
                        "params": {
                            "run_mode": "tuned",
                            "algorithm": "random_forest_classifier",
                            "tuning_config": {"n_trials": 0},
                        },
                    },
                ],
            },
        )
    assert response.status_code == 422
    assert "n_trials" in response.text
    create_job.assert_not_called()


def test_regression_ignores_retained_classification_calibration_options(client):
    """Classifier-only inactive settings must not block switching an ensemble to regression."""
    response = client.post(
        "/run",
        json={
            "pipeline_id": "p",
            "nodes": [
                {
                    "node_id": "n",
                    "step_type": "training",
                    "params": {
                        "model_type": "voting_regressor",
                        "hyperparameters": {"calibrate_base_models": True, "calibration_cv": None},
                    },
                },
            ],
        },
    )
    assert response.status_code == 200


def test_oversized_search_candidate_is_not_a_valid_finite_value(client):
    """Numeric search candidates obey the same finite range as numeric controls."""
    response = client.post(
        "/run",
        json={
            "pipeline_id": "p",
            "nodes": [
                {
                    "node_id": "n",
                    "step_type": "training",
                    "params": {
                        "run_mode": "tuned",
                        "tuning_config": {"search_space": {"alpha": [10**400]}},
                    },
                },
            ],
        },
    )
    assert response.status_code == 422
