"""The mounted defaults route must distinguish missing searches from fixed controls."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.ml_pipeline.api import router


@pytest.fixture
def client():
    """Exercise production pipeline routes without starting unrelated application services."""
    app = FastAPI()
    app.include_router(router, prefix="/pipeline")
    with TestClient(app) as session:
        yield session


@pytest.mark.parametrize("strategy", ["random", "grid", "optuna", "halving_grid", "halving_random"])
@pytest.mark.parametrize(
    ("model_key", "expected_params"),
    [
        ("kmeans", {"n_clusters", "n_init"}),
        ("minibatch_kmeans", {"n_clusters", "n_init", "batch_size"}),
        ("gaussian_mixture", {"n_components", "covariance_type"}),
        ("birch", {"n_clusters", "threshold", "branching_factor"}),
    ],
)
def test_clustering_defaults_populate_tunable_controls(
    client, model_key, expected_params, strategy
):
    """Every supported strategy must supply candidate lists for the advertised clustering fields."""
    response = client.get(
        f"/pipeline/hyperparameters/{model_key}/defaults", params={"strategy": strategy}
    )
    assert response.status_code == 200
    space = response.json()
    assert set(space) == expected_params
    assert all(isinstance(values, list) and values for values in space.values())


def test_voting_regressor_empty_defaults_have_no_top_level_tuning_controls(client):
    """The UI must receive explicit fixed-only metadata for an intentionally empty search."""
    defaults = client.get("/pipeline/hyperparameters/voting_regressor/defaults")
    definitions = client.get("/pipeline/hyperparameters/voting_regressor")
    assert defaults.status_code == definitions.status_code == 200
    assert defaults.json() == {}
    fields = definitions.json()
    assert {field["name"] for field in fields} == {"base_estimators", "n_jobs"}
    assert all(field["tunable"] is False for field in fields)
