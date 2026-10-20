"""Serving grants can differ per Bundle target; each target applies only its own principals."""

import pytest
import yaml
from test_serving_endpoint_job import endpoint_project  # noqa: F401 - pytest fixture

QUERY = "CAN_QUERY"
PER_TARGET = {
    "dev": [{"group_name": "test-callers", "level": QUERY}, {"user_name": "alice@example.com"}],
    "prod": [{"group_name": "prod-callers", "level": QUERY}],
}


def _settings(endpoint_project, permissions, target="dev"):  # noqa: F811
    """Read the deployed YAML exactly as the serving job does for one target."""
    from skyulf.integrations.databricks.jobs.serving_endpoint_job import (
        read_serving_endpoint_settings,
    )

    document, path, values = endpoint_project
    document["serving_endpoint"]["permissions"] = permissions
    path.write_text(yaml.safe_dump(document))
    return read_serving_endpoint_settings({**values, "bundle_target": target})


def test_mapping_applies_only_the_current_targets_principals(endpoint_project):  # noqa: F811
    """A prod deployment must never try to grant the test group, which may not exist there."""
    assert _settings(endpoint_project, PER_TARGET, "dev")["permissions"] == [
        {"group_name": "test-callers", "level": QUERY},
        {"user_name": "alice@example.com", "level": QUERY},
    ]
    assert _settings(endpoint_project, PER_TARGET, "prod")["permissions"] == [
        {"group_name": "prod-callers", "level": QUERY}
    ]


def test_unlisted_target_grants_nothing(endpoint_project):  # noqa: F811
    """Personal development endpoints stay private unless their target is listed."""
    assert _settings(endpoint_project, PER_TARGET, "dev_alice")["permissions"] == []


def test_flat_list_still_applies_to_every_target(endpoint_project):  # noqa: F811
    """Single-environment projects keep the existing list form."""
    principals = [{"group_name": "data_products"}]
    for target in ("dev", "prod"):
        assert _settings(endpoint_project, principals, target)["permissions"] == [
            {"group_name": "data_products", "level": QUERY}
        ]


@pytest.mark.parametrize(
    "permissions",
    [
        {"dev": [], "prod": [{"group_name": "prod-callers", "level": "CAN_MANAGE"}]},
        {"dev": {"group_name": "test-callers"}},
        {"bad target!": []},
        {"": []},
    ],
)
def test_every_targets_entries_are_checked(endpoint_project, permissions):  # noqa: F811
    """A malformed entry for another target fails now, not only when that target deploys."""
    with pytest.raises(ValueError, match="permissions"):
        _settings(endpoint_project, permissions, "dev")


def _smoke_project(tmp_path, endpoint_project, permissions):  # noqa: F811
    """Write a minimal project with real target names and the serving document."""
    document, _, values = endpoint_project
    document["serving_endpoint"]["permissions"] = permissions
    (tmp_path / "config").mkdir()
    (tmp_path / "deployment").mkdir()
    (tmp_path / "config/serving.yml").write_text(yaml.safe_dump(document))
    targets = {"targets": {name: {"mode": "production"} for name in ("test", "syst", "prod")}}
    (tmp_path / "deployment/targets.yml").write_text(yaml.safe_dump(targets))
    return {key: value for key, value in values.items() if key != "serving_config_path"}


def test_smoke_check_rejects_a_mistyped_target(tmp_path, endpoint_project):  # noqa: F811
    """A typo such as porod would silently grant nothing in prod; the offline check stops it."""
    from skyulf.integrations.databricks.projects.project_checks import _check_serving

    bindings = _smoke_project(tmp_path, endpoint_project, {"test": [], "porod": []})
    with pytest.raises(ValueError, match="porod"):
        _check_serving(tmp_path, bindings)


def test_smoke_check_accepts_known_targets(tmp_path, endpoint_project):  # noqa: F811
    """Listing only some of the project's targets is valid."""
    from skyulf.integrations.databricks.projects.project_checks import _check_serving

    bindings = _smoke_project(tmp_path, endpoint_project, {"test": [], "prod": []})
    _check_serving(tmp_path, bindings)
