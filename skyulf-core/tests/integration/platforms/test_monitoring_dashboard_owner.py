"""A project can explicitly own the shared dashboard without unsafe resource removal."""

import runpy
from pathlib import Path

import pytest
import yaml

TEMPLATE = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"
WAREHOUSE_ID = "d047a4d9aa276958"
DASHBOARD_ID = "01f1c090238e1b6da5d633032ad9960b"


@pytest.fixture
def project(tmp_path):
    """Create the shipped directory layout without invoking cloud services."""
    (tmp_path / "resources").mkdir()
    asset = tmp_path / "src/monitoring/monitoring.lvdash.json"
    asset.parent.mkdir(parents=True)
    asset.write_text('{"datasets": [], "pages": []}', encoding="utf-8")
    return tmp_path


def _configure(project, **kwargs):
    """Execute the actual dependency-free configuration helper."""
    configure = runpy.run_path(str(TEMPLATE / "src/tools/configure_monitoring_dashboard.py"))[
        "configure"
    ]
    return configure(project, **kwargs)


def test_owner_configuration_binds_refresh_and_shared_store(project):
    """An explicit owner must publish one dashboard and refresh that same resource."""
    path = _configure(project, create_dashboard=True, warehouse_id=WAREHOUSE_ID, targets=("123",))
    owner = yaml.safe_load((project / "resources/monitoring.dashboard.yml").read_text())
    assert set(owner["targets"]) == {"123"}
    target = owner["targets"]["123"]
    dashboard = target["resources"]["dashboards"]["monitoring_dashboard"]
    assert dashboard["file_path"] == "../src/monitoring/monitoring.lvdash.json"
    assert dashboard["warehouse_id"] == WAREHOUSE_ID
    assert dashboard["dataset_catalog"] == "${var.monitoring_catalog}"
    assert dashboard["dataset_schema"] == "${var.monitoring_schema}"
    assert dashboard["embed_credentials"] is False
    assert target["variables"]["monitoring_dashboard_url"] == (
        "${workspace.host}/dashboardsv3/${resources.dashboards.monitoring_dashboard.id}/published"
    )
    tasks = yaml.safe_load(path.read_text())["targets"]["123"]["resources"]["jobs"]["monitoring"][
        "tasks"
    ]
    assert tasks[1]["dashboard_task"] == {
        "dashboard_id": "${resources.dashboards.monitoring_dashboard.id}",
        "warehouse_id": WAREHOUSE_ID,
    }


def test_owner_configuration_is_stable_and_disable_preserves_dashboard(project):
    """Refresh can be disabled without scheduling deletion of the managed dashboard."""
    path = _configure(project, create_dashboard=True, warehouse_id=WAREHOUSE_ID)
    owner_path = project / "resources/monitoring.dashboard.yml"
    before = (owner_path.read_bytes(), path.read_bytes())
    _configure(project, create_dashboard=True, warehouse_id=WAREHOUSE_ID)
    assert (owner_path.read_bytes(), path.read_bytes()) == before
    assert _configure(project) == path
    assert owner_path.read_bytes() == before[0]
    assert not path.exists()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"warehouse_id": "invalid"},
        {"warehouse_id": WAREHOUSE_ID, "dashboard_id": DASHBOARD_ID},
        {"warehouse_id": WAREHOUSE_ID, "dashboard_url": "https://example.com"},
        {"warehouse_id": WAREHOUSE_ID, "targets": ("test", "prod")},
        {"warehouse_id": WAREHOUSE_ID, "targets": ()},
        {"warehouse_id": WAREHOUSE_ID, "targets": ("bad:target",)},
    ],
)
def test_invalid_owner_options_preserve_existing_refresh(project, kwargs):
    """Invalid ownership input must leave the existing shared-dashboard link untouched."""
    refresh = _configure(project, dashboard_id=DASHBOARD_ID)
    before = refresh.read_bytes()
    with pytest.raises(ValueError):
        _configure(project, create_dashboard=True, **kwargs)
    assert refresh.read_bytes() == before
    assert not (project / "resources/monitoring.dashboard.yml").exists()


def test_missing_dashboard_asset_fails_before_writing_either_overlay(project):
    """A partial template must not replace a working refresh with an undeployable owner."""
    refresh = _configure(project, dashboard_id=DASHBOARD_ID)
    before = refresh.read_bytes()
    (project / "src/monitoring/monitoring.lvdash.json").unlink()
    with pytest.raises(ValueError, match="dashboard asset"):
        _configure(project, create_dashboard=True, warehouse_id=WAREHOUSE_ID)
    assert refresh.read_bytes() == before
    assert not (project / "resources/monitoring.dashboard.yml").exists()


@pytest.mark.parametrize("unowned", ["monitoring.dashboard.yml", "monitoring_dashboard.job.yml"])
def test_owner_preflights_both_files_before_writing(project, unowned):
    """Neither generated file may be written before detecting an operator-owned collision."""
    path = project / "resources" / unowned
    path.write_text("# operator managed\n", encoding="utf-8")
    before = {item.name: item.read_bytes() for item in path.parent.iterdir()}
    with pytest.raises(ValueError, match="not generated"):
        _configure(project, create_dashboard=True, warehouse_id=WAREHOUSE_ID)
    assert {item.name: item.read_bytes() for item in path.parent.iterdir()} == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dashboard_id": DASHBOARD_ID},
        {"create_dashboard": True, "warehouse_id": WAREHOUSE_ID, "targets": ("prod",)},
    ],
)
def test_owner_cannot_silently_switch_to_external_link_or_another_target(project, kwargs):
    """Changing the owner identity must not implicitly orphan or delete the shared dashboard."""
    _configure(project, create_dashboard=True, warehouse_id=WAREHOUSE_ID)
    resources = project / "resources"
    before = {item.name: item.read_bytes() for item in resources.iterdir()}
    with pytest.raises(ValueError, match="owner|ownership"):
        _configure(project, **kwargs)
    assert {item.name: item.read_bytes() for item in resources.iterdir()} == before
