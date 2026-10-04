"""Coherent model-set MLflow packaging and explicit alias lifecycle tests."""

import json
import subprocess
import sys
from dataclasses import replace
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, cast

import pandas as pd
import polars as pl
import pytest

mlflow = pytest.importorskip("mlflow")
from skyulf.integrations.mlflow.model_set import (  # noqa: E402
    SkyulfModelSetPythonModel,
    load_registered_model_set,
    log_model_set,
)
from skyulf.integrations.mlflow.model_set_lifecycle import (  # noqa: E402
    approve_model_set,
    rollback_model_set,
)
from skyulf.integrations.mlflow.promotion import (  # noqa: E402
    AliasConflictError,
    AliasOutcomeUnknownError,
    LocalAliasAdmission,
)
from skyulf.integrations.mlflow.registry import register_model, resolve_model  # noqa: E402


def test_unloaded_model_set_fails_without_context() -> None:
    """An unloaded pyfunc cannot silently bypass its validated artifact context."""
    with pytest.raises(RuntimeError, match="load_context"):
        SkyulfModelSetPythonModel().predict(None, pd.DataFrame())


def test_set_challenger_history_rejection_and_approval(registered_sets):
    """Displaced and rejected contenders remain inspectable without partial set activation."""
    from skyulf.integrations.mlflow.model_set_challenger import (
        nominate_model_set,
        reject_model_set,
    )

    client, uri, versions, query, admission = registered_sets
    options = {"admission": admission, "tracking_uri": uri, "registry_uri": uri}
    nominate_model_set(versions[0], expected_champion_version=None, **options)
    assert str(client.get_model_version_by_alias("coherent", "challenger").version) == "1"
    nominate_model_set(versions[1], expected_champion_version=None, **options)
    aliases = client.get_registered_model("coherent").aliases
    assert {key: str(value) for key, value in aliases.items()} == {
        "challenger": "2",
        "previous_challenger": "1",
    }
    with pytest.raises(AliasConflictError, match="[Cc]hallenger"):
        approve_model_set(
            versions[0],
            query,
            expected_champion_version=None,
            max_rows=10,
            max_bytes=10000,
            **options,
        )
    receipt = reject_model_set(
        versions[1], reason="Business review failed", expected_champion_version=None, **options
    )
    assert (
        reject_model_set(
            versions[1], reason="Business review failed", expected_champion_version=None, **options
        )
        == receipt
    )
    with pytest.raises(AliasConflictError, match="reason"):
        reject_model_set(
            versions[1], reason="Changed reason", expected_champion_version=None, **options
        )
    with pytest.raises(AliasConflictError, match="rejected"):
        approve_model_set(
            versions[1],
            query,
            expected_champion_version=None,
            max_rows=10,
            max_bytes=10000,
            **options,
        )
    assert client.get_registered_model("coherent").aliases == aliases
    assert client.get_model_version("coherent", "2").tags["approval_status"] == "rejected"


def test_set_nomination_is_cleared_on_activation_and_rollback_preserves_new_candidate(
    registered_sets,
):
    """Champion activation removes its challenger alias and rollback preserves unrelated contenders."""
    from skyulf.integrations.mlflow.model_set_challenger import nominate_model_set

    client, uri, versions, query, admission = registered_sets
    options = {"admission": admission, "tracking_uri": uri, "registry_uri": uri}
    nominate_model_set(versions[0], expected_champion_version=None, **options)
    approve_model_set(
        versions[0], query, expected_champion_version=None, max_rows=10, max_bytes=10000, **options
    )
    assert "challenger" not in client.get_registered_model("coherent").aliases
    with pytest.raises(AliasConflictError, match="[Cc]hampion"):
        nominate_model_set(versions[1], expected_champion_version=None, **options)
    nominate_model_set(versions[1], expected_champion_version="1", **options)
    receipt = approve_model_set(
        versions[1], query, expected_champion_version="1", max_rows=10, max_bytes=10000, **options
    )
    assert "challenger" not in client.get_registered_model("coherent").aliases
    result = rollback_model_set(receipt, expected_current_version="2", **options)
    assert result.new_version == "1"
    assert str(client.get_model_version_by_alias("coherent", "champion").version) == "1"


def test_cached_set_is_not_serialized() -> None:
    """Only saved context assets may determine the set restored in another process."""
    model = SkyulfModelSetPythonModel()
    model._artifact = cast(Any, object())
    assert model.__getstate__()["_artifact"] is None


@pytest.mark.parametrize("legacy", [False, True])
def test_plain_digest_metadata_preserves_legacy_registry_replay(registered_sets, legacy) -> None:
    """New metadata uses the plain name while previously published sets stay readable."""
    from pathlib import Path

    client, uri, versions, _, _ = registered_sets
    resolved = versions[0]
    source = client.get_model_version(resolved.name, resolved.version).source
    path = Path(mlflow.artifacts.download_artifacts(artifact_uri=source, tracking_uri=uri))
    metadata = mlflow.models.Model.load(path)
    assert metadata.metadata["model_set_digest"] == resolved.digest
    assert "skyulf_model_set_digest" not in metadata.metadata
    if legacy:
        metadata.metadata["skyulf_model_set_digest"] = metadata.metadata.pop("model_set_digest")
        metadata.save(str(path / "MLmodel"))
    pinned = resolve_model(
        resolved.name, version=resolved.version, tracking_uri=uri, registry_uri=uri
    )
    restored = load_registered_model_set(pinned, tracking_uri=uri, registry_uri=uri)
    assert restored.manifest.set_sha256 == resolved.digest


@pytest.fixture
def registered_sets(tmp_path, monkeypatch):
    """Publish two real fitted set versions in an isolated SQLite MLflow registry."""
    from skyulf.data.dataset import SplitDataset
    from skyulf.inference.bundle import ColumnSpec
    from skyulf.inference.local_pipeline import load_local_pipeline, save_local_pipeline
    from skyulf.inference.model_set import ComponentReference, save_model_set
    from skyulf.pipeline import SkyulfPipeline

    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.create_experiment("sets", artifact_location=(tmp_path / "runs").as_uri())
    run = client.create_run(experiment)
    versions = []
    for number in (1, 2):
        components = {}
        for branch in ("left", "right"):
            frame = pd.DataFrame({branch: [1.0, 2.0, 3.0], "target": [2.0, 4.0, 6.0]})
            pipeline = SkyulfPipeline(
                {
                    "preprocessing": [],
                    "modeling": {"type": "linear_regression"},
                    "project_python_source": "import pandas as pd\ndef eligible(frame, params):\n"
                    "    return pd.Series(['zero' if value == 0 else None for value in "
                    "frame[params['column']]], index=frame.index)\n",
                    "project_scoring": {
                        "eligibility": [
                            {
                                "name": "nonzero",
                                "version": "1",
                                "function": "eligible",
                                "params": {"column": branch},
                            }
                        ],
                        "outputs": [],
                    },
                }
            )
            native = pl.from_pandas(frame) if branch == "right" else frame
            pipeline.fit(SplitDataset(train=native, test=native.head(0)), target_column="target")
            path = tmp_path / f"{branch}-{number}"
            save_local_pipeline(pipeline, path)
            local = load_local_pipeline(path)
            components[branch] = (
                ComponentReference(
                    name=branch, version=str(number), digest=local.manifest.pipeline_sha256
                ),
                path,
            )
        path = tmp_path / f"set-{number}"
        save_model_set(
            path,
            components,
            record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
            composition_source="import pandas as pd\ndef combine(inputs, predictions, params):\n"
            "    if (inputs['left'] < 0).any():\n        raise ValueError('invalid business input')\n"
            "    return pd.DataFrame({'total': predictions['left__prediction'] + "
            "predictions['right__prediction']}, index=inputs.index)\n",
            composition_config={
                "outputs": [
                    {
                        "name": "combined",
                        "version": str(number),
                        "function": "combine",
                        "params": {},
                        "required_components": ["left", "right"],
                        "columns": [{"name": "total", "dtype": "float64"}],
                    }
                ]
            },
        )
        logged = log_model_set(
            path, run_id=run.info.run_id, artifact_path=f"set{number}", tracking_uri=uri
        )
        registered = register_model(logged, "coherent", tracking_uri=uri, registry_uri=uri)
        versions.append(
            resolve_model(
                "coherent", version=registered.version, tracking_uri=uri, registry_uri=uri
            )
        )
    query = pd.DataFrame({"id": [12, 11], "left": [5.0, 4.0], "right": [7.0, 8.0]})
    return client, uri, versions, query, LocalAliasAdmission(tmp_path / "locks")


def test_model_set_package_loads_in_fresh_process(registered_sets, tmp_path) -> None:
    """The complete registry package preserves keys without original component paths."""
    client, uri, versions, query, _ = registered_sets
    resolved = versions[0]
    artifact = load_registered_model_set(resolved, tracking_uri=uri, registry_uri=uri)
    assert artifact.manifest.set_sha256 == resolved.digest
    source = client.get_model_version(resolved.name, resolved.version).source
    local = mlflow.artifacts.download_artifacts(artifact_uri=source, tracking_uri=uri)
    from pathlib import Path

    metadata = (Path(local) / "MLmodel").read_text(encoding="utf-8")
    assert str(tmp_path) not in metadata
    assert "uri: model_set" in metadata
    for name in ("left-1", "left-2", "right-1", "right-2", "set-1", "set-2"):
        (tmp_path / name).rename(tmp_path / f"retired-{name}")
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import json,mlflow,pandas as pd,sys; "
            "m=mlflow.pyfunc.load_model(sys.argv[1]); "
            "print(json.dumps(m.predict(pd.DataFrame(json.loads(sys.argv[2]))).to_dict('list')))",
            local,
            query.to_json(),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(child.stdout)
    assert result["id"] == [12, 11]
    assert result["left__prediction"] == pytest.approx([10.0, 8.0])
    assert result["total"] == pytest.approx([24.0, 24.0])


def test_explicit_set_activation_replacement_and_rollback(registered_sets) -> None:
    """One durable set champion switches and restores complete validated releases."""
    client, uri, versions, query, admission = registered_sets
    from skyulf.integrations.mlflow.local_model import log_local_model

    artifact = load_registered_model_set(versions[0], tracking_uri=uri, registry_uri=uri)
    run_id = client.get_model_version("coherent", "1").run_id
    for branch in ("left", "right"):
        logged = log_local_model(
            artifact.directory / "components" / branch,
            run_id=run_id,
            artifact_path=branch,
            tracking_uri=uri,
        )
        register_model(logged, branch, tracking_uri=uri, registry_uri=uri)
        client.set_registered_model_alias(branch, "champion", "1")
        component = resolve_model(branch, version="1", tracking_uri=uri, registry_uri=uri)
        with pytest.raises(ValueError, match="kind"):
            load_registered_model_set(component, tracking_uri=uri, registry_uri=uri)
    options: dict[str, Any] = {
        "admission": admission,
        "max_rows": 10,
        "max_bytes": 10000,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    first = approve_model_set(versions[0], query, expected_champion_version=None, **options)
    assert first.kind == "initial"
    with pytest.raises(AliasConflictError, match="expected"):
        approve_model_set(versions[1], query, expected_champion_version=None, **options)
    promotion = approve_model_set(versions[1], query, expected_champion_version="1", **options)
    assert str(client.get_model_version_by_alias("coherent", "champion").version) == "2"
    rollback = rollback_model_set(
        promotion,
        expected_current_version="2",
        admission=admission,
        tracking_uri=uri,
        registry_uri=uri,
    )
    assert rollback.new_version == "1"
    assert str(client.get_model_version_by_alias("coherent", "champion").version) == "1"
    for branch in ("left", "right"):
        assert str(client.get_model_version_by_alias(branch, "champion").version) == "1"


def test_failed_validation_does_not_create_champion(registered_sets) -> None:
    """Empty or unbounded representative data cannot certify a release."""
    client, uri, versions, query, admission = registered_sets
    with pytest.raises(ValueError, match="nonempty"):
        approve_model_set(
            versions[0],
            query.head(0),
            expected_champion_version=None,
            admission=admission,
            max_rows=10,
            max_bytes=10000,
            tracking_uri=uri,
            registry_uri=uri,
        )
    assert "champion" not in client.get_registered_model("coherent").aliases


def test_set_loader_rejects_digest_mismatch(registered_sets) -> None:
    """A caller cannot substitute another digest into a pinned set identity."""
    _, uri, versions, _, _ = registered_sets
    with pytest.raises(ValueError, match="digest"):
        load_registered_model_set(
            replace(versions[0], digest="0" * 64), tracking_uri=uri, registry_uri=uri
        )


def test_unknown_alias_write_blocks_retry(registered_sets, monkeypatch) -> None:
    """A lost alias response leaves a durable pending event that forbids blind retries."""
    client, uri, versions, query, admission = registered_sets
    original = mlflow.MlflowClient.set_registered_model_alias

    def lose_response(self, name, alias, version):
        """Simulate only an uncertain transport response after the real SQLite mutation."""
        original(self, name, alias, version)
        raise ConnectionError("response lost")

    monkeypatch.setattr(mlflow.MlflowClient, "set_registered_model_alias", lose_response)
    options: dict[str, Any] = {
        "admission": admission,
        "max_rows": 10,
        "max_bytes": 10000,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    with pytest.raises(AliasOutcomeUnknownError):
        approve_model_set(versions[0], query, expected_champion_version=None, **options)
    assert client.get_registered_model("coherent").tags["pending_alias_event"]
    with pytest.raises(AliasConflictError, match="pending"):
        approve_model_set(versions[0], query, expected_champion_version="1", **options)
    assert str(client.get_model_version_by_alias("coherent", "champion").version) == "1"


def test_uncontrolled_alias_cannot_be_adopted(registered_sets) -> None:
    """A manual alias change is not sufficient evidence of a controlled release."""
    client, uri, versions, query, admission = registered_sets
    client.set_registered_model_alias("coherent", "champion", "1")
    with pytest.raises(AliasConflictError, match="receipt"):
        approve_model_set(
            versions[1],
            query,
            expected_champion_version="1",
            admission=admission,
            max_rows=10,
            max_bytes=10000,
            tracking_uri=uri,
            registry_uri=uri,
        )
    assert str(client.get_model_version_by_alias("coherent", "champion").version) == "1"


def test_rollback_requires_prior_validation_evidence(registered_sets) -> None:
    """Missing prior-version validation must fail before rollback changes any alias."""
    client, uri, versions, query, admission = registered_sets
    options: dict[str, Any] = {
        "admission": admission,
        "max_rows": 10,
        "max_bytes": 10000,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    approve_model_set(versions[0], query, expected_champion_version=None, **options)
    receipt = approve_model_set(versions[1], query, expected_champion_version="1", **options)
    client.delete_model_version_tag("coherent", "1", "model_set_validation_sha256")
    with pytest.raises(ValueError, match="no durable"):
        rollback_model_set(
            receipt,
            expected_current_version="2",
            admission=admission,
            tracking_uri=uri,
            registry_uri=uri,
        )
    assert str(client.get_model_version_by_alias("coherent", "champion").version) == "2"


def test_external_previous_alias_blocks_activation(registered_sets) -> None:
    """Unexpected history aliases must be detected before another release replaces them."""
    client, uri, versions, query, admission = registered_sets
    options: dict[str, Any] = {
        "admission": admission,
        "max_rows": 10,
        "max_bytes": 10000,
        "tracking_uri": uri,
        "registry_uri": uri,
    }
    approve_model_set(versions[0], query, expected_champion_version=None, **options)
    client.set_registered_model_alias("coherent", "previous_champion", "2")
    with pytest.raises(AliasConflictError, match="previous champion"):
        approve_model_set(versions[1], query, expected_champion_version="1", **options)
    assert str(client.get_model_version_by_alias("coherent", "champion").version) == "1"


def test_composition_failure_does_not_activate_set(registered_sets) -> None:
    """A representative row rejected by saved business code cannot approve partial output."""
    client, uri, versions, query, admission = registered_sets
    invalid = query.copy()
    invalid.loc[0, "left"] = -1.0
    with pytest.raises(ValueError, match="invalid business input"):
        approve_model_set(
            versions[0],
            invalid,
            expected_champion_version=None,
            admission=admission,
            max_rows=10,
            max_bytes=10000,
            tracking_uri=uri,
            registry_uri=uri,
        )
    assert "champion" not in client.get_registered_model("coherent").aliases


@pytest.mark.parametrize(
    "left,right,unexercised",
    [([0.0, 0.0], [1.0, 2.0], "left"), ([0.0, 1.0], [1.0, 0.0], "combined")],
)
def test_every_component_and_rule_must_be_exercised(
    registered_sets, left, right, unexercised
) -> None:
    """Partial branch successes do not establish that a dependent composition rule executes."""
    client, uri, versions, query, admission = registered_sets
    query["left"], query["right"] = left, right
    with pytest.raises(ValueError, match=f"did not exercise model set output {unexercised}"):
        approve_model_set(
            versions[0],
            query,
            expected_champion_version=None,
            admission=admission,
            max_rows=10,
            max_bytes=10000,
            tracking_uri=uri,
            registry_uri=uri,
        )
    assert "champion" not in client.get_registered_model("coherent").aliases


def _capture_composition(tmp_path, source, pins):
    """Capture the same organized project dependency contract as the project adapter."""
    from skyulf.integrations.databricks.model_set_project import capture_set_composition

    project = tmp_path / "composition_project"
    package = project / "src" / "composition"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(source, encoding="utf-8")
    (package / "requirements.txt").write_text("\n".join(pins), encoding="utf-8")
    return capture_set_composition({"config_path": str(project / "configs" / "pipeline.json")})


def test_composition_only_dependency_is_in_logged_requirements(registered_sets, tmp_path) -> None:
    """A fresh serving environment must install dependencies used only by saved composition."""
    from skyulf.inference.model_set import save_model_set

    client, uri, versions, query, _ = registered_sets
    artifact = load_registered_model_set(versions[0], tracking_uri=uri, registry_uri=uri)
    pin = f"pytest=={package_version('pytest')}"
    source = _capture_composition(
        tmp_path, "import pytest\n" + (artifact.directory / "composition.py").read_text(), [pin]
    )
    destination = tmp_path / "set_with_composition_dependency"
    save_model_set(
        destination,
        {
            c.branch: (c.reference, artifact.directory / "components" / c.branch)
            for c in artifact.manifest.components
        },
        record_key_schema=artifact.manifest.record_key_schema,
        composition_source=source,
        composition_config=artifact.manifest.composition_config,
    )
    run_id = client.get_model_version("coherent", "1").run_id
    logged = log_model_set(
        destination, run_id=run_id, artifact_path="composition_dependency", tracking_uri=uri
    )
    downloaded = Path(mlflow.artifacts.download_artifacts(artifact_uri=logged, tracking_uri=uri))
    requirements = (downloaded / "requirements.txt").read_text().splitlines()
    assert pin in requirements
    assert not any(line.startswith("-e") or " @ " in line for line in requirements)
    assert mlflow.pyfunc.load_model(str(downloaded)).predict(query)[
        "total"
    ].tolist() == pytest.approx([24.0, 24.0])


def test_composition_pin_conflicts_with_component_runtime(registered_sets, tmp_path) -> None:
    """Pin aggregation must reject contradictory component and composition dependencies."""
    from skyulf.integrations.mlflow.model_set import _set_requirements

    _, uri, versions, _, _ = registered_sets
    artifact = load_registered_model_set(versions[0], tracking_uri=uri, registry_uri=uri)
    source = _capture_composition(tmp_path, "", ["numpy==0.0.0"])
    # Isolate metadata aggregation from the earlier runtime/checksum rejection:
    # no imported project code or fitted component is replaced in this test.
    (artifact.directory / "composition.py").write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="Conflicting model set requirement pins for numpy"):
        _set_requirements(artifact)
