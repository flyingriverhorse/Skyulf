"""Model sets retain verified component bytes without mutable external references."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from tests.integration.platforms.test_local_pipeline_artifact import _fitted_pipeline

from skyulf.inference._manifest import ColumnSpec
from skyulf.inference.local_pipeline import (
    load_local_pipeline,
    predict_local_pipeline,
    save_local_pipeline,
)


def _api():
    """Fail explicitly when the requested package boundary is absent."""
    assert importlib.util.find_spec("skyulf.inference.model_set") is not None
    from skyulf.inference import model_set

    return model_set


@pytest.fixture
def components(tmp_path):
    """Save two genuine fitted engines before their source directories disappear."""
    result = {}
    for engine in ("pandas", "polars"):
        pipeline, _ = _fitted_pipeline(engine)
        path = tmp_path / engine
        save_local_pipeline(pipeline, path)
        digest = hashlib.sha256((path / "pipeline.pkl").read_bytes()).hexdigest()
        result[engine] = (digest, path)
    return result


def _save(tmp_path, components, **kwargs):
    """Build a package through the public API with concrete version references."""
    api = _api()
    refs = {
        branch: (
            api.ComponentReference(name=f"catalog.schema.{branch}", version="1", digest=digest),
            path,
        )
        for branch, (digest, path) in components.items()
    }
    kwargs.setdefault("record_key_schema", (ColumnSpec(name="record_id", dtype="int64"),))
    return api.save_model_set(tmp_path / "set", refs, **kwargs)


def test_set_replays_exact_component_bytes_after_source_deletion(tmp_path, components):
    """Transport must preserve both fitted engines and their original payload identity."""
    artifact = _save(tmp_path, components)
    for branch, (digest, source) in components.items():
        copied = artifact.directory / "components" / branch
        assert (copied / "pipeline.pkl").read_bytes() == (source / "pipeline.pkl").read_bytes()
        shutil.rmtree(source)
        restored = load_local_pipeline(copied)
        _, query = _fitted_pipeline(branch)
        np.testing.assert_allclose(
            predict_local_pipeline(query, restored)["prediction"], [29, 18, 41]
        )
        assert restored.manifest.pipeline_sha256 == digest
    loaded = _api().load_model_set(artifact.directory)
    assert loaded.manifest == artifact.manifest
    assert loaded.manifest.record_key_columns == ("record_id",)
    assert [c.name for c in loaded.manifest.output_schema] == [
        "record_id",
        "pandas__prediction",
        "pandas__scoring_status",
        "pandas__exclusion_reason",
        "polars__prediction",
        "polars__scoring_status",
        "polars__exclusion_reason",
    ]


def test_legacy_manifest_keeps_original_shape_and_digest(tmp_path, components):
    """Packages without quality pins must remain readable with their original digest."""
    artifact = _save(tmp_path, components)
    payload = json.loads((artifact.directory / "manifest.json").read_text())
    assert "quality_evidence_json" not in payload
    expected = payload.pop("set_sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == expected
    restored = _api().load_model_set(artifact.directory)
    assert restored.manifest.set_sha256 == expected
    assert restored.manifest.quality_evidence is None


def test_composition_config_is_defensively_captured(tmp_path, components):
    """Caller mutations must never change the rules covered by the set digest."""
    config = {
        "outputs": [
            {
                "name": "weighted",
                "version": "1",
                "function": "output",
                "params": {"weight": 2},
                "columns": [{"name": "weighted_value", "dtype": "float64"}],
                "required_components": ["pandas"],
            }
        ]
    }
    artifact = _save(
        tmp_path,
        components,
        composition_config=config,
        composition_source="def output(inputs, predictions, params):\n    return predictions\n",
    )
    config["outputs"][0]["params"]["weight"] = 9
    artifact.manifest.composition_config["outputs"][0]["params"]["weight"] = 17
    assert artifact.manifest.composition_config["outputs"][0]["params"]["weight"] == 2
    assert (
        _api().load_model_set(artifact.directory).manifest.set_sha256
        == artifact.manifest.set_sha256
    )


@pytest.mark.parametrize(
    "target",
    ["composition.py", "components/pandas/pipeline.pkl", "components/pandas/manifest.json"],
)
def test_tampered_bytes_are_rejected(tmp_path, components, target):
    """All executable and metadata bytes must remain inside the package identity."""
    artifact = _save(tmp_path, components)
    path = artifact.directory / target
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum|digest"):
        _api().load_model_set(artifact.directory)


@pytest.mark.parametrize("branch", ["../escape", "a/b", "a\\b", "a__b", "CON", "a."])
def test_unsafe_component_names_are_rejected(tmp_path, components, branch):
    """Branch names must be unambiguous portable directory and output prefixes."""
    with pytest.raises(ValueError):
        _save(tmp_path, {branch: components["pandas"]})
    assert not (tmp_path / "set").exists()


@pytest.mark.parametrize(
    "keys",
    [
        (),
        (ColumnSpec(name="id", dtype="object"),),
        (ColumnSpec(name="id", dtype="float64"),),
        (ColumnSpec(name="id", dtype="uint64"),),
        (ColumnSpec(name="id", dtype="int64"),) * 2,
    ],
)
def test_record_keys_require_unique_supported_schema(tmp_path, components, keys):
    """Key contracts must be available for replay before any live source exists."""
    with pytest.raises(ValueError):
        _save(tmp_path, components, record_key_schema=keys)
    assert not (tmp_path / "set").exists()


def test_wrong_reference_digest_is_rejected(tmp_path, components):
    """A registry version claim cannot silently select another model payload."""
    _, source = components["pandas"]
    with pytest.raises(ValueError, match="digest"):
        _save(tmp_path, {"pandas": ("0" * 64, source)})
    assert not (tmp_path / "set").exists()


def test_duplicate_identity_and_casefold_branch_are_rejected(tmp_path, components):
    """Multiple branches must not silently alias one version or one Windows directory."""
    api = _api()
    digest, source = components["pandas"]
    ref = api.ComponentReference(name="same", version="1", digest=digest)
    for branches in (("a", "b"), ("a", "A")):
        with pytest.raises(ValueError):
            api.save_model_set(
                tmp_path / "set",
                dict.fromkeys(branches, (ref, source)),
                record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
            )
    assert not (tmp_path / "set").exists()


def test_manifest_schema_cannot_be_forged_even_with_recomputed_digest(tmp_path, components):
    """Schema declarations must agree with fitted components, beyond a self-reported hash."""
    artifact = _save(tmp_path, components)
    manifest_path = artifact.directory / "manifest.json"
    document = json.loads(manifest_path.read_text())
    document["components"][0]["output_schema"][0]["dtype"] = "string"
    document.pop("set_sha256")
    document["set_sha256"] = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="schema|component"):
        _api().load_model_set(artifact.directory)


def test_source_and_aggregate_limits_leave_no_destination(tmp_path, components, monkeypatch):
    """Bounded packages must fail before publishing a partial directory."""
    api = _api()
    with pytest.raises(ValueError, match="size limit"):
        _save(tmp_path, components, composition_source="x" * (65536 + 1))
    monkeypatch.setattr(api, "_MAX_PACKAGE_BYTES", 100)
    with pytest.raises(ValueError, match="size limit"):
        _save(tmp_path, components)
    assert not (tmp_path / "set").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", "champion"),
        ("version", "0"),
        ("version", "01"),
        ("version", 1),
        ("digest", "missing"),
        ("name", "../outside"),
        ("name", "name\\path"),
    ],
)
def test_nonconcrete_model_references_are_rejected(field, value):
    """Local loading must never need mutable alias selection or path-based model names."""
    values = {"name": "catalog.schema.model", "version": "1", "digest": "a" * 64}
    values[field] = value
    with pytest.raises(ValueError):
        _api().ComponentReference(**values)


def test_conflicting_key_and_component_dtype_fails(tmp_path, components):
    """A shared source column cannot satisfy incompatible key and model schemas."""
    with pytest.raises(ValueError, match="dtype conflict"):
        _save(tmp_path, components, record_key_schema=(ColumnSpec(name="amount", dtype="string"),))
    assert not (tmp_path / "set").exists()


def test_conflicting_component_input_dtypes_fail(tmp_path, components):
    """Independent fitted components must agree on every shared raw input column."""
    import pandas as pd

    from skyulf.data.dataset import SplitDataset
    from skyulf.pipeline import SkyulfPipeline

    pipeline = SkyulfPipeline({"modeling": {"type": "linear_regression"}})
    train = pd.DataFrame({"amount": [1, 2, 3, 4], "target": [2.0, 4.0, 6.0, 8.0]})
    pipeline.fit(SplitDataset(train=train, test=train), target_column="target")
    source = tmp_path / "integers"
    save_local_pipeline(pipeline, source)
    digest = hashlib.sha256((source / "pipeline.pkl").read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="dtype conflict"):
        _save(tmp_path, {"pandas": components["pandas"], "integers": (digest, source)})
    assert not (tmp_path / "set").exists()


def test_namespaced_output_key_collision_fails(tmp_path, components):
    """Output names must never overwrite record keys during downstream joining."""
    with pytest.raises(ValueError, match="collision"):
        _save(
            tmp_path,
            components,
            record_key_schema=(ColumnSpec(name="pandas__prediction", dtype="string"),),
        )
    assert not (tmp_path / "set").exists()


@pytest.mark.parametrize("config", [{"weight": float("nan")}, {"nested": (1, 2)}, {1: "coerced"}])
def test_non_json_configuration_fails(tmp_path, components, config):
    """Rule parameters must retain their exact finite JSON meaning across replay."""
    with pytest.raises(ValueError, match="JSON"):
        _save(tmp_path, components, composition_config=config)
    assert not (tmp_path / "set").exists()


def test_manifest_duplicate_keys_are_rejected(tmp_path, components):
    """Duplicate JSON entries must not let parsers disagree about package identity."""
    artifact = _save(tmp_path, components)
    path = artifact.directory / "manifest.json"
    path.write_text(
        path.read_text().replace('"format_version":1', '"format_version":1,"format_version":1'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _api().load_model_set(artifact.directory)


def test_component_symlink_is_rejected(tmp_path, components):
    """Package copying must not dereference an artifact path outside its own tree."""
    source = components["pandas"][1]
    link = source / "outside.pkl"
    try:
        link.symlink_to(components["polars"][1] / "pipeline.pkl")
    except OSError:
        pytest.skip("Symlink creation requires Windows developer mode or privilege.")
    with pytest.raises(ValueError, match="symlink"):
        _save(tmp_path, components)
    assert not (tmp_path / "set").exists()


def test_aggregate_bound_rejects_multiple_individually_small_components(
    tmp_path, components, monkeypatch
):
    """The memory budget must apply to the entire package, not each model separately."""
    size = sum(file.stat().st_size for file in components["pandas"][1].iterdir())
    monkeypatch.setattr(_api(), "_MAX_PACKAGE_BYTES", size + 1024)
    with pytest.raises(ValueError, match="size limit"):
        _save(tmp_path, components)
    assert not (tmp_path / "set").exists()


def test_fresh_process_replays_after_sources_disappear(tmp_path, components):
    """Offline loading must not depend on live Python objects or original artifact paths."""
    artifact = _save(tmp_path, components)
    for _, source in components.values():
        shutil.rmtree(source)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from skyulf.inference.model_set import load_model_set; import sys; a=load_model_set(sys.argv[1]); print(a.manifest.set_sha256)",
            str(artifact.directory),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == artifact.manifest.set_sha256


@pytest.mark.parametrize(
    "target", ["composition.py", "components/pandas/pipeline.pkl", "manifest.json"]
)
def test_loaded_artifact_byte_verification_detects_later_changes(tmp_path, components, target):
    """Scoring must reject changed disk contents before executing saved callbacks."""
    artifact = _save(tmp_path, components)
    api = _api()
    assert hasattr(api, "verify_model_set_files")
    api.verify_model_set_files(artifact)
    path = artifact.directory / target
    if target == "manifest.json":
        document = json.loads(path.read_text())
        document["set_sha256"] = "f" * 64
        path.write_text(json.dumps(document), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum|digest|manifest"):
        api.verify_model_set_files(artifact)


def test_manifest_limit_rejects_oversized_component_metadata(tmp_path, components):
    """Small model payloads must not permit an unbounded metadata document."""
    api = _api()
    digest, source = components["pandas"]
    reference = api.ComponentReference(name="model" * 14000, version="1", digest=digest)
    with pytest.raises(ValueError, match="manifest exceeds the size limit"):
        api.save_model_set(
            tmp_path / "set",
            {"a": (reference, source)},
            record_key_schema=(ColumnSpec(name="id", dtype="int64"),),
        )
    assert not (tmp_path / "set").exists()


def test_manifest_component_path_traversal_fails_before_loading(tmp_path, components):
    """Even a self-consistent manifest cannot redirect component loading outside the set."""
    artifact = _save(tmp_path, components)
    path = artifact.directory / "manifest.json"
    document = json.loads(path.read_text())
    document["components"][0]["branch"] = "../../pandas"
    document.pop("set_sha256")
    document["set_sha256"] = hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="branch name"):
        _api().load_model_set(artifact.directory)
