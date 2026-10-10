"""Self-contained, bounded packages of immutable local model artifacts.

Loading executes trusted component pickle and saved preprocessing code. Digests
detect changed bytes; they do not authenticate the producer. Saved composition
source is imported when validating its declared callback contract.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ..core.portable_state import _bad_constant, _unique_object
from ._manifest import ColumnSpec, checksum
from ._model_set_manifest import (
    ComponentFile,
    ComponentManifest,
    ComponentReference,
    ModelSetManifest,
    canonical_json,
    combined_schemas,
    manifest_digest,
    quality_evidence_json,
    validate_branch,
    validate_components,
    validate_keys,
)
from .fitted_pipeline import load_pipeline as load_local_pipeline
from .fitted_pipeline import read_bounded_artifact
from .pipeline_scoring import scoring_output_schema

_MAX_PACKAGE_BYTES = 256 * 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_SOURCE_BYTES = 64 * 1024
_LOCAL_FILES = {"manifest.json", "pipeline.pkl", "preprocessing.py"}


@dataclass(frozen=True)
class ModelSetArtifact:
    """Retain a validated directory and contract without retaining fitted models."""

    manifest: ModelSetManifest
    directory: Path
    feature_lookup_json: str | None = None


def _safe_directory(path: Path) -> Path:
    """Reject symlink traversal including symlinks in ancestor directories."""
    if any(part.is_symlink() or part.is_junction() for part in (path, *path.parents)):
        raise ValueError("Model set paths must not contain symlinks or junctions.")
    return path.resolve()


def _files(path: Path, limit: int) -> tuple[Path, ...]:
    """Check the full tree and its aggregate byte budget before deserialization."""
    root = _safe_directory(path)
    result = []
    total = 0
    for child in root.rglob("*"):
        if child.is_symlink() or child.is_junction() or not child.resolve().is_relative_to(root):
            raise ValueError("Model set paths must not contain symlinks or junctions.")
        if child.is_file():
            total += child.stat().st_size
            if total > limit:
                raise ValueError("Model set package exceeds the size limit.")
            result.append(child)
        elif not child.is_dir():
            raise ValueError("Model set contains a nonregular file.")
    return tuple(result)


def _component(path: Path, branch: str, reference: ComponentReference) -> ComponentManifest:
    """Derive component metadata from one validated model, releasing it on return."""
    validate_branch(branch)
    files = _files(path, _MAX_PACKAGE_BYTES)
    names = {file.relative_to(path.resolve()).as_posix() for file in files}
    if not {"manifest.json", "pipeline.pkl"} <= names <= _LOCAL_FILES:
        raise ValueError("Unexpected or missing component artifact files.")
    artifact = load_local_pipeline(path)
    if artifact.manifest.pipeline_sha256 != reference.digest:
        raise ValueError("Component reference digest disagrees with saved pipeline.")
    expected = {"manifest.json", "pipeline.pkl"}
    if artifact.manifest.project_source_sha256 is not None:
        expected.add("preprocessing.py")
    if names != expected:
        raise ValueError("Component source files disagree with its fitted artifact.")
    schema = tuple(
        ColumnSpec(name=name, dtype=dtype)
        for name, dtype in zip(
            artifact.manifest.input_columns, artifact.manifest.input_dtypes, strict=True
        )
    )
    return ComponentManifest(
        branch=branch,
        reference=reference,
        input_schema=schema,
        output_schema=scoring_output_schema(artifact),
        files=tuple(
            ComponentFile.model_validate(
                {
                    "name": file.name,
                    "sha256": checksum(read_bounded_artifact(file, _MAX_PACKAGE_BYTES)),
                }
            )
            for file in sorted(files)
        ),
    )


def _configuration(config: dict | None) -> str:
    """Capture only finite JSON objects without mutable aliases or lossy coercion."""
    if config is None:
        config = {"outputs": []}
    if type(config) is not dict:
        raise ValueError("Composition configuration must be a JSON object.")
    try:
        encoded = canonical_json(config)
        if json.loads(encoded) != config:
            raise ValueError("Composition configuration must contain JSON values.")
    except (TypeError, ValueError) as exc:
        raise ValueError("Composition configuration must contain finite JSON values.") from exc
    return encoded


def _new_manifest(
    components: tuple[ComponentManifest, ...],
    keys: tuple[ColumnSpec, ...],
    source: bytes,
    config: str,
    quality_evidence: dict | None = None,
) -> ModelSetManifest:
    """Build canonical schemas and immutable metadata independently of source paths."""
    from .model_set_scoring import (  # noqa: PLC0415 - avoid runtime inference import cycle
        model_set_schema,
        validate_model_set_composition,
    )

    validate_keys(keys)
    validate_components(components)
    inputs, _ = combined_schemas(components, keys)
    rules = validate_model_set_composition(
        json.loads(config), source.decode("utf-8"), components, keys
    )
    outputs = model_set_schema(components, keys, rules)
    manifest = ModelSetManifest(
        components=components,
        record_key_schema=keys,
        input_schema=inputs,
        output_schema=outputs,
        composition_source_sha256=checksum(source),
        composition_config_json=_configuration(rules),
        quality_evidence_json=quality_evidence_json(
            quality_evidence, {c.branch for c in components}
        ),
        set_sha256="0" * 64,
    )
    return manifest.model_copy(update={"set_sha256": manifest_digest(manifest)})


def save_model_set(
    path: str | Path,
    components: dict[str, tuple[ComponentReference, Path]],
    *,
    record_key_schema: tuple[ColumnSpec, ...],
    composition_source: str = "",
    composition_config: dict | None = None,
    quality_evidence: dict | None = None,
) -> ModelSetArtifact:
    """Copy validated trusted component files into a new atomic offline package."""
    destination = _safe_directory(Path(path))
    if destination.exists():
        raise FileExistsError(destination)
    source = composition_source.encode("utf-8")
    if len(source) > _MAX_SOURCE_BYTES:
        raise ValueError("Composition source exceeds the size limit.")
    config = _configuration(composition_config)
    _check_total_size(components, len(source))
    records = tuple(
        _component(Path(directory), branch, reference)
        for branch, (reference, directory) in sorted(components.items())
    )
    manifest = _new_manifest(records, record_key_schema, source, config, quality_evidence)
    metadata = manifest.model_dump_json(exclude_none=True).encode()
    if len(metadata) > _MAX_MANIFEST_BYTES:
        raise ValueError("Model set manifest exceeds the size limit.")
    _check_total_size(components, len(source) + len(metadata))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".model-set-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        (staging / "manifest.json").write_bytes(metadata)
        (staging / "composition.py").write_bytes(source)
        _copy_components(components, records, staging)
        load_model_set(staging)
        staging.rename(destination)
    return ModelSetArtifact(manifest, destination)


def _check_total_size(components: dict[str, tuple[ComponentReference, Path]], total: int) -> None:
    """Account for all component and set bytes before writing the package."""
    for _, directory in components.values():
        total += sum(file.stat().st_size for file in _files(Path(directory), _MAX_PACKAGE_BYTES))
    if total > _MAX_PACKAGE_BYTES:
        raise ValueError("Model set package exceeds the size limit.")


def _copy_components(
    components: dict[str, tuple[ComponentReference, Path]],
    records: tuple[ComponentManifest, ...],
    destination: Path,
) -> None:
    """Copy exact saved bytes without serializing fitted models again."""
    for record in records:
        target = destination / "components" / record.branch
        target.mkdir(parents=True)
        source = Path(components[record.branch][1])
        for file in record.files:
            shutil.copyfile(source / file.name, target / file.name)


def _read_manifest(source: Path) -> ModelSetManifest:
    """Reject duplicate keys and nonfinite JSON before strict schema parsing."""
    metadata = read_bounded_artifact(source / "manifest.json", _MAX_MANIFEST_BYTES)
    document = json.loads(metadata, object_pairs_hook=_unique_object, parse_constant=_bad_constant)
    if type(document) is not dict or type(document.get("format_version")) is not int:
        raise ValueError("Invalid model set format_version.")
    manifest = ModelSetManifest.model_validate_json(metadata)
    if manifest_digest(manifest) != manifest.set_sha256:
        raise ValueError("Model set manifest digest mismatch.")
    return manifest


def _verify_files(source: Path, manifest: ModelSetManifest, files: tuple[Path, ...]) -> None:
    """Verify the declared exact file inventory before loading executable components."""
    expected = {"manifest.json", "composition.py"}
    for component in manifest.components:
        validate_branch(component.branch)
        for file in component.files:
            relative = f"components/{component.branch}/{file.name}"
            expected.add(relative)
            if (
                checksum(read_bounded_artifact(source / relative, _MAX_PACKAGE_BYTES))
                != file.sha256
            ):
                raise ValueError("Model set component file checksum mismatch.")
    actual = {file.relative_to(source).as_posix() for file in files}
    if actual != expected:
        raise ValueError("Model set file inventory mismatch.")


def _verified_contents(source: Path) -> tuple[ModelSetManifest, bytes]:
    """Verify all recorded package bytes before importing executable contents."""
    files = _files(source, _MAX_PACKAGE_BYTES)
    manifest = _read_manifest(source)
    validate_components(manifest.components)
    _verify_files(source, manifest, files)
    code = read_bounded_artifact(source / "composition.py", _MAX_SOURCE_BYTES)
    code.decode("utf-8")
    if checksum(code) != manifest.composition_source_sha256:
        raise ValueError("Composition source checksum mismatch.")
    return manifest, code


def verify_model_set_files(artifact: ModelSetArtifact) -> None:
    """Recheck an already loaded package without retaining or unpickling models.

    Use immediately before scoring to detect changes since ``load_model_set``.
    Callers must still use a trusted, stable directory throughout execution.
    """
    source = _safe_directory(artifact.directory)
    manifest, _ = _verified_contents(source)
    if manifest != artifact.manifest:
        raise ValueError("Model set manifest changed after loading.")


def load_model_set(path: str | Path) -> ModelSetArtifact:
    """Validate trusted local bytes and fitted schemas without resolving any aliases."""
    source = _safe_directory(Path(path))
    manifest, code = _verified_contents(source)
    records = tuple(
        _component(source / "components" / record.branch, record.branch, record.reference)
        for record in manifest.components
    )
    expected = _new_manifest(
        records,
        manifest.record_key_schema,
        code,
        _configuration(manifest.composition_config),
        manifest.quality_evidence,
    )
    if expected != manifest:
        raise ValueError("Model set manifest disagrees with component schemas or configuration.")
    return ModelSetArtifact(manifest, source)
