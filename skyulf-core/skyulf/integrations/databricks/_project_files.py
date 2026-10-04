"""Capture bounded project source and locate hooks in legacy or organized layouts."""

import json
import keyword
from base64 import b64encode
from pathlib import Path

from ...inference.project_code import MAX_PROJECT_SOURCE_BYTES, project_source_digest
from ...inference.project_dependencies import parse_project_requirements


def read_source(path: Path) -> str:
    """Read a bounded UTF-8 source file without truncating the saved program."""
    with path.open("rb") as stream:
        payload = stream.read(MAX_PROJECT_SOURCE_BYTES + 1)
    if len(payload) > MAX_PROJECT_SOURCE_BYTES:
        raise ValueError(f"{path.name} source exceeds 64 KiB.")
    return payload.decode("utf-8")


def _module_path(path: Path, root: Path) -> str:
    """Reject escaping paths and filenames that cannot be imported unambiguously."""
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Project package source must stay inside its root.")
    relative = path.relative_to(root)
    for name in (*relative.parts[:-1], relative.stem):
        if not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"Project Python module needs an importable name: {relative}.")
    return relative.as_posix()


def _contained_file(root: Path, relative: str) -> Path:
    """Require canonical relative paths without symlinks or platform-specific aliases."""
    parts = relative.split("/")
    if any(part in {"", ".", ".."} for part in parts) or any(
        character in relative for character in "\\:\x00"
    ):
        raise ValueError(f"Project asset path must be a canonical relative path: {relative}.")
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink() or candidate.is_junction():
            raise ValueError(f"Project asset/metadata cannot use a symlink: {relative}.")
    if not candidate.resolve().is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError(f"Project asset/metadata must be a file inside its root: {relative}.")
    return candidate


def _project_assets(root: Path) -> dict[str, str]:
    """Capture only explicitly declared data assets within the total snapshot bound."""
    manifest = root / "assets.json"
    if not manifest.exists() and not manifest.is_symlink():
        return {}
    try:
        entries = _asset_entries(json.loads(read_source(_contained_file(root, "assets.json"))))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Project assets.json must contain a JSON list of asset paths.") from exc
    entries = _validate_asset_entries(entries)
    assets = {}
    size = 0
    for relative in sorted(entries):
        path = _contained_file(root, relative)
        with path.open("rb") as stream:
            content = stream.read(MAX_PROJECT_SOURCE_BYTES + 1)
        size += len(content)
        if size > MAX_PROJECT_SOURCE_BYTES:
            raise ValueError("Project assets exceed 64 KiB.")
        assets[relative] = b64encode(content).decode("ascii")
    return assets


def _asset_entries(manifest: object) -> object:
    """Accept legacy lists or a documented file list without capturing its help text."""
    if not isinstance(manifest, dict):
        return manifest
    if "files" not in manifest or set(manifest) - {"files", "_help"}:
        raise ValueError("Project assets.json object requires files and optional _help only.")
    help_text = manifest.get("_help", [])
    if not isinstance(help_text, list) or any(not isinstance(line, str) for line in help_text):
        raise ValueError("Project assets.json _help must be a list of text instructions.")
    return manifest["files"]


def _validate_asset_entries(entries: object) -> list[str]:
    """Reject ambiguous declarations and keep executable source out of data assets."""
    if not isinstance(entries, list) or any(not isinstance(item, str) for item in entries):
        raise ValueError("Project assets.json must contain a JSON list of asset paths.")
    if len(entries) != len(set(entries)):
        raise ValueError("Project asset paths must be distinct.")
    if any(
        Path(item).suffix.lower() == ".py" or item in {"assets.json", "requirements.txt"}
        for item in entries
    ):
        raise ValueError("Project assets must be data files, separate from source and metadata.")
    return entries


def _project_requirements(root: Path) -> tuple[str, ...]:
    """Read optional pinned external dependencies without installing or importing them."""
    manifest = root / "requirements.txt"
    if not manifest.exists() and not manifest.is_symlink():
        return ()
    return parse_project_requirements(read_source(_contained_file(root, "requirements.txt")))


def _validate_package_files(files: dict[str, str]) -> None:
    """Require explicit package boundaries and reject module/package name collisions."""
    if "__init__.py" not in files:
        raise ValueError("Project package must contain __init__.py.")
    for filename in files:
        for parent in Path(filename).parents:
            if (parent / "__init__.py").as_posix() not in files:
                raise ValueError(f"Project package needs {parent}/__init__.py.")
        if (
            filename.endswith("/__init__.py")
            and filename.removesuffix("/__init__.py") + ".py" in files
        ):
            raise ValueError(f"Project module conflicts with its package: {filename}.")


def project_source(path: Path) -> str:
    """Keep single-file snapshots compatible or archive a complete Python package."""
    if not path.is_dir():
        return read_source(path)
    files = {}
    size = 0
    for filename in sorted(path.rglob("*.py"), key=lambda item: item.relative_to(path).as_posix()):
        relative = _module_path(filename, path)
        source = read_source(filename)
        size += len(source.encode("utf-8"))
        if size > MAX_PROJECT_SOURCE_BYTES:
            raise ValueError("Project package source exceeds 64 KiB.")
        files[relative] = source
    _validate_package_files(files)
    assets = _project_assets(path)
    requirements = _project_requirements(path)
    extras = f", assets={assets!r}, requirements={requirements!r}" if assets or requirements else ""
    source = (
        "from skyulf.inference.project_package import install_project_package\n"
        f"install_project_package(__name__, {files!r}{extras})\n"
    )
    project_source_digest(source)
    return source


def modeling_hook(path: Path, name: str) -> Path:
    """Resolve organized feature packages beside modeling, or legacy sibling hooks."""
    return path.parent / "modeling" / name if path.is_dir() else path.with_name(name)


def renamed_modeling_hook(path: Path, legacy_name: str) -> Path:
    """Accept a legacy filename only when it cannot compete with its replacement."""
    legacy = path.with_name(legacy_name)
    if path.exists() and legacy.exists():
        raise ValueError(f"Ambiguous modeling files: {path.name} and {legacy_name}; keep only one.")
    return legacy if legacy.is_file() else path
