"""Snapshot the installed pure-Python Core without build tools or package-index access."""

import base64
import csv
import hashlib
import io
import os
import re
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from functools import lru_cache
from importlib.metadata import Distribution, distribution, distributions
from pathlib import Path

import skyulf


@dataclass(frozen=True)
class CoreWheel:
    """Keep cached source identity and archive bytes immutable across export requests."""

    filename: str
    content: bytes
    source_sha256: str


def _package_root() -> Path:
    """Use the actual imported Core, including editable development installations."""
    return Path(skyulf.__file__).resolve().parent


def _contained_file(root: Path, path: Path) -> bytes:
    """Refuse symbolic links and resolved escapes before copying package content."""
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_relative_to(root):
        raise ValueError("Core package contains a symbolic link or file outside its root.")
    return resolved.read_bytes()


def _package_files(root: Path) -> dict[str, bytes]:
    """Retain Python modules and package resources while excluding interpreter caches."""
    files = {}
    for directory, children, names in os.walk(root, onerror=_walk_error):
        parent = Path(directory)
        children[:] = _package_directories(root, parent, children)
        for name in names:
            if not name.startswith(".") and Path(name).suffix not in {".pyc", ".pyo"}:
                path = parent / name
                files["skyulf/" + path.relative_to(root).as_posix()] = _contained_file(root, path)
    if any(Path(name).suffix in {".so", ".pyd", ".dll", ".dylib"} for name in files):
        raise ValueError("Standalone export requires a pure-Python skyulf-core installation.")
    if "skyulf/__init__.py" not in files:
        raise ValueError("The installed Core package is missing its Python source.")
    return files


def _walk_error(error: OSError) -> None:
    """Abort rather than emit an incomplete wheel when a package directory is unreadable."""
    raise ValueError("Core package files are unreadable; repair the installation.") from error


def _package_directories(root: Path, parent: Path, children: list[str]) -> list[str]:
    """Exclude caches and reject directory links before recursive discovery."""
    result = [name for name in children if name != "__pycache__" and not name.startswith(".")]
    for child in result:
        path = parent / child
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Core package contains a symbolic directory outside its root.")
    return result


def _core_distribution() -> Distribution:
    """Prefer installed wheel metadata when a source checkout also exposes egg-info."""
    installed = distribution("skyulf-core")
    if installed.read_text("WHEEL") is not None:
        return installed
    for candidate in distributions(name="skyulf-core"):
        if candidate.version == installed.version and candidate.read_text("WHEEL") is not None:
            return candidate
    raise ValueError("Reinstall skyulf-core to provide wheel metadata for standalone export.")


def _metadata_root(installed: Distribution) -> Path:
    """Locate standard installed metadata without using private distribution attributes."""
    candidates = [
        item
        for item in installed.files or ()
        if len(item.parts) == 2 and item.parts[0].endswith(".dist-info") and item.name == "METADATA"
    ]
    if len(candidates) != 1:
        raise ValueError("Core wheel export requires installed .dist-info metadata.")
    return Path(str(installed.locate_file(candidates[0]))).resolve().parent


def _validate_distribution(installed: Distribution, root: Path) -> None:
    """Restrict repacking to Core's known single-package portable wheel layout."""
    wheel = Parser().parsestr(_contained_file(root, root / "WHEEL").decode())
    valid_wheel = (
        wheel["Wheel-Version"] == "1.0"
        and wheel["Root-Is-Purelib"] == "true"
        and wheel.get_all("Tag") == ["py3-none-any"]
    )
    top_level = installed.read_text("top_level.txt")
    valid_package = top_level is None or top_level.split() == ["skyulf"]
    if not valid_wheel or not valid_package:
        raise ValueError("Standalone export requires a pure-Python skyulf-core installation.")
    if not re.fullmatch(r"[A-Za-z0-9.+!_]+", installed.version):
        raise ValueError("Installed Core version cannot be represented in a wheel filename.")


def _metadata_files(installed: Distribution) -> dict[str, bytes]:
    """Retain dependencies, entry points and licenses without local install provenance."""
    root = _metadata_root(installed)
    _validate_distribution(installed, root)
    metadata = f"skyulf_core-{installed.version}.dist-info"
    names = {"METADATA", "WHEEL", "entry_points.txt", "top_level.txt"}
    for license_name in installed.metadata.get_all("License-File") or ():
        relative = Path(license_name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in license_name:
            raise ValueError("Core license metadata contains an unsafe relative path.")
        modern = root / "licenses" / relative
        names.add(("licenses/" if modern.is_file() else "") + relative.as_posix())
    files = {}
    for name in sorted(names):
        path = root / name
        if path.is_file():
            files[f"{metadata}/{name}"] = _contained_file(root, path)
        elif name not in {"entry_points.txt", "top_level.txt"}:
            raise ValueError(f"Core wheel metadata is missing {name}.")
    return files


def _source_digest(entries: tuple[tuple[str, bytes], ...]) -> str:
    """Hash every emitted filename and byte, including changed resources and metadata."""
    digest = hashlib.sha256()
    for name, content in entries:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(len(content)).encode())
        digest.update(b"\0")
        digest.update(content)
    return digest.hexdigest()


def _write_member(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    """Use deterministic wheel timestamps and readable permissions on every platform."""
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = 0o644 << 16
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED)


@lru_cache(maxsize=2)
def _build_wheel(
    version: str, source_sha256: str, entries: tuple[tuple[str, bytes], ...]
) -> CoreWheel:
    """Cache two immutable snapshots, rebuilding only after content or version changes."""
    prefix = f"skyulf_core-{version}"
    build = "1" + source_sha256[:16]
    filename = f"{prefix}-{build}-py3-none-any.whl"
    files = dict(entries)
    files[f"{prefix}.dist-info/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: Skyulf project export\n"
        f"Root-Is-Purelib: true\nTag: py3-none-any\nBuild: {build}\n"
    ).encode()
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            _write_member(archive, name, content)
            checksum = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
            writer.writerow((name, "sha256=" + checksum.decode(), len(content)))
        record_name = f"{prefix}.dist-info/RECORD"
        writer.writerow((record_name, "", ""))
        _write_member(archive, record_name, records.getvalue().encode())
    return CoreWheel(filename, output.getvalue(), source_sha256)


def bundled_core_wheel() -> CoreWheel:
    """Return the current Core source as a self-contained wheel with a fresh RECORD."""
    installed = _core_distribution()
    files = _package_files(_package_root().resolve())
    files.update(_metadata_files(installed))
    entries = tuple(sorted(files.items()))
    return _build_wheel(installed.version, _source_digest(entries), entries)
