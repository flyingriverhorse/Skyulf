"""Exact-source pure-Python wheels for isolated MLflow Spark worker environments."""

import base64
import csv
import hashlib
import inspect
import io
import shutil
import zipfile
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import skyulf


def pyfunc_environment(
    directory: Path, requirements: list[str], *, spark_certified: bool
) -> tuple[dict[str, Any], str | None]:
    """Prepare local save options and snapshot runtime code only for certified workers."""
    import mlflow  # noqa: PLC0415 - optional packaging boundary  # ty: ignore[unresolved-import]

    options: dict[str, Any] = {"pip_requirements": requirements}
    if "uv_project_path" in inspect.signature(mlflow.pyfunc.save_model).parameters:
        options["uv_project_path"] = str(directory)
    source_sha256 = None
    if spark_certified:
        code_paths, options["pip_requirements"], source_sha256 = snapshot_worker_environment(
            directory, requirements
        )
        options["code_paths"] = code_paths
    return options, source_sha256


def package_source_root() -> Path:
    """Locate installed Skyulf sources independently of adapter package nesting."""
    return Path(skyulf.__file__).resolve().parent


def source_digest(root: Path) -> str:
    """Hash package-relative Python paths and bytes independently of the install path."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _snapshot_source(destination: Path) -> Path:
    """Copy package source and resources while excluding machine bytecode caches."""
    root = package_source_root()
    snapshot = destination / "skyulf"
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError("Spark worker source snapshot cannot include symlinks.")
        if path.is_file():
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return snapshot


def _wheel_entries(snapshot: Path, metadata: str, version: str) -> dict[str, bytes]:
    """Build portable wheel entries without copying producer RECORD or entrypoints."""
    entries = {
        f"skyulf/{path.relative_to(snapshot).as_posix()}": path.read_bytes()
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    }
    info = f"skyulf_core-{version}.dist-info"
    entries[f"{info}/METADATA"] = metadata.encode("utf-8")
    entries[f"{info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: skyulf-spark-snapshot\n"
        b"Root-Is-Purelib: true\nTag: py3-none-any\n"
    )
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, content in sorted(entries.items()):
        checksum = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
        writer.writerow([name, f"sha256={checksum}", len(content)])
    writer.writerow([f"{info}/RECORD", "", ""])
    entries[f"{info}/RECORD"] = record.getvalue().encode("utf-8")
    return entries


def snapshot_worker_environment(
    directory: Path,
    requirements: list[str],
) -> tuple[list[str], list[str], str]:
    """Ship exact runtime code as an installable wheel referenced relative to MLmodel.

    MLflow restores virtualenv requirements with the model directory available,
    so ``code/<wheel>.whl`` avoids looking up unpublished Skyulf builds on PyPI.
    Other saved requirements and the installed distribution metadata are retained.
    The adjacent source directory also supports explicit ``env_manager='local'``.
    """
    package = distribution("skyulf-core")
    metadata = package.read_text("METADATA") or package.read_text("PKG-INFO")
    if metadata is None:
        raise ValueError("Spark worker packaging requires installed Skyulf distribution metadata.")
    pin = f"skyulf-core=={package.version}"
    if pin not in requirements:
        raise ValueError("Spark worker runtime pin differs from installed Skyulf metadata.")
    snapshot = _snapshot_source(directory)
    digest = source_digest(snapshot)
    wheel = directory / f"skyulf_core-{package.version}-0{digest[:16]}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(_wheel_entries(snapshot, metadata, package.version).items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, content)
    pins = [
        f"code/{wheel.name}" if requirement == pin else requirement for requirement in requirements
    ]
    return [str(snapshot), str(wheel)], pins, digest
