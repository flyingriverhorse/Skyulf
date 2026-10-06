"""Prepare and verify the declared Core wheel before Bundle deployment."""

import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def validate_wheel(wheel: Path, expected_version: str) -> None:
    """Reject the wrong distribution or release before changing deployment files."""
    with zipfile.ZipFile(wheel) as archive:
        _validate_runtime_layout(archive.namelist())
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise ValueError("Core wheel must contain exactly one package metadata record.")
        message = BytesParser().parsebytes(archive.read(metadata[0]))
    if message["Name"].lower().replace("_", "-") != "skyulf-core":
        raise ValueError("Expected a skyulf-core wheel.")
    if message["Version"] != expected_version:
        raise ValueError(
            f"Core wheel version {message['Version']} differs from {expected_version}."
        )
    if wheel.name != f"skyulf_core-{expected_version}-py3-none-any.whl":
        raise ValueError("Core wheel filename differs from its declared release.")


def _validate_runtime_layout(names: list[str]) -> None:
    """Reject cached flat modules that would shadow compatibility aliases after a move."""
    _validate_integration_layout(
        names, "databricks", "Databricks", ("jobs/job_runtime.py", "jobs/shared/job_runtime.py")
    )
    _validate_integration_layout(names, "mlflow", "MLflow", ("runs/tracking.py",))


def _validate_integration_layout(
    names: list[str], package: str, label: str, markers: tuple[str, ...]
) -> None:
    """Keep legacy releases valid while rejecting mixed layouts in reorganized wheels."""
    prefix = f"skyulf/integrations/{package}/"
    if not any(prefix + marker in names for marker in markers):
        return
    flat = [name.removeprefix(prefix) for name in names if name.startswith(prefix)]
    if any(
        "/" not in name and name != "__init__.py" and name.endswith((".py", ".pyc"))
        for name in flat
    ):
        raise ValueError(
            f"Core wheel contains stale flat {label} modules. Clean the source checkout's "
            "generated build directory and rebuild the wheel before deployment."
        )
    _validate_unique_runtime_modules(flat, label)


def _validate_unique_runtime_modules(names: list[str], label: str) -> None:
    """Reject previous package locations retained by an incremental wheel build."""
    modules = [
        Path(name).stem
        for name in names
        if name.endswith((".py", ".pyc"))
        and not name.startswith("_compat/")
        and Path(name).stem != "__init__"
    ]
    if len(modules) != len(set(modules)):
        raise ValueError(
            f"Core wheel contains stale duplicate {label} modules. Clean the source "
            "checkout's generated build directory and rebuild the wheel before deployment."
        )


def prepare_source(source: Path, staging: Path) -> Path:
    """Build a source checkout or stage a caller-selected release wheel."""
    if source.is_dir():
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(staging), str(source)],
            check=True,
        )
    elif source.is_file() and source.suffix == ".whl":
        shutil.copyfile(source, staging / source.name)
    else:
        raise ValueError("Set deployment/artifact.json source to a Core source directory or wheel.")
    wheels = list(staging.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Core artifact build must produce exactly one wheel.")
    return wheels[0]


def build(project: Path) -> dict[str, str]:
    """Publish only a verified wheel and its content digest to the owned artifact directory."""
    project = project.resolve()
    settings = json.loads((project / "deployment/artifact.json").read_text(encoding="utf-8"))
    source = Path(settings["source"])
    source = (project / source).resolve() if not source.is_absolute() else source.resolve()
    output = project / "dist/skyulf"
    if not output.resolve().is_relative_to(project):
        raise ValueError("Core artifact output must remain within the project.")
    if source.is_relative_to(output.resolve()):
        raise ValueError("Keep the Core source or release wheel outside dist/skyulf.")
    with tempfile.TemporaryDirectory(prefix="skyulf-wheel-") as temporary:
        wheel = prepare_source(source, Path(temporary))
        validate_wheel(wheel, settings["version"])
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        # A wheel build tag changes its cache identity without changing the
        # distribution version required by already registered model manifests.
        filename = f"skyulf_core-{settings['version']}-1{digest}-py3-none-any.whl"
        receipt = {
            "version": settings["version"],
            "source_filename": wheel.name,
            "filename": filename,
            "sha256": digest,
        }
        output.mkdir(parents=True, exist_ok=True)
        staged = output / (filename + ".tmp")
        shutil.copyfile(wheel, staged)
        staged.replace(output / filename)
        for previous in output.glob("skyulf_core-*.whl"):
            if previous.name != filename:
                previous.unlink()
        (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    print(json.dumps(build(Path(__file__).resolve().parents[2]), indent=2))
