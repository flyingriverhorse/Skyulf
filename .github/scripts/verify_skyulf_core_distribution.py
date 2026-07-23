"""Verify that a built skyulf-core wheel works outside its source checkout."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def run(command: list[str], *, cwd: Path) -> None:
    """Run a checked command in the isolated verification directory."""
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    """Install the supplied wheel into a fresh venv and import public APIs."""
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_skyulf_core_distribution.py <wheel-path>")

    wheel = Path(sys.argv[1]).resolve()
    if wheel.suffix != ".whl" or not wheel.is_file():
        raise SystemExit(f"wheel does not exist: {wheel}")

    with tempfile.TemporaryDirectory(prefix="skyulf-core-wheel-") as directory:
        root = Path(directory)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        executable = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        python = environment / executable

        run([str(python), "-m", "pip", "install", str(wheel)], cwd=root)
        run(
            [
                str(python),
                "-c",
                (
                    "from skyulf import ("
                    "SkyulfPipeline, EDAAnalyzer, DriftCalculator, NodeRegistry"
                    "); print('skyulf-core wheel import passed')"
                ),
            ],
            cwd=root,
        )


if __name__ == "__main__":
    main()
