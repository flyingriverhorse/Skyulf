"""Optional package sources for logged models: a private index, or wheels saved in the model.

Model Serving builds its container from the model's requirements. When public PyPI is
blocked, set these environment variables on the job that logs the model (for example
``spark_env_vars`` in the ``job_cluster`` Bundle variable):

- ``SKYULF_MODEL_PIP_INDEX_URL`` / ``SKYULF_MODEL_PIP_EXTRA_INDEX_URLS``: written as
  ``--index-url`` / ``--extra-index-url`` lines (space-separated extras).
- ``SKYULF_MODEL_PIP_BUNDLE_WHEELS=true``: download every dependency wheel for Linux
  x86_64 and this Python into the model and install with ``--no-index``. Use this when
  the serving build network cannot reach any index. The index variables are then
  used only for the download.

Unset, requirements are unchanged.
"""

import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from urllib.parse import urlsplit

INDEX_URL = "SKYULF_MODEL_PIP_INDEX_URL"
EXTRA_INDEX_URLS = "SKYULF_MODEL_PIP_EXTRA_INDEX_URLS"
BUNDLE_WHEELS = "SKYULF_MODEL_PIP_BUNDLE_WHEELS"
MAX_BUNDLE_BYTES = 2 * 1024**3
# Serving images are glibc-based Linux x86_64; pure-Python wheels match any platform.
_PLATFORMS = ("manylinux2014_x86_64", "manylinux_2_17_x86_64", "manylinux_2_28_x86_64")

Downloader = Callable[[list[str], Path, list[str]], None]


def pip_download(requirements: list[str], target: Path, index_args: list[str]) -> None:
    """Download binary wheels, with dependencies, for the serving platform and Python."""
    version = f"{sys.version_info.major}{sys.version_info.minor}"
    platforms = [part for name in _PLATFORMS for part in ("--platform", name)]
    command = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary=:all:",
        "--python-version",
        version,
        *platforms,
        "--dest",
        str(target),
        *index_args,
        *requirements,
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-2000:]
        raise RuntimeError(f"Downloading model dependency wheels failed: {detail}")


def _index_url(value: str) -> str:
    """Accept a plain http(s) index URL; requirement lines are stored in the registry."""
    parts = urlsplit(value)
    if parts.scheme not in {"https", "http"} or not parts.hostname:
        raise ValueError(f"{INDEX_URL}/{EXTRA_INDEX_URLS} need http(s) URLs: {value!r}.")
    if parts.username or parts.password:
        raise ValueError(
            f"{INDEX_URL}/{EXTRA_INDEX_URLS} cannot contain credentials; they would be "
            "saved with the model. Use pip configuration on the cluster instead."
        )
    return value


def _settings(environ: Mapping[str, str]) -> tuple[list[str], bool]:
    """Read pip index arguments and the bundle switch from the environment."""
    index = [("--index-url", environ[INDEX_URL].strip())] if environ.get(INDEX_URL) else []
    extras = environ.get(EXTRA_INDEX_URLS, "").split()
    pairs = [*index, *(("--extra-index-url", url) for url in extras)]
    flag = environ.get(BUNDLE_WHEELS, "false")
    if flag not in {"true", "false"}:
        raise ValueError(f"{BUNDLE_WHEELS} must be true or false.")
    return [part for option, url in pairs for part in (option, _index_url(url))], flag == "true"


def apply_package_sources(
    directory: Path,
    requirements: list[str],
    *,
    environ: Mapping[str, str] = os.environ,
    download: Downloader = pip_download,
) -> tuple[list[str], list[str]]:
    """Return the model requirements and extra ``code_paths`` for the configured sources."""
    index_args, bundle = _settings(environ)
    if not bundle:
        lines = [f"{index_args[i]} {index_args[i + 1]}" for i in range(0, len(index_args), 2)]
        return [*lines, *requirements], []
    return _bundle_wheels(Path(directory), requirements, index_args, download)


def _bundle_wheels(
    directory: Path, requirements: list[str], index_args: list[str], download: Downloader
) -> tuple[list[str], list[str]]:
    """Save dependency wheels under the model and point requirements at them offline."""
    local = [line for line in requirements if line.startswith("code/")]
    remote = [line for line in requirements if not line.startswith("code/")]
    wheels = directory / "wheels"
    wheels.mkdir()
    saved = [str(directory / line.removeprefix("code/")) for line in local]
    download([*remote, *saved], wheels, index_args)
    for line in local:  # Already saved under code/; keep one copy.
        (wheels / Path(line).name).unlink(missing_ok=True)
    files = sorted(wheels.glob("*.whl"))
    if sum(path.stat().st_size for path in files) > MAX_BUNDLE_BYTES:
        raise ValueError(f"Model dependency wheels exceed {MAX_BUNDLE_BYTES} bytes.")
    lines = ["--no-index", *local, *(f"code/wheels/{path.name}" for path in files)]
    return lines, [str(wheels)]
