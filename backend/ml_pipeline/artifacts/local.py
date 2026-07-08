import logging
import os
from pathlib import Path
from typing import Any

import joblib

from .store import ArtifactStore

logger = logging.getLogger(__name__)


class LocalArtifactStore(ArtifactStore):
    def __init__(self, base_path: str):
        self.base_path = base_path
        Path(self.base_path).mkdir(parents=True, exist_ok=True)

    def _get_path(self, key: str) -> str:
        # Ensure key is safe and ends with .joblib if not present
        safe_key = key.replace("/", "_").replace("\\", "_")
        if not safe_key.endswith(".joblib"):
            safe_key += ".joblib"
        resolved = os.path.realpath(os.path.join(self.base_path, safe_key))
        base = os.path.realpath(self.base_path)
        if not resolved.startswith(base + os.sep) and resolved != base:
            raise PermissionError(f"Access denied: artifact key '{key}' resolves outside the store")
        return resolved

    def save(self, key: str, data: Any) -> None:
        path = self._get_path(key)
        joblib.dump(data, path)

    def load(self, key: str) -> Any:
        """Load a joblib artifact.

        Warning:
            ``joblib.load`` uses pickle internally and can execute arbitrary code.
            Only load artifacts that were saved by this application.
        """
        path = self._get_path(key)
        if not Path(path).exists():
            raise FileNotFoundError(f"Artifact not found: {key}")
        logger.debug("Loading artifact from %s", path)
        return joblib.load(path)

    def exists(self, key: str) -> bool:
        path = self._get_path(key)
        return Path(path).exists()

    def list_artifacts(self) -> list[str]:
        """List all artifacts in the store."""
        if not Path(self.base_path).exists():
            return []
        # Return keys (filenames without extension if possible, or just filenames)
        # We appended .joblib in _get_path, so we should strip it if present
        files = [f.name for f in Path(self.base_path).iterdir()]
        keys = []
        for f in files:
            if f.endswith(".joblib"):
                keys.append(f[:-7])  # Remove .joblib
            else:
                keys.append(f)
        return keys

    def get_artifact_uri(self, key: str) -> str:
        """Get the full URI/Path for a given artifact key."""
        return self._get_path(key)
