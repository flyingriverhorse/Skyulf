import os
import joblib
from typing import Any
from .store import ArtifactStore

class LocalArtifactStore(ArtifactStore):
    def __init__(self, base_path: str):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)

    def _get_path(self, key: str) -> str:
        # Ensure key is safe and ends with .joblib if not present
        safe_key = key.replace("/", "_").replace("\\", "_")
        if not safe_key.endswith(".joblib"):
            safe_key += ".joblib"
        return os.path.join(self.base_path, safe_key)

    def save(self, key: str, data: Any) -> None:
        path = self._get_path(key)
        joblib.dump(data, path)

    def load(self, key: str) -> Any:
        path = self._get_path(key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Artifact not found: {key}")
        return joblib.load(path)

    def exists(self, key: str) -> bool:
        path = self._get_path(key)
        return os.path.exists(path)

    def list_artifacts(self) -> list[str]:
        """List all artifacts in the store."""
        if not os.path.exists(self.base_path):
            return []
        # Return keys (filenames without extension if possible, or just filenames)
        # We appended .joblib in _get_path, so we should strip it if present
        files = os.listdir(self.base_path)
        keys = []
        for f in files:
            if f.endswith(".joblib"):
                keys.append(f[:-7]) # Remove .joblib
            else:
                keys.append(f)
        return keys
