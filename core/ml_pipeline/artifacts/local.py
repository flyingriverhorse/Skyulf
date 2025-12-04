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
