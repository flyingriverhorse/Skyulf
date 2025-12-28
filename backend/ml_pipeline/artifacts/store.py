from abc import ABC, abstractmethod
from typing import Any


class ArtifactStore(ABC):
    @abstractmethod
    def save(self, key: str, data: Any) -> None:
        """Save data to the store with the given key."""
        pass

    @abstractmethod
    def load(self, key: str) -> Any:
        """Load data from the store using the given key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the store."""
        pass

    @abstractmethod
    def list_artifacts(self) -> list[str]:
        """List all artifacts in the store."""
        pass

    @abstractmethod
    def get_artifact_uri(self, key: str) -> str:
        """Get the full URI/Path for a given artifact key."""
        pass
