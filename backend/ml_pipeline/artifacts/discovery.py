"""Artifact discovery seam.

Monitoring needs to *discover* job folders and their reference-data artifacts at the
root of the artifact location. The :class:`ArtifactStore` ABC only operates within a
single job store, so discovery (a root-level scan) lives here behind its own
abstraction.

This keeps ``monitoring`` free of direct ``Path.iterdir()`` calls so a future
``UCVolumeArtifactStore`` / S3 backend can drop in without touching the routers.
Local behaviour is identical to the previous inline filesystem scan.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from backend.ml_pipeline.artifacts.local import LocalArtifactStore
from backend.ml_pipeline.artifacts.store import ArtifactStore

logger = logging.getLogger(__name__)

# Folder timestamp pattern, e.g. "mydata_20240131_120000_<job_id>".
_TIMESTAMP_RE = re.compile(r"(\d{8})_(\d{6})")
_REFERENCE_PREFIX = "reference_data_"


@dataclass(frozen=True)
class ReferenceArtifact:
    """A reference-data artifact discovered under the artifact root."""

    job_id: str
    dataset_name: str
    filename: str
    created_at: Optional[str]
    folder: str


class ArtifactDiscovery(ABC):
    """Root-level discovery of job folders and their reference artifacts."""

    @abstractmethod
    def list_reference_artifacts(self) -> List[ReferenceArtifact]:
        """List every ``reference_data_*`` artifact across all job folders."""

    @abstractmethod
    def get_store_for_job(self, job_id: str) -> ArtifactStore:
        """Return the artifact store rooted at the folder owning ``job_id``.

        Falls back to a store rooted at the artifact root for backward
        compatibility when no matching subfolder is found.
        """


class LocalArtifactDiscovery(ArtifactDiscovery):
    """Filesystem implementation rooted at ``TRAINING_ARTIFACT_DIR``."""

    def __init__(self, root_path: str):
        self.root_path = Path(root_path).expanduser().resolve()

    @staticmethod
    def _parse_created_at(folder_name: str) -> Optional[str]:
        match = _TIMESTAMP_RE.search(folder_name)
        if not match:
            return None
        try:
            from datetime import datetime

            dt = datetime.strptime(f"{match.group(1)}{match.group(2)}", "%Y%m%d%H%M%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None

    def list_reference_artifacts(self) -> List[ReferenceArtifact]:
        if not self.root_path.exists():
            return []

        artifacts: List[ReferenceArtifact] = []
        try:
            root_items = list(self.root_path.iterdir())
        except OSError:
            logger.warning("Could not scan artifact root %s", self.root_path, exc_info=True)
            return []

        for item_path in root_items:
            if not item_path.is_dir():
                continue
            created_at = self._parse_created_at(item_path.name)
            try:
                for file_path in item_path.glob(f"{_REFERENCE_PREFIX}*.joblib"):
                    remainder = file_path.stem[len(_REFERENCE_PREFIX) :]
                    parts = remainder.rsplit("_", 1)
                    if len(parts) != 2:
                        continue
                    dataset_name, job_id = parts
                    artifacts.append(
                        ReferenceArtifact(
                            job_id=job_id,
                            dataset_name=dataset_name,
                            filename=file_path.name,
                            created_at=created_at,
                            folder=str(item_path),
                        )
                    )
            except OSError:
                logger.warning("Could not scan artifact folder %s", item_path, exc_info=True)
                continue
        return artifacts

    def get_store_for_job(self, job_id: str) -> ArtifactStore:
        job_folder = str(self.root_path)
        if self.root_path.exists():
            try:
                for item_path in self.root_path.iterdir():
                    if item_path.is_dir() and (
                        item_path.name == job_id or item_path.name.endswith(f"_{job_id}")
                    ):
                        job_folder = str(item_path)
                        break
            except OSError:
                logger.warning("Could not inspect artifact root %s", self.root_path, exc_info=True)
        return LocalArtifactStore(base_path=job_folder)
