"""Optional MLflow adapters; importing this package does not import MLflow."""

from pathlib import Path

# Resolve released module names lazily without loading optional model adapters.
__path__ = [*__path__, str(Path(__file__).with_name("_compat"))]

from .runs.tracking import TrackingConfig, TrackingRun, track_run

__all__ = ["TrackingConfig", "TrackingRun", "track_run"]
