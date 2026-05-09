"""Per-run capture of node-level `logger.warning(...)` calls.

The skyulf-core nodes (TargetEncoder, OneHotEncoder, Resampling, ...) emit
soft advisories via the standard ``logging`` module. Until now those went
straight to the server log and the user never saw them. This module
provides a lightweight ``logging.Handler`` that buffers warnings emitted
under the ``skyulf.*`` logger tree while a pipeline is running, tagged
with the currently-executing node id, so the engine can return them on
``PipelineExecutionResult.node_warnings`` and the UI can surface them as
toasts / a notification panel.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# Logger trees we want to mirror to the user. ``skyulf`` covers all
# preprocessing / modeling node logs; ``backend.ml_pipeline`` covers
# engine-level advisories that aren't already routed via merge_warnings.
_CAPTURED_LOGGERS = ("skyulf", "backend.ml_pipeline")


class WarningCaptureHandler(logging.Handler):
    """Buffer ``WARNING``+ records emitted by skyulf-core during a run.

    Usage::

        handler = WarningCaptureHandler()
        with handler.attach():
            engine.run(...)
        warnings = handler.drain()
    """

    def __init__(self, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self._buffer: List[Dict[str, Any]] = []
        self._current_node_id: Optional[str] = None
        self._current_node_type: Optional[str] = None
        # Track which loggers we attached to so we can detach cleanly.
        self._attached: List[logging.Logger] = []

    # ------------------------------------------------------------------
    # Engine integration
    # ------------------------------------------------------------------

    def set_current_node(self, node_id: Optional[str], node_type: Optional[str]) -> None:
        """Tag subsequent warnings with this node id (called by the engine
        before each ``_execute_node``). Pass ``None`` to clear."""
        self._current_node_id = node_id
        self._current_node_type = node_type

    def drain(self) -> List[Dict[str, Any]]:
        """Return and clear the buffered warnings."""
        out = self._buffer
        self._buffer = []
        return out

    # ------------------------------------------------------------------
    # logging.Handler implementation
    # ------------------------------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        try:
            msg = record.getMessage()
        except Exception:
            return
        self._buffer.append(
            {
                "node_id": self._current_node_id,
                "node_type": self._current_node_type,
                "level": record.levelname.lower(),
                "logger": record.name,
                "message": msg,
            }
        )

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self) -> "WarningCaptureHandler":
        """Attach this handler to the captured logger trees and return self
        so callers can use it as a context manager."""
        for name in _CAPTURED_LOGGERS:
            lg = logging.getLogger(name)
            lg.addHandler(self)
            self._attached.append(lg)
        return self

    def detach(self) -> None:
        for lg in self._attached:
            try:
                lg.removeHandler(self)
            except Exception:
                pass
        self._attached.clear()

    # Context-manager sugar so engine code can do ``with handler.attach():``.
    def __enter__(self) -> "WarningCaptureHandler":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.detach()


__all__ = ["WarningCaptureHandler"]
