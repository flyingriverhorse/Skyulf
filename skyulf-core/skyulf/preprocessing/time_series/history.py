"""Explicit, immutable continuation state for bounded temporal preprocessing.

The caller owns persistence. A session proposes state only after its body succeeds;
publish that state together with the predictions, never before them.
"""

import json
from contextvars import ContextVar
from copy import deepcopy
from typing import Any

_ACTIVE: ContextVar[Any] = ContextVar("skyulf_temporal_history", default=None)


class TemporalHistorySession:
    """Collect per-step next histories without modifying a fitted model or input state."""

    def __init__(self, model_id: str, state: dict[str, Any] | None = None) -> None:
        """Bind continuation to an immutable model identity supplied by the caller."""
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Temporal history requires an immutable model identity.")
        if state is not None and len(json.dumps(state, allow_nan=False).encode()) > 4 * 1024 * 1024:
            raise ValueError("Temporal history session exceeds the 4 MiB state budget.")
        if state is not None and (
            state.get("version") != 1
            or state.get("model_id") != model_id
            or not isinstance(state.get("steps"), dict)
        ):
            raise ValueError("Temporal history belongs to a different model or format.")
        self.model_id = model_id
        self.continuing = state is not None
        self.previous = deepcopy(state["steps"]) if state is not None else {}
        self.proposed: dict[str, Any] = {}
        self.state: dict[str, Any] | None = None
        self._token: Any = None

    def __enter__(self) -> "TemporalHistorySession":
        """Activate this session for synchronous preprocessing calls in the context."""
        if _ACTIVE.get() is not None or self._token is not None:
            raise ValueError("Temporal history sessions cannot be nested or reused.")
        self._token = _ACTIVE.set(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Expose a detached proposal only if the complete operation succeeded."""
        _ACTIVE.reset(self._token)
        if exc_type is None:
            if set(self.previous) - set(self.proposed):
                raise ValueError("Temporal history contains steps absent from this prediction.")
            self.state = {"version": 1, "model_id": self.model_id, "steps": deepcopy(self.proposed)}


def current_history(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the original state on every prediction pass, including predict_proba."""
    session = _ACTIVE.get()
    if session is None:
        return params["history_seed"]
    if session.continuing and params["history_id"] not in session.previous:
        raise ValueError("Temporal history is missing a fitted step's context.")
    return session.previous.get(params["history_id"], params["history_seed"])


def propose_history(params: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Record one deterministic proposal; repeated model passes must agree."""
    session = _ACTIVE.get()
    if session is None:
        return
    key = params["history_id"]
    previous = session.proposed.get(key)
    if previous is not None and previous != rows:
        raise ValueError("Temporal steps share a history identity but received different rows.")
    session.proposed[key] = rows
