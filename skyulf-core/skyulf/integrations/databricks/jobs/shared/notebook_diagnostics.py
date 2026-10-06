"""Human-readable task progress and errors, independent of machine-readable receipts."""

import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from time import monotonic
from typing import Any

_CONTEXT_FIELDS = ("job_id", "job_run_id", "task_key", "model_key", "bundle_target")
_DETAIL_FIELDS = {"reference", "reference_json", "config", "pipeline", "request"}


def _visible_items(payload: dict) -> Iterator[tuple[str, Any]]:
    """Keep meaningful false/zero results while omitting absent options and receipt hashes."""
    for key, value in payload.items():
        if str(key).endswith(("sha256", "digest")) or key in _DETAIL_FIELDS:
            continue
        if value is None or (isinstance(value, (str, dict, list, tuple)) and len(value) == 0):
            continue
        yield str(key).replace("_", " ").capitalize(), value


def _summary_lines(payload: dict, *, depth: int = 0) -> list[str]:
    """Format small nested result sections without flooding the notebook with raw receipts."""
    lines: list[str] = []
    prefix = "  " * depth
    for label, value in _visible_items(payload):
        if isinstance(value, dict):
            children = _summary_lines(value, depth=depth + 1) if depth < 3 else []
            if children:
                lines.extend([f"{prefix}{label}:", *children])
        elif isinstance(value, list):
            lines.extend(_list_lines(label, value, depth))
        else:
            lines.append(f"{prefix}{label}: {_short_value(value)}")
    return lines


def _list_lines(label: str, values: list, depth: int) -> list[str]:
    """Bound long leaderboards and reports while making omitted details explicit."""
    prefix = "  " * depth
    lines = [f"{prefix}{label} ({len(values)}):"]
    for value in values[:10]:
        if isinstance(value, dict):
            lines.extend(
                _summary_lines(value, depth=depth + 1)
                if depth < 3
                else [f"{prefix}  Details in the JSON result."]
            )
            lines.append("")
        elif value is not None:
            lines.append(f"{prefix}  - {_short_value(value)}")
    if len(values) > 10:
        lines.append(f"{prefix}  ... {len(values) - 10} more entries in the JSON result.")
    return lines


def _short_value(value: Any) -> str:
    """Point to the full result when a displayed scalar has been abbreviated."""
    text = str(value)
    return text if len(text) <= 500 else text[:500] + "... (see JSON result)"


def _error_location(error: BaseException) -> str:
    """Identify the raising frame without substituting the diagnostic wrapper."""
    frames = traceback.extract_tb(error.__traceback__)
    if not frames:
        return "See original traceback"
    origin = frames[-1]
    return f"{origin.filename}:{origin.lineno} ({origin.name})"


def result_summary(payload: dict[str, Any]) -> str:
    """Return a presentation-only summary; the original payload is never modified."""
    lines = _summary_lines(payload)
    return "\n".join(
        ["Result", "------", *lines, "Full details: notebook JSON result and saved run artifacts."]
    )


def _task_context(dbutils: Any) -> str:
    """Read only diagnostic identifiers; missing widgets must not mask task failures."""
    try:
        values = dbutils.widgets.getAll()
        return " | ".join(f"{key}={values[key]}" for key in _CONTEXT_FIELDS if values.get(key))
    except Exception:  # noqa: BLE001 - diagnostics must work even when widgets are unavailable
        return ""


@contextmanager
def notebook_task(name: str, dbutils: Any) -> Iterator[None]:
    """Identify a task failure without replacing its exception, traceback or retry semantics.

    Keep dbutils.notebook.exit outside this context: it is a separate notebook
    cell, not part of task execution. No job parameters or receipt bodies are dumped.
    """
    started = monotonic()
    context = _task_context(dbutils)
    print(f"STARTED | {name}" + (f" | {context}" if context else ""), flush=True)
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - annotate then re-raise the identical task failure
        location = _error_location(exc)
        print(f"FAILED | {name} | {type(exc).__name__}: {exc}", flush=True)
        print(f"Source: {location}\nElapsed: {monotonic() - started:.2f}s", flush=True)
        if exc.__cause__ is not None:
            cause = exc.__cause__
            print(f"Caused by: {type(cause).__name__}: {cause}", flush=True)
            print(f"Cause source: {_error_location(cause)}", flush=True)
        print("Inspect the traceback below and this task's inputs before retrying.", flush=True)
        exc.add_note(f"Skyulf task: {name}; {context}; source: {location}")
        raise
    else:
        print(f"COMPLETED | {name} | Elapsed: {monotonic() - started:.2f}s", flush=True)
