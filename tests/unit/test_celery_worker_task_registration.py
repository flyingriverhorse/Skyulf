"""Pin that the Celery worker imports every module that registers a task.

Task registration in this repo is import-only: ``backend/celery_app.py`` declares
no tasks and configures no autodiscovery, so a task module that
``celery_worker.py`` never imports is invisible to the worker even though the web
process can still enqueue it. The worker then rejects the message with
"Received unregistered task" and the job hangs in PENDING forever.

``backend.eda.tasks`` was missing this way — ``eda.generate_profile`` enqueued
fine from ``eda/router.py`` and never ran. It stayed hidden because the default
``USE_CELERY=False`` uses the FastAPI ``BackgroundTasks`` path instead.

The checks are static (AST over source) on purpose: importing the task modules
would register them with the shared app and make the very gap under test
disappear.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "backend"
WORKER = REPO_ROOT / "celery_worker.py"

# `@celery_app.task` / `@shared_task`, bare or called. Matched on the attribute
# name so any app object is covered, not just the one spelled `celery_app`.
_TASK_DECORATORS = {"task", "shared_task"}

# The modules known to register tasks today. Asserted explicitly so a change in
# decorator style that silently empties the discovery cannot pass the real test.
_EXPECTED_TASK_MODULES = {
    "backend.data_ingestion.tasks",
    "backend.eda.tasks",
    "backend.ml_pipeline.tasks",
    "backend.monitoring.tasks",
}


def _registers_tasks(path: Path) -> bool:
    """Return True if any function in the module carries a Celery task decorator.

    Handles both spellings in use: ``@celery_app.task`` (attribute access) and
    ``@shared_task`` (a bare imported name), each bare or called.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = getattr(target, "attr", None) or getattr(target, "id", None)
            if name in _TASK_DECORATORS:
                return True
    return False


def _task_modules() -> set[str]:
    """Dotted names of every module under ``backend`` that registers a task."""
    return {
        ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
        for path in sorted(BACKEND.rglob("*.py"))
        if _registers_tasks(path)
    }


def _worker_imports() -> set[str]:
    """Dotted names reachable from ``celery_worker.py``'s import statements.

    Both halves of a ``from package import module`` are recorded, so
    ``from backend.eda import tasks as _eda_tasks`` yields ``backend.eda.tasks``.
    """
    tree = ast.parse(WORKER.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_discovery_finds_the_known_task_modules():
    """Guard the guard: discovery must find today's four task modules.

    Without this, a decorator style the scanner does not recognise would empty
    ``_task_modules()`` and the registration test below would pass vacuously.
    """
    assert _task_modules() == _EXPECTED_TASK_MODULES


def test_worker_imports_every_task_registering_module():
    """Every task module must be imported by the worker, or its tasks never run."""
    missing = sorted(_task_modules() - _worker_imports())
    assert not missing, (
        f"celery_worker.py does not import {missing}; tasks registered there are "
        "invisible to the worker and will be rejected as unregistered. Add "
        "`from <package> import tasks as _<name>_tasks  # noqa: F401` alongside "
        "the existing task imports."
    )
