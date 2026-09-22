"""Single-host publish admission; distributed providers must supply the same contract."""

import errno
import hashlib
import importlib
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Protocol


class BatchConflictError(RuntimeError):
    """Another writer or request owns the expected target state."""


class PublishAdmission(Protocol):
    """Hold an exclusive table lock until the commit receipt has been read.

    All publishers must use the same authority keyed by immutable Delta table ID.
    Ownership must not expire while the context is active. This API does not
    pass fencing tokens to Delta; time-limited distributed leases are unsupported.
    """

    local_only: bool

    def hold(self, table_id: str) -> AbstractContextManager[None]:
        """Reject a competing writer, releasing ownership on context exit."""
        ...


def validate_admission(spark: Any, admission: PublishAdmission | None) -> PublishAdmission:
    """Reject absent admission or local locks on a distributed runtime before I/O."""
    if admission is None or not callable(getattr(admission, "hold", None)):
        raise ValueError("A shared publish admission is required.")
    if admission.local_only:
        master = spark.sparkContext.master
        if master != "local" and not master.startswith("local["):
            raise ValueError("LocalTableLock cannot coordinate distributed drivers.")
    return admission


class LocalTableLock:
    """Use OS file locks for local Spark drivers sharing one lock directory.

    This implementation is deliberately rejected for distributed Spark runtimes.
    Lock files remain on disk: unlinking them can split the lock authority.
    OS ownership is released when a process exits, including abnormal exits.
    """

    local_only = True

    def __init__(self, directory: str | Path) -> None:
        """Bind admission to a caller-owned local directory."""
        self.directory = Path(directory).resolve()

    @contextmanager
    def hold(self, table_id: str) -> Iterator[None]:
        """Acquire a nonblocking exclusive lock for the whole Delta table."""
        self.directory.mkdir(parents=True, exist_ok=True)
        filename = hashlib.sha256(table_id.encode()).hexdigest() + ".lock"
        with (self.directory / filename).open("a+b") as handle:
            windows = os.name == "nt"
            api = importlib.import_module("msvcrt" if windows else "fcntl")
            if windows:
                if handle.tell() == 0:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
            try:
                if windows:
                    api.locking(handle.fileno(), api.LK_NBLCK, 1)
                else:
                    api.flock(handle.fileno(), api.LOCK_EX | api.LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
                raise BatchConflictError("Another publisher holds this table's admission.") from exc
            try:
                yield
            finally:
                if windows:
                    handle.seek(0)
                    api.locking(handle.fileno(), api.LK_UNLCK, 1)
                else:
                    api.flock(handle.fileno(), api.LOCK_UN)
