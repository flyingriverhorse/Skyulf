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
    """Control publication until the commit receipt has been read.

    Shared providers must use the same authority keyed by immutable Delta table
    ID. ``SingleWriterAdmission`` instead relies on caller-enforced exclusive
    write access and serialized job runs. This API does not pass fencing tokens
    to Delta; time-limited distributed leases are unsupported.
    """

    local_only: bool

    def hold(self, table_id: str) -> AbstractContextManager[None]:
        """Reject a competing writer, releasing ownership on context exit."""
        ...


def validate_admission(spark: Any, admission: PublishAdmission | None) -> PublishAdmission:
    """Require an explicit shared or externally enforced single-writer policy."""
    if admission is None or not callable(getattr(admission, "hold", None)):
        raise ValueError("An explicit publish admission is required.")
    if admission.local_only:
        master = spark.sparkContext.master
        if master != "local" and not master.startswith("local["):
            raise ValueError("LocalTableLock cannot coordinate distributed drivers.")
    return admission


class SingleWriterAdmission:
    """Opt in to table-free publication when one external writer is guaranteed.

    This provider does not lock. Use it only when one job owns all target writes,
    its runs cannot overlap, and other principals cannot modify the target.
    Delta receipts and transaction IDs still detect/replay committed increments,
    but they do not replace cross-job admission if another writer is allowed.
    """

    local_only = False

    @contextmanager
    def hold(self, table_id: str) -> Iterator[None]:
        """Enter the caller's externally serialized single-writer scope."""
        if type(table_id) is not str or not table_id:
            raise ValueError("Single-writer publication needs a target table ID.")
        yield


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
