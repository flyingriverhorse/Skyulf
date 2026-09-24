"""Cross-process admission checks for the local Delta harness."""

import os
import subprocess
import sys
from types import SimpleNamespace


def test_independent_process_cannot_enter_held_table_lock(tmp_path):
    """Different local drivers must serialize even when periods overlap partially."""
    from skyulf.integrations.databricks.admission import LocalTableLock

    lock = LocalTableLock(tmp_path / "locks")
    script = """
import sys
from skyulf.integrations.databricks.admission import LocalTableLock, BatchConflictError
try:
    with LocalTableLock(sys.argv[1]).hold("table-id"): pass
except BatchConflictError:
    print("BUSY")
else:
    print("ACQUIRED")
"""
    with lock.hold("table-id"):
        child = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path / "locks")],
            capture_output=True,
            text=True,
            check=True,
            env=os.environ.copy(),
        )
    assert child.stdout.strip() == "BUSY"
    with lock.hold("table-id"):
        assert (tmp_path / "locks").is_dir()


def test_explicit_single_writer_path_needs_no_control_table(tmp_path):
    """A caller-controlled sole writer can publish without provisioning lock state."""
    from skyulf.integrations.databricks.admission import (
        SingleWriterAdmission,
        validate_admission,
    )

    spark = SimpleNamespace(sparkContext=SimpleNamespace(master="spark://remote"))
    provider = validate_admission(spark, SingleWriterAdmission())
    with provider.hold("prediction-table-id"):
        assert list(tmp_path.iterdir()) == []
