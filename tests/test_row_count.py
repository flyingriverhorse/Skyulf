import csv
import tempfile
import time
from pathlib import Path


def _write_sample_csv(path: Path, rows: int = 10000) -> None:
    """Write a small sample CSV with a header and `rows` data rows."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["col1", "col2", "col3"])
        for i in range(rows):
            writer.writerow([i, f"val{i}", i * 2])


def test_row_count_methods_agree(tmp_path: Path) -> None:
    """Create a sample CSV, count rows with two different methods and assert they match.

    This replaces the previous dependency on an external dataset file so tests are self-contained.
    """
    sample = tmp_path / "sample.csv"
    _write_sample_csv(sample, rows=5000)

    # Old method - text iteration
    start = time.time()
    with sample.open("r", encoding="utf-8", errors="ignore") as f:
        count_old = sum(1 for _ in f) - 1
    old_time = time.time() - start

    # New method - binary buffered counting
    start = time.time()
    count_new = 0
    with sample.open("rb") as f:
        buffer_size = 65536
        buf = f.read(buffer_size)
        while buf:
            count_new += buf.count(b"\n")
            buf = f.read(buffer_size)
    count_new = count_new - 1  # Subtract header
    new_time = time.time() - start

    assert count_old == count_new
    # Sanity: ensure counts equal expected
    assert count_old == 5000

    # Print timings (pytest captures output only on failure or with -s)
    print(f"Old method: {count_old:,} rows in {old_time:.4f}s")
    print(f"New method: {count_new:,} rows in {new_time:.4f}s")
