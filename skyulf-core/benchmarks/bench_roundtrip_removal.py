"""Benchmark: Category B round-trip removal (migration plan Phase 4/5).

Measures the polars-native node paths against a faithful reconstruction of the
old whole-frame ``to_pandas()`` + ``from_pandas()`` round-trip they replaced:

* TrainTestSplitter — index split + gather vs. pandas train_test_split on frames
* EllipticEnvelope  — native boolean-mask filter vs. pandas filter + conversion
* CountVectorizer   — native hstack attach vs. pandas concat + conversion

Run from the repo root:
    .venv/Scripts/python.exe skyulf-core/benchmarks/bench_roundtrip_removal.py
"""

import statistics
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

from skyulf.preprocessing.outliers.elliptic import (
    EllipticEnvelopeApplier,
    EllipticEnvelopeCalculator,
)
from skyulf.preprocessing.split import DataSplitter
from skyulf.preprocessing.vectorization import (
    CountVectorizerApplier,
    CountVectorizerCalculator,
)

N_ROWS = 500_000
N_COLS = 20
REPEATS = 3


def _time(fn: Callable[[], Any]) -> float:
    """Median wall time (seconds) over REPEATS runs."""
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def _frame(n_rows: int = N_ROWS, n_cols: int = N_COLS) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    data = {f"f{i}": rng.normal(0, 1, n_rows) for i in range(n_cols)}
    data["id"] = np.arange(n_rows)
    return pl.DataFrame(data)


def bench_split() -> tuple[float, float]:
    df = _frame()
    splitter = DataSplitter(test_size=0.2, random_state=42)

    def old() -> None:
        df_pd = df.to_pandas()
        train, test = train_test_split(df_pd, test_size=0.2, random_state=42, shuffle=True)
        pl.from_pandas(train)
        pl.from_pandas(test)

    def new() -> None:
        splitter.split(df)

    return _time(old), _time(new)


def bench_elliptic() -> tuple[float, float]:
    df = _frame(n_rows=200_000)  # fit cost grows with rows; keep total runtime sane
    fit_df = df.head(50_000).to_pandas()
    params = EllipticEnvelopeCalculator().fit(
        fit_df, {"columns": ["f0", "f1"], "contamination": 0.01}
    )
    applier = EllipticEnvelopeApplier()

    from skyulf.preprocessing.outliers.elliptic import _elliptic_filter_pandas

    def old() -> None:
        df_pd = df.to_pandas()
        mask = _elliptic_filter_pandas(df_pd, params["models"])
        pl.from_pandas(df_pd[mask])

    def new() -> None:
        applier.apply(df, params)

    return _time(old), _time(new)


def bench_count_vectorizer() -> tuple[float, float]:
    rng = np.random.default_rng(1)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    n_rows = 200_000
    texts = [" ".join(rng.choice(vocab, size=8)) for _ in range(n_rows)]
    df = pl.DataFrame(
        {
            "text": texts,
            "id": np.arange(n_rows),
            **{f"f{i}": rng.normal(0, 1, n_rows) for i in range(10)},
        }
    )
    art = CountVectorizerCalculator().fit(df.to_pandas(), {"columns": ["text"]})
    applier = CountVectorizerApplier()

    from skyulf.preprocessing.vectorization._common import _sklearn_vectorizer_apply_pandas

    def old() -> None:
        df_pd = df.to_pandas()
        out_pd, _ = _sklearn_vectorizer_apply_pandas(df_pd, None, art)
        pl.from_pandas(out_pd)

    def new() -> None:
        applier.apply(df, art)

    return _time(old), _time(new)


def main() -> None:
    print(f"rows={N_ROWS:,} cols={N_COLS} repeats={REPEATS} (median)\n")
    print(f"{'node':<20} {'old (round-trip)':>18} {'new (native)':>14} {'speedup':>9}")
    for name, bench in (
        ("TrainTestSplitter", bench_split),
        ("EllipticEnvelope", bench_elliptic),
        ("CountVectorizer", bench_count_vectorizer),
    ):
        old, new = bench()
        print(f"{name:<20} {old:>16.3f}s {new:>12.3f}s {old / new:>8.2f}x")


if __name__ == "__main__":
    main()
