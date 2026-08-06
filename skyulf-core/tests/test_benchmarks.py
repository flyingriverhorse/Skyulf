"""Performance benchmark harness for skyulf-core (pytest-benchmark).

These benchmarks are **opt-in** — they are skipped during normal test runs and
only execute when invoked explicitly:

    pytest skyulf-core/tests/test_benchmarks.py --benchmark-only

CI can wire this into a dedicated job and use ``--benchmark-compare`` /
``--benchmark-compare-fail=mean:10%`` to alert on regressions against a saved
baseline (``--benchmark-save=baseline`` on the main branch).

Each benchmark builds a synthetic frame and times a single preprocessing +
modeling fit, parametrised over the pandas and polars engines so we can spot
per-engine regressions independently.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.pipeline import SkyulfPipeline
from skyulf.preprocessing.feature_selection import correlation as correlation_module
from skyulf.preprocessing.feature_selection.correlation import CorrelationThresholdCalculator

try:
    import polars as pl

    _ENGINES = ["pandas", "polars"]
except ImportError:  # pragma: no cover - polars is a core dep, kept defensive
    pl = None  # ty: ignore[invalid-assignment]
    _ENGINES = ["pandas"]

# Skip the whole module unless the user explicitly asks for benchmarks. Without
# the plugin's ``--benchmark-only`` flag the ``benchmark`` fixture still runs the
# callable once, so these would otherwise slow every normal test invocation.
pytestmark = pytest.mark.benchmark


_PIPELINE_CONFIG = {
    "preprocessing": [
        {"name": "imputer", "transformer": "SimpleImputer", "params": {"strategy": "mean"}},
        {
            "name": "scaler",
            "transformer": "StandardScaler",
            "params": {"columns": ["numeric_1", "numeric_2"]},
        },
        {"name": "encoder", "transformer": "OneHotEncoder", "params": {"columns": ["categorical"]}},
    ],
    "modeling": {"type": "logistic_regression", "params": {}},
}


def _synthetic_frame(n_rows: int = 20_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "numeric_1": rng.standard_normal(n_rows),
            "numeric_2": rng.random(n_rows) * 100,
            "categorical": rng.choice(["A", "B", "C", "D"], n_rows),
            "target": rng.integers(0, 2, n_rows),
        }
    )


def _as_engine(df: pd.DataFrame, engine: str):
    if engine == "polars" and pl is not None:
        return pl.from_pandas(df)
    return df


_RUN_LARGE_CORRELATION_BENCHMARKS = os.environ.get("SKYULF_RUN_LARGE_BENCHMARKS") == "1"
_LARGE_CORRELATION_CASE = pytest.mark.skipif(
    not _RUN_LARGE_CORRELATION_BENCHMARKS,
    reason="set SKYULF_RUN_LARGE_BENCHMARKS=1 to run large correlation benchmarks",
)
_CORRELATION_BENCHMARK_CASES = [
    pytest.param(100_000, 50, id="100k-x-50"),
    pytest.param(1_000_000, 20, marks=_LARGE_CORRELATION_CASE, id="1m-x-20"),
    pytest.param(50_000, 500, marks=_LARGE_CORRELATION_CASE, id="50k-x-500"),
]


def _correlated_polars_frame(rows: int, columns: int):
    """Build a deterministic numeric Polars frame with missing and correlated values."""
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(20260806)
    values = rng.normal(size=(rows, columns))
    values[rng.random(values.shape) < 0.05] = np.nan
    values[:, 1] = values[:, 0] * 2.0 + 1.0
    return pl.DataFrame(values, schema=[f"feature_{index}" for index in range(columns)])


def _correlation_benchmark_config(columns: int) -> dict[str, object]:
    """Return an explicit candidate list so every route measures identical work."""
    return {
        "columns": [f"feature_{index}" for index in range(columns)],
        "threshold": 0.95,
        "correlation_method": "pearson",
    }


def _peak_rss_bytes() -> int:
    """Return the process maximum RSS in bytes on supported benchmark platforms."""
    if sys.platform == "win32":
        pytest.skip("isolated RSS measurement is not implemented on Windows")
    import resource

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


@pytest.mark.parametrize("engine", _ENGINES)
def test_pipeline_fit_benchmark(benchmark, engine):
    """Benchmark a full preprocessing + modeling fit on each engine."""
    df = _synthetic_frame()

    def _fit():
        pipeline = SkyulfPipeline(_PIPELINE_CONFIG)
        pipeline.fit(_as_engine(df.copy(), engine), target_column="target")
        return pipeline

    pipeline = benchmark(_fit)
    assert pipeline is not None


@pytest.mark.parametrize(("rows", "columns"), _CORRELATION_BENCHMARK_CASES)
@pytest.mark.parametrize("wrapped", [False, True], ids=["raw", "wrapped"])
@pytest.mark.parametrize("route", ["legacy", "native"])
def test_correlation_threshold_fit_benchmark(benchmark, rows, columns, wrapped, route):
    """Benchmark equivalent legacy and native correlation-threshold fitting."""
    raw = _correlated_polars_frame(rows, columns)
    frame = SkyulfPolarsWrapper(raw) if wrapped else raw
    config = _correlation_benchmark_config(columns)

    if route == "legacy":
        artifact = benchmark(correlation_module._fit_correlation_threshold_pandas, frame, config)
    else:
        artifact = benchmark(CorrelationThresholdCalculator().fit, frame, config)

    assert "feature_1" in artifact["columns_to_drop"]


@pytest.mark.skipif(
    os.environ.get("SKYULF_MEASURE_CORRELATION_RSS") != "1",
    reason="set SKYULF_MEASURE_CORRELATION_RSS=1 for isolated RSS output",
)
@pytest.mark.parametrize(("rows", "columns"), _CORRELATION_BENCHMARK_CASES)
@pytest.mark.parametrize("wrapped", [False, True], ids=["raw", "wrapped"])
@pytest.mark.parametrize("route", ["legacy", "native"])
def test_correlation_threshold_fit_peak_rss(benchmark, rows, columns, wrapped, route):
    """Print incremental process RSS for one separately invoked fit route."""
    raw = _correlated_polars_frame(rows, columns)
    frame = SkyulfPolarsWrapper(raw) if wrapped else raw
    config = _correlation_benchmark_config(columns)
    baseline = _peak_rss_bytes()

    def _fit():
        if route == "legacy":
            return correlation_module._fit_correlation_threshold_pandas(frame, config)
        return CorrelationThresholdCalculator().fit(frame, config)

    artifact = benchmark.pedantic(_fit, rounds=1, iterations=1, warmup_rounds=0)

    delta = max(0, _peak_rss_bytes() - baseline)
    print(f"route={route} wrapped={wrapped} rows={rows} columns={columns} peak_rss_delta={delta}")
    assert "feature_1" in artifact["columns_to_drop"]
