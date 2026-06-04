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

import numpy as np
import pandas as pd
import pytest

from skyulf.pipeline import SkyulfPipeline

try:
    import polars as pl

    _ENGINES = ["pandas", "polars"]
except ImportError:  # pragma: no cover - polars is a core dep, kept defensive
    pl = None
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
