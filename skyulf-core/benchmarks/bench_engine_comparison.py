"""Benchmark: Polars vs Pandas engine across preprocessing nodes and models.

Runs the same node (fit + apply) on an identical dataset once as a pandas
frame and once as a Polars frame, so the numbers measure what a
``SKYULF_ENGINE`` choice costs end to end — including the legitimate
pandas/sklearn boundaries that Polars still has to cross.

Run from the repo root:
    .venv/Scripts/python.exe skyulf-core/benchmarks/bench_engine_comparison.py
"""

import statistics
import time
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from skyulf.registry import NodeRegistry

N_ROWS = 200_000
REPEATS = 3
RNG = np.random.default_rng(0)

NUMERIC = [f"f{i}" for i in range(12)]
NULLABLE_FLOAT = NUMERIC[:4]
INTS = [f"i{i}" for i in range(4)]
CATS = [f"c{i}" for i in range(3)]
TEXT = "text"
TARGET = "target"


def _make_pandas() -> pd.DataFrame:
    data: dict[str, Any] = {}
    for col in NUMERIC:
        data[col] = RNG.normal(0, 1, N_ROWS)
    for col in NULLABLE_FLOAT:
        data[col][RNG.random(N_ROWS) < 0.05] = np.nan
    for col in INTS:
        values = RNG.integers(0, 100, N_ROWS).astype(float)
        values[RNG.random(N_ROWS) < 0.05] = np.nan
        data[col] = values
    for col in CATS:
        data[col] = RNG.choice([f"cat_{i}" for i in range(8)], N_ROWS)
    vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    data[TEXT] = [" ".join(RNG.choice(vocab, 8)) for _ in range(N_ROWS)]
    data[TARGET] = (data["f0"] > 0).astype(int)
    return pd.DataFrame(data)


# (label, node, config, kind) — kind: "frame" fits/applies on the full frame,
# "model" fits on (numeric X, y).
NODES: list[tuple[str, str, dict[str, Any], str]] = [
    ("SimpleImputer", "SimpleImputer", {"strategy": "mean", "columns": NULLABLE_FLOAT}, "frame"),
    ("StandardScaler", "StandardScaler", {"columns": NUMERIC}, "frame"),
    ("MinMaxScaler", "MinMaxScaler", {"columns": NUMERIC}, "frame"),
    ("RobustScaler", "RobustScaler", {"columns": NUMERIC}, "frame"),
    ("PowerTransformer", "PowerTransformer", {"columns": NUMERIC}, "frame"),
    ("Winsorize", "Winsorize", {"columns": NUMERIC[:4]}, "frame"),
    ("IQR", "IQR", {"columns": NUMERIC[:4]}, "frame"),
    ("ZScore", "ZScore", {"columns": NUMERIC[:4]}, "frame"),
    (
        "EllipticEnvelope",
        "EllipticEnvelope",
        {"columns": NUMERIC[:2], "contamination": 0.01},
        "frame",
    ),
    ("OneHotEncoder", "OneHotEncoder", {"columns": CATS}, "frame"),
    ("OrdinalEncoder", "OrdinalEncoder", {"columns": CATS}, "frame"),
    ("HashEncoder", "HashEncoder", {"columns": CATS, "n_features": 8}, "frame"),
    ("LabelEncoder", "LabelEncoder", {"columns": CATS}, "frame"),
    ("GeneralBinning", "GeneralBinning", {"columns": ["f0"], "n_bins": 5}, "frame"),
    (
        "TrainTestSplitter",
        "TrainTestSplitter",
        {"test_size": 0.2, "target_column": TARGET},
        "frame",
    ),
    ("CountVectorizer", "count_vectorizer", {"columns": [TEXT]}, "frame"),
    ("TfidfVectorizer", "tfidf_vectorizer", {"columns": [TEXT]}, "frame"),
    ("Tokenizer", "tokenizer", {"columns": [TEXT]}, "frame"),
    (
        "LogisticRegression",
        "logistic_regression",
        {"max_iter": 200, "solver": "lbfgs", "random_state": 42},
        "model",
    ),
    (
        "RandomForest(n=20)",
        "random_forest_classifier",
        {"n_estimators": 20, "max_depth": 8, "random_state": 42},
        "model",
    ),
    (
        "GradientBoosting(n=30)",
        "gradient_boosting_classifier",
        {"n_estimators": 30, "learning_rate": 0.1, "max_depth": 3},
        "model",
    ),
]


def _run_node(
    name: str, config: dict[str, Any], kind: str, frame_pd: pd.DataFrame
) -> tuple[float, float]:
    """Return (pandas_seconds, polars_seconds) for one node's fit+apply."""
    calc_cls = NodeRegistry._calculators[name]
    frame_pl = pl.from_pandas(frame_pd)

    if kind == "model":
        X_pd = frame_pd[NUMERIC].fillna(0.0)
        y_pd = frame_pd[TARGET]
        X_pl = pl.from_pandas(X_pd)
        y_pl = pl.from_pandas(y_pd.to_frame()).to_series()

        def run(X: Any, y: Any) -> None:
            calc_cls().fit(X, y, config)

        pd_t = _time(lambda: run(X_pd, y_pd), repeats=1)
        pl_t = _time(lambda: run(X_pl, y_pl), repeats=1)
        return pd_t, pl_t

    def run(frame: Any) -> None:
        params = calc_cls().fit(frame, config)
        applier = NodeRegistry._appliers[name]()
        applier.apply(frame, params)

    return _time(lambda: run(frame_pd)), _time(lambda: run(frame_pl))


def _time(fn: Any, repeats: int = REPEATS) -> float:
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main() -> None:
    print(
        f"building frame: {N_ROWS:,} rows ({len(NUMERIC)} float, {len(INTS)} int-null, "
        f"{len(CATS)} cat, 1 text, 1 target)"
    )
    frame_pd = _make_pandas()
    print(f"\n{'node':<22} {'pandas':>10} {'polars':>10} {'speedup':>9}")
    for label, name, config, kind in NODES:
        try:
            pd_t, pl_t = _run_node(name, config, kind, frame_pd)
        except Exception as e:  # keep the table alive if one node can't run
            print(f"{label:<22} {'SKIP':>10} {type(e).__name__}: {e}")
            continue
        speedup = pd_t / pl_t if pl_t > 0 else float("inf")
        print(f"{label:<22} {pd_t:>9.3f}s {pl_t:>9.3f}s {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
