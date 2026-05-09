"""Mega smoke test — exercises every registered node end-to-end.

Goal: a single regression test that catches "I broke the registry / a node
type signature drifted / a node throws on default-shaped input" without us
having to maintain 76 hand-rolled tests. For each node it tries:

1. Build the Calculator + Applier from `NodeRegistry`.
2. Run `Calculator.fit(...)` with sane defaults derived from
   `__node_meta__.params` and the synthetic dataset.
3. Run `Applier.apply(...)` with the resulting params.

It does *not* assert deep semantic correctness — that is what the per-node
unit tests do. Failure modes captured:
  * NotImplementedError from a half-finished node.
  * TypeError from a refactor that broke `(X, y, params)` signatures.
  * Plain Python errors (AttributeError, ImportError) from regressions.

Skip list `_NEEDS_SPECIAL_INPUT` covers nodes where a generic call shape
will never work (e.g. `Oversampling` requires imblearn + numeric-only
features + a binary `y`). Those are covered by their own targeted tests.
"""

from __future__ import annotations

from typing import Any, Dict, Set, Tuple

import numpy as np
import pandas as pd
import pytest

import skyulf  # noqa: F401  (side-effect: populate the registry)
from skyulf.registry import NodeRegistry


# ---------------------------------------------------------------------------
# Synthetic dataset — wide enough that every node finds at least one valid
# column to operate on. Has numeric (with outliers + missing), categorical
# strings, a small-cardinality "label-ish" target, and a high-cardinality
# string column so HashEncoder / OneHotEncoder both have grist.
# ---------------------------------------------------------------------------


def _make_dataset(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, n),
            "num_b": rng.normal(5, 2, n),
            "num_skew": rng.exponential(1.0, n) + 0.1,
            "with_outliers": np.concatenate(
                [rng.normal(0, 1, n - 5), np.array([50, -50, 100, -100, 75])]
            ),
            "with_missing": rng.normal(0, 1, n),
            "cat_low": rng.choice(["red", "green", "blue"], n),
            "cat_mid": rng.choice([f"c{i}" for i in range(8)], n),
            "cat_high": rng.choice([f"h{i}" for i in range(40)], n),
            "target": rng.choice([0, 1], n),
        }
    )
    # Inject a few NaNs into `with_missing` so imputers have work to do.
    df.loc[df.sample(frac=0.1, random_state=seed).index, "with_missing"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Per-node hints: column lists that map well to the node's purpose. Avoids
# every test fanning out into "every column" which often trips dtype checks.
# ---------------------------------------------------------------------------

_NUMERIC_COLS = ["num_a", "num_b", "with_outliers", "with_missing"]
_CATEGORICAL_COLS = ["cat_low", "cat_mid"]

_COLUMN_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # Scalers / transformers — numeric only.
    "MinMaxScaler": {"columns": _NUMERIC_COLS},
    "MaxAbsScaler": {"columns": _NUMERIC_COLS},
    "StandardScaler": {"columns": _NUMERIC_COLS},
    "RobustScaler": {"columns": _NUMERIC_COLS},
    "PowerTransformer": {"columns": ["num_a", "num_skew"]},
    "GeneralTransformation": {"columns": ["num_skew"], "transformation": "log"},
    "SimpleTransformation": {"columns": ["num_a"], "transformation": "abs"},
    # Imputers.
    "SimpleImputer": {"columns": ["with_missing"], "strategy": "mean"},
    "KNNImputer": {"columns": ["with_missing", "num_a"], "n_neighbors": 3},
    "IterativeImputer": {"columns": ["with_missing", "num_a"]},
    # Outlier nodes — pure numeric.
    "IQR": {"columns": ["with_outliers"]},
    "ZScore": {"columns": ["with_outliers"]},
    "Winsorize": {"columns": ["with_outliers"]},
    "ManualBounds": {
        "columns": ["with_outliers"],
        "lower_bound": -10,
        "upper_bound": 10,
    },
    "EllipticEnvelope": {"columns": ["num_a", "num_b"], "contamination": 0.1},
    # Bucketing.
    "GeneralBinning": {"columns": ["num_a"], "n_bins": 4, "strategy": "uniform"},
    "KBinsDiscretizer": {"columns": ["num_a"], "n_bins": 4, "strategy": "uniform"},
    "CustomBinning": {
        "columns": ["num_a"],
        "bin_edges": [-3, -1, 0, 1, 3],
    },
    # Encoders — categorical.
    "DummyEncoder": {"columns": _CATEGORICAL_COLS},
    "OneHotEncoder": {"columns": _CATEGORICAL_COLS},
    "OrdinalEncoder": {"columns": _CATEGORICAL_COLS},
    "LabelEncoder": {"columns": _CATEGORICAL_COLS},
    "HashEncoder": {"columns": ["cat_high"], "n_features": 4},
    "TargetEncoder": {"columns": _CATEGORICAL_COLS, "target_column": "target"},
}

# ---------------------------------------------------------------------------
# Nodes that need a tuple input or external libraries we don't want to spam
# in this smoke test. They have their own dedicated tests.
# ---------------------------------------------------------------------------

_NEEDS_SPECIAL_INPUT: Set[str] = {
    # Resamplers need imblearn + a binary y as a separate Series.
    "Oversampling",
    "Undersampling",
}

# Optional: nodes where we expect fit to silently no-op (return {} or None);
# we still call them but skip the apply step.
_FIT_MAY_BE_EMPTY: Set[str] = set()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def _build_config(node_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    """Merge the node's declared default params with our column overrides."""
    config: Dict[str, Any] = dict(meta.get("params") or {})
    if node_id in _COLUMN_OVERRIDES:
        config.update(_COLUMN_OVERRIDES[node_id])
    return config


def _maybe_call_apply(
    applier: Any, X: pd.DataFrame, y: Any, params: Dict[str, Any]
) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` after attempting ``applier.apply``."""
    try:
        # Most nodes use the (X, params) shape; encoders that touch a target
        # need (X, y). We try both.
        try:
            applier.apply(X, params)
        except TypeError:
            applier.apply((X, y), params)
    except Exception as exc:  # noqa: BLE001
        return False, f"apply failed: {type(exc).__name__}: {exc}"
    return True, ""


@pytest.mark.parametrize(
    "node_id",
    sorted(
        node_id
        for node_id, meta in NodeRegistry.get_all_metadata().items()
        if meta.get("category") in {"Preprocessing", "Cleaning"}
        and node_id not in _NEEDS_SPECIAL_INPUT
    ),
)
def test_node_smoke(node_id: str) -> None:
    """Smoke test: every Preprocessing/Cleaning node fits + applies on a
    realistic synthetic dataset without raising."""
    meta = NodeRegistry.get_all_metadata()[node_id]
    calculator = NodeRegistry.get_calculator(node_id)()
    applier = NodeRegistry.get_applier(node_id)()
    config = _build_config(node_id, meta)

    df = _make_dataset()
    y = df["target"]
    X = df.drop(columns=["target"])

    # Some encoders (TargetEncoder, LabelEncoder when target_column is set)
    # require the (X, y) tuple shape for fit. Try the simple shape first.
    try:
        params = calculator.fit(X, config)
    except TypeError:
        params = calculator.fit((X, y), config)

    if not params:
        # Nothing to apply (node opted out at fit time, e.g. user_picked_no_columns
        # short-circuit). That's a valid path — don't fail the smoke.
        return

    ok, reason = _maybe_call_apply(applier, X, y, params)
    assert ok, f"{node_id}: {reason}"


def test_registry_minimum_population() -> None:
    """Guard rail: if someone accidentally breaks the auto-import sweep in
    `skyulf.preprocessing.__init__`, the registry shrinks silently. Pin a
    minimum size so CI catches it."""
    meta = NodeRegistry.get_all_metadata()
    preprocessing = [k for k, m in meta.items() if m.get("category") == "Preprocessing"]
    assert (
        len(preprocessing) >= 25
    ), f"Preprocessing registry shrunk: only {len(preprocessing)} nodes registered"
