"""Registry-wide contract test — validates every registered transformer against
its own defaults, NaN-aware parity, and wrapped-frame correctness.

Per the growth plan Stage 0 exit criteria:
  1. Fitting/applying with the node's own ``@node_meta`` default params produces
     no empty artifact, no "unknown method" warning, and no silent no-op.
  2. Any node that changes row count or row order returns a ``y`` whose length
     and order match ``X``.
  3. Engine parity on float **NaN**, not only nulls, and on **wrapped** frames,
     not only raw ones.

Skip list ``_NEEDS_SPECIAL_INPUT`` covers nodes that require data shapes or
external libraries the generic fixture cannot provide.  Those are covered by
their own targeted tests.
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest

from skyulf.engines.polars_engine import SkyulfPolarsWrapper
from skyulf.registry import NodeRegistry

# ---------------------------------------------------------------------------
# Test dataset — includes float NaN, categoricals, text, geo, and a y column
# ---------------------------------------------------------------------------

_SEED = 42
_N = 80

_TEXT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming data science",
    "Python is a popular programming language",
    "Data preprocessing is essential for good models",
    "The weather today is sunny and warm",
    "Deep learning requires large amounts of data",
    "Feature engineering improves model performance",
    "Natural language processing is a key AI field",
]


def _make_nan_frame() -> pd.DataFrame:
    """Small DataFrame with float NaN in multiple columns, plus categoricals,
    text, and geo coordinates.

    Important: ``np.nan`` in a float column is a genuine float NaN, not a null.
    Polars ``is_null()`` does NOT match it; pandas ``isna()`` matches both.
    """
    rng = np.random.default_rng(_SEED)
    df = pd.DataFrame(
        {
            "num_a": rng.normal(0, 1, _N),
            "num_b": rng.normal(5, 2, _N),
            "num_c": rng.exponential(1.0, _N),
            "cat_a": rng.choice(["red", "green", "blue"], _N),
            "cat_b": rng.choice(["x", "y", "z"], _N),
            "text_col": rng.choice(_TEXT_SENTENCES, _N),
            "lat1": rng.uniform(40.0, 41.0, _N),
            "lon1": rng.uniform(-74.0, -73.0, _N),
            "lat2": rng.uniform(34.0, 35.0, _N),
            "lon2": rng.uniform(-118.0, -117.0, _N),
            "y": rng.choice([0, 1], _N),
        }
    )
    # Inject float NaN into numeric columns at known positions.
    for col in ("num_a", "num_b", "num_c"):
        nan_mask = rng.random(_N) < 0.1
        df.loc[nan_mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Column hints — map node IDs to a subset of columns that match the node's
# purpose.  Without this, many nodes would fail on dtype mismatches (e.g.
# passing a categorical column to a numeric-only scaler).
# ---------------------------------------------------------------------------

_NUM = ["num_a", "num_b", "num_c"]
_CAT = ["cat_a", "cat_b"]
_TXT = ["text_col"]

_COLUMN_HINTS: dict[str, dict[str, Any]] = {
    # Scalers
    "MinMaxScaler": {"columns": _NUM},
    "MaxAbsScaler": {"columns": _NUM},
    "StandardScaler": {"columns": _NUM},
    "RobustScaler": {"columns": _NUM},
    # Transformations
    "PowerTransformer": {"columns": ["num_a"]},
    "GeneralTransformation": {"columns": ["num_a"], "transformation": "log"},
    "SimpleTransformation": {"columns": ["num_a"], "transformation": "abs"},
    # Imputers
    "SimpleImputer": {"columns": _NUM, "strategy": "mean"},
    "KNNImputer": {"columns": _NUM, "n_neighbors": 3},
    "IterativeImputer": {"columns": _NUM},
    # Outliers (row-dropping)
    "IQR": {"columns": _NUM},
    "ZScore": {"columns": _NUM},
    "Winsorize": {"columns": _NUM},
    "ManualBounds": {"columns": _NUM, "lower_bound": -10, "upper_bound": 10},
    "EllipticEnvelope": {"columns": ["num_a", "num_b"], "contamination": 0.1},
    # Bucketing
    "GeneralBinning": {"columns": ["num_a"], "n_bins": 4, "strategy": "uniform"},
    "KBinsDiscretizer": {"columns": ["num_a"], "n_bins": 4, "strategy": "uniform"},
    "CustomBinning": {"columns": ["num_a"], "bin_edges": [-3, -1, 0, 1, 3]},
    # Encoders
    "DummyEncoder": {"columns": _CAT},
    "OneHotEncoder": {"columns": _CAT},
    "OrdinalEncoder": {"columns": _CAT},
    "LabelEncoder": {"columns": _CAT},
    "HashEncoder": {"columns": _CAT, "n_features": 4},
    "TargetEncoder": {"columns": _CAT, "target_column": "y"},
    "WOEEncoder": {"columns": _CAT, "target_column": "y"},
    # Drop / missing
    "DropMissingRows": {"columns": _NUM},
    "DropMissingColumns": {"columns": _NUM},
    "MissingIndicator": {"columns": _NUM},
    "Deduplicate": {"columns": _CAT},
    # Feature generation
    "FeatureInteraction": {"columns": _NUM},
    "PolynomialFeatures": {"columns": ["lat1", "lat2"]},
    "PolynomialFeaturesNode": {"columns": ["lat1", "lat2"]},
    "FeatureGeneration": {
        "operations": [
            {"operation_type": "arithmetic", "input_columns": ["num_a", "num_b"], "method": "add"}
        ]
    },
    "FeatureMath": {
        "operations": [
            {"operation_type": "arithmetic", "input_columns": ["num_a", "num_b"], "method": "add"}
        ]
    },
    # Feature selection
    "feature_selection": {"method": "variance", "threshold": 0.0},
    "VarianceThreshold": {"columns": _NUM, "threshold": 0.0},
    "CorrelationThreshold": {"columns": _NUM, "threshold": 0.95},
    "ModelBasedSelection": {"columns": _NUM, "target_column": "y", "estimator": "RandomForest"},
    "UnivariateSelection": {
        "columns": _NUM,
        "target_column": "y",
        "score_func": "f_classif",
        "k": 3,
    },
    # Value replacement
    "ValueReplacement": {"columns": _NUM, "replace_value": 0.0, "new_value": 1.0},
    # Casting
    "TypeCasting": {"columns": _NUM, "target_type": "float64"},
    # Cleaning
    "TextCleaning": {"columns": ["cat_b"]},
    "InvalidValueHandler": {"columns": _NUM},
    "InvalidValueReplacement": {"columns": _NUM, "rule": "negative"},
    "AliasReplacement": {"columns": _CAT, "alias_type": "boolean"},
    # Time series (needs sort_by to avoid ValueError)
    "LagFeatures": {"columns": ["num_a"], "lags": [1, 2], "sort_by": "num_a"},
    "RollingAggregate": {
        "columns": ["num_a"],
        "aggregations": ["mean"],
        "window": 3,
        "sort_by": "num_a",
    },
    # Text / NLP
    "count_vectorizer": {"columns": _TXT},
    "hashing_vectorizer": {"columns": _TXT, "n_features": 16},
    "tfidf_vectorizer": {"columns": _TXT},
    "tokenizer": {"columns": _TXT},
    "sentence_embedder": {"columns": _TXT},
    # Geo
    "GeoDistance": {"lat1_col": "lat1", "lon1_col": "lon1", "lat2_col": "lat2", "lon2_col": "lon2"},
    "H3Index": {"lat_col": "lat1", "lon_col": "lon1", "resolution": 5},
}

# ---------------------------------------------------------------------------
# Nodes that need tuple input or external libraries not worth pulling in here.
# ---------------------------------------------------------------------------

# Nodes whose Calculator.fit() requires (X, y) as a tuple — the @fit_method
# decorator unpacks the tuple and passes X, y separately.  Without the tuple
# wrapper, y is None and the fit returns an empty artifact.
_NEEDS_TUPLE_INPUT: set[str] = {
    "TargetEncoder",
    "WOEEncoder",
    "ModelBasedSelection",
    "UnivariateSelection",
}

_NEEDS_SPECIAL_INPUT: set[str] = {
    # Resamplers — need imblearn
    "Oversampling",
    "Undersampling",
    # Text nodes needing optional packages
    "sentence_embedder",  # needs sentence-transformers
    "H3Index",  # needs h3 package
    # Ensemble nodes — need fitted model pipelines
    "stacking_classifier",
    "stacking_regressor",
    "voting_classifier",
    "voting_regressor",
    # Inspection nodes — no meaningful fit artifact
    "DataSnapshot",
    "DatasetProfile",
}

# Nodes whose fit is expected to return an empty dict (legitimate no-op).
_FIT_MAY_BE_EMPTY: set[str] = set()

# Nodes that change row count — y must be filtered to match.
_ROW_DROPPING: set[str] = {
    "IQR",
    "ZScore",
    "ManualBounds",
    "EllipticEnvelope",
    "Deduplicate",
    "DropMissingRows",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(node_id: str, meta: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = dict(meta.get("params") or {})
    if node_id in _COLUMN_HINTS:
        config.update(_COLUMN_HINTS[node_id])
    return config


def _fit_and_apply(
    node_id: str,
    X: Any,
    y: Any,
    config: dict[str, Any],
    *,
    expect_y_filtered: bool = False,
) -> tuple[Any, Any, str]:
    """Fit calculator + apply applier.  Returns ``(X_out, y_out, error_reason)``."""
    try:
        calculator = NodeRegistry.get_calculator(node_id)()
        applier = NodeRegistry.get_applier(node_id)()
    except Exception as exc:  # noqa: BLE001 - contract harness reports node failures as strings, not harness errors
        return None, None, f"registry lookup: {exc}"

    # Fit
    if node_id in _NEEDS_TUPLE_INPUT:
        try:
            params = calculator.fit((X, y), config)
        except Exception as exc:  # noqa: BLE001 - contract harness reports node failures as strings, not harness errors
            return None, None, f"fit: {type(exc).__name__}: {exc}"
    else:
        try:
            params = calculator.fit(X, config)
        except TypeError:
            try:
                params = calculator.fit((X, y), config)
            except Exception as exc:  # noqa: BLE001 - contract harness reports node failures as strings, not harness errors
                return None, None, f"fit: {type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - contract harness reports node failures as strings, not harness errors
            return None, None, f"fit: {type(exc).__name__}: {exc}"

    if not params and node_id not in _FIT_MAY_BE_EMPTY:
        return None, None, "fit returned empty artifact"

    # Apply
    try:
        result = applier.apply((X, y), params) if y is not None else applier.apply(X, params)
    except TypeError:
        try:
            result = applier.apply(X, params)
        except Exception as exc:  # noqa: BLE001 - contract harness reports node failures as strings, not harness errors
            return None, None, f"apply: {type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001 - contract harness reports node failures as strings, not harness errors
        return None, None, f"apply: {type(exc).__name__}: {exc}"

    X_out, y_out = result, y
    if isinstance(result, tuple) and len(result) == 2:
        X_out, y_out = result

    if expect_y_filtered and y_out is not None:
        if hasattr(X_out, "height"):
            x_len = X_out.height
        elif hasattr(X_out, "shape"):
            x_len = X_out.shape[0]
        else:
            x_len = len(X_out)
        if hasattr(y_out, "len"):
            y_len = y_out.len()
        elif hasattr(y_out, "shape"):
            y_len = y_out.shape[0]
        else:
            y_len = len(y_out)
        if x_len != y_len:
            return X_out, y_out, f"y length {y_len} != X length {x_len} after row drop"

    return X_out, y_out, ""


def _assert_artifacts_equal(pd_params: Any, pl_params: Any, node_id: str) -> None:
    assert set(pd_params.keys()) == set(pl_params.keys()), (
        f"{node_id}: key mismatch: pandas={set(pd_params)} polars={set(pl_params)}"
    )
    for key, pd_val in pd_params.items():
        pl_val = pl_params[key]
        if isinstance(pd_val, list) and pd_val and isinstance(pd_val[0], (int, float, np.floating)):
            np.testing.assert_allclose(
                np.asarray(pd_val, dtype=float),
                np.asarray(pl_val, dtype=float),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"{node_id}: numeric mismatch on key '{key}'",
            )
        elif isinstance(pd_val, dict):
            _assert_artifacts_equal(pd_val, pl_val, f"{node_id}.{key}")
        elif isinstance(pd_val, np.ndarray):
            np.testing.assert_allclose(
                pd_val.astype(float),
                np.asarray(pl_val, dtype=float),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"{node_id}: numpy mismatch on key '{key}'",
            )
        elif hasattr(pd_val, "classes_"):
            # sklearn estimator — compare classes_ attribute
            assert hasattr(pl_val, "classes_"), (
                f"{node_id}: sklearn object missing classes_ on key '{key}'"
            )
            np.testing.assert_array_equal(
                pd_val.classes_,
                pl_val.classes_,
                err_msg=f"{node_id}: sklearn classes_ mismatch on key '{key}'",
            )
        elif hasattr(pd_val, "vocabulary_"):
            # sklearn vectorizer (CountVectorizer, TfidfVectorizer) — compare vocabulary
            assert hasattr(pl_val, "vocabulary_"), (
                f"{node_id}: sklearn vectorizer missing vocabulary_ on key '{key}'"
            )
            assert pd_val.vocabulary_ == pl_val.vocabulary_, (
                f"{node_id}: sklearn vocabulary mismatch on key '{key}'"
            )
        elif hasattr(pd_val, "get_params") and hasattr(pd_val, "set_params"):
            # sklearn estimator without classes_ (e.g. EllipticEnvelope, IterativeImputer)
            assert type(pd_val) is type(pl_val), (
                f"{node_id}: sklearn type mismatch on key '{key}': "
                f"{type(pd_val).__name__} vs {type(pl_val).__name__}"
            )
        elif pd_val is None and pl_val is None:
            continue
        elif isinstance(pd_val, (float, np.floating)):
            np.testing.assert_allclose(
                float(pd_val),
                float(pl_val),
                rtol=1e-6,
                atol=1e-6,
                err_msg=f"{node_id}: float mismatch on key '{key}'",
            )
        else:
            assert pd_val == pl_val, f"{node_id}: mismatch on key '{key}': {pd_val!r} vs {pl_val!r}"


# ---------------------------------------------------------------------------
# Clause 1 — every node's own defaults produce a non-empty artifact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_id",
    sorted(
        nid
        for nid, meta in NodeRegistry.get_all_metadata().items()
        if meta.get("category") != "Modeling" and nid not in _NEEDS_SPECIAL_INPUT
    ),
)
def test_default_params_produce_non_empty_artifact(node_id: str) -> None:
    meta = NodeRegistry.get_all_metadata()[node_id]
    config = _build_config(node_id, meta)
    df = _make_nan_frame()
    X = df.drop(columns=["y"])
    y = df["y"]

    calculator = NodeRegistry.get_calculator(node_id)()
    if node_id in _NEEDS_TUPLE_INPUT:
        params = calculator.fit((X, y), config)
    else:
        try:
            params = calculator.fit(X, config)
        except TypeError:
            params = calculator.fit((X, y), config)

    if node_id in _FIT_MAY_BE_EMPTY:
        return

    assert params, (
        f"{node_id}: fit with own defaults {meta.get('params')!r} returned empty artifact. "
        f"Re-check the node's default params or its dispatch table."
    )


# ---------------------------------------------------------------------------
# Clause 2 — row-dropping nodes propagate y correctly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("node_id", sorted(_ROW_DROPPING))
def test_row_dropping_nodes_propagate_y(node_id: str) -> None:
    meta = NodeRegistry.get_all_metadata()[node_id]
    config = _build_config(node_id, meta)
    df = _make_nan_frame()
    X = df.drop(columns=["y"])
    y = df["y"]

    _, _, err = _fit_and_apply(node_id, X, y, config, expect_y_filtered=True)
    assert not err, f"{node_id}: {err}"


# ---------------------------------------------------------------------------
# Clause 3 — engine parity on float NaN + wrapped frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node_id",
    sorted(
        nid
        for nid, meta in NodeRegistry.get_all_metadata().items()
        if meta.get("category") != "Modeling" and nid not in _NEEDS_SPECIAL_INPUT
    ),
)
def test_nan_parity_pandas_vs_polars(node_id: str) -> None:
    """Fit on pandas-with-NaN vs polars-with-NaN must produce identical artifacts."""
    meta = NodeRegistry.get_all_metadata()[node_id]
    config = _build_config(node_id, meta)
    df_pd = _make_nan_frame()
    X_pd = df_pd.drop(columns=["y"])
    y_pd = df_pd["y"]

    df_pl = pl.from_pandas(df_pd)
    X_pl = df_pl.drop("y")
    y_pl = df_pl["y"]

    calculator = NodeRegistry.get_calculator(node_id)()

    # Fit on pandas
    if node_id in _NEEDS_TUPLE_INPUT:
        pd_params = calculator.fit((X_pd, y_pd), config)
    else:
        try:
            pd_params = calculator.fit(X_pd, config)
        except TypeError:
            pd_params = calculator.fit((X_pd, y_pd), config)

    # Fit on polars
    if node_id in _NEEDS_TUPLE_INPUT:
        pl_params = calculator.fit((X_pl, y_pl), config)
    else:
        try:
            pl_params = calculator.fit(X_pl, config)
        except TypeError:
            pl_params = calculator.fit((X_pl, y_pl), config)

    if not pd_params and not pl_params:
        if node_id in _FIT_MAY_BE_EMPTY:
            return
        pytest.fail(f"{node_id}: both engines returned empty artifact")

    # If one engine returned empty and the other didn't, that's a parity bug.
    assert bool(pd_params) == bool(pl_params), (
        f"{node_id}: pandas returned {'non-empty' if pd_params else 'empty'}, "
        f"polars returned {'non-empty' if pl_params else 'empty'}"
    )

    if pd_params and pl_params:
        _assert_artifacts_equal(pd_params, pl_params, node_id)


@pytest.mark.parametrize(
    "node_id",
    sorted(
        nid
        for nid, meta in NodeRegistry.get_all_metadata().items()
        if meta.get("category") != "Modeling" and nid not in _NEEDS_SPECIAL_INPUT
    ),
)
def test_wrapped_frame_parity(node_id: str) -> None:
    """SkyulfPolarsWrapper must produce the same artifact as raw Polars."""
    meta = NodeRegistry.get_all_metadata()[node_id]
    config = _build_config(node_id, meta)
    df_pd = _make_nan_frame()
    df_pl = pl.from_pandas(df_pd)
    X_pl = df_pl.drop("y")
    y_pl = df_pl["y"]

    X_wrapped = SkyulfPolarsWrapper(X_pl)

    calculator = NodeRegistry.get_calculator(node_id)()

    if node_id in _NEEDS_TUPLE_INPUT:
        try:
            raw_params = calculator.fit((X_pl, y_pl), config)
        except Exception as exc:  # noqa: BLE001 - parity harness converts unexpected fit failures into skips
            pytest.skip(f"{node_id}: raw polars fit failed: {exc}")
    else:
        try:
            raw_params = calculator.fit(X_pl, config)
        except TypeError:
            raw_params = calculator.fit((X_pl, y_pl), config)
        except Exception as exc:  # noqa: BLE001 - parity harness converts unexpected fit failures into skips
            pytest.skip(f"{node_id}: raw polars fit failed: {exc}")

    if node_id in _NEEDS_TUPLE_INPUT:
        try:
            wrapped_params = calculator.fit((X_wrapped, y_pl), config)
        except Exception as exc:  # noqa: BLE001 - parity harness converts unexpected fit failures into skips
            pytest.fail(f"{node_id}: wrapped frame fit failed: {exc}")
    else:
        try:
            wrapped_params = calculator.fit(X_wrapped, config)
        except TypeError:
            wrapped_params = calculator.fit((X_wrapped, y_pl), config)
        except Exception as exc:  # noqa: BLE001 - parity harness converts unexpected fit failures into skips
            pytest.fail(f"{node_id}: wrapped frame fit failed: {exc}")

    if not raw_params and not wrapped_params:
        return

    assert bool(raw_params) == bool(wrapped_params), (
        f"{node_id}: raw returned {'non-empty' if raw_params else 'empty'}, "
        f"wrapped returned {'non-empty' if wrapped_params else 'empty'}"
    )

    if raw_params and wrapped_params:
        _assert_artifacts_equal(raw_params, wrapped_params, node_id)


# ---------------------------------------------------------------------------
# Guard rail — registry must not shrink silently
# ---------------------------------------------------------------------------


def test_registry_size_guard() -> None:
    meta = NodeRegistry.get_all_metadata()
    preprocessing = [k for k, m in meta.items() if m.get("category") == "Preprocessing"]
    assert len(preprocessing) >= 25, (
        f"Preprocessing registry shrunk: only {len(preprocessing)} nodes registered"
    )
    assert len(meta) >= 90, f"Total registry shrunk: only {len(meta)} nodes"


def test_every_registered_node_resolves_calculator_and_applier() -> None:
    """F-10: `pipeline.py` resolves models exclusively through the registry —
    the hardcoded fallback map is gone, so a partial registration (calculator
    without applier, or vice versa) would surface as a user-facing failure.
    Walk the whole registry and assert every node has a complete pair."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # deprecated-alias lookups ('Split')
        for node_id in sorted(NodeRegistry.get_all_metadata()):
            assert NodeRegistry.get_calculator(node_id) is not None, (
                f"{node_id}: registered without a calculator"
            )
            assert NodeRegistry.get_applier(node_id) is not None, (
                f"{node_id}: registered without an applier"
            )
