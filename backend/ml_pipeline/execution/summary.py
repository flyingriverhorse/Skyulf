"""Per-node one-line summary builder rendered on the canvas card.

Each successful node writes a short ``metadata.summary`` string (≤ ~50
chars) the frontend stamps under the node title. This lets the user see
*what each node actually did* without opening the inspector.

Design constraints
------------------
* **Pure / cheap.** No I/O. Called for every node on every run.
* **Defensive.** Any exception → ``None`` → the card falls back to its
  static description. Never block a pipeline.
* **Family-aware.** Each registered transformer is classified by what
  it *does* (drop rows, encode, scale, impute, …) and the phrasing
  borrows the most informative fact for that family — strategy name,
  row delta with %, column delta with target count, headline metric.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import pandas as pd

from skyulf.data.dataset import SplitDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node-family classification
# ---------------------------------------------------------------------------
# Mapping registered step_type id (case/sep-insensitive) -> family tag.

_FAMILY: Dict[str, str] = {
    # Encoders — typically expand column count.
    "dummyencoder": "encode",
    "onehotencoder": "encode",
    "ordinalencoder": "encode",
    "labelencoder": "encode",
    "targetencoder": "encode",
    "hashencoder": "encode",
    # Scalers — same shape, just rescaled values.
    "minmaxscaler": "scale",
    "maxabsscaler": "scale",
    "robustscaler": "scale",
    "standardscaler": "scale",
    "powertransformer": "scale",
    "zscore": "scale",
    # Imputers — same shape, fewer NaNs.
    "simpleimputer": "impute",
    "knnimputer": "impute",
    "iterativeimputer": "impute",
    "missingindicator": "impute",
    # Drops / dedup.
    "dropmissingcolumns": "drop",
    "dropmissingrows": "drop",
    "deduplicate": "drop",
    # Outlier handlers.
    "iqr": "outlier",
    "ellipticenvelope": "outlier",
    "winsorize": "outlier",
    "manualbounds": "outlier",
    # Feature selection — fewer cols.
    "correlationthreshold": "select",
    "variancethreshold": "select",
    "univariateselection": "select",
    "modelbasedselection": "select",
    "featureselection": "select",
    # Binning — typically same col count, value-replaced.
    "custombinning": "bin",
    "generalbinning": "bin",
    "kbinsdiscretizer": "bin",
    # Feature generation — adds cols.
    "featuregeneration": "generate",
    "featuregenerationnode": "generate",
    "featuremath": "generate",
    "polynomialfeatures": "generate",
    "polynomialfeaturesnode": "generate",
    "simpletransformation": "generate",
    "generaltransformation": "generate",
    # Replacements / cleaning — same shape.
    "valuereplacement": "replace",
    "aliasreplacement": "replace",
    "invalidvaluereplacement": "replace",
    "textcleaning": "replace",
    "casting": "replace",
    # Sampling — changes rows.
    "oversampling": "resample",
    "undersampling": "resample",
    # Splitters — produce SplitDataset (handled separately).
    "traintestsplitter": "split",
    "split": "split",
    "featuretargetsplit": "split",
    # Profiling / snapshot — pass-through frames.
    "datasnapshot": "snapshot",
    "datasetprofile": "snapshot",
}

# Verb word stamped onto same-shape transformers when no delta exists.
_VERB_BY_FAMILY: Dict[str, str] = {
    "scale": "scaled",
    "impute": "imputed",
    "replace": "cleaned",
    "bin": "binned",
    "encode": "encoded",
    "generate": "generated",
}

# Pretty short labels for scaler implementations.
_SCALER_LABEL: Dict[str, str] = {
    "standardscaler": "standard",
    "minmaxscaler": "min-max",
    "maxabsscaler": "max-abs",
    "robustscaler": "robust",
    "powertransformer": "power",
    "zscore": "z-score",
}


def _normalize(step_type: Any) -> str:
    return str(step_type or "").lower().replace(" ", "").replace("-", "").replace("_", "")


def _family_of(step_type: Any) -> str:
    """Classify a step_type into one of the known families."""
    raw = str(step_type or "")
    key = _normalize(raw)
    if key in _FAMILY:
        return _FAMILY[key]
    low = raw.lower()
    if "training" in low or "tuning" in low or "regress" in low or "classif" in low:
        return "train"
    if "split" in low and "feature" not in low:
        return "split"
    if "encod" in low:
        return "encode"
    if "impute" in low:
        return "impute"
    if "outlier" in low:
        return "outlier"
    if "select" in low:
        return "select"
    if "bin" in low or "discret" in low:
        return "bin"
    if "scal" in low or "transform" in low:
        return "scale"
    if "loader" in low or "snapshot" in low or "profile" in low:
        return "snapshot"
    return "generic"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _shape_of(payload: Any) -> Optional[Tuple[int, int]]:
    """Return ``(rows, cols)`` for anything frame-shaped; ``None`` otherwise."""
    if isinstance(payload, pd.DataFrame):
        return payload.shape
    if isinstance(payload, tuple) and len(payload) >= 1:
        first = payload[0]
        if isinstance(first, pd.DataFrame):
            extra = 1 if len(payload) >= 2 else 0
            return (first.shape[0], first.shape[1] + extra)
    return None


def _split_summary(data: SplitDataset) -> Optional[str]:
    """Format ``train / test [/ val]`` row counts plus column count."""
    train = _shape_of(data.train)
    test = _shape_of(data.test)
    if train is None or test is None:
        return None
    parts = [_fmt_int(train[0]), _fmt_int(test[0])]
    if data.validation is not None:
        val = _shape_of(data.validation)
        if val is not None:
            parts.append(_fmt_int(val[0]))
    return f"{' / '.join(parts)} rows × {train[1]} cols"


def _delta_phrase(in_shape: Tuple[int, int], out_shape: Tuple[int, int]) -> Optional[str]:
    """Render the row/col delta between input and output shapes."""
    in_rows, in_cols = in_shape
    out_rows, out_cols = out_shape
    drow = out_rows - in_rows
    dcol = out_cols - in_cols
    if drow == 0 and dcol == 0:
        return None
    chunks = []
    if drow != 0:
        sign = "+" if drow > 0 else "−"
        if in_rows:
            pct = abs(drow) / in_rows * 100
            chunks.append(f"{sign}{_fmt_int(abs(drow))} rows ({pct:.1f}%)")
        else:
            chunks.append(f"{sign}{_fmt_int(abs(drow))} rows")
    if dcol != 0:
        sign = "+" if dcol > 0 else "−"
        chunks.append(f"{sign}{abs(dcol)} cols")
    return " · ".join(chunks)


# ---------------------------------------------------------------------------
# Family-specific phrasing for DataFrame outputs
# ---------------------------------------------------------------------------


def _strategy_label(params: Mapping[str, Any], *keys: str) -> Optional[str]:
    """Pull the first non-empty string param under any of ``keys``."""
    for key in keys:
        v = params.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _scaler_label(step_type: Any) -> str:
    return _SCALER_LABEL.get(_normalize(step_type), "scaled")


def _frame_branch(
    family: str,
    step_type: Any,
    out: pd.DataFrame,
    in_shape: Optional[Tuple[int, int]],
    params: Mapping[str, Any],
) -> str:
    """Render a one-liner for a node whose output is a single DataFrame."""
    out_rows, out_cols = out.shape

    # Prefer a row/col delta when we have a baseline — that's the most
    # informative single fact for transformers that change the frame.
    if in_shape is not None:
        delta = _delta_phrase(in_shape, out.shape)
        if delta:
            if family in {"select", "encode", "generate", "bin"}:
                return f"{delta} → {out_cols} cols"
            return delta

        # Same shape — name what we did, with strategy detail when known.
        if family == "scale":
            return f"{_scaler_label(step_type)} · {out_cols} cols"
        if family == "impute":
            strat = _strategy_label(params, "strategy", "method")
            base = f"{strat} imputer" if strat else "imputed"
            return f"{base} · {out_cols} cols"
        if family == "bin":
            n = params.get("n_bins") or params.get("bins")
            head = f"binned ({n} bins)" if isinstance(n, int) else "binned"
            return f"{head} · {out_cols} cols"
        if family == "replace":
            return f"cleaned · {out_cols} cols"
        if family == "encode":
            return f"encoded · {out_cols} cols"

    # No baseline available — fall back to plain shape line.
    return f"{_fmt_int(out_rows)} rows × {out_cols} cols"


# ---------------------------------------------------------------------------
# Trainer phrasing — robust to the prefixed key layout produced by
# ``_run_basic_training`` / ``_run_advanced_tuning`` (test_*, train_*,
# val_* and the bare-key form some unit tests use).
# ---------------------------------------------------------------------------


def _first_finite(metrics: Mapping[str, Any], candidates: Iterable[str]) -> Optional[float]:
    """Return the first numeric value found under any of ``candidates``."""
    for key in candidates:
        v = metrics.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            # Skip NaN.
            if f == f:
                return f
    return None


# Search order matters: prefer test_, then bare, then val_, then train_.
def _expand(name: str) -> Tuple[str, ...]:
    return (f"test_{name}", name, f"val_{name}", f"train_{name}")


def _training_summary(metrics: Mapping[str, Any]) -> Optional[str]:
    """Build the headline metric line for a trained model.

    Strategy: probe for classification metrics first (acc / f1 / auc),
    then regression (r² / rmse / mae). Each metric is searched under
    the most informative prefix order.
    """
    inner: Mapping[str, Any]
    nested = metrics.get("metrics") if isinstance(metrics, Mapping) else None
    inner = nested if isinstance(nested, Mapping) else metrics
    if not isinstance(inner, Mapping):
        return None

    # ---- Classification ----
    acc = _first_finite(inner, _expand("accuracy"))
    f1 = _first_finite(
        inner,
        _expand("f1") + _expand("f1_weighted") + _expand("f1_score"),
    )
    if acc is not None and f1 is not None:
        return f"acc {acc:.2f} · f1 {f1:.2f}"
    if acc is not None:
        return f"acc {acc:.2f}"
    auc = _first_finite(inner, _expand("roc_auc") + _expand("auc") + _expand("roc_auc_weighted"))
    if auc is not None:
        return f"auc {auc:.2f}"

    # ---- Regression ----
    r2 = _first_finite(inner, _expand("r2") + _expand("r2_score"))
    rmse = _first_finite(inner, _expand("rmse") + _expand("root_mean_squared_error"))
    if r2 is not None and rmse is not None:
        return f"r² {r2:.2f} · rmse {rmse:.3g}"
    if r2 is not None:
        return f"r² {r2:.2f}"
    mae = _first_finite(inner, _expand("mae") + _expand("mean_absolute_error"))
    if mae is not None:
        return f"mae {mae:.3g}"
    mse = _first_finite(inner, _expand("mse") + _expand("mean_squared_error"))
    if mse is not None:
        return f"mse {mse:.3g}"

    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_summary(
    *,
    step_type: Any,
    output: Any,
    metrics: Dict[str, Any],
    input_shape: Optional[Tuple[int, int]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Build the one-line node-card summary.

    Parameters
    ----------
    step_type:
        Node kind string (e.g. ``"DropMissingRows"``, ``"basic_training"``).
    output:
        The artifact written by the node (DataFrame, SplitDataset, model, …).
    metrics:
        Dict returned by the runner. Training runners flatten each
        split's metrics under ``train_*`` / ``test_*`` / ``val_*``;
        feature-engineering runners often nest under ``"metrics"``.
        Both layouts are tolerated.
    input_shape:
        Optional ``(rows, cols)`` of the upstream artifact. Lets us
        render ``"−127 rows (1.8%)"`` for filters / encoders / selectors.
    params:
        Optional node config params — used to name strategies
        (``mean imputer``, ``standard scaler``, ``binned (5 bins)``).
    """
    try:
        family = _family_of(step_type)
        params = params or {}

        # Splitters: per-slot row counts beat any other phrasing.
        if isinstance(output, SplitDataset):
            return _split_summary(output)

        # Trainers: model output (or anything non-frame) — headline metric.
        if family == "train":
            return _training_summary(metrics)

        # Frames: family-aware delta or shape line.
        if isinstance(output, pd.DataFrame):
            return _frame_branch(family, step_type, output, input_shape, params)

        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("summary build failed for step=%s: %s", step_type, exc)
        return None


# Re-export for tests / external callers that want to inspect families.
__all__ = ["build_summary", "_family_of"]
