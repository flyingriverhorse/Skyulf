"""Build the short human summary that the canvas renders inside a node card.

Each post-run node carries a `metadata.summary` string (≤ ~40 chars) so
the user can read what happened without opening the inspector. This
module is the single place that knows how to format those strings — the
engine calls :func:`build_summary` after every successful node and the
result is stitched onto :class:`NodeExecutionResult.metadata`.

Design constraints:

* **Pure / cheap.** No I/O, no model introspection beyond what's already
  in memory. We're called for every node on every run.
* **Defensive.** A bad summary must never break a pipeline. Any
  exception is swallowed (the node simply renders without a summary).
* **Stable across job types.** The frontend treats the string as opaque,
  so we can change phrasing freely without breaking contracts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import pandas as pd

from skyulf.data.dataset import SplitDataset

logger = logging.getLogger(__name__)


def _fmt_int(value: int) -> str:
    """Thousands-separated integer (`7000` -> `'7,000'`)."""
    return f"{value:,}"


def _shape_of(payload: Any) -> Optional[tuple[int, int]]:
    """Return (rows, cols) for anything frame-shaped; ``None`` otherwise."""
    if isinstance(payload, pd.DataFrame):
        return payload.shape
    # SplitDataset payload may be (X, y) — count y as a column.
    if isinstance(payload, tuple) and len(payload) >= 1:
        first = payload[0]
        if isinstance(first, pd.DataFrame):
            extra = 1 if len(payload) >= 2 else 0
            return (first.shape[0], first.shape[1] + extra)
    return None


def _split_summary(data: SplitDataset) -> Optional[str]:
    """Format `train / test [/ val]` row counts."""
    train = _shape_of(data.train)
    test = _shape_of(data.test)
    if train is None or test is None:
        return None
    parts = [_fmt_int(train[0]), _fmt_int(test[0])]
    if data.validation is not None:
        val = _shape_of(data.validation)
        if val is not None:
            parts.append(_fmt_int(val[0]))
    return " / ".join(parts)


def _frame_summary(data: pd.DataFrame) -> str:
    rows, cols = data.shape
    return f"{_fmt_int(rows)} rows × {cols} cols"


def _training_summary(metrics: Dict[str, Any]) -> Optional[str]:
    """Pick the most relevant 1–2 metrics for a model card.

    Falls back gracefully when the trainer didn't surface anything we
    recognise — there's no point hand-coding every algorithm's metric
    set.
    """
    # Common shapes: {'metrics': {'accuracy': 0.87, ...}} or flat dict.
    inner = metrics.get("metrics") if isinstance(metrics.get("metrics"), dict) else metrics
    if not isinstance(inner, dict):
        return None

    def _pick(key: str) -> Optional[float]:
        v = inner.get(key)
        return v if isinstance(v, (int, float)) else None

    # Classification: prefer accuracy + f1; fall back to roc_auc.
    acc = _pick("accuracy") or _pick("test_accuracy")
    f1 = _pick("f1") or _pick("f1_score") or _pick("test_f1")
    if acc is not None and f1 is not None:
        return f"acc {acc:.2f} · f1 {f1:.2f}"
    if acc is not None:
        return f"acc {acc:.2f}"
    auc = _pick("roc_auc") or _pick("auc")
    if auc is not None:
        return f"auc {auc:.2f}"

    # Regression: r2 + rmse / mae.
    r2 = _pick("r2") or _pick("r2_score")
    rmse = _pick("rmse") or _pick("root_mean_squared_error")
    if r2 is not None and rmse is not None:
        return f"r² {r2:.2f} · rmse {rmse:.3g}"
    if r2 is not None:
        return f"r² {r2:.2f}"
    mae = _pick("mae") or _pick("mean_absolute_error")
    if mae is not None:
        return f"mae {mae:.3g}"

    return None


def build_summary(
    *,
    step_type: Any,
    output: Any,
    metrics: Dict[str, Any],
) -> Optional[str]:
    """Build the one-line summary for a node, given its run result.

    Returns ``None`` when nothing useful can be said — the frontend then
    falls back to the static description.
    """
    try:
        step = str(step_type or "").lower()

        # Splitters: the most informative single fact is per-split row
        # counts. We treat both train_test_split and the train/val/test
        # variant the same way thanks to SplitDataset's optional .validation.
        if isinstance(output, SplitDataset):
            return _split_summary(output)

        # Trainers: surface the headline metric(s).
        if "training" in step or "train" in step or "tuning" in step:
            return _training_summary(metrics)

        # Everything else with a frame output: shape line. This covers
        # data loaders + every preprocessing transformer (drop, encode,
        # scale, impute, outlier, bin, feature_select, ...).
        if isinstance(output, pd.DataFrame):
            return _frame_summary(output)

        return None
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("summary build failed for step=%s: %s", step_type, exc)
        return None
