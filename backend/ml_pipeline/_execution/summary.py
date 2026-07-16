"""Per-node one-line summary builder rendered on the canvas card.

Each successful node writes a short ``metadata.summary`` string (≤ ~60
chars) the frontend stamps under the node title. The goal is to let
the user see *what each node actually did* without opening the
inspector.

Design constraints
------------------
* **Pure / cheap.** No I/O. Called for every node on every run.
* **Defensive.** Any exception → ``None`` → the card falls back to its
  static description. Never block a pipeline.
* **Family-aware.** Each registered transformer is classified by what
  it *does* (drop rows, encode, scale, impute, …) and the phrasing
  borrows the most informative facts for that family — strategy name,
  row delta with %, column delta with target count, headline metric +
  per-split detail (test/train) for trained models.
"""

import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from skyulf.data.dataset import SplitDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node-family classification
# ---------------------------------------------------------------------------

_FAMILY: dict[str, str] = {
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
    # Loaders.
    "dataloader": "loader",
    "data_loader": "loader",
}

# Pretty short labels for scaler implementations.
_SCALER_LABEL: dict[str, str] = {
    "standardscaler": "standard",
    "minmaxscaler": "min-max",
    "maxabsscaler": "max-abs",
    "robustscaler": "robust",
    "powertransformer": "power",
    "zscore": "z-score",
}


def _normalize(step_type: Any) -> str:
    # `step_type` may arrive as either a `StepType` member or a plain string
    # (e.g. from a Pydantic field typed as `str`). Prefer `.value` explicitly
    # rather than `str(step_type)` so both inputs normalise consistently.
    raw = getattr(step_type, "value", step_type)
    return str(raw or "").lower().replace(" ", "").replace("-", "").replace("_", "")


# Ordered keyword fallback rules used when a step_type isn't in `_FAMILY`
# verbatim. Checked in order; first match wins (mirrors the previous
# if/elif chain).
_FAMILY_KEYWORD_RULES: list[tuple[str, Callable[[str], bool]]] = [
    ("tune", lambda low: "tuning" in low),
    ("train", lambda low: "training" in low or "regress" in low or "classif" in low),
    ("split", lambda low: "split" in low and "feature" not in low),
    ("encode", lambda low: "encod" in low),
    ("impute", lambda low: "impute" in low),
    ("outlier", lambda low: "outlier" in low),
    ("select", lambda low: "select" in low),
    ("bin", lambda low: "bin" in low or "discret" in low),
    ("scale", lambda low: "scal" in low or "transform" in low),
    ("loader", lambda low: "loader" in low),
    ("snapshot", lambda low: "snapshot" in low or "profile" in low),
]


def _keyword_family_of(low: str) -> str:
    """Classify a lowercased step_type string via the keyword fallback rules."""
    for family, matches in _FAMILY_KEYWORD_RULES:
        if matches(low):
            return family
    return "generic"


def _family_of(step_type: Any) -> str:
    """Classify a step_type into one of the known families."""
    raw = getattr(step_type, "value", step_type)
    raw_str = str(raw or "")
    key = _normalize(raw_str)
    if key in _FAMILY:
        return _FAMILY[key]
    return _keyword_family_of(raw_str.lower())


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _shape_of(payload: Any) -> tuple[int, int] | None:
    """Return ``(rows, cols)`` for anything frame-shaped; ``None`` otherwise."""
    if isinstance(payload, pd.DataFrame):
        return payload.shape
    if isinstance(payload, tuple) and len(payload) >= 1:
        first = payload[0]
        if isinstance(first, pd.DataFrame):
            extra = 1 if len(payload) >= 2 else 0
            return (first.shape[0], first.shape[1] + extra)
    return None


def _dtype_breakdown(df: pd.DataFrame) -> str | None:
    """Return ``"10 num · 2 cat"`` style breakdown when a DataFrame has a
    meaningful mix of dtypes. ``None`` when it's all one kind (avoids
    redundant noise)."""
    try:
        num = int(df.select_dtypes(include="number").shape[1])
        cat = int(df.select_dtypes(include=["object", "category", "string"]).shape[1])
        dt = int(df.select_dtypes(include="datetime").shape[1])
        bo = int(df.select_dtypes(include="bool").shape[1])
        parts = []
        if num:
            parts.append(f"{num} num")
        if cat:
            parts.append(f"{cat} cat")
        if dt:
            parts.append(f"{dt} dt")
        if bo:
            parts.append(f"{bo} bool")
        # Only render when there's actual mix; a single bucket adds no info.
        if len(parts) >= 2:
            return " · ".join(parts)
        return None
    except Exception:
        return None


def _split_ratio_str(train_n: int, test_n: int, val_n: int, total: int) -> str:
    """Format train/test[/val] percentages, rounded to int and normalized to sum to 100."""
    pcts = [round(train_n / total * 100), round(test_n / total * 100)]
    if val_n:
        pcts.append(round(val_n / total * 100))
    drift = 100 - sum(pcts)
    if drift:
        pcts[pcts.index(max(pcts))] += drift
    return "/".join(str(p) for p in pcts) + "%"


def _split_summary(data: SplitDataset) -> str | None:
    """Format ``train / test [/ val]`` row counts with ratio + column count."""
    train = _shape_of(data.train)
    test = _shape_of(data.test)
    if train is None or test is None:
        return None
    train_n, test_n = train[0], test[0]
    val_n = 0
    if data.validation is not None:
        v = _shape_of(data.validation)
        if v is not None:
            val_n = v[0]
    total = train_n + test_n + val_n
    if total <= 0:
        return None

    # Absolute row counts.
    parts = [_fmt_int(train_n), _fmt_int(test_n)]
    if val_n:
        parts.append(_fmt_int(val_n))
    abs_str = " / ".join(parts)

    # Ratio percentages — the single most useful piece of context the
    # absolute numbers are missing.
    ratio_str = _split_ratio_str(train_n, test_n, val_n, total)

    return f"{ratio_str} · {abs_str} × {train[1]} cols"


def _delta_phrase(in_shape: tuple[int, int], out_shape: tuple[int, int]) -> str | None:
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


def _strategy_label(params: Mapping[str, Any], *keys: str) -> str | None:
    """Pull the first non-empty string param under any of ``keys``."""
    for key in keys:
        v = params.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _scaler_label(step_type: Any) -> str:
    return _SCALER_LABEL.get(_normalize(step_type), "scaled")


def _loader_or_snapshot(out: pd.DataFrame) -> str:
    """Loader / snapshot phrasing — adds dtype mix when meaningful."""
    rows, cols = out.shape
    breakdown = _dtype_breakdown(out)
    head = f"{_fmt_int(rows)} rows × {cols} cols"
    return f"{head} ({breakdown})" if breakdown else head


def _scale_same_shape_label(step_type: Any, out_cols: int, params: Mapping[str, Any]) -> str:
    """Same-shape phrasing for the ``scale`` family."""
    return f"{_scaler_label(step_type)} · {out_cols} cols"


def _impute_same_shape_label(step_type: Any, out_cols: int, params: Mapping[str, Any]) -> str:
    """Same-shape phrasing for the ``impute`` family."""
    strat = _strategy_label(params, "strategy", "method")
    base = f"{strat} imputer" if strat else "imputed"
    return f"{base} · {out_cols} cols"


def _bin_same_shape_label(step_type: Any, out_cols: int, params: Mapping[str, Any]) -> str:
    """Same-shape phrasing for the ``bin`` family."""
    n = params.get("n_bins") or params.get("bins")
    head = f"binned ({n} bins)" if isinstance(n, int) else "binned"
    return f"{head} · {out_cols} cols"


def _replace_same_shape_label(step_type: Any, out_cols: int, params: Mapping[str, Any]) -> str:
    """Same-shape phrasing for the ``replace`` family."""
    return f"cleaned · {out_cols} cols"


def _encode_same_shape_label(step_type: Any, out_cols: int, params: Mapping[str, Any]) -> str:
    """Same-shape phrasing for the ``encode`` family."""
    return f"encoded · {out_cols} cols"


# Family → same-shape renderer. Mirrors the family-renderer registry below —
# a one-line table edit instead of a growing if/elif chain.
_SAME_SHAPE_RENDERERS: dict[str, Callable[[Any, int, Mapping[str, Any]], str]] = {
    "scale": _scale_same_shape_label,
    "impute": _impute_same_shape_label,
    "bin": _bin_same_shape_label,
    "replace": _replace_same_shape_label,
    "encode": _encode_same_shape_label,
}


def _same_shape_label(
    family: str, step_type: Any, out_cols: int, params: Mapping[str, Any]
) -> str | None:
    """Name what a same-shape transform did, with strategy detail when known.

    Returns ``None`` when ``family`` has no dedicated same-shape phrasing.
    """
    renderer = _SAME_SHAPE_RENDERERS.get(family)
    if renderer is None:
        return None
    return renderer(step_type, out_cols, params)


def _delta_baseline_phrase(
    family: str, in_shape: tuple[int, int], out_shape: tuple[int, int]
) -> str | None:
    """Render a row/col delta phrase when a baseline shape is available.

    Returns ``None`` when there is no baseline delta (same shape).
    """
    delta = _delta_phrase(in_shape, out_shape)
    if not delta:
        return None
    out_cols = out_shape[1]
    if family in {"select", "encode", "generate", "bin"}:
        return f"{delta} → {out_cols} cols"
    return delta


def _frame_branch(
    family: str,
    step_type: Any,
    out: pd.DataFrame,
    in_shape: tuple[int, int] | None,
    params: Mapping[str, Any],
) -> str:
    """Render a one-liner for a node whose output is a single DataFrame."""
    out_rows, out_cols = out.shape

    # Loaders + snapshots: enrich the shape line with a dtype mix when
    # a mixed schema is present.
    if family in {"loader", "snapshot"}:
        return _loader_or_snapshot(out)

    # Prefer a row/col delta when we have a baseline.
    if in_shape is not None:
        delta = _delta_baseline_phrase(family, in_shape, out.shape)
        if delta:
            return delta

        # Same shape — name what we did, with strategy detail when known.
        same_shape = _same_shape_label(family, step_type, out_cols, params)
        if same_shape is not None:
            return same_shape

    # No baseline available — fall back to plain shape line.
    return f"{_fmt_int(out_rows)} rows × {out_cols} cols"


# ---------------------------------------------------------------------------
# Trainer / tuner phrasing
# ---------------------------------------------------------------------------


def _first_finite(metrics: Mapping[str, Any], candidates: Iterable[str]) -> float | None:
    """Return the first finite numeric value found under any of ``candidates``."""
    for key in candidates:
        v = metrics.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            if f == f:  # filter NaN
                return f
    return None


def _expand(name: str) -> tuple[str, ...]:
    """Search order: prefer ``test_*``, then bare, then ``val_*``, then ``train_*``."""
    return (f"test_{name}", name, f"val_{name}", f"train_{name}")


# Per-metric search keys, ordered. Both binary and multiclass variants
# fold into the same family so the headline picks whichever exists.
_F1_KEYS = _expand("f1") + _expand("f1_weighted") + _expand("f1_score")
_AUC_KEYS = _expand("roc_auc") + _expand("auc") + _expand("roc_auc_weighted")
_R2_KEYS = _expand("r2") + _expand("r2_score")
_RMSE_KEYS = _expand("rmse") + _expand("root_mean_squared_error")
_MAE_KEYS = _expand("mae") + _expand("mean_absolute_error")
_MSE_KEYS = _expand("mse") + _expand("mean_squared_error")


def _train_only(metrics: Mapping[str, Any], candidates: Iterable[str]) -> float | None:
    """Same as ``_first_finite`` but only checks ``train_*`` keys."""
    for key in candidates:
        if not key.startswith("train_"):
            continue
        v = metrics.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                f = float(v)
                if f == f:
                    return f
            except (TypeError, ValueError):
                continue
    return None


def _gap_suffix(test_v: float, train_v: float | None, higher_is_better: bool) -> str:
    """Render ``"  ▲0.12"`` when the train→test gap is meaningful overfit.

    Empty string when train is missing or the gap is small (< 0.05).
    """
    if train_v is None:
        return ""
    diff = (train_v - test_v) if higher_is_better else (test_v - train_v)
    if diff < 0.05:
        return ""
    return f" · ▲{diff:.2f}"


def _classification_headline(inner: Mapping[str, Any]) -> str | None:
    """Build the classification headline (acc / f1 / auc), or ``None`` if absent."""
    acc = _first_finite(inner, _expand("accuracy"))
    f1 = _first_finite(inner, _F1_KEYS)
    if acc is not None and f1 is not None:
        gap = _gap_suffix(acc, _train_only(inner, _expand("accuracy")), higher_is_better=True)
        return f"acc {acc:.2f} · f1 {f1:.2f}{gap}"
    if acc is not None:
        gap = _gap_suffix(acc, _train_only(inner, _expand("accuracy")), higher_is_better=True)
        return f"acc {acc:.2f}{gap}"
    auc = _first_finite(inner, _AUC_KEYS)
    if auc is not None:
        return f"auc {auc:.2f}"
    return None


def _regression_headline(inner: Mapping[str, Any]) -> str | None:
    """Build the regression headline (r² / rmse / mae / mse), or ``None`` if absent."""
    r2 = _first_finite(inner, _R2_KEYS)
    rmse = _first_finite(inner, _RMSE_KEYS)
    if r2 is not None and rmse is not None:
        gap = _gap_suffix(r2, _train_only(inner, _R2_KEYS), higher_is_better=True)
        return f"r² {r2:.2f} · rmse {rmse:.3g}{gap}"
    if r2 is not None:
        gap = _gap_suffix(r2, _train_only(inner, _R2_KEYS), higher_is_better=True)
        return f"r² {r2:.2f}{gap}"
    mae = _first_finite(inner, _MAE_KEYS)
    if mae is not None:
        return f"mae {mae:.3g}"
    mse = _first_finite(inner, _MSE_KEYS)
    if mse is not None:
        return f"mse {mse:.3g}"
    return None


def _training_summary(metrics: Mapping[str, Any]) -> str | None:
    """Build the headline metric line for a trained model.

    Strategy: probe for classification metrics first (acc / f1 / auc),
    then regression (r² / rmse / mae). For each headline, append a
    train→test overfit gap badge when both splits exist and the gap is
    meaningful (≥ 0.05).
    """
    inner: Mapping[str, Any]
    nested = metrics.get("metrics") if isinstance(metrics, Mapping) else None
    inner = nested if isinstance(nested, Mapping) else metrics
    if not isinstance(inner, Mapping):
        return None

    classification = _classification_headline(inner)
    if classification is not None:
        return classification

    return _regression_headline(inner)


# Short labels used in the tuning headline. Mirrors what the JobsDrawer
# renders next to "Best Score (…)" so the card and the drawer agree on
# wording. Anything not listed falls back to the raw scoring name.
_SCORING_LABEL: dict[str, str] = {
    "accuracy": "acc",
    "balanced_accuracy": "bal-acc",
    "f1": "f1",
    "f1_weighted": "f1w",
    "f1_macro": "f1m",
    "f1_micro": "f1µ",
    "precision": "prec",
    "precision_weighted": "precw",
    "recall": "rec",
    "recall_weighted": "recw",
    "roc_auc": "auc",
    "roc_auc_weighted": "aucw",
    "roc_auc_ovr": "auc",
    "roc_auc_ovr_weighted": "aucw",
    "average_precision": "ap",
    "neg_log_loss": "logloss",
    "r2": "r²",
    "neg_mean_absolute_error": "mae",
    "neg_mean_squared_error": "mse",
    "neg_root_mean_squared_error": "rmse",
    "neg_mean_absolute_percentage_error": "mape",
}


def _n_trials(metrics: Mapping[str, Any]) -> int | None:
    """Return the trial count from either a list of trials or an int field."""
    trials = metrics.get("trials")
    if isinstance(trials, list):
        return len(trials)
    if isinstance(trials, int):
        return trials
    return None


def _best_score_value_and_label(metrics: Mapping[str, Any], best: float) -> tuple[float, str]:
    """Compute the display value (sign-flipped for neg_* scorers) and label for best_score."""
    scoring_raw = str(metrics.get("scoring_metric") or "").strip()
    # Most sklearn losses are reported as `neg_*` (higher is better).
    # Flip the sign so the card shows the natural magnitude.
    value = float(best)
    if scoring_raw.startswith("neg_") and value <= 0:
        value = -value
    label = _SCORING_LABEL.get(scoring_raw) or (scoring_raw.replace("_", " ") or "score")
    return value, label


def _best_score_headline(metrics: Mapping[str, Any], n_trials: int | None) -> str | None:
    """Render the "<label> <value>" headline from ``best_score``, or ``None`` if absent."""
    best = metrics.get("best_score")
    if not isinstance(best, int | float) or isinstance(best, bool):
        return None
    value, label = _best_score_value_and_label(metrics, best)
    head = f"{label} {value:.3f}"
    return f"{head} · {n_trials} trials" if n_trials else head


def _tuning_summary(metrics: Mapping[str, Any]) -> str | None:
    """Hyperparameter-tuning summary.

    Leads with the tuner's own ``best_score`` + scoring-metric short
    label, because that is the headline number the JobsDrawer shows
    ("Best Score (F1 Weighted) 0.9324"). The post-tuning test-set
    evaluation lives further down in the drawer; using *that* as the
    card line caused two recurring complaints:

    * It didn't match the prominent "Best Score" the user just read.
    * Small test sets often round identically across re-tunes, so the
      card looked stale even when the tuner improved.

    Falls back to the eval headline when no ``best_score`` is present
    (e.g. legacy job rows), then to nothing.
    """
    n_trials = _n_trials(metrics)

    headline = _best_score_headline(metrics, n_trials)
    if headline is not None:
        return headline

    # Legacy fallback: no best_score recorded — use eval headline.
    base = _training_summary(metrics)
    if base is not None:
        return f"{base} · {n_trials} trials" if n_trials else base
    return None


# ---------------------------------------------------------------------------
# Renderer registry — one base shape, one entry per family
# ---------------------------------------------------------------------------
#
# L1 follow-up: the previous ``build_summary`` body was an inline
# if/elif union over every family, which made adding a new node kind
# touch the orchestrator. A small registry keeps families isolated:
# each renderer takes the same ``SummaryContext`` and returns
# ``Optional[str]``. Adding a family is now a one-line table edit, not
# an orchestrator surgery.


@dataclass(frozen=True)
class SummaryContext:
    """Inputs every family renderer receives.

    Frozen so renderers can't mutate it by accident — the orchestrator
    builds one and hands the same instance to whichever renderer the
    family table dispatches to.
    """

    step_type: Any
    output: Any
    metrics: Mapping[str, Any]
    input_shape: tuple[int, int] | None
    params: Mapping[str, Any]


SummaryRenderer = Callable[[SummaryContext], str | None]


def _render_tune(ctx: SummaryContext) -> str | None:
    return _tuning_summary(ctx.metrics)


def _render_train(ctx: SummaryContext) -> str | None:
    return _training_summary(ctx.metrics)


def _render_split(ctx: SummaryContext) -> str | None:
    if isinstance(ctx.output, SplitDataset):
        return _split_summary(ctx.output)
    return None


def _render_frame(ctx: SummaryContext) -> str | None:
    """Default renderer for any family whose output is a DataFrame.

    Covers loaders, snapshots, encoders, scalers, imputers, drops,
    outliers, selectors, binning, generators, replacements and
    sampling — _frame_branch already dispatches on family for the
    actual phrasing.
    """
    if isinstance(ctx.output, pd.DataFrame):
        family = _family_of(ctx.step_type)
        return _frame_branch(family, ctx.step_type, ctx.output, ctx.input_shape, ctx.params)
    return None


# Family → renderer. Anything not listed falls through to the
# DataFrame renderer when applicable (covers all "shape-changing"
# data-prep families with one entry).
_RENDERERS: dict[str, SummaryRenderer] = {
    "tune": _render_tune,
    "train": _render_train,
    "split": _render_split,
    # Frame-output families share one renderer.
    "loader": _render_frame,
    "snapshot": _render_frame,
    "encode": _render_frame,
    "scale": _render_frame,
    "impute": _render_frame,
    "drop": _render_frame,
    "outlier": _render_frame,
    "select": _render_frame,
    "bin": _render_frame,
    "generate": _render_frame,
    "replace": _render_frame,
    "resample": _render_frame,
    "generic": _render_frame,
}


def register_renderer(family: str, renderer: SummaryRenderer) -> None:
    """Plug a custom renderer in for a family. Used by tests / plugins."""
    _RENDERERS[family] = renderer


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_summary(
    *,
    step_type: Any,
    output: Any,
    metrics: dict[str, Any],
    input_shape: tuple[int, int] | None = None,
    params: Mapping[str, Any] | None = None,
) -> str | None:
    """Build the one-line node-card summary.

    Parameters
    ----------
    step_type:
        Node kind string (e.g. ``"DropMissingRows"``, ``"basic_training"``).
    output:
        The artifact written by the node (DataFrame, SplitDataset, model,
        …). Trainers do not need a usable output — the summary is built
        purely from ``metrics`` for the ``train`` / ``tune`` families,
        so passing ``None`` is safe there.
    metrics:
        Dict returned by the runner. Training runners flatten each
        split's metrics under ``train_*`` / ``test_*`` / ``val_*``;
        tuning adds ``best_score`` / ``scoring_metric`` / ``trials``.
        Both layouts are tolerated.
    input_shape:
        Optional ``(rows, cols)`` of the upstream artifact. Lets us
        render ``"−127 rows (1.8%)"`` for filters / encoders / selectors.
    params:
        Optional node config params — used to name strategies
        (``mean imputer``, ``standard``, ``binned (5 bins)``).
    """
    try:
        family = _family_of(step_type)
        ctx = SummaryContext(
            step_type=step_type,
            output=output,
            metrics=metrics,
            input_shape=input_shape,
            params=params or {},
        )
        # Splitter outputs are dispatched by *type* not family — some
        # nodes registered as `split` only emit SplitDataset, while
        # others (legacy) hand back a tuple. The renderer handles both.
        if isinstance(output, SplitDataset):
            return _render_split(ctx)
        renderer = _RENDERERS.get(family, _render_frame)
        return renderer(ctx)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("summary build failed for step=%s: %s", step_type, exc)
        return None


__all__ = [
    "build_summary",
    "register_renderer",
    "SummaryContext",
    "SummaryRenderer",
    "_family_of",
]
