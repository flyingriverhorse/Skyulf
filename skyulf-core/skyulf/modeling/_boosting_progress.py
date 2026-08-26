"""Per-boosting-iteration progress adapters.

XGBoost and LightGBM build their trees one boosting round at a time and both
expose native callback hooks. These adapters translate those hooks into the
``IterationCallback`` protocol used by the calculators — so a boosting fit can
stream ``(current, total, score, metric, direction)`` points without changing
what the model learns: the eval set is display-only (no early stopping is
configured anywhere), so fits with and without a callback produce identical
models.

The score is evaluated on the training frame itself — these curves are
training-progress telemetry for the live job chart, not generalization
estimates (the job's real metrics still come from the held-out test split).
"""

import contextlib
from collections.abc import Callable
from typing import Any

IterationCallback = Callable[[int, int, float, str, str], None]
"""``callback(current, total, score, metric, direction)`` — 1-based ``current``,
``direction`` is ``"minimize"`` or ``"maximize"``."""

_MAXIMIZE_METRICS = frozenset({"auc", "aucpr", "map", "ndcg", "precession", "precision"})


def direction_for_xgb_metric(metric: str) -> str:
    """XGBoost's callback env doesn't say which way is better; losses and
    error rates dominate its metric names, so default to minimize and
    whitelist the ranking metrics."""
    return "maximize" if metric.lower() in _MAXIMIZE_METRICS else "minimize"


XgboostIterationAdapter: type | None

try:
    import xgboost.callback as _xgb_callback

    class _XgboostIterationAdapter(_xgb_callback.TrainingCallback):
        """Fires the iteration callback after every boosting round.

        XGBoost 3.x's callback protocol is ``after_iteration(model, epoch,
        evals_log)`` and exposes no total, so the final round count is
        captured up front from the estimator's ``n_estimators``.
        """

        def __init__(self, callback: IterationCallback, total: int) -> None:
            super().__init__()
            self._callback = callback
            self._total = int(total)

        def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
            for metrics in (evals_log or {}).values():
                for metric, values in metrics.items():
                    if values:
                        with contextlib.suppress(Exception):  # telemetry never breaks training
                            self._callback(
                                int(epoch) + 1,
                                self._total,
                                float(values[-1]),
                                str(metric),
                                direction_for_xgb_metric(str(metric)),
                            )
                break  # first eval set only
            return False

    XgboostIterationAdapter = _XgboostIterationAdapter
except ImportError:  # pragma: no cover - optional dependency
    XgboostIterationAdapter = None


class LightGBMIterationAdapter:
    """LightGBM callbacks are plain callables receiving the booster env."""

    def __init__(self, callback: IterationCallback) -> None:
        self._callback = callback

    def __call__(self, env: Any) -> None:
        results = getattr(env, "evaluation_result_list", None) or []
        if not results:
            return
        _dataset, metric, value, is_higher_better = results[0]
        with contextlib.suppress(Exception):  # telemetry never breaks training
            self._callback(
                int(env.iteration) + 1,
                int(env.end_iteration),
                float(value),
                str(metric),
                "maximize" if is_higher_better else "minimize",
            )
