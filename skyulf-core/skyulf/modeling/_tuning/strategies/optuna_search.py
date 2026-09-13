"""Optuna study search with fold-local fitting and genuine pruning checkpoints."""

import math
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError
from typing import Any

import optuna  # ty: ignore[unresolved-import]  # optional module, loaded only for Optuna search
from sklearn.base import clone, is_classifier
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.utils import _safe_indexing, indexable

from .optuna_folds import fit_and_score_fold


class _TrialFitFailure(Exception):
    """Mark ordinary training failures that Optuna may record and continue past."""


def _suggest_parameter(trial: Any, name: str, distribution: Any) -> Any:
    """Sample one supported Optuna distribution using its public suggestion API."""
    if isinstance(distribution, optuna.distributions.CategoricalDistribution):
        return trial.suggest_categorical(name, distribution.choices)
    if isinstance(distribution, optuna.distributions.IntDistribution):
        return trial.suggest_int(
            name, distribution.low, distribution.high, step=distribution.step, log=distribution.log
        )
    if isinstance(distribution, optuna.distributions.FloatDistribution):
        return trial.suggest_float(
            name, distribution.low, distribution.high, step=distribution.step, log=distribution.log
        )
    raise ValueError(f"Unsupported Optuna distribution for {name}: {type(distribution).__name__}")


def _finite_score(score: Any) -> float:
    """Reject a fold or native checkpoint that cannot support a valid CV comparison."""
    value = float(score)
    if not math.isfinite(value):
        raise ValueError("Cross-validation produced a non-finite score.")
    return value


def _native_iteration_count(estimator: Any, fallback: int) -> int:
    """Read the candidate's native budget without changing the shared reporting stride."""
    parameters = estimator.get_params(deep=True)
    for name in ("n_estimators", "model__estimator__n_estimators", "estimator__n_estimators"):
        value = parameters.get(name)
        if isinstance(value, int) and value > 0:
            return value
    return fallback


class OptunaPruningSearchCV:
    """Search ordinary estimators between folds or native boosters between iterations.

    The outer estimator is cloned for each candidate; the fold helper owns a
    fresh clone and preprocessing state for each training split. Pruned or
    failed candidates never expose parameters or partial CV scores for refit.
    """

    def __init__(
        self,
        estimator: Any,
        param_distributions: Mapping[str, Any],
        cv: Any,
        scoring: Any,
        study: Any,
        n_trials: int,
        timeout: float | None,
        n_jobs: int,
        callbacks: list[Callable[..., None]],
        mode: str,
        iteration_budget: int,
    ) -> None:
        """Keep the strategy's study, CV contract, and common iteration reporting budget."""
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.cv = cv
        self.scoring = scoring
        self.study = study
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.callbacks = callbacks
        self.mode = mode
        self.iteration_budget = iteration_budget
        self.enable_pruning = True
        self.refit = False

    def fit(self, X: Any, y: Any) -> "OptunaPruningSearchCV":
        """Materialize shared CV splits once and run independent candidate objectives."""
        if self.mode not in {"folds", "iterations"}:
            raise ValueError(f"Unsupported pruning mode: {self.mode}")
        if self.iteration_budget < 1:
            raise ValueError("The common iteration budget must be positive.")
        X, y = indexable(X, y)
        cv = check_cv(self.cv, y, classifier=is_classifier(self.estimator))
        splits = list(cv.split(X, y))
        if not splits:
            raise ValueError("Cross-validation must provide at least one split.")
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
        self.study_ = self.study
        self.study_.optimize(
            lambda trial: self._objective(trial, X, y, splits),
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            callbacks=self.callbacks,
            catch=(_TrialFitFailure,),
        )
        self.n_trials_ = len(self.study_.trials)
        return self

    def _objective(self, trial: Any, X: Any, y: Any, splits: list[Any]) -> float:
        """Continue ordinary bad candidates while letting pruning and cancellation escape."""
        try:
            parameters = {
                name: _suggest_parameter(trial, name, distribution)
                for name, distribution in self.param_distributions.items()
            }
            candidate = clone(self.estimator).set_params(**parameters)
            return self._score_candidate(trial, candidate, X, y, splits)
        except (optuna.TrialPruned, CancelledError):
            raise
        except Exception as error:
            # Only this marker is caught by Study.optimize: cancellation must
            # propagate instead of quietly starting the next expensive trial.
            raise _TrialFitFailure(f"{type(error).__name__}: {error}") from error

    def _score_candidate(
        self, trial: Any, candidate: Any, X: Any, y: Any, splits: list[Any]
    ) -> float:
        """Score complete held-out folds and reject the entire trial if any fold fails."""
        scores: list[float] = []
        for fold_index, (train, valid) in enumerate(splits):
            report = None
            if self.mode == "iterations":
                report = self._iteration_reporter(trial, candidate, scores, fold_index, len(splits))
            score = fit_and_score_fold(
                candidate,
                _safe_indexing(X, train),
                _safe_indexing(y, train),
                _safe_indexing(X, valid),
                _safe_indexing(y, valid),
                self.scorer_,
                report=report,
            )
            scores.append(_finite_score(score))
            if self.mode == "folds":
                trial.report(math.fsum(scores) / len(scores), fold_index)
                if fold_index + 1 < len(splits) and trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned after CV fold {fold_index + 1}.")
        return math.fsum(scores) / len(scores)

    def _iteration_reporter(
        self,
        trial: Any,
        candidate: Any,
        completed_scores: list[float],
        fold_index: int,
        fold_count: int,
    ) -> Callable[[float, int], None]:
        """Compare equal stages across candidates with a fixed stride between CV folds."""
        last_iteration = -1
        final_iteration = _native_iteration_count(candidate, self.iteration_budget) - 1

        def report(score: float, iteration: int) -> None:
            """Report each native checkpoint once without discarding a fully scored trial."""
            nonlocal last_iteration
            if iteration <= last_iteration:
                return
            if iteration < 0 or iteration >= self.iteration_budget:
                raise ValueError("Native iteration exceeds the common reporting budget.")
            last_iteration = iteration
            mean_score = math.fsum([*completed_scores, _finite_score(score)]) / (
                len(completed_scores) + 1
            )
            trial.report(mean_score, fold_index * self.iteration_budget + iteration)
            work_remains = fold_index + 1 < fold_count or iteration < final_iteration
            if work_remains and trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at CV fold {fold_index + 1}, iteration {iteration + 1}."
                )

        return report

    @property
    def best_trial_(self) -> Any:
        """Expose a completed winner or explain why pruning left no model to refit."""
        trials = self.study_.trials
        if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in trials):
            pruned = sum(trial.state == optuna.trial.TrialState.PRUNED for trial in trials)
            if pruned:
                failed = sum(trial.state == optuna.trial.TrialState.FAIL for trial in trials)
                raise ValueError(
                    "No complete trial is available for refit: "
                    f"{pruned} trials were pruned and {failed} failed. "
                    "Reduce pruning or increase the number of trials."
                )
        return self.study_.best_trial

    @property
    def best_params_(self) -> dict[str, Any]:
        """Return parameters exclusively from the best fully evaluated candidate."""
        return self.best_trial_.params

    @property
    def best_score_(self) -> float:
        """Return the best complete mean CV score, never a pruned partial score."""
        return float(self.best_trial_.value)
