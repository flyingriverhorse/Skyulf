"""Tests for the core-only console trial reporter (``progress=True``).

TTY mode must stay tidy (one self-updating line, never a flood); piped
mode must stay quiet per trial; the end-of-run summary always prints and
carries best score, best params, and the top trials.
"""

import io

from skyulf.modeling._tuning.reporter import ConsoleTrialReporter
from skyulf.modeling._tuning.schemas import TuningResult


def _result() -> TuningResult:
    return TuningResult(
        best_params={"C": 1.0},
        best_score=0.9,
        n_trials=4,
        trials=[
            {"params": {"C": 0.1}, "score": 0.7},
            {"params": {"C": 1.0}, "score": 0.9},
            {"params": {"C": 10.0}, "score": 0.8},
            {"params": {}, "score": None},
        ],
        scoring_metric="accuracy",
    )


def test_live_mode_updates_one_line_and_tracks_best():
    stream = io.StringIO()
    reporter = ConsoleTrialReporter(stream=stream, force_live=True)
    reporter(1, 3, 0.5)
    reporter(2, 3, 0.9)
    reporter(3, 3, 0.7)
    out = stream.getvalue()
    # Every update rewinds the same line; only the final trial closes it.
    assert out.count("\r") == 3
    assert out.count("\n") == 1
    assert "trial 3/3" in out
    assert "score 0.7000" in out
    assert "best 0.9000 (#2)" in out


def test_live_mode_notes_scoreless_trials():
    stream = io.StringIO()
    reporter = ConsoleTrialReporter(stream=stream, force_live=True)
    reporter(1, 2, None)
    assert "failed or pruned" in stream.getvalue()


def test_piped_mode_is_silent_per_trial():
    stream = io.StringIO()
    reporter = ConsoleTrialReporter(stream=stream, force_live=False)
    reporter(1, 3, 0.5)
    reporter(2, 3, 0.9)
    assert stream.getvalue() == ""


def test_summary_carries_best_and_top_trials():
    stream = io.StringIO()
    ConsoleTrialReporter(stream=stream, force_live=False).finish(_result())
    out = stream.getvalue()
    assert "Tuning complete | 3/4 scored trials | metric accuracy" in out
    assert "best 0.9000 | {'C': 1.0}" in out
    # Top trials sorted best-first; the scoreless trial is excluded.
    top = out.split("top: ")[1]
    assert top.startswith("#2 0.9000 | #3 0.8000 | #1 0.7000")


def test_finish_closes_an_open_live_line():
    stream = io.StringIO()
    reporter = ConsoleTrialReporter(stream=stream, force_live=True)
    reporter(1, 5, 0.5)  # not the final trial -> line left open
    reporter.finish(_result())
    out = stream.getvalue()
    assert out.startswith("\r")
    assert "\nTuning complete" in out


def test_reporter_satisfies_progress_callback_protocol():
    # Same positional shape the engine's loops already emit.
    stream = io.StringIO()
    reporter = ConsoleTrialReporter(stream=stream, force_live=True)
    reporter(1, 1, 0.8, {"C": 1.0})
    assert "trial 1/1" in stream.getvalue()
