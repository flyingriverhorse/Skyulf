"""Tidy console progress for core-only tuning runs.

Backend/UI users watch trials on the live chart; someone running
skyulf-core from a terminal used to get silence (or had to hand-write a
``progress_callback``). With ``progress=True`` in the tuning config, the
engine attaches :class:`ConsoleTrialReporter`:

- on a TTY, one self-updating line (``\\r``) — ``Tuning trial 12/60 |
  score 0.8530 | best 0.8710 (#8)`` — never a 200-line flood;
- on completion, a compact summary (best score + params, top trials);
- when stdout is piped (CI, logs), per-trial lines are skipped and only
  the summary prints.

Output is deliberately ASCII-only: a Unicode glyph can raise
``UnicodeEncodeError`` on cp1252 Windows consoles, and a progress printer
must never break the run it reports on.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from .schemas import TuningResult


class ConsoleTrialReporter:
    """``progress_callback``-shaped reporter; call ``finish`` after the run."""

    def __init__(self, stream: TextIO | None = None, force_live: bool | None = None) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._live = force_live if force_live is not None else self._stream.isatty()
        self._best: float | None = None
        self._best_trial: int | None = None
        self._line_len = 0
        self._line_open = False

    def __call__(
        self,
        current: int,
        total: int,
        score: float | None,
        params: dict[str, Any] | None = None,
    ) -> None:
        if score is not None and (self._best is None or score > self._best):
            self._best = float(score)
            self._best_trial = current
        if not self._live:
            return
        if score is None:
            body = f"Tuning trial {current}/{total} | failed or pruned"
        else:
            best_part = (
                f" | best {self._best:.4f} (#{self._best_trial})" if self._best is not None else ""
            )
            body = f"Tuning trial {current}/{total} | score {score:.4f}{best_part}"
        closed = current >= total
        # \r + pad to erase any leftover chars from a longer previous line.
        self._stream.write("\r" + body.ljust(self._line_len) + ("\n" if closed else ""))
        self._stream.flush()
        self._line_len = 0 if closed else len(body)
        self._line_open = not closed

    def finish(self, result: TuningResult) -> None:
        """Print the compact end-of-run summary (always, TTY or not)."""
        if self._line_open:
            self._stream.write("\n")
            self._line_open = False
        scored = [
            (i + 1, t["score"])
            for i, t in enumerate(result.trials)
            if isinstance(t.get("score"), (int, float))
        ]
        metric = f" | metric {result.scoring_metric}" if result.scoring_metric else ""
        self._stream.write(
            f"Tuning complete | {len(scored)}/{result.n_trials} scored trials{metric}\n"
        )
        self._stream.write(f"  best {result.best_score:.4f} | {result.best_params}\n")
        top = sorted(scored, key=lambda pair: pair[1], reverse=True)[:3]
        if top:
            rendered = " | ".join(f"#{trial} {score:.4f}" for trial, score in top)
            self._stream.write(f"  top: {rendered}\n")
        self._stream.flush()
