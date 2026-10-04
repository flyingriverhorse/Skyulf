"""Outlier-handling nodes package.

Split from a single 539-LOC module into per-node files:
  _common.py        — shared mask helpers (y filter, pandas mask apply)
  iqr.py            — IQR
  zscore.py         — ZScore
  winsorize.py      — Winsorize
  manual_bounds.py  — ManualBounds
  elliptic.py       — EllipticEnvelope
  clip_values.py    — ClipValues (fixed bounds, clips instead of dropping rows)

All public names are re-exported here so existing imports such as
``from skyulf.preprocessing.outliers import IQRCalculator`` continue to work.
"""

from .clip_values import ClipValuesApplier, ClipValuesCalculator
from .elliptic import EllipticEnvelopeApplier, EllipticEnvelopeCalculator
from .iqr import IQRApplier, IQRCalculator
from .manual_bounds import ManualBoundsApplier, ManualBoundsCalculator
from .winsorize import WinsorizeApplier, WinsorizeCalculator
from .zscore import ZScoreApplier, ZScoreCalculator

__all__ = [
    "ClipValuesApplier",
    "ClipValuesCalculator",
    "EllipticEnvelopeApplier",
    "EllipticEnvelopeCalculator",
    "IQRApplier",
    "IQRCalculator",
    "ManualBoundsApplier",
    "ManualBoundsCalculator",
    "WinsorizeApplier",
    "WinsorizeCalculator",
    "ZScoreApplier",
    "ZScoreCalculator",
]
