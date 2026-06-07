"""Encoding nodes package.

Split from a single 832-LOC module into per-encoder files:
  _common.py   — shared helpers (detect_categorical_columns, _exclude_target_column, …)
  one_hot.py   — OneHotEncoder
  ordinal.py   — OrdinalEncoder
  label.py     — LabelEncoder
  target.py    — TargetEncoder
  hash.py      — HashEncoder
  dummy.py     — DummyEncoder
  woe.py       — WOEEncoder (Weight-of-Evidence / Information Value)

All public names are re-exported here so existing imports such as
``from skyulf.preprocessing.encoding import OneHotEncoderCalculator``
continue to work unchanged.
"""

from ._common import _exclude_target_column, detect_categorical_columns
from .dummy import DummyEncoderApplier, DummyEncoderCalculator
from .hash import HashEncoderApplier, HashEncoderCalculator
from .label import LabelEncoderApplier, LabelEncoderCalculator
from .one_hot import OneHotEncoderApplier, OneHotEncoderCalculator
from .ordinal import OrdinalEncoderApplier, OrdinalEncoderCalculator
from .target import TargetEncoderApplier, TargetEncoderCalculator
from .woe import WOEEncoderApplier, WOEEncoderCalculator

__all__ = [
    "detect_categorical_columns",
    "_exclude_target_column",
    "DummyEncoderApplier",
    "DummyEncoderCalculator",
    "HashEncoderApplier",
    "HashEncoderCalculator",
    "LabelEncoderApplier",
    "LabelEncoderCalculator",
    "OneHotEncoderApplier",
    "OneHotEncoderCalculator",
    "OrdinalEncoderApplier",
    "OrdinalEncoderCalculator",
    "TargetEncoderApplier",
    "TargetEncoderCalculator",
    "WOEEncoderApplier",
    "WOEEncoderCalculator",
]
