"""Internal mixins backing :class:`skyulf.profiling.analyzer.EDAAnalyzer`.

Public import path stays ``from skyulf.profiling.analyzer import EDAAnalyzer``.
This package is private (leading underscore); mixins are split by analytical
concern so each file stays small and reviewable.
"""

from .categorical import CategoricalMixin
from .causal import CausalMixin
from .column import ColumnMixin
from .dates import DatesMixin
from .decomposition import DecompositionMixin
from .geo import GeoMixin
from .multivariate import MultivariateMixin
from .numeric import NumericMixin
from .recommendations import RecommendationsMixin
from .rules import RulesMixin
from .target import TargetMixin
from .temporal import TemporalMixin
from .text import TextMixin

__all__ = [
    "CategoricalMixin",
    "CausalMixin",
    "ColumnMixin",
    "DatesMixin",
    "DecompositionMixin",
    "GeoMixin",
    "MultivariateMixin",
    "NumericMixin",
    "RecommendationsMixin",
    "RulesMixin",
    "TargetMixin",
    "TemporalMixin",
    "TextMixin",
]
