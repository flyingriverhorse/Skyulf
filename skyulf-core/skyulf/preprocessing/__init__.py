from .base import BaseApplier, BaseCalculator, StatefulTransformer
from ._schema import SkyulfSchema
from .bucketing import (
    CustomBinningApplier,
    CustomBinningCalculator,
    GeneralBinningApplier,
    GeneralBinningCalculator,
    KBinsDiscretizerApplier,
    KBinsDiscretizerCalculator,
)
from .casting import CastingApplier, CastingCalculator
from .cleaning import (
    TextCleaningApplier,
    TextCleaningCalculator,
    ValueReplacementApplier,
    ValueReplacementCalculator,
)
from .drop_and_missing import (
    DeduplicateApplier,
    DeduplicateCalculator,
    DropMissingColumnsApplier,
    DropMissingColumnsCalculator,
    DropMissingRowsApplier,
    DropMissingRowsCalculator,
    MissingIndicatorApplier,
    MissingIndicatorCalculator,
)
from .encoding import (
    LabelEncoderApplier,
    LabelEncoderCalculator,
    OneHotEncoderApplier,
    OneHotEncoderCalculator,
    OrdinalEncoderApplier,
    OrdinalEncoderCalculator,
    TargetEncoderApplier,
    TargetEncoderCalculator,
)
from .feature_generation import (
    FeatureGenerationApplier,
    FeatureGenerationCalculator,
    PolynomialFeaturesApplier,
    PolynomialFeaturesCalculator,
)
from .feature_selection import (
    CorrelationThresholdApplier,
    CorrelationThresholdCalculator,
    ModelBasedSelectionApplier,
    ModelBasedSelectionCalculator,
    UnivariateSelectionApplier,
    UnivariateSelectionCalculator,
    VarianceThresholdApplier,
    VarianceThresholdCalculator,
)
from .imputation import (
    IterativeImputerApplier,
    IterativeImputerCalculator,
    KNNImputerApplier,
    KNNImputerCalculator,
    SimpleImputerApplier,
    SimpleImputerCalculator,
)
from .inspection import (
    DatasetProfileApplier,
    DatasetProfileCalculator,
    DataSnapshotApplier,
    DataSnapshotCalculator,
)
from .outliers import (
    EllipticEnvelopeApplier,
    EllipticEnvelopeCalculator,
    IQRApplier,
    IQRCalculator,
    WinsorizeApplier,
    WinsorizeCalculator,
    ZScoreApplier,
    ZScoreCalculator,
)
from .pipeline import FeatureEngineer
from .resampling import (
    OversamplingApplier,
    OversamplingCalculator,
    UndersamplingApplier,
    UndersamplingCalculator,
)
from .scaling import (
    MinMaxScalerApplier,
    MinMaxScalerCalculator,
    RobustScalerApplier,
    RobustScalerCalculator,
    StandardScalerApplier,
    StandardScalerCalculator,
)
from .split import SplitApplier, SplitCalculator
from .transformations import (
    GeneralTransformationApplier,
    GeneralTransformationCalculator,
    PowerTransformerApplier,
    PowerTransformerCalculator,
    SimpleTransformationApplier,
    SimpleTransformationCalculator,
)

__all__ = [
    "BaseCalculator",
    "BaseApplier",
    "StatefulTransformer",
    "SkyulfSchema",
    "FeatureEngineer",
    "SplitCalculator",
    "SplitApplier",
    "TextCleaningCalculator",
    "TextCleaningApplier",
    "ValueReplacementCalculator",
    "ValueReplacementApplier",
    "DeduplicateCalculator",
    "DeduplicateApplier",
    "DropMissingColumnsCalculator",
    "DropMissingColumnsApplier",
    "DropMissingRowsCalculator",
    "DropMissingRowsApplier",
    "MissingIndicatorCalculator",
    "MissingIndicatorApplier",
    "SimpleImputerCalculator",
    "SimpleImputerApplier",
    "KNNImputerCalculator",
    "KNNImputerApplier",
    "IterativeImputerCalculator",
    "IterativeImputerApplier",
    "OneHotEncoderCalculator",
    "OneHotEncoderApplier",
    "OrdinalEncoderCalculator",
    "OrdinalEncoderApplier",
    "LabelEncoderCalculator",
    "LabelEncoderApplier",
    "TargetEncoderCalculator",
    "TargetEncoderApplier",
    "StandardScalerCalculator",
    "StandardScalerApplier",
    "MinMaxScalerCalculator",
    "MinMaxScalerApplier",
    "RobustScalerCalculator",
    "RobustScalerApplier",
    "IQRCalculator",
    "IQRApplier",
    "ZScoreCalculator",
    "ZScoreApplier",
    "WinsorizeCalculator",
    "WinsorizeApplier",
    "EllipticEnvelopeCalculator",
    "EllipticEnvelopeApplier",
    "PowerTransformerCalculator",
    "PowerTransformerApplier",
    "SimpleTransformationCalculator",
    "SimpleTransformationApplier",
    "GeneralTransformationCalculator",
    "GeneralTransformationApplier",
    "GeneralBinningCalculator",
    "GeneralBinningApplier",
    "CustomBinningCalculator",
    "CustomBinningApplier",
    "KBinsDiscretizerCalculator",
    "KBinsDiscretizerApplier",
    "CastingCalculator",
    "CastingApplier",
    "FeatureGenerationCalculator",
    "FeatureGenerationApplier",
    "PolynomialFeaturesCalculator",
    "PolynomialFeaturesApplier",
    "VarianceThresholdCalculator",
    "VarianceThresholdApplier",
    "CorrelationThresholdCalculator",
    "CorrelationThresholdApplier",
    "UnivariateSelectionCalculator",
    "UnivariateSelectionApplier",
    "ModelBasedSelectionCalculator",
    "ModelBasedSelectionApplier",
    "DatasetProfileCalculator",
    "DatasetProfileApplier",
    "DataSnapshotCalculator",
    "DataSnapshotApplier",
    "OversamplingCalculator",
    "OversamplingApplier",
    "UndersamplingCalculator",
    "UndersamplingApplier",
]

# Auto-import any submodule added to this package so its @NodeRegistry.register
# decorators run at import time.  New node files need no __init__.py edits.
import importlib as _importlib
import pkgutil as _pkgutil

for _mi in _pkgutil.iter_modules(__path__, __name__ + "."):  # type: ignore[name-defined]
    _importlib.import_module(_mi.name)
