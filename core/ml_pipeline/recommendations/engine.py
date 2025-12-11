from typing import List
from .schemas import AnalysisProfile, Recommendation
from .plugins.base import BaseAdvisorPlugin

# Import plugins
from .plugins.imputation import ImputationAdvisor
from .plugins.scaling import ScalingAdvisor
from .plugins.cleaning import CleaningAdvisor
from .plugins.encoding import EncodingAdvisor
from .plugins.outliers import OutlierAdvisor
from .plugins.transformation import TransformationAdvisor
from .plugins.resampling import ResamplingAdvisor
from .plugins.feature_generation import FeatureGenerationAdvisor
from .plugins.bucketing import BucketingAdvisor

class AdvisorEngine:
    def __init__(self):
        self.plugins: List[BaseAdvisorPlugin] = []
        self._register_default_plugins()

    def _register_default_plugins(self):
        self.plugins.append(CleaningAdvisor())
        self.plugins.append(ImputationAdvisor())
        self.plugins.append(ScalingAdvisor())
        self.plugins.append(EncodingAdvisor())
        self.plugins.append(OutlierAdvisor())
        self.plugins.append(TransformationAdvisor())
        self.plugins.append(ResamplingAdvisor())
        self.plugins.append(FeatureGenerationAdvisor())
        self.plugins.append(BucketingAdvisor())

    def register_plugin(self, plugin: BaseAdvisorPlugin):
        self.plugins.append(plugin)

    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        recommendations = []
        for plugin in self.plugins:
            try:
                recs = plugin.analyze(profile)
                recommendations.extend(recs)
            except Exception as e:
                # Log error but don't crash the whole analysis
                print(f"Error in recommendation plugin {plugin.__class__.__name__}: {e}")
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations
