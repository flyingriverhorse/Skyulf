from abc import ABC, abstractmethod
from typing import List
from ..schemas import AnalysisProfile, Recommendation

class BaseAdvisorPlugin(ABC):
    @abstractmethod
    def analyze(self, profile: AnalysisProfile) -> List[Recommendation]:
        """
        Analyzes the data profile and returns a list of recommendations.
        """
        pass
