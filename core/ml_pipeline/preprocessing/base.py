from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import pandas as pd
from ..data.container import SplitDataset
from ..artifacts.store import ArtifactStore

class BaseCalculator(ABC):
    @abstractmethod
    def fit(self, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates parameters from the training data.
        Returns a dictionary of fitted parameters (serializable).
        """
        pass

class BaseApplier(ABC):
    @abstractmethod
    def apply(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Applies the transformation using fitted parameters.
        """
        pass

class StatefulTransformer:
    def __init__(self, calculator: BaseCalculator, applier: BaseApplier, artifact_store: ArtifactStore, node_id: str, apply_on_test: bool = True, apply_on_validation: bool = True):
        self.calculator = calculator
        self.applier = applier
        self.artifact_store = artifact_store
        self.node_id = node_id
        self.apply_on_test = apply_on_test
        self.apply_on_validation = apply_on_validation

    def fit_transform(self, dataset: Union[SplitDataset, pd.DataFrame], config: Dict[str, Any]) -> Union[SplitDataset, pd.DataFrame]:
        if isinstance(dataset, pd.DataFrame):
            # Fit on the whole dataframe (be careful about leakage!)
            params = self.calculator.fit(dataset, config)
            self.artifact_store.save(self.node_id, params)
            return self.applier.apply(dataset, params)
        
        # If dataset is a tuple (e.g. from FeatureTargetSplitter), we can't fit on it like a SplitDataset
        if isinstance(dataset, tuple):
             # This case shouldn't happen for standard transformers, but if it does, we might need to handle it.
             # For now, assume standard transformers don't run AFTER a FeatureTargetSplitter in this pipeline structure.
             raise ValueError(f"Unexpected input type 'tuple' for node {self.node_id}. Did you place a transformer after a Feature-Target Split?")

        # 1. Calculate on Train
        params = self.calculator.fit(dataset.train, config)
        
        # 2. Save Artifact
        self.artifact_store.save(self.node_id, params)
        
        # 3. Apply to all splits
        new_train = self.applier.apply(dataset.train, params)
        
        new_test = dataset.test
        if self.apply_on_test:
            new_test = self.applier.apply(dataset.test, params)
            
        new_val = dataset.validation
        if self.apply_on_validation and dataset.validation is not None:
            new_val = self.applier.apply(dataset.validation, params)
        
        return SplitDataset(train=new_train, test=new_test, validation=new_val)

    def transform(self, dataset: Union[SplitDataset, pd.DataFrame]) -> Union[SplitDataset, pd.DataFrame]:
        # 1. Load Artifact
        params = self.artifact_store.load(self.node_id)
        
        if isinstance(dataset, pd.DataFrame):
            return self.applier.apply(dataset, params)
        
        # 2. Apply
        new_train = self.applier.apply(dataset.train, params)
        
        new_test = dataset.test
        if self.apply_on_test:
            new_test = self.applier.apply(dataset.test, params)
            
        new_val = dataset.validation
        if self.apply_on_validation and dataset.validation is not None:
            new_val = self.applier.apply(dataset.validation, params)
        
        return SplitDataset(train=new_train, test=new_test, validation=new_val)
