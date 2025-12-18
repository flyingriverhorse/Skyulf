# Extending Skyulf-Core

Skyulf-core is intentionally simple: calculators learn parameters, appliers apply them.

## Add a new preprocessing node

1. Create a new module in `skyulf.preprocessing`.
2. Implement a `Calculator` and an `Applier`.
3. Register the node type string in the `FeatureEngineer` dispatcher.

### Example skeleton

```python
from typing import Any, Dict, Tuple, Union

import pandas as pd

from skyulf.preprocessing.base import BaseApplier, BaseCalculator
from skyulf.utils import pack_pipeline_output, unpack_pipeline_input


class MyNodeCalculator(BaseCalculator):
    def fit(
        self,
        df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        # Learn something from X...
        return {"type": "my_node", "columns": config.get("columns", [])}


class MyNodeApplier(BaseApplier):
    def apply(
        self,
        df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        # Apply transformation...
        return pack_pipeline_output(X, y, is_tuple)
```

## Add a new modeling estimator

1. Implement a new `BaseModelCalculator` and `BaseModelApplier` (or subclass `SklearnCalculator/SklearnApplier`).
2. Add a mapping entry in `SkyulfPipeline._init_model_estimator()`.

## Testing guidance

Prefer integration tests that run:

`Calculator.fit` → `Applier.apply` on a small synthetic dataframe.
