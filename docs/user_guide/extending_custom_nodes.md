# Extending Skyulf-Core

Skyulf-core uses a Calculator / Applier architecture: calculators learn parameters during `fit`, appliers apply them during `transform`. New nodes are registered via the `@node_meta` decorator and `NodeRegistry`.

## Add a new preprocessing node

1. Create a new module in `skyulf/preprocessing/`.
2. Implement a `Calculator` (extends `BaseCalculator`) and an `Applier` (extends `BaseApplier`).
3. Decorate the Calculator with `@NodeRegistry.register()` and `@node_meta()`.

### Step-by-step example

```python
from typing import Any, Dict, Tuple, Union

import pandas as pd

from skyulf.preprocessing.base import BaseApplier, BaseCalculator
from skyulf.core.meta.decorators import node_meta
from skyulf.registry import NodeRegistry
from skyulf.utils import pack_pipeline_output, unpack_pipeline_input


class MyNodeApplier(BaseApplier):
    """Applies the learned transformation."""

    def apply(
        self,
        df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        params: Dict[str, Any],
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        X, y, is_tuple = unpack_pipeline_input(df)
        columns = params.get("columns", [])
        # Apply your transformation to X using the fitted params...
        return pack_pipeline_output(X, y, is_tuple)


@NodeRegistry.register("MyNode", MyNodeApplier)
@node_meta(
    id="MyNode",
    name="My Custom Node",
    category="Preprocessing",
    description="A short description of what this node does.",
    params={"columns": "list[str] — columns to transform"},
)
class MyNodeCalculator(BaseCalculator):
    """Learns parameters from training data."""

    def fit(
        self,
        df: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        X, _, _ = unpack_pipeline_input(df)
        columns = config.get("columns", [])
        # Learn something from X...
        return {"type": "MyNode", "columns": columns}
```

### What happens under the hood

- `@NodeRegistry.register("MyNode", MyNodeApplier)` registers the Calculator and Applier classes so that `FeatureEngineer` can resolve `"transformer": "MyNode"` in a pipeline config.
- `@node_meta(...)` attaches a `NodeMetadata` dataclass to the class, used for auto-documentation and the frontend node palette.

### Real-world reference

See `skyulf/preprocessing/encoding/one_hot.py` for the `OneHotEncoder` implementation — it follows this exact pattern.

## Add a new modeling estimator

1. Implement a new Calculator (extends `BaseModelCalculator`) and Applier (extends `BaseModelApplier`), or subclass `SklearnCalculator` / `SklearnApplier`.
2. Register with `@NodeRegistry.register("my_model_key", MyModelApplier)`.

The model key can then be used as `"type": "my_model_key"` in the modeling config.

## Advanced: duck-typed steps without subclassing

Registering a node with `@NodeRegistry.register(...)` (so it's usable from a
JSON pipeline config / the frontend palette) still requires subclassing
`BaseCalculator`/`BaseApplier` as shown above — the registry and `@node_meta`
decorator are built around those ABCs.

If you're calling `StatefulTransformer` directly in Python (not going through
the registry — e.g. scripting an ad-hoc pipeline step, or wrapping a
third-party object), you don't need to subclass anything. `skyulf.core.protocols`
defines `CalculatorProtocol`/`ApplierProtocol` as structural (`typing.Protocol`)
types — any object with matching `fit`/`apply` methods satisfies them:

```python
from skyulf.preprocessing.base import StatefulTransformer


class MeanCenterer:
    """Plain class — no BaseCalculator/BaseApplier inheritance."""

    def fit(self, df, config):
        return {"mean": df["x"].mean()}

    def apply(self, df, params):
        out = df.copy()
        out["x"] = out["x"] - params["mean"]
        return out


centerer = MeanCenterer()
step = StatefulTransformer(calculator=centerer, applier=centerer, node_id="center_x")
```

This is useful for quick experiments or wrapping existing objects (e.g. an
sklearn-like class that already has compatible methods) without writing an
ABC subclass just to satisfy a type check.

## Testing guidance

Write integration tests that run the full cycle:

```python
calc = MyNodeCalculator()
artifact = calc.fit(sample_df, {"columns": ["col_a"]})

applier = MyNodeApplier()
result = applier.apply(sample_df, artifact)

assert "col_a" in result.columns  # or whatever your node guarantees
```

Prefer real DataFrames over mocks — see `tests/` for examples.
