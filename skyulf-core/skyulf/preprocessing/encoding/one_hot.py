"""One-Hot Encoder node (Calculator + Applier)."""

import logging
from typing import Any, Dict, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ...utils import (
    resolve_columns,
    user_picked_no_columns,
)
from ..base import BaseApplier, BaseCalculator, apply_method, fit_method
from .._artifacts import OneHotArtifact
from ...core.meta.decorators import node_meta
from ...registry import NodeRegistry
from ...engines import EngineName, get_engine
from ...engines.sklearn_bridge import SklearnBridge
from ._common import detect_categorical_columns, _exclude_target_column

logger = logging.getLogger(__name__)


class OneHotEncoderApplier(BaseApplier):
    @apply_method
    def apply(
        self,
        X: Any,
        _y: Any,
        params: Dict[str, Any],
    ) -> Any:
        engine = get_engine(X)

        if not params or not params.get("columns"):
            return X

        cols = params["columns"]
        encoder = params.get("encoder_object")
        feature_names = params.get("feature_names")
        drop_original = params.get("drop_original", True)
        include_missing = params.get("include_missing", False)

        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols or not encoder:
            return X

        # Polars Path
        if engine.name == EngineName.POLARS:
            import polars as pl

            X_pl: Any = X

            try:
                X_subset = X_pl.select(valid_cols)
                if include_missing:
                    X_subset = X_subset.fill_null("__mlops_missing__")

                X_np, _ = SklearnBridge.to_sklearn(X_subset)
                encoded_array = encoder.transform(X_np)

                if hasattr(encoded_array, "toarray"):
                    encoded_array = encoded_array.toarray()

                encoded_df = pl.DataFrame(encoded_array, schema=feature_names)
                X_out = pl.concat([X_pl, encoded_df], how="horizontal")

                if drop_original:
                    X_out = X_out.drop(valid_cols)

                return X_out
            except Exception as e:
                logger.error(f"OneHot Encoding failed: {e}")
                return X

        # Pandas Path
        X_out = X.copy()
        X_subset = X_out[valid_cols]

        if include_missing:
            X_subset = X_subset.fillna("__mlops_missing__")

        try:
            X_input = X_subset.values if hasattr(X_subset, "values") else X_subset
            encoded_array = encoder.transform(X_input)

            if hasattr(encoded_array, "toarray"):
                encoded_array = encoded_array.toarray()
            elif hasattr(encoded_array, "values"):
                encoded_array = encoded_array.values

            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=X_out.index)
            X_out = pd.concat(cast(Any, [X_out, encoded_df]), axis=1)

            if drop_original:
                X_out = X_out.drop(columns=valid_cols)

        except Exception as e:
            logger.error(f"OneHot Encoding failed: {e}")

        return X_out


@NodeRegistry.register("OneHotEncoder", OneHotEncoderApplier)
@node_meta(
    id="OneHotEncoder",
    name="One-Hot Encoder",
    category="Preprocessing",
    description="Encodes categorical features as a one-hot numeric array.",
    params={
        "handle_unknown": "ignore",
        "drop_first": False,
        "max_categories": 20,
        "columns": [],
        "include_missing": False,
    },
)
class OneHotEncoderCalculator(BaseCalculator):
    @fit_method
    def fit(
        self,
        X: Any,
        y: Any,
        config: Dict[str, Any],
    ) -> OneHotArtifact:
        engine = get_engine(X)

        if user_picked_no_columns(config):
            return {}

        cols = resolve_columns(X, config, detect_categorical_columns)
        cols = _exclude_target_column(cols, config, "OneHotEncoder", y)

        if not cols:
            return {}

        drop = "first" if config.get("drop_first", False) else None
        max_categories = config.get("max_categories", 20)
        handle_unknown = "ignore" if config.get("handle_unknown", "ignore") == "ignore" else "error"
        prefix_separator = config.get("prefix_separator", "_")
        drop_original = config.get("drop_original", True)
        include_missing = config.get("include_missing", False)

        if engine.name == EngineName.POLARS:
            X_pl: Any = X
            X_subset = X_pl.select(cols)
            if include_missing:
                X_subset = X_subset.fill_null("__mlops_missing__")
        else:
            X_subset = X[cols]
            if include_missing:
                X_subset = X_subset.fillna("__mlops_missing__")

        X_np, _ = SklearnBridge.to_sklearn(X_subset)

        encoder = OneHotEncoder(
            drop=drop,
            max_categories=max_categories,
            handle_unknown=handle_unknown,
            sparse_output=False,
            dtype=np.int8,
        )
        encoder.fit(X_np)

        if hasattr(encoder, "categories_"):
            for i, col in enumerate(cols):
                n_cats = len(encoder.categories_[i])
                if n_cats == 0:
                    logger.warning(
                        f"OneHotEncoder: Column '{col}' has 0 categories "
                        "(empty or all missing). It will be dropped."
                    )
                elif drop == "first" and n_cats == 1:
                    logger.warning(
                        f"OneHotEncoder: Column '{col}' has only 1 category "
                        f"('{encoder.categories_[i][0]}') and 'Drop First' is enabled. "
                        "This results in 0 encoded features."
                    )

        return {
            "type": "onehot",
            "columns": cols,
            "encoder_object": encoder,
            "feature_names": encoder.get_feature_names_out(cols).tolist(),
            "prefix_separator": prefix_separator,
            "drop_original": drop_original,
            "include_missing": include_missing,
        }


__all__ = ["OneHotEncoderApplier", "OneHotEncoderCalculator"]
