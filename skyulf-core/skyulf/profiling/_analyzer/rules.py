"""Decision-tree surrogate model for human-readable rule extraction."""

from typing import List, Optional

import numpy as np
import polars as pl

from ..schemas import RuleNode, RuleTree
from ._utils import SKLEARN_AVAILABLE, _AnalyzerState


class RulesMixin(_AnalyzerState):
    """Rule-discovery helpers for :class:`EDAAnalyzer`."""

    def _discover_rules(  # noqa: C901
        self,
        feature_cols: List[str],
        target_col: str,
        task_type: Optional[str] = None,
    ) -> Optional[RuleTree]:
        """Train a depth-4 surrogate tree, return its structure + IF/THEN rules."""
        if not SKLEARN_AVAILABLE:
            return None
        try:
            from sklearn.tree import (  # ty: ignore[unresolved-import]
                DecisionTreeClassifier,
                DecisionTreeRegressor,
                _tree,  # ty: ignore[unresolved-import]
            )

            limit = 100000
            df_sample = self.df.select(  # type: ignore[attr-defined]
                feature_cols + [target_col]
            ).head(limit)

            is_regression = False
            if task_type:
                if task_type.lower() == "regression":
                    is_regression = True
                elif task_type.lower() == "classification":
                    is_regression = False
            else:
                target_type = self._get_semantic_type(df_sample[target_col])  # type: ignore[attr-defined]
                is_regression = target_type == "Numeric"

            cat_cols = [
                c
                for c in feature_cols
                if self._get_semantic_type(df_sample[c])  # type: ignore[attr-defined]
                in ["Categorical", "Boolean", "Text"]
            ]
            num_cols = [c for c in feature_cols if c not in cat_cols]

            X_data = {}
            feature_names: List[str] = []

            for col in num_cols:
                mean_val = df_sample[col].mean()
                X_data[col] = df_sample[col].fill_null(mean_val).to_numpy()
                feature_names.append(col)

            for col in cat_cols:
                s = df_sample[col].cast(pl.Utf8).fill_null("Missing")
                # Ordinal encode (factorize) — keeps tree splits readable.
                codes = s.cast(pl.Categorical).to_physical().to_numpy()
                X_data[col] = codes
                feature_names.append(col)

            X_list = [X_data[col] for col in feature_names]
            X = np.column_stack(X_list)

            if is_regression:
                y_mean = df_sample[target_col].mean()
                y = df_sample[target_col].fill_null(y_mean).to_numpy()
                class_names: List[str] = []
                clf = DecisionTreeRegressor(max_depth=4, random_state=42)
            else:
                y_series = df_sample[target_col].cast(pl.Utf8).fill_null("Missing")

                # Cap classes at 10 + "Other" so the tree stays interpretable.
                if y_series.n_unique() > 10:
                    top_10 = (
                        y_series.value_counts()
                        .sort("count", descending=True)
                        .head(10)[target_col]
                        .to_list()
                    )
                    temp_df = pl.DataFrame({"y": y_series})
                    y_series = temp_df.select(
                        pl.when(pl.col("y").is_in(top_10))
                        .then(pl.col("y"))
                        .otherwise(pl.lit("Other"))
                    ).to_series()

                y = y_series.cast(pl.Categorical).to_physical().to_numpy()
                class_names = y_series.cast(pl.Categorical).cat.get_categories().to_list()
                clf = DecisionTreeClassifier(max_depth=4, random_state=42)

            clf.fit(X, y)

            importances = clf.feature_importances_
            feature_importance_list = [
                {"feature": feature_names[idx], "importance": float(imp)}
                for idx, imp in enumerate(importances)
                if imp > 0
            ]
            feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)

            tree_ = clf.tree_
            nodes: List[RuleNode] = []

            def recurse(node_id):
                is_leaf = bool(tree_.children_left[node_id] == _tree.TREE_LEAF)

                feature = None
                threshold = None
                if not is_leaf:
                    feature_idx = tree_.feature[node_id]
                    feature = feature_names[feature_idx]
                    threshold = float(tree_.threshold[node_id])

                if is_regression:
                    val = float(tree_.value[node_id][0][0])
                    value = [val]
                    class_name = f"{val:.2f}"
                else:
                    value = tree_.value[node_id][0].tolist()
                    class_idx = np.argmax(value)
                    class_name = (
                        str(class_names[class_idx]) if class_idx < len(class_names) else "Unknown"
                    )

                children = []
                if not is_leaf:
                    left_id = tree_.children_left[node_id]
                    right_id = tree_.children_right[node_id]
                    children = [int(left_id), int(right_id)]
                    recurse(left_id)
                    recurse(right_id)

                nodes.append(
                    RuleNode(
                        id=int(node_id),
                        feature=feature,
                        threshold=threshold,
                        impurity=float(tree_.impurity[node_id]),
                        samples=int(tree_.n_node_samples[node_id]),
                        value=value,
                        class_name=class_name,
                        is_leaf=is_leaf,
                        children=children,
                    )
                )

            recurse(0)
            nodes.sort(key=lambda x: x.id)

            rules_text: List[str] = []

            def recurse_rules(node_id, current_rule):
                if tree_.children_left[node_id] == _tree.TREE_LEAF:
                    total_samples = int(tree_.n_node_samples[node_id])
                    if is_regression:
                        val = float(tree_.value[node_id][0][0])
                        rule_str = (
                            f"IF {current_rule} THEN Value = {val:.2f} (Samples: {total_samples})"
                        )
                    else:
                        value = tree_.value[node_id][0]
                        class_idx = np.argmax(value)
                        class_name = (
                            str(class_names[class_idx])
                            if class_idx < len(class_names)
                            else "Unknown"
                        )
                        total = np.sum(value)
                        confidence = (value[class_idx] / total) * 100 if total > 0 else 0
                        rule_str = (
                            f"IF {current_rule} THEN {class_name} "
                            f"(Confidence: {confidence:.1f}%, Samples: {int(total)})"
                        )
                    rules_text.append(rule_str)
                    return

                feature_idx = tree_.feature[node_id]
                feature_name = feature_names[feature_idx]
                threshold = tree_.threshold[node_id]

                left_rule = (
                    f"{current_rule} AND {feature_name} <= {threshold:.2f}"
                    if current_rule
                    else f"{feature_name} <= {threshold:.2f}"
                )
                recurse_rules(tree_.children_left[node_id], left_rule)

                right_rule = (
                    f"{current_rule} AND {feature_name} > {threshold:.2f}"
                    if current_rule
                    else f"{feature_name} > {threshold:.2f}"
                )
                recurse_rules(tree_.children_right[node_id], right_rule)

            recurse_rules(0, "")

            return RuleTree(
                nodes=nodes,
                accuracy=float(clf.score(X, y)),
                rules=rules_text,
                feature_importances=feature_importance_list,
            )

        except Exception as e:
            print(f"Error in rule discovery: {e}")
            return None
