"""Decision-tree surrogate model for human-readable rule extraction."""

import logging
from typing import Any

import numpy as np
import polars as pl

from ..schemas import RuleNode, RuleTree
from ._utils import SKLEARN_AVAILABLE, _AnalyzerState

logger = logging.getLogger(__name__)


class RulesMixin(_AnalyzerState):
    """Rule-discovery helpers for :class:`EDAAnalyzer`."""

    def _discover_rules(
        self,
        feature_cols: list[str],
        target_col: str,
        task_type: str | None = None,
    ) -> RuleTree | None:
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

            is_regression = self._determine_is_regression(task_type, df_sample, target_col)
            cat_cols, num_cols = self._split_feature_columns(df_sample, feature_cols)

            X, feature_names, cat_categories = self._build_feature_matrix(
                df_sample, num_cols, cat_cols
            )

            if is_regression:
                y, class_names, clf = self._prepare_regression_target(
                    df_sample, target_col, DecisionTreeRegressor
                )
            else:
                y, class_names, clf = self._prepare_classification_target(
                    df_sample, target_col, DecisionTreeClassifier
                )

            clf.fit(X, y)

            feature_importance_list = self._extract_feature_importances(clf, feature_names)

            tree_ = clf.tree_
            nodes = self._build_rule_nodes(tree_, feature_names, is_regression, class_names, _tree)
            rules_text = self._build_rules_text(
                tree_, feature_names, is_regression, class_names, cat_categories, _tree
            )

            return RuleTree(
                nodes=nodes,
                accuracy=float(clf.score(X, y)),
                rules=rules_text,
                feature_importances=feature_importance_list,
                categories=cat_categories or None,
            )

        except Exception as e:
            logger.warning(f"Error in rule discovery: {e}")
            return None

    def _determine_is_regression(
        self, task_type: str | None, df_sample: pl.DataFrame, target_col: str
    ) -> bool:
        """Decide whether the surrogate model should be a regressor or classifier."""
        if task_type:
            return task_type.lower() == "regression"

        target_type = self._get_semantic_type(  # type: ignore[attr-defined]  # pylint: disable=assignment-from-no-return
            df_sample[target_col]
        )
        return target_type == "Numeric"

    def _split_feature_columns(
        self, df_sample: pl.DataFrame, feature_cols: list[str]
    ) -> tuple[list[str], list[str]]:
        """Split feature columns into categorical and numeric groups by semantic type."""
        cat_cols = [
            c
            for c in feature_cols
            if self._get_semantic_type(df_sample[c])  # type: ignore[attr-defined]
            in ["Categorical", "Boolean", "Text"]
        ]
        num_cols = [c for c in feature_cols if c not in cat_cols]
        return cat_cols, num_cols

    def _build_feature_matrix(
        self, df_sample: pl.DataFrame, num_cols: list[str], cat_cols: list[str]
    ) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
        """Build the numpy feature matrix, ordinal-encoding categoricals.

        Returns the feature matrix, the ordered feature names, and a mapping
        from categorical feature name to its ordered category labels
        (physical code i == cat_categories[col][i]), so rule text can show
        human-readable category names instead of the raw ordinal code the
        tree actually split on (e.g. "color <= 2.00" is meaningless without
        this mapping).
        """
        X_data = {}
        feature_names: list[str] = []
        cat_categories: dict[str, list[str]] = {}

        for col in num_cols:
            mean_val = df_sample[col].mean()
            X_data[col] = df_sample[col].fill_null(mean_val).to_numpy()
            feature_names.append(col)

        for col in cat_cols:
            s = df_sample[col].cast(pl.Utf8).fill_null("Missing")
            # Ordinal encode (factorize) — keeps tree splits readable.
            cat_series = s.cast(pl.Categorical)
            codes = cat_series.to_physical().to_numpy()
            X_data[col] = codes
            feature_names.append(col)
            cat_categories[col] = cat_series.cat.get_categories().to_list()

        X_list = [X_data[col] for col in feature_names]
        X = np.column_stack(X_list)
        return X, feature_names, cat_categories

    def _prepare_regression_target(
        self, df_sample: pl.DataFrame, target_col: str, regressor_cls: Any
    ) -> tuple[np.ndarray, list[str], Any]:
        """Prepare the regression target array, model, and (empty) class names."""
        y_mean = df_sample[target_col].mean()
        y = df_sample[target_col].fill_null(y_mean).to_numpy()
        class_names: list[str] = []
        clf = regressor_cls(max_depth=4, random_state=42)
        return y, class_names, clf

    def _prepare_classification_target(
        self, df_sample: pl.DataFrame, target_col: str, classifier_cls: Any
    ) -> tuple[np.ndarray, list[str], Any]:
        """Prepare the classification target array, model, and class names.

        Caps classes at 10 + "Other" so the tree stays interpretable.
        """
        y_series = df_sample[target_col].cast(pl.Utf8).fill_null("Missing")

        if y_series.n_unique() > 10:
            top_10 = (
                y_series.value_counts()
                .sort("count", descending=True)
                .head(10)[target_col]
                .to_list()
            )
            temp_df = pl.DataFrame({"y": y_series})
            y_series = temp_df.select(
                pl.when(pl.col("y").is_in(top_10)).then(pl.col("y")).otherwise(pl.lit("Other"))
            ).to_series()

        y = y_series.cast(pl.Categorical).to_physical().to_numpy()
        class_names = y_series.cast(pl.Categorical).cat.get_categories().to_list()
        clf = classifier_cls(max_depth=4, random_state=42)
        return y, class_names, clf

    def _extract_feature_importances(self, clf: Any, feature_names: list[str]) -> list[dict]:
        """Build a sorted list of non-zero feature importances."""
        importances = clf.feature_importances_
        feature_importance_list = [
            {"feature": feature_names[idx], "importance": float(imp)}
            for idx, imp in enumerate(importances)
            if imp > 0
        ]
        feature_importance_list.sort(key=lambda x: x["importance"], reverse=True)
        return feature_importance_list

    def _leaf_value_and_class_name(
        self, tree_, node_id: int, is_regression: bool, class_names: list[str]
    ) -> tuple[list[float], str]:
        """Compute a node's value vector and display class/label name."""
        if is_regression:
            val = float(tree_.value[node_id][0][0])
            return [val], f"{val:.2f}"

        value = tree_.value[node_id][0].tolist()
        class_idx = np.argmax(value)
        class_name = str(class_names[class_idx]) if class_idx < len(class_names) else "Unknown"
        return value, class_name

    def _build_rule_nodes(
        self,
        tree_,
        feature_names: list[str],
        is_regression: bool,
        class_names: list[str],
        tree_module,
    ) -> list[RuleNode]:
        """Recursively walk the fitted tree, building a flat, id-sorted list of RuleNodes."""
        nodes: list[RuleNode] = []
        self._collect_rule_node(
            0, tree_, feature_names, is_regression, class_names, tree_module, nodes
        )
        nodes.sort(key=lambda x: x.id)
        return nodes

    def _collect_rule_node(
        self,
        node_id: int,
        tree_,
        feature_names: list[str],
        is_regression: bool,
        class_names: list[str],
        tree_module,
        nodes: list[RuleNode],
    ) -> None:
        """Append the RuleNode for ``node_id`` and recurse into its children."""
        is_leaf = bool(tree_.children_left[node_id] == tree_module.TREE_LEAF)

        feature = None
        threshold = None
        if not is_leaf:
            feature_idx = tree_.feature[node_id]
            feature = feature_names[feature_idx]
            threshold = float(tree_.threshold[node_id])

        value, class_name = self._leaf_value_and_class_name(
            tree_, node_id, is_regression, class_names
        )

        children = []
        if not is_leaf:
            left_id = tree_.children_left[node_id]
            right_id = tree_.children_right[node_id]
            children = [int(left_id), int(right_id)]
            self._collect_rule_node(
                left_id, tree_, feature_names, is_regression, class_names, tree_module, nodes
            )
            self._collect_rule_node(
                right_id, tree_, feature_names, is_regression, class_names, tree_module, nodes
            )

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

    def _build_rules_text(
        self,
        tree_,
        feature_names: list[str],
        is_regression: bool,
        class_names: list[str],
        cat_categories: dict[str, list[str]],
        tree_module,
    ) -> list[str]:
        """Recursively derive human-readable IF/THEN rule strings from the fitted tree."""
        rules_text: list[str] = []
        self._collect_rule_text(
            0,
            "",
            tree_,
            feature_names,
            is_regression,
            class_names,
            cat_categories,
            tree_module,
            rules_text,
        )
        return rules_text

    def _leaf_rule_text(
        self, tree_, node_id: int, is_regression: bool, class_names: list[str]
    ) -> str:
        """Build the trailing THEN clause + confidence/sample count for a leaf node."""
        total_samples = int(tree_.n_node_samples[node_id])
        if is_regression:
            val = float(tree_.value[node_id][0][0])
            return f"Value = {val:.2f} (Samples: {total_samples})"

        value = tree_.value[node_id][0]
        class_idx = np.argmax(value)
        class_name = str(class_names[class_idx]) if class_idx < len(class_names) else "Unknown"
        total = np.sum(value)
        confidence = (value[class_idx] / total) * 100 if total > 0 else 0
        return f"{class_name} (Confidence: {confidence:.1f}%, Samples: {int(total)})"

    def _split_clauses(
        self,
        feature_name: str,
        threshold: float,
        cat_categories: dict[str, list[str]],
    ) -> tuple[str, str]:
        """Build the left/right branch clauses for a split, using category names when relevant."""
        if feature_name in cat_categories:
            categories = cat_categories[feature_name]
            left_categories = [c for i, c in enumerate(categories) if i <= threshold]
            right_categories = [c for i, c in enumerate(categories) if i > threshold]
            return (
                f"{feature_name} in {left_categories}",
                f"{feature_name} in {right_categories}",
            )
        return (
            f"{feature_name} <= {threshold:.2f}",
            f"{feature_name} > {threshold:.2f}",
        )

    def _collect_rule_text(
        self,
        node_id: int,
        current_rule: str,
        tree_,
        feature_names: list[str],
        is_regression: bool,
        class_names: list[str],
        cat_categories: dict[str, list[str]],
        tree_module,
        rules_text: list[str],
    ) -> None:
        """Recursively build an IF/THEN rule string for each leaf reachable from ``node_id``."""
        if tree_.children_left[node_id] == tree_module.TREE_LEAF:
            then_clause = self._leaf_rule_text(tree_, node_id, is_regression, class_names)
            rules_text.append(f"IF {current_rule} THEN {then_clause}")
            return

        feature_idx = tree_.feature[node_id]
        feature_name = feature_names[feature_idx]
        threshold = tree_.threshold[node_id]

        left_clause, right_clause = self._split_clauses(feature_name, threshold, cat_categories)

        left_rule = f"{current_rule} AND {left_clause}" if current_rule else left_clause
        self._collect_rule_text(
            tree_.children_left[node_id],
            left_rule,
            tree_,
            feature_names,
            is_regression,
            class_names,
            cat_categories,
            tree_module,
            rules_text,
        )

        right_rule = f"{current_rule} AND {right_clause}" if current_rule else right_clause
        self._collect_rule_text(
            tree_.children_right[node_id],
            right_rule,
            tree_,
            feature_names,
            is_regression,
            class_names,
            cat_categories,
            tree_module,
            rules_text,
        )
