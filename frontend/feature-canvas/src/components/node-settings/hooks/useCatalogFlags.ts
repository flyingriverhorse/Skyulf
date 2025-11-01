import { useMemo } from 'react';
import type { Node } from 'react-flow-renderer';
import { DATA_CONSISTENCY_TYPES, INSPECTION_NODE_TYPES, isDatasetNode } from '../utils/catalogTypes';

export type CatalogFlagMap = {
  catalogType: string;
  isDataset: boolean;
  isImputerNode: boolean;
  isMissingIndicatorNode: boolean;
  isReplaceAliasesNode: boolean;
  isTrimWhitespaceNode: boolean;
  isRemoveSpecialCharsNode: boolean;
  isReplaceInvalidValuesNode: boolean;
  isRegexCleanupNode: boolean;
  isNormalizeTextCaseNode: boolean;
  isStandardizeDatesNode: boolean;
  isLabelEncodingNode: boolean;
  isTargetEncodingNode: boolean;
  isHashEncodingNode: boolean;
  isTrainModelDraftNode: boolean;
  isModelEvaluationNode: boolean;
  isModelRegistryNode: boolean;
  isHyperparameterTuningNode: boolean;
  isFeatureTargetSplitNode: boolean;
  isTrainTestSplitNode: boolean;
  isClassUndersamplingNode: boolean;
  isClassOversamplingNode: boolean;
  isOrdinalEncodingNode: boolean;
  isDummyEncodingNode: boolean;
  isOneHotEncodingNode: boolean;
  isFeatureMathNode: boolean;
  isCastNode: boolean;
  isBinningNode: boolean;
  isScalingNode: boolean;
  isPolynomialFeaturesNode: boolean;
  isFeatureSelectionNode: boolean;
  isSkewnessNode: boolean;
  isSkewnessDistributionNode: boolean;
  isBinnedDistributionNode: boolean;
  isTransformerAuditNode: boolean;
  isOutlierNode: boolean;
  isDataConsistencyNode: boolean;
  isInspectionNode: boolean;
  isRemoveDuplicatesNode: boolean;
  isDropMissingColumnsNode: boolean;
  isDropMissingRowsNode: boolean;
  isDropMissingNode: boolean;
};

export const useCatalogFlags = (node: Node | null | undefined): CatalogFlagMap => {
  const catalogType = node?.data?.catalogType ?? '';

  return useMemo<CatalogFlagMap>(() => {
    const isDataset = isDatasetNode(node);
    const isImputerNode =
      catalogType === 'imputation_methods' ||
      catalogType === 'advanced_imputer' ||
      catalogType === 'simple_imputer';
    const isMissingIndicatorNode = catalogType === 'missing_value_indicator';
    const isReplaceAliasesNode = catalogType === 'replace_aliases_typos';
    const isTrimWhitespaceNode = catalogType === 'trim_whitespace';
    const isRemoveSpecialCharsNode = catalogType === 'remove_special_characters';
    const isReplaceInvalidValuesNode = catalogType === 'replace_invalid_values';
    const isRegexCleanupNode = catalogType === 'regex_replace_fix';
    const isNormalizeTextCaseNode = catalogType === 'normalize_text_case';
    const isStandardizeDatesNode = catalogType === 'standardize_date_formats';
    const isLabelEncodingNode = catalogType === 'label_encoding';
    const isTargetEncodingNode = catalogType === 'target_encoding';
    const isHashEncodingNode = catalogType === 'hash_encoding';
    const isTrainModelDraftNode = catalogType === 'train_model_draft';
    const isModelEvaluationNode = catalogType === 'model_evaluation';
    const isModelRegistryNode = catalogType === 'model_registry_overview';
    const isHyperparameterTuningNode = catalogType === 'hyperparameter_tuning';
    const isFeatureTargetSplitNode = catalogType === 'feature_target_split';
    const isTrainTestSplitNode = catalogType === 'train_test_split';
    const isClassUndersamplingNode = catalogType === 'class_undersampling';
    const isClassOversamplingNode = catalogType === 'class_oversampling';
    const isOrdinalEncodingNode = catalogType === 'ordinal_encoding';
    const isDummyEncodingNode = catalogType === 'dummy_encoding';
    const isOneHotEncodingNode = catalogType === 'one_hot_encoding';
    const isFeatureMathNode = catalogType === 'feature_math';
    const isCastNode = catalogType === 'cast_column_types';
    const isBinningNode = catalogType === 'binning_discretization';
    const isScalingNode = catalogType === 'scale_numeric_features';
    const isPolynomialFeaturesNode = catalogType === 'polynomial_features';
    const isFeatureSelectionNode = catalogType === 'feature_selection';
    const isTransformerAuditNode = catalogType === 'transformer_audit';
    const isOutlierNode = catalogType === 'outlier_removal';
    const isSkewnessNode = catalogType === 'skewness_transform';
    const isSkewnessDistributionNode = catalogType === 'skewness_distribution';
    const isBinnedDistributionNode = catalogType === 'binned_distribution';
    const isDataConsistencyNode = DATA_CONSISTENCY_TYPES.has(catalogType);
    const isInspectionNode = INSPECTION_NODE_TYPES.has(catalogType);
    const isRemoveDuplicatesNode = catalogType === 'remove_duplicates';
    const isDropMissingColumnsNode = catalogType === 'drop_missing_columns';
    const isDropMissingRowsNode = catalogType === 'drop_missing_rows';
    const isDropMissingNode = isDropMissingColumnsNode || isDropMissingRowsNode;

    return {
      catalogType,
      isDataset,
      isImputerNode,
      isMissingIndicatorNode,
      isReplaceAliasesNode,
      isTrimWhitespaceNode,
      isRemoveSpecialCharsNode,
      isReplaceInvalidValuesNode,
      isRegexCleanupNode,
      isNormalizeTextCaseNode,
      isStandardizeDatesNode,
      isLabelEncodingNode,
      isTargetEncodingNode,
      isHashEncodingNode,
      isTrainModelDraftNode,
      isModelEvaluationNode,
      isModelRegistryNode,
      isHyperparameterTuningNode,
      isFeatureTargetSplitNode,
      isTrainTestSplitNode,
      isClassUndersamplingNode,
      isClassOversamplingNode,
      isOrdinalEncodingNode,
      isDummyEncodingNode,
      isOneHotEncodingNode,
      isFeatureMathNode,
      isCastNode,
      isBinningNode,
      isScalingNode,
      isPolynomialFeaturesNode,
      isFeatureSelectionNode,
      isTransformerAuditNode,
      isOutlierNode,
      isSkewnessNode,
      isSkewnessDistributionNode,
      isBinnedDistributionNode,
      isDataConsistencyNode,
      isInspectionNode,
      isRemoveDuplicatesNode,
      isDropMissingColumnsNode,
      isDropMissingRowsNode,
      isDropMissingNode,
    };
  }, [catalogType, node]);
};
