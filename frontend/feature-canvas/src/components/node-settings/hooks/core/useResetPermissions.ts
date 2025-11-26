import { useMemo } from 'react';
import type { CatalogFlagMap } from './useCatalogFlags';

interface ResetPermissionOptions {
  isResetAvailable?: boolean;
  defaultConfigTemplate?: Record<string, any> | null;
  catalogFlags: CatalogFlagMap;
}

interface ResetPermissionResult {
  canResetNode: boolean;
  headerCanResetNode: boolean;
  footerCanResetNode: boolean;
}

/**
 * Centralizes the logic for deciding whether a node can be reset from the header/footer.
 * Keeping this here keeps NodeSettingsModal leaner as we continue modularization.
 */
export const useResetPermissions = ({
  isResetAvailable = false,
  defaultConfigTemplate,
  catalogFlags,
}: ResetPermissionOptions): ResetPermissionResult => {
  const {
    isDataset,
    isScalingNode,
    isBinningNode,
    isBinnedDistributionNode,
    isSkewnessDistributionNode,
    isFeatureMathNode,
    isFeatureTargetSplitNode,
    isTrainTestSplitNode,
    isModelEvaluationNode,
    isHashEncodingNode,
    isPolynomialFeaturesNode,
    isFeatureSelectionNode,
    isDataConsistencyNode,
    isTrainModelDraftNode,
    isHyperparameterTuningNode,
    isClassUndersamplingNode,
    isClassOversamplingNode,
    isLabelEncodingNode,
    isTargetEncodingNode,
    isDummyEncodingNode,
    isOneHotEncodingNode,
    isOrdinalEncodingNode,
    isImputerNode,
    isMissingIndicatorNode,
    isReplaceAliasesNode,
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isReplaceInvalidValuesNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
    isStandardizeDatesNode,
    isCastNode,
    isRemoveDuplicatesNode,
    isDropMissingNode,
    isOutlierNode,
  } = catalogFlags;

  const canResetNode = useMemo(
    () => Boolean(isResetAvailable && !isDataset && defaultConfigTemplate),
    [defaultConfigTemplate, isDataset, isResetAvailable],
  );

  const isClassResamplingNode = isClassUndersamplingNode || isClassOversamplingNode;

  const footerResetFlags = [
    isFeatureMathNode,
    isFeatureTargetSplitNode,
    isTrainTestSplitNode,
    isModelEvaluationNode,
    isHashEncodingNode,
    isPolynomialFeaturesNode,
    isFeatureSelectionNode,
    isDataConsistencyNode,
    isTrainModelDraftNode,
    isHyperparameterTuningNode,
    isClassResamplingNode,
    isLabelEncodingNode,
    isTargetEncodingNode,
    isDummyEncodingNode,
    isOneHotEncodingNode,
    isOrdinalEncodingNode,
    isImputerNode,
    isMissingIndicatorNode,
    isReplaceAliasesNode,
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isReplaceInvalidValuesNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
    isStandardizeDatesNode,
    isCastNode,
    isRemoveDuplicatesNode,
    isDropMissingNode,
    isOutlierNode,
  ];

  const footerOnlyResetNodes = footerResetFlags.some(Boolean);
  const disableAllGlobalResets = isBinnedDistributionNode || isSkewnessDistributionNode;
  const headerResetDisabled = disableAllGlobalResets || isScalingNode || isBinningNode || footerOnlyResetNodes;

  const headerCanResetNode = canResetNode && !headerResetDisabled;
  const footerCanResetNode = canResetNode && !disableAllGlobalResets;

  return {
    canResetNode,
    headerCanResetNode,
    footerCanResetNode,
  };
};
