import { useMemo } from 'react';
import type { FeatureNodeParameter } from '../../../api';

type FilterFlags = {
  isBinningNode: boolean;
  isCastNode: boolean;
  isDropMissingColumnsNode: boolean;
  isDropMissingRowsNode: boolean;
  isImputerNode: boolean;
  isMissingIndicatorNode: boolean;
  isReplaceAliasesNode: boolean;
  isTrimWhitespaceNode: boolean;
  isRemoveSpecialCharsNode: boolean;
  isReplaceInvalidValuesNode: boolean;
  isRegexCleanupNode: boolean;
  isNormalizeTextCaseNode: boolean;
  isStandardizeDatesNode: boolean;
  isScalingNode: boolean;
  isClassUndersamplingNode: boolean;
  isClassOversamplingNode: boolean;
  isTrainModelDraftNode: boolean;
  isLabelEncodingNode: boolean;
  isOrdinalEncodingNode: boolean;
  isDummyEncodingNode: boolean;
  isOneHotEncodingNode: boolean;
  isRemoveDuplicatesNode: boolean;
  isSkewnessNode: boolean;
  isFeatureTargetSplitNode: boolean;
};

export const useFilteredParameters = (
  parameters: FeatureNodeParameter[],
  flags: FilterFlags,
): FeatureNodeParameter[] => {
  const {
    isBinningNode,
    isCastNode,
    isDropMissingColumnsNode,
    isDropMissingRowsNode,
    isImputerNode,
    isMissingIndicatorNode,
    isReplaceAliasesNode,
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isReplaceInvalidValuesNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
    isStandardizeDatesNode,
    isScalingNode,
    isClassUndersamplingNode,
    isClassOversamplingNode,
    isTrainModelDraftNode,
    isLabelEncodingNode,
    isOrdinalEncodingNode,
    isDummyEncodingNode,
    isOneHotEncodingNode,
    isRemoveDuplicatesNode,
    isSkewnessNode,
    isFeatureTargetSplitNode,
  } = flags;

  return useMemo(() => {
    let result = parameters;
    result = result.filter((parameter) => parameter?.source?.type !== 'drop_column_recommendations');
    if (isDropMissingColumnsNode || isDropMissingRowsNode) {
      result = result.filter((parameter) => parameter.name !== 'missing_threshold');
    }
    if (isDropMissingRowsNode) {
      result = result.filter((parameter) => parameter.name !== 'drop_if_any_missing');
    }
    if (isImputerNode) {
      result = result.filter((parameter) => parameter.name !== 'strategies');
    }
    if (isSkewnessNode) {
      result = result.filter((parameter) => parameter.name !== 'transformations');
    }
    if (isCastNode) {
      result = result.filter((parameter) => parameter.name !== 'columns');
    }
    if (isBinningNode) {
      result = result.filter((parameter) => parameter.name !== 'columns');
    }
    if (isScalingNode) {
      result = result.filter(
        (parameter) => !['columns', 'default_method', 'auto_detect'].includes(parameter.name),
      );
    }
    if (isRemoveDuplicatesNode) {
      result = result.filter((parameter) => parameter.name !== 'columns' && parameter.name !== 'keep');
    }
    if (isMissingIndicatorNode) {
      result = result.filter((parameter) => parameter.name !== 'columns' && parameter.name !== 'flag_suffix');
    }
    if (isReplaceAliasesNode) {
      result = result.filter(
        (parameter) => !['columns', 'mode', 'custom_pairs'].includes(parameter.name),
      );
    }
    if (isTrimWhitespaceNode) {
      result = result.filter((parameter) => !['columns', 'mode'].includes(parameter.name));
    }
    if (isRemoveSpecialCharsNode) {
      result = result.filter((parameter) => !['columns', 'mode', 'replacement'].includes(parameter.name));
    }
    if (isReplaceInvalidValuesNode) {
      result = result.filter(
        (parameter) => !['columns', 'mode', 'min_value', 'max_value'].includes(parameter.name),
      );
    }
    if (isRegexCleanupNode) {
      result = result.filter((parameter) => !['columns', 'mode', 'pattern', 'replacement'].includes(parameter.name));
    }
    if (isNormalizeTextCaseNode) {
      result = result.filter((parameter) => !['columns', 'mode'].includes(parameter.name));
    }
    if (isStandardizeDatesNode) {
      result = result.filter((parameter) => !['columns', 'mode'].includes(parameter.name));
    }
    if (isClassUndersamplingNode) {
      result = result.filter(
        (parameter) =>
          !['method', 'target_column', 'sampling_strategy', 'random_state', 'replacement'].includes(parameter.name),
      );
    }
    if (isClassOversamplingNode) {
      result = result.filter(
        (parameter) => !['method', 'target_column', 'sampling_strategy', 'random_state'].includes(parameter.name),
      );
    }
    if (isTrainModelDraftNode) {
      result = result.filter(
        (parameter) =>
          ![
            'target_column',
            'problem_type',
            'cv_enabled',
            'cv_strategy',
            'cv_folds',
            'cv_shuffle',
            'cv_random_state',
            'cv_refit_strategy',
          ].includes(parameter.name),
      );
    }
    if (isFeatureTargetSplitNode) {
      result = result.filter((parameter) => parameter.name !== 'target_column');
    }
    if (isLabelEncodingNode) {
      result = result.filter(
        (parameter) =>
          ![
            'columns',
            'auto_detect',
            'max_unique_values',
            'output_suffix',
            'drop_original',
            'missing_strategy',
            'missing_code',
          ].includes(parameter.name),
      );
    }
    if (isOrdinalEncodingNode) {
      result = result.filter(
        (parameter) =>
          ![
            'columns',
            'auto_detect',
            'max_categories',
            'output_suffix',
            'drop_original',
            'encode_missing',
            'handle_unknown',
            'unknown_value',
          ].includes(parameter.name),
      );
    }
    if (isDummyEncodingNode) {
      result = result.filter(
        (parameter) =>
          ![
            'columns',
            'auto_detect',
            'max_categories',
            'drop_first',
            'include_missing',
            'drop_original',
            'prefix_separator',
          ].includes(parameter.name),
      );
    }
    if (isOneHotEncodingNode) {
      result = result.filter(
        (parameter) =>
          ![
            'columns',
            'auto_detect',
            'max_categories',
            'drop_first',
            'include_missing',
            'drop_original',
            'prefix_separator',
          ].includes(parameter.name),
      );
    }
    return result;
  }, [
    parameters,
    isBinningNode,
    isCastNode,
    isDropMissingColumnsNode,
    isDropMissingRowsNode,
    isImputerNode,
    isMissingIndicatorNode,
    isReplaceAliasesNode,
    isTrimWhitespaceNode,
    isRemoveSpecialCharsNode,
    isReplaceInvalidValuesNode,
    isRegexCleanupNode,
    isNormalizeTextCaseNode,
    isStandardizeDatesNode,
    isScalingNode,
    isClassUndersamplingNode,
    isClassOversamplingNode,
    isTrainModelDraftNode,
    isLabelEncodingNode,
    isOrdinalEncodingNode,
    isDummyEncodingNode,
    isOneHotEncodingNode,
    isRemoveDuplicatesNode,
    isSkewnessNode,
    isFeatureTargetSplitNode,
  ]);
};
