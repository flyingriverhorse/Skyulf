import { useMemo } from 'react';
import type { FeatureNodeParameter } from '../../../api';
import type { CatalogFlagMap } from './useCatalogFlags';

export const useNodeSpecificParameters = (
  getParameter: (name: string) => FeatureNodeParameter | null,
  flags: CatalogFlagMap
) => {
  const getParameterIf = (condition: boolean, name: string) => (condition ? getParameter(name) : null);

  return useMemo(() => {
    const {
      isBinningNode,
      isScalingNode,
      isRemoveDuplicatesNode,
      isReplaceAliasesNode,
      isTrimWhitespaceNode,
      isRemoveSpecialCharsNode,
      isReplaceInvalidValuesNode,
      isRegexCleanupNode,
      isNormalizeTextCaseNode,
      isLabelEncodingNode,
      isHashEncodingNode,
      isPolynomialFeaturesNode,
      isFeatureSelectionNode,
      isFeatureMathNode,
      isClassUndersamplingNode,
      isClassOversamplingNode,
      isFeatureTargetSplitNode,
      isTrainModelDraftNode,
      isHyperparameterTuningNode,
      isTargetEncodingNode,
      isOrdinalEncodingNode,
      isDummyEncodingNode,
      isOneHotEncodingNode,
      isMissingIndicatorNode,
      isDropMissingRowsNode,
    } = flags;

    const isClassResamplingNode = isClassUndersamplingNode || isClassOversamplingNode;
    const hasTrainOrTuningNode = isTrainModelDraftNode || isHyperparameterTuningNode;

    return {
      binningColumnsParameter: getParameterIf(isBinningNode, 'columns'),
      scalingColumnsParameter: getParameterIf(isScalingNode, 'columns'),
      removeDuplicatesColumnsParameter: getParameterIf(isRemoveDuplicatesNode, 'columns'),
      removeDuplicatesKeepParameter: getParameterIf(isRemoveDuplicatesNode, 'keep'),
      replaceAliasesCustomPairsParameter: getParameterIf(isReplaceAliasesNode, 'custom_pairs'),
      trimWhitespaceColumnsParameter: getParameterIf(isTrimWhitespaceNode, 'columns'),
      trimWhitespaceModeParameter: getParameterIf(isTrimWhitespaceNode, 'mode'),
      removeSpecialColumnsParameter: getParameterIf(isRemoveSpecialCharsNode, 'columns'),
      removeSpecialModeParameter: getParameterIf(isRemoveSpecialCharsNode, 'mode'),
      removeSpecialReplacementParameter: getParameterIf(isRemoveSpecialCharsNode, 'replacement'),
      replaceInvalidColumnsParameter: getParameterIf(isReplaceInvalidValuesNode, 'columns'),
      replaceInvalidModeParameter: getParameterIf(isReplaceInvalidValuesNode, 'mode'),
      replaceInvalidMinValueParameter: getParameterIf(isReplaceInvalidValuesNode, 'min_value'),
      replaceInvalidMaxValueParameter: getParameterIf(isReplaceInvalidValuesNode, 'max_value'),
      regexCleanupColumnsParameter: getParameterIf(isRegexCleanupNode, 'columns'),
      regexCleanupModeParameter: getParameterIf(isRegexCleanupNode, 'mode'),
      regexCleanupPatternParameter: getParameterIf(isRegexCleanupNode, 'pattern'),
      regexCleanupReplacementParameter: getParameterIf(isRegexCleanupNode, 'replacement'),
      normalizeCaseColumnsParameter: getParameterIf(isNormalizeTextCaseNode, 'columns'),
      normalizeCaseModeParameter: getParameterIf(isNormalizeTextCaseNode, 'mode'),

      labelEncodingColumnsParameter: isLabelEncodingNode ? getParameter('columns') : null,
      labelEncodingAutoDetectParameter: isLabelEncodingNode ? getParameter('auto_detect') : null,
      labelEncodingMaxUniqueParameter: isLabelEncodingNode ? getParameter('max_unique_values') : null,
      labelEncodingOutputSuffixParameter: isLabelEncodingNode ? getParameter('output_suffix') : null,
      labelEncodingDropOriginalParameter: isLabelEncodingNode ? getParameter('drop_original') : null,
      labelEncodingMissingStrategyParameter: isLabelEncodingNode ? getParameter('missing_strategy') : null,
      labelEncodingMissingCodeParameter: isLabelEncodingNode ? getParameter('missing_code') : null,

      hashEncodingColumnsParameter: isHashEncodingNode ? getParameter('columns') : null,
      hashEncodingAutoDetectParameter: isHashEncodingNode ? getParameter('auto_detect') : null,
      hashEncodingMaxCategoriesParameter: isHashEncodingNode ? getParameter('max_categories') : null,
      hashEncodingBucketsParameter: isHashEncodingNode ? getParameter('n_buckets') : null,
      hashEncodingOutputSuffixParameter: isHashEncodingNode ? getParameter('output_suffix') : null,
      hashEncodingDropOriginalParameter: isHashEncodingNode ? getParameter('drop_original') : null,
      hashEncodingEncodeMissingParameter: isHashEncodingNode ? getParameter('encode_missing') : null,

      polynomialColumnsParameter: getParameterIf(isPolynomialFeaturesNode, 'columns'),
      polynomialAutoDetectParameter: getParameterIf(isPolynomialFeaturesNode, 'auto_detect'),
      polynomialDegreeParameter: getParameterIf(isPolynomialFeaturesNode, 'degree'),
      polynomialIncludeBiasParameter: getParameterIf(isPolynomialFeaturesNode, 'include_bias'),
      polynomialInteractionOnlyParameter: getParameterIf(isPolynomialFeaturesNode, 'interaction_only'),
      polynomialIncludeInputFeaturesParameter: getParameterIf(
        isPolynomialFeaturesNode,
        'include_input_features'
      ),
      polynomialOutputPrefixParameter: getParameterIf(isPolynomialFeaturesNode, 'output_prefix'),

      featureSelectionColumnsParameter: getParameterIf(isFeatureSelectionNode, 'columns'),
      featureSelectionAutoDetectParameter: getParameterIf(isFeatureSelectionNode, 'auto_detect'),
      featureSelectionTargetColumnParameter: getParameterIf(isFeatureSelectionNode, 'target_column'),
      featureSelectionMethodParameter: getParameterIf(isFeatureSelectionNode, 'method'),
      featureSelectionScoreFuncParameter: getParameterIf(isFeatureSelectionNode, 'score_func'),
      featureSelectionProblemTypeParameter: getParameterIf(isFeatureSelectionNode, 'problem_type'),
      featureSelectionKParameter: getParameterIf(isFeatureSelectionNode, 'k'),
      featureSelectionPercentileParameter: getParameterIf(isFeatureSelectionNode, 'percentile'),
      featureSelectionAlphaParameter: getParameterIf(isFeatureSelectionNode, 'alpha'),
      featureSelectionThresholdParameter: getParameterIf(isFeatureSelectionNode, 'threshold'),
      featureSelectionModeParameter: getParameterIf(isFeatureSelectionNode, 'mode'),
      featureSelectionEstimatorParameter: getParameterIf(isFeatureSelectionNode, 'estimator'),
      featureSelectionStepParameter: getParameterIf(isFeatureSelectionNode, 'step'),
      featureSelectionMinFeaturesParameter: getParameterIf(isFeatureSelectionNode, 'min_features'),
      featureSelectionMaxFeaturesParameter: getParameterIf(isFeatureSelectionNode, 'max_features'),
      featureSelectionDropUnselectedParameter: getParameterIf(
        isFeatureSelectionNode,
        'drop_unselected'
      ),

      featureMathErrorHandlingParameter: getParameterIf(isFeatureMathNode, 'error_handling'),
      featureMathAllowOverwriteParameter: getParameterIf(isFeatureMathNode, 'allow_overwrite'),
      featureMathDefaultTimezoneParameter: getParameterIf(isFeatureMathNode, 'default_timezone'),
      featureMathEpsilonParameter: getParameterIf(isFeatureMathNode, 'epsilon'),
      resamplingMethodParameter: getParameterIf(isClassResamplingNode, 'method'),
      resamplingTargetColumnParameter: getParameterIf(isClassResamplingNode, 'target_column'),
      resamplingSamplingStrategyParameter: getParameterIf(
        isClassResamplingNode,
        'sampling_strategy'
      ),
      resamplingRandomStateParameter: getParameterIf(isClassResamplingNode, 'random_state'),
      resamplingKNeighborsParameter: getParameterIf(isClassOversamplingNode, 'k_neighbors'),
      resamplingReplacementParameter: getParameterIf(isClassUndersamplingNode, 'replacement'),
      featureTargetSplitTargetColumnParameter: getParameterIf(
        isFeatureTargetSplitNode,
        'target_column'
      ),
      trainModelTargetColumnParameter: getParameterIf(hasTrainOrTuningNode, 'target_column'),
      trainModelProblemTypeParameter: getParameterIf(hasTrainOrTuningNode, 'problem_type'),
      trainModelModelTypeParameter: getParameterIf(hasTrainOrTuningNode, 'model_type'),
      trainModelHyperparametersParameter: getParameterIf(
        isTrainModelDraftNode,
        'hyperparameters'
      ),
      hyperparameterTuningSearchStrategyParameter: getParameterIf(
        isHyperparameterTuningNode,
        'search_strategy'
      ),
      hyperparameterTuningSearchIterationsParameter: getParameterIf(
        isHyperparameterTuningNode,
        'search_iterations'
      ),
      hyperparameterTuningSearchRandomStateParameter: getParameterIf(
        isHyperparameterTuningNode,
        'search_random_state'
      ),
      hyperparameterTuningScoringMetricParameter: getParameterIf(
        isHyperparameterTuningNode,
        'scoring_metric'
      ),
      trainModelCvEnabledParameter: getParameterIf(hasTrainOrTuningNode, 'cv_enabled'),
      trainModelCvStrategyParameter: getParameterIf(hasTrainOrTuningNode, 'cv_strategy'),
      trainModelCvFoldsParameter: getParameterIf(hasTrainOrTuningNode, 'cv_folds'),
      trainModelCvShuffleParameter: getParameterIf(hasTrainOrTuningNode, 'cv_shuffle'),
      trainModelCvRandomStateParameter: getParameterIf(hasTrainOrTuningNode, 'cv_random_state'),
      trainModelCvRefitStrategyParameter: getParameterIf(
        isTrainModelDraftNode,
        'cv_refit_strategy'
      ),

      targetEncodingColumnsParameter: isTargetEncodingNode ? getParameter('columns') : null,
      targetEncodingTargetColumnParameter: isTargetEncodingNode ? getParameter('target_column') : null,
      targetEncodingAutoDetectParameter: isTargetEncodingNode ? getParameter('auto_detect') : null,
      targetEncodingMaxCategoriesParameter: isTargetEncodingNode ? getParameter('max_categories') : null,
      targetEncodingOutputSuffixParameter: isTargetEncodingNode ? getParameter('output_suffix') : null,
      targetEncodingDropOriginalParameter: isTargetEncodingNode ? getParameter('drop_original') : null,
      targetEncodingSmoothingParameter: isTargetEncodingNode ? getParameter('smoothing') : null,
      targetEncodingEncodeMissingParameter: isTargetEncodingNode ? getParameter('encode_missing') : null,
      targetEncodingHandleUnknownParameter: isTargetEncodingNode ? getParameter('handle_unknown') : null,

      ordinalEncodingColumnsParameter: isOrdinalEncodingNode ? getParameter('columns') : null,
      ordinalEncodingAutoDetectParameter: isOrdinalEncodingNode ? getParameter('auto_detect') : null,
      ordinalEncodingMaxCategoriesParameter: isOrdinalEncodingNode ? getParameter('max_categories') : null,
      ordinalEncodingOutputSuffixParameter: isOrdinalEncodingNode ? getParameter('output_suffix') : null,
      ordinalEncodingDropOriginalParameter: isOrdinalEncodingNode ? getParameter('drop_original') : null,
      ordinalEncodingEncodeMissingParameter: isOrdinalEncodingNode ? getParameter('encode_missing') : null,
      ordinalEncodingHandleUnknownParameter: isOrdinalEncodingNode ? getParameter('handle_unknown') : null,
      ordinalEncodingUnknownValueParameter: isOrdinalEncodingNode ? getParameter('unknown_value') : null,

      dummyEncodingColumnsParameter: isDummyEncodingNode ? getParameter('columns') : null,
      dummyEncodingAutoDetectParameter: isDummyEncodingNode ? getParameter('auto_detect') : null,
      dummyEncodingMaxCategoriesParameter: isDummyEncodingNode ? getParameter('max_categories') : null,
      dummyEncodingDropFirstParameter: isDummyEncodingNode ? getParameter('drop_first') : null,
      dummyEncodingIncludeMissingParameter: isDummyEncodingNode ? getParameter('include_missing') : null,
      dummyEncodingDropOriginalParameter: isDummyEncodingNode ? getParameter('drop_original') : null,
      dummyEncodingPrefixSeparatorParameter: isDummyEncodingNode ? getParameter('prefix_separator') : null,

      oneHotEncodingColumnsParameter: isOneHotEncodingNode ? getParameter('columns') : null,
      oneHotEncodingAutoDetectParameter: isOneHotEncodingNode ? getParameter('auto_detect') : null,
      oneHotEncodingMaxCategoriesParameter: isOneHotEncodingNode ? getParameter('max_categories') : null,
      oneHotEncodingDropFirstParameter: isOneHotEncodingNode ? getParameter('drop_first') : null,
      oneHotEncodingIncludeMissingParameter: isOneHotEncodingNode ? getParameter('include_missing') : null,
      oneHotEncodingDropOriginalParameter: isOneHotEncodingNode ? getParameter('drop_original') : null,
      oneHotEncodingPrefixSeparatorParameter: isOneHotEncodingNode ? getParameter('prefix_separator') : null,

      missingIndicatorColumnsParameter: getParameterIf(isMissingIndicatorNode, 'columns'),
      missingIndicatorSuffixParameter: getParameterIf(isMissingIndicatorNode, 'flag_suffix'),
      scalingDefaultMethodParameter: getParameterIf(isScalingNode, 'default_method'),
      scalingAutoDetectParameter: getParameterIf(isScalingNode, 'auto_detect'),
      dropRowsAnyParameter: getParameterIf(isDropMissingRowsNode, 'drop_if_any_missing'),
    };
  }, [flags, getParameter]);
};
