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
      isDropMissingColumnsNode,
    } = flags;

    const isClassResamplingNode = isClassUndersamplingNode || isClassOversamplingNode;
    const hasTrainOrTuningNode = isTrainModelDraftNode || isHyperparameterTuningNode;

    return {
      binning: {
        columns: getParameterIf(isBinningNode, 'columns'),
      },
      scaling: {
        columns: getParameterIf(isScalingNode, 'columns'),
        defaultMethod: getParameterIf(isScalingNode, 'default_method'),
        autoDetect: getParameterIf(isScalingNode, 'auto_detect'),
      },
      removeDuplicates: {
        columns: getParameterIf(isRemoveDuplicatesNode, 'columns'),
        keep: getParameterIf(isRemoveDuplicatesNode, 'keep'),
      },
      replaceAliases: {
        customPairs: getParameterIf(isReplaceAliasesNode, 'custom_pairs'),
      },
      trimWhitespace: {
        columns: getParameterIf(isTrimWhitespaceNode, 'columns'),
        mode: getParameterIf(isTrimWhitespaceNode, 'mode'),
      },
      removeSpecial: {
        columns: getParameterIf(isRemoveSpecialCharsNode, 'columns'),
        mode: getParameterIf(isRemoveSpecialCharsNode, 'mode'),
        replacement: getParameterIf(isRemoveSpecialCharsNode, 'replacement'),
      },
      replaceInvalid: {
        columns: getParameterIf(isReplaceInvalidValuesNode, 'columns'),
        mode: getParameterIf(isReplaceInvalidValuesNode, 'mode'),
        minValue: getParameterIf(isReplaceInvalidValuesNode, 'min_value'),
        maxValue: getParameterIf(isReplaceInvalidValuesNode, 'max_value'),
      },
      regexCleanup: {
        columns: getParameterIf(isRegexCleanupNode, 'columns'),
        mode: getParameterIf(isRegexCleanupNode, 'mode'),
        pattern: getParameterIf(isRegexCleanupNode, 'pattern'),
        replacement: getParameterIf(isRegexCleanupNode, 'replacement'),
      },
      normalizeCase: {
        columns: getParameterIf(isNormalizeTextCaseNode, 'columns'),
        mode: getParameterIf(isNormalizeTextCaseNode, 'mode'),
      },
      labelEncoding: {
        columns: isLabelEncodingNode ? getParameter('columns') : null,
        autoDetect: isLabelEncodingNode ? getParameter('auto_detect') : null,
        maxUnique: isLabelEncodingNode ? getParameter('max_unique_values') : null,
        outputSuffix: isLabelEncodingNode ? getParameter('output_suffix') : null,
        dropOriginal: isLabelEncodingNode ? getParameter('drop_original') : null,
        missingStrategy: isLabelEncodingNode ? getParameter('missing_strategy') : null,
        missingCode: isLabelEncodingNode ? getParameter('missing_code') : null,
      },
      hashEncoding: {
        columns: isHashEncodingNode ? getParameter('columns') : null,
        autoDetect: isHashEncodingNode ? getParameter('auto_detect') : null,
        maxCategories: isHashEncodingNode ? getParameter('max_categories') : null,
        buckets: isHashEncodingNode ? getParameter('n_buckets') : null,
        outputSuffix: isHashEncodingNode ? getParameter('output_suffix') : null,
        dropOriginal: isHashEncodingNode ? getParameter('drop_original') : null,
        encodeMissing: isHashEncodingNode ? getParameter('encode_missing') : null,
      },
      polynomial: {
        columns: getParameterIf(isPolynomialFeaturesNode, 'columns'),
        autoDetect: getParameterIf(isPolynomialFeaturesNode, 'auto_detect'),
        degree: getParameterIf(isPolynomialFeaturesNode, 'degree'),
        includeBias: getParameterIf(isPolynomialFeaturesNode, 'include_bias'),
        interactionOnly: getParameterIf(isPolynomialFeaturesNode, 'interaction_only'),
        includeInputFeatures: getParameterIf(isPolynomialFeaturesNode, 'include_input_features'),
        outputPrefix: getParameterIf(isPolynomialFeaturesNode, 'output_prefix'),
      },
      featureSelection: {
        columns: getParameterIf(isFeatureSelectionNode, 'columns'),
        autoDetect: getParameterIf(isFeatureSelectionNode, 'auto_detect'),
        targetColumn: getParameterIf(isFeatureSelectionNode, 'target_column'),
        method: getParameterIf(isFeatureSelectionNode, 'method'),
        scoreFunc: getParameterIf(isFeatureSelectionNode, 'score_func'),
        problemType: getParameterIf(isFeatureSelectionNode, 'problem_type'),
        k: getParameterIf(isFeatureSelectionNode, 'k'),
        percentile: getParameterIf(isFeatureSelectionNode, 'percentile'),
        alpha: getParameterIf(isFeatureSelectionNode, 'alpha'),
        threshold: getParameterIf(isFeatureSelectionNode, 'threshold'),
        mode: getParameterIf(isFeatureSelectionNode, 'mode'),
        estimator: getParameterIf(isFeatureSelectionNode, 'estimator'),
        step: getParameterIf(isFeatureSelectionNode, 'step'),
        minFeatures: getParameterIf(isFeatureSelectionNode, 'min_features'),
        maxFeatures: getParameterIf(isFeatureSelectionNode, 'max_features'),
        dropUnselected: getParameterIf(isFeatureSelectionNode, 'drop_unselected'),
      },
      featureMath: {
        errorHandling: getParameterIf(isFeatureMathNode, 'error_handling'),
        allowOverwrite: getParameterIf(isFeatureMathNode, 'allow_overwrite'),
        defaultTimezone: getParameterIf(isFeatureMathNode, 'default_timezone'),
        epsilon: getParameterIf(isFeatureMathNode, 'epsilon'),
      },
      resampling: {
        method: getParameterIf(isClassResamplingNode, 'method'),
        targetColumn: getParameterIf(isClassResamplingNode, 'target_column'),
        samplingStrategy: getParameterIf(isClassResamplingNode, 'sampling_strategy'),
        randomState: getParameterIf(isClassResamplingNode, 'random_state'),
        kNeighbors: getParameterIf(isClassOversamplingNode, 'k_neighbors'),
        replacement: getParameterIf(isClassUndersamplingNode, 'replacement'),
      },
      featureTargetSplit: {
        targetColumn: getParameterIf(isFeatureTargetSplitNode, 'target_column'),
      },
      trainModel: {
        targetColumn: getParameterIf(hasTrainOrTuningNode, 'target_column'),
        problemType: getParameterIf(hasTrainOrTuningNode, 'problem_type'),
        modelType: getParameterIf(hasTrainOrTuningNode, 'model_type'),
        hyperparameters: getParameterIf(isTrainModelDraftNode, 'hyperparameters'),
        cvEnabled: getParameterIf(hasTrainOrTuningNode, 'cv_enabled'),
        cvStrategy: getParameterIf(hasTrainOrTuningNode, 'cv_strategy'),
        cvFolds: getParameterIf(hasTrainOrTuningNode, 'cv_folds'),
        cvShuffle: getParameterIf(hasTrainOrTuningNode, 'cv_shuffle'),
        cvRandomState: getParameterIf(hasTrainOrTuningNode, 'cv_random_state'),
        cvRefitStrategy: getParameterIf(isTrainModelDraftNode, 'cv_refit_strategy'),
      },
      hyperparameterTuning: {
        searchStrategy: getParameterIf(isHyperparameterTuningNode, 'search_strategy'),
        searchIterations: getParameterIf(isHyperparameterTuningNode, 'search_iterations'),
        searchRandomState: getParameterIf(isHyperparameterTuningNode, 'search_random_state'),
        scoringMetric: getParameterIf(isHyperparameterTuningNode, 'scoring_metric'),
      },
      targetEncoding: {
        columns: isTargetEncodingNode ? getParameter('columns') : null,
        targetColumn: isTargetEncodingNode ? getParameter('target_column') : null,
        autoDetect: isTargetEncodingNode ? getParameter('auto_detect') : null,
        maxCategories: isTargetEncodingNode ? getParameter('max_categories') : null,
        outputSuffix: isTargetEncodingNode ? getParameter('output_suffix') : null,
        dropOriginal: isTargetEncodingNode ? getParameter('drop_original') : null,
        smoothing: isTargetEncodingNode ? getParameter('smoothing') : null,
        encodeMissing: isTargetEncodingNode ? getParameter('encode_missing') : null,
        handleUnknown: isTargetEncodingNode ? getParameter('handle_unknown') : null,
      },
      ordinalEncoding: {
        columns: isOrdinalEncodingNode ? getParameter('columns') : null,
        autoDetect: isOrdinalEncodingNode ? getParameter('auto_detect') : null,
        maxCategories: isOrdinalEncodingNode ? getParameter('max_categories') : null,
        outputSuffix: isOrdinalEncodingNode ? getParameter('output_suffix') : null,
        dropOriginal: isOrdinalEncodingNode ? getParameter('drop_original') : null,
        encodeMissing: isOrdinalEncodingNode ? getParameter('encode_missing') : null,
        handleUnknown: isOrdinalEncodingNode ? getParameter('handle_unknown') : null,
        unknownValue: isOrdinalEncodingNode ? getParameter('unknown_value') : null,
      },
      dummyEncoding: {
        columns: isDummyEncodingNode ? getParameter('columns') : null,
        autoDetect: isDummyEncodingNode ? getParameter('auto_detect') : null,
        maxCategories: isDummyEncodingNode ? getParameter('max_categories') : null,
        dropFirst: isDummyEncodingNode ? getParameter('drop_first') : null,
        includeMissing: isDummyEncodingNode ? getParameter('include_missing') : null,
        dropOriginal: isDummyEncodingNode ? getParameter('drop_original') : null,
        prefixSeparator: isDummyEncodingNode ? getParameter('prefix_separator') : null,
      },
      oneHotEncoding: {
        columns: isOneHotEncodingNode ? getParameter('columns') : null,
        autoDetect: isOneHotEncodingNode ? getParameter('auto_detect') : null,
        maxCategories: isOneHotEncodingNode ? getParameter('max_categories') : null,
        dropFirst: isOneHotEncodingNode ? getParameter('drop_first') : null,
        includeMissing: isOneHotEncodingNode ? getParameter('include_missing') : null,
        dropOriginal: isOneHotEncodingNode ? getParameter('drop_original') : null,
        prefixSeparator: isOneHotEncodingNode ? getParameter('prefix_separator') : null,
      },
      missingIndicator: {
        columns: getParameterIf(isMissingIndicatorNode, 'columns'),
        suffix: getParameterIf(isMissingIndicatorNode, 'flag_suffix'),
      },
      dropRows: {
        any: getParameterIf(isDropMissingRowsNode, 'drop_if_any_missing'),
      },
      dropMissing: {
        threshold: getParameterIf(isDropMissingColumnsNode || isDropMissingRowsNode, 'missing_threshold'),
      },
    };
  }, [flags, getParameter]);
};
