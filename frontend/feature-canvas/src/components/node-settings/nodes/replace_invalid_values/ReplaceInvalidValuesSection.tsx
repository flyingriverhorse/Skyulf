import React from 'react';
import type { FeatureNodeParameter } from '../../../../api';
import {
  type InvalidValueColumnSummary,
  type InvalidValueMode,
  type InvalidValueModeDetails,
  type InvalidValueSampleMap,
  summarizeInvalidRecommendations,
  summarizeInvalidSamples,
  summarizeInvalidWarnings,
} from './replaceInvalidValuesSettings';

export type ReplaceInvalidValuesSectionProps = {
  sourceId?: string | null;
  hasReachableSource: boolean;
  columnsParameter: FeatureNodeParameter | null;
  modeParameter: FeatureNodeParameter | null;
  minValueParameter: FeatureNodeParameter | null;
  maxValueParameter: FeatureNodeParameter | null;
  renderMultiSelectField: (parameter: FeatureNodeParameter) => React.ReactNode;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  columnSummary: InvalidValueColumnSummary;
  modeDetails: InvalidValueModeDetails;
  sampleMap: InvalidValueSampleMap;
  selectedMode: InvalidValueMode;
  minValue: number | null;
  maxValue: number | null;
};

const formatBoundsDescription = (mode: InvalidValueMode, minValue: number | null, maxValue: number | null): string | null => {
  if (mode === 'custom_range') {
    if (minValue === null && maxValue === null) {
      return 'Specify a minimum and/or maximum to enable the custom range rule.';
    }
    const lower = minValue === null ? '-∞' : String(minValue);
    const upper = maxValue === null ? '∞' : String(maxValue);
    return `Custom bounds active: [${lower}, ${upper}].`;
  }
  if (mode === 'percentage_bounds') {
    const lower = minValue ?? 0;
    const upper = maxValue ?? 100;
    return `Using percentage bounds ${lower}–${upper}.`;
  }
  if (mode === 'age_bounds') {
    const lower = minValue ?? 0;
    const upper = maxValue ?? 120;
    return `Using age bounds ${lower}–${upper}.`;
  }
  if (mode === 'negative_to_nan' && (minValue !== null || maxValue !== null)) {
    const lower = minValue ?? 0;
    return maxValue === null
      ? `Negative-to-missing rule overrides values below ${lower}.`
      : `Negative-to-missing rule overrides values < ${lower} or > ${maxValue}.`;
  }
  return null;
};

export const ReplaceInvalidValuesSection: React.FC<ReplaceInvalidValuesSectionProps> = ({
  sourceId,
  hasReachableSource,
  columnsParameter,
  modeParameter,
  minValueParameter,
  maxValueParameter,
  renderMultiSelectField,
  renderParameterField,
  columnSummary,
  modeDetails,
  sampleMap,
  selectedMode,
  minValue,
  maxValue,
}) => {
  if (!columnsParameter && !modeParameter && !minValueParameter && !maxValueParameter) {
    return null;
  }

  const { selectedColumns, autoDetectionActive } = columnSummary;
  const selectedCount = selectedColumns.length;
  const recommendationSummary = summarizeInvalidRecommendations(columnSummary.recommendedColumns);
  const negativeSummary = summarizeInvalidWarnings(columnSummary.negativeCandidates);
  const zeroSummary = summarizeInvalidWarnings(columnSummary.zeroCandidates);
  const percentageSummary = summarizeInvalidWarnings(columnSummary.percentageOutliers);
  const ageSummary = summarizeInvalidWarnings(columnSummary.ageOutliers);
  const nonNumericSummary = summarizeInvalidWarnings(columnSummary.nonNumericSelected);
  const samplePreviews = summarizeInvalidSamples(sampleMap, selectedColumns, selectedMode, minValue, maxValue);
  const boundsDescription = formatBoundsDescription(selectedMode, minValue, maxValue);
  const hasDatasetContextWarning = !sourceId || !hasReachableSource;

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Replace invalid numeric values</h3>
      </div>
      <p className="canvas-modal__note">
        <strong>{modeDetails.label}</strong> rule. {modeDetails.guidance} Example: {modeDetails.example}.
      </p>
      {boundsDescription && <p className="canvas-modal__note">{boundsDescription}</p>}
      {autoDetectionActive ? (
        <p className="canvas-modal__note">
          No columns selected&mdash;numeric columns will be auto-detected when the node runs.
        </p>
      ) : (
        <p className="canvas-modal__note">
          Targeting <strong>{selectedCount}</strong> column{selectedCount === 1 ? '' : 's'} for invalid value cleanup.
        </p>
      )}
      {samplePreviews.length > 0 && (
        <p className="canvas-modal__note">
          Samples: {samplePreviews.map((preview) => `${preview.column} (${preview.values.join('; ')})`).join(' | ')}.
        </p>
      )}
      {negativeSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Columns with negative values: {negativeSummary.preview.join(', ')}
          {negativeSummary.remaining > 0 ? `, ... (${negativeSummary.remaining} more)` : ''}.
        </p>
      )}
      {zeroSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Columns with placeholder zeros: {zeroSummary.preview.join(', ')}
          {zeroSummary.remaining > 0 ? `, ... (${zeroSummary.remaining} more)` : ''}.
        </p>
      )}
      {percentageSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Percentage candidates with values outside 0-100: {percentageSummary.preview.join(', ')}
          {percentageSummary.remaining > 0 ? `, ... (${percentageSummary.remaining} more)` : ''}.
        </p>
      )}
      {ageSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Age-like values outside 0-120: {ageSummary.preview.join(', ')}
          {ageSummary.remaining > 0 ? `, ... (${ageSummary.remaining} more)` : ''}.
        </p>
      )}
      {recommendationSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Other numeric columns: {recommendationSummary.preview.join(', ')}
          {recommendationSummary.remaining > 0 ? `, ... (${recommendationSummary.remaining} more)` : ''}.
        </p>
      )}
      {nonNumericSummary.preview.length > 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          {nonNumericSummary.preview.join(', ')} {nonNumericSummary.preview.length === 1 ? 'is' : 'are'} not numeric-type columns.
          {nonNumericSummary.remaining > 0
            ? ` ${nonNumericSummary.remaining} more column${nonNumericSummary.remaining === 1 ? ' is' : 's are'} also non-numeric.`
            : ' Verify the selection or adjust the rule.'}
        </p>
      )}
      {hasDatasetContextWarning ? (
        !sourceId ? (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Select a dataset to inspect numeric samples before applying replacements.
          </p>
        ) : (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Connect this node to an upstream output to review numeric insights.
          </p>
        )
      ) : null}
      {columnsParameter && (
        <div className="canvas-modal__subsection">
          <h4>Columns to clean</h4>
          {renderMultiSelectField(columnsParameter)}
        </div>
      )}
      {(modeParameter || minValueParameter || maxValueParameter) && (
        <div className="canvas-modal__parameter-list">
          {modeParameter ? renderParameterField(modeParameter) : null}
          {minValueParameter ? renderParameterField(minValueParameter) : null}
          {maxValueParameter ? renderParameterField(maxValueParameter) : null}
        </div>
      )}
    </section>
  );
};
