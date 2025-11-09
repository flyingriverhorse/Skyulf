import React from 'react';
import type { FeatureNodeParameter } from '../../../../api';
import {
  type TrimColumnSummary,
  type TrimModeDetails,
  type TrimSampleMap,
  summarizeNonTextSelections,
  summarizeTrimRecommendations,
  summarizeTrimSamples,
  summarizeWhitespaceCandidates,
} from './trimWhitespaceSettings';

export type TrimWhitespaceSectionProps = {
  sourceId?: string | null;
  hasReachableSource: boolean;
  columnsParameter: FeatureNodeParameter | null;
  modeParameter: FeatureNodeParameter | null;
  renderMultiSelectField: (parameter: FeatureNodeParameter) => React.ReactNode;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  columnSummary: TrimColumnSummary;
  modeDetails: TrimModeDetails;
  sampleMap: TrimSampleMap;
};

export const TrimWhitespaceSection: React.FC<TrimWhitespaceSectionProps> = ({
  sourceId,
  hasReachableSource,
  columnsParameter,
  modeParameter,
  renderMultiSelectField,
  renderParameterField,
  columnSummary,
  modeDetails,
  sampleMap,
}) => {
  if (!columnsParameter && !modeParameter) {
    return null;
  }

  const selectedCount = columnSummary.selectedColumns.length;
  const recommendationSummary = summarizeTrimRecommendations(columnSummary.recommendedColumns);
  const nonTextSummary = summarizeNonTextSelections(columnSummary.nonTextSelected);
  const whitespaceSummary = summarizeWhitespaceCandidates(columnSummary.whitespaceCandidates);
  const samplePreviews = summarizeTrimSamples(sampleMap, columnSummary.selectedColumns);
  const hasDatasetContextWarning = !sourceId || !hasReachableSource;

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Whitespace trimming</h3>
      </div>
      <p className="canvas-modal__note">
        <strong>{modeDetails.label}</strong> mode. {modeDetails.guidance} Example: {modeDetails.example}.
      </p>
      {columnSummary.autoDetectionActive ? (
        <p className="canvas-modal__note">
          No columns selected&mdash;text columns will be auto-detected when the node runs.
        </p>
      ) : (
        <p className="canvas-modal__note">
          Targeting <strong>{selectedCount}</strong> column{selectedCount === 1 ? '' : 's'} for trimming.
        </p>
      )}
      {samplePreviews.length > 0 && (
        <p className="canvas-modal__note">
          Samples: {samplePreviews.map((preview) => `${preview.column} (${preview.values.join(', ')})`).join('; ')}.
        </p>
      )}
      {whitespaceSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Columns with leading/trailing whitespace detected: {whitespaceSummary.preview.join(', ')}
          {whitespaceSummary.remaining > 0 ? `, ... (${whitespaceSummary.remaining} more)` : ''}.
        </p>
      )}
      {recommendationSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Other text columns: {recommendationSummary.preview.join(', ')}
          {recommendationSummary.remaining > 0 ? `, ... (${recommendationSummary.remaining} more)` : ''}.
        </p>
      )}
      {nonTextSummary.preview.length > 0 && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          {nonTextSummary.preview.join(', ')} {nonTextSummary.preview.length === 1 ? 'is' : 'are'} not text-like.
          {nonTextSummary.remaining > 0
            ? ` ${nonTextSummary.remaining} more column${nonTextSummary.remaining === 1 ? ' is' : 's are'} also non-text.`
            : ' Consider removing them or confirming whitespace cleanup is intentional.'}
        </p>
      )}
      {hasDatasetContextWarning ? (
        !sourceId ? (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Select a dataset to inspect column samples for trimming opportunities.
          </p>
        ) : (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Connect this node to an upstream output to review column insights.
          </p>
        )
      ) : null}
      {columnsParameter && (
        <div className="canvas-modal__subsection">
          <h4>Columns to trim</h4>
          {renderMultiSelectField(columnsParameter)}
        </div>
      )}
      {modeParameter && (
        <div className="canvas-modal__parameter-list">{renderParameterField(modeParameter)}</div>
      )}
    </section>
  );
};
