import React from 'react';
import type { FeatureNodeParameter } from '../../../../api';
import {
  type CaseColumnSummary,
  type CaseModeDetails,
  type CaseSampleMap,
  CaseMode,
  summarizeCaseRecommendations,
  summarizeCaseSamples,
  summarizeCaseWarnings,
} from './normalizeTextCaseSettings';

export type NormalizeTextCaseSectionProps = {
  sourceId?: string | null;
  hasReachableSource: boolean;
  columnsParameter: FeatureNodeParameter | null;
  modeParameter: FeatureNodeParameter | null;
  renderMultiSelectField: (parameter: FeatureNodeParameter) => React.ReactNode;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  columnSummary: CaseColumnSummary;
  modeDetails: CaseModeDetails;
  sampleMap: CaseSampleMap;
  selectedMode: CaseMode;
};

export const NormalizeTextCaseSection: React.FC<NormalizeTextCaseSectionProps> = ({
  sourceId,
  hasReachableSource,
  columnsParameter,
  modeParameter,
  renderMultiSelectField,
  renderParameterField,
  columnSummary,
  modeDetails,
  sampleMap,
  selectedMode,
}) => {
  if (!columnsParameter && !modeParameter) {
    return null;
  }

  const selectedCount = columnSummary.selectedColumns.length;
  const recommendationSummary = summarizeCaseRecommendations(columnSummary.recommendedColumns);
  const inconsistentSummary = summarizeCaseWarnings(columnSummary.inconsistentColumns);
  const nonTextSummary = summarizeCaseWarnings(columnSummary.nonTextSelected);
  const samplePreviews = summarizeCaseSamples(sampleMap, columnSummary.selectedColumns, selectedMode);
  const hasDatasetContextWarning = !sourceId || !hasReachableSource;

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Text case normalization</h3>
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
          Targeting <strong>{selectedCount}</strong> column{selectedCount === 1 ? '' : 's'} for case normalization.
        </p>
      )}
      {samplePreviews.length > 0 && (
        <p className="canvas-modal__note">
          Samples: {samplePreviews.map((preview) => `${preview.column} (${preview.values.join('; ')})`).join(' | ')}.
        </p>
      )}
      {inconsistentSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Columns with mixed casing: {inconsistentSummary.preview.join(', ')}
          {inconsistentSummary.remaining > 0 ? `, ... (${inconsistentSummary.remaining} more)` : ''}.
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
            : ' Confirm case normalization is necessary for these values.'}
        </p>
      )}
      {hasDatasetContextWarning ? (
        !sourceId ? (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Select a dataset to inspect column samples before adjusting case.
          </p>
        ) : (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Connect this node to an upstream output to review column insights.
          </p>
        )
      ) : null}
      {columnsParameter && (
        <div className="canvas-modal__subsection">
          <h4>Columns to normalize</h4>
          {renderMultiSelectField(columnsParameter)}
        </div>
      )}
      {modeParameter && (
        <div className="canvas-modal__parameter-list">{renderParameterField(modeParameter)}</div>
      )}
    </section>
  );
};
