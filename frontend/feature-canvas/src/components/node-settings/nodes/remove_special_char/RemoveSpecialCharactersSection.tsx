import React from 'react';
import type { FeatureNodeParameter } from '../../../../api';
import {
  type SpecialColumnSummary,
  type SpecialSampleMap,
  type RemoveSpecialModeDetails,
  summarizeSpecialRecommendations,
  summarizeSpecialSamples,
  summarizeSpecialWarnings,
} from './removeSpecialCharactersSettings';

export type RemoveSpecialCharactersSectionProps = {
  sourceId?: string | null;
  hasReachableSource: boolean;
  columnsParameter: FeatureNodeParameter | null;
  modeParameter: FeatureNodeParameter | null;
  replacementParameter: FeatureNodeParameter | null;
  renderMultiSelectField: (parameter: FeatureNodeParameter) => React.ReactNode;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  columnSummary: SpecialColumnSummary;
  modeDetails: RemoveSpecialModeDetails;
  sampleMap: SpecialSampleMap;
};

export const RemoveSpecialCharactersSection: React.FC<RemoveSpecialCharactersSectionProps> = ({
  sourceId,
  hasReachableSource,
  columnsParameter,
  modeParameter,
  replacementParameter,
  renderMultiSelectField,
  renderParameterField,
  columnSummary,
  modeDetails,
  sampleMap,
}) => {
  if (!columnsParameter && !modeParameter && !replacementParameter) {
    return null;
  }

  const selectedCount = columnSummary.selectedColumns.length;
  const recommendationSummary = summarizeSpecialRecommendations(columnSummary.recommendedColumns);
  const specialSummary = summarizeSpecialWarnings(columnSummary.specialCandidates);
  const digitsOnlySummary = summarizeSpecialWarnings(columnSummary.digitsOnlyCandidates);
  const lettersOnlySummary = summarizeSpecialWarnings(columnSummary.lettersOnlyCandidates);
  const nonTextSummary = summarizeSpecialWarnings(columnSummary.nonTextSelected);
  const samplePreviews = summarizeSpecialSamples(sampleMap, columnSummary.selectedColumns);
  const hasDatasetContextWarning = !sourceId || !hasReachableSource;

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Special character cleanup</h3>
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
          Targeting <strong>{selectedCount}</strong> column{selectedCount === 1 ? '' : 's'} for cleanup.
        </p>
      )}
      {samplePreviews.length > 0 && (
        <p className="canvas-modal__note">
          Samples: {samplePreviews.map((preview) => `${preview.column} (${preview.values.join(', ')})`).join('; ')}.
        </p>
      )}
      {specialSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Columns containing punctuation or symbols: {specialSummary.preview.join(', ')}
          {specialSummary.remaining > 0 ? `, ... (${specialSummary.remaining} more)` : ''}.
        </p>
      )}
      {digitsOnlySummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Numeric-only samples detected in: {digitsOnlySummary.preview.join(', ')}
          {digitsOnlySummary.remaining > 0 ? `, ... (${digitsOnlySummary.remaining} more)` : ''}. Consider `digits only` mode.
        </p>
      )}
      {lettersOnlySummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Letter-only samples detected in: {lettersOnlySummary.preview.join(', ')}
          {lettersOnlySummary.remaining > 0 ? `, ... (${lettersOnlySummary.remaining} more)` : ''}. Consider `letters only` mode.
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
            : ' Double-check that symbol stripping is necessary for these fields.'}
        </p>
      )}
      {hasDatasetContextWarning ? (
        !sourceId ? (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Select a dataset to inspect column samples before removing characters.
          </p>
        ) : (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Connect this node to an upstream output to review column insights.
          </p>
        )
      ) : null}
      {columnsParameter && (
        <div className="canvas-modal__subsection">
          <h4>Columns to sanitize</h4>
          {renderMultiSelectField(columnsParameter)}
        </div>
      )}
      {(modeParameter || replacementParameter) && (
        <div className="canvas-modal__parameter-list">
          {modeParameter ? renderParameterField(modeParameter) : null}
          {replacementParameter ? renderParameterField(replacementParameter) : null}
        </div>
      )}
    </section>
  );
};
