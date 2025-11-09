import React from 'react';
import type { FeatureNodeParameter } from '../../../../api';
import {
  type RegexColumnSummary,
  type RegexModeDetails,
  type RegexSampleMap,
  RegexMode,
  summarizeRegexRecommendations,
  summarizeRegexSamples,
  summarizeRegexWarnings,
} from './regexCleanupSettings';

export type RegexCleanupSectionProps = {
  sourceId?: string | null;
  hasReachableSource: boolean;
  columnsParameter: FeatureNodeParameter | null;
  modeParameter: FeatureNodeParameter | null;
  patternParameter: FeatureNodeParameter | null;
  replacementParameter: FeatureNodeParameter | null;
  renderMultiSelectField: (parameter: FeatureNodeParameter) => React.ReactNode;
  renderParameterField: (parameter: FeatureNodeParameter) => React.ReactNode;
  columnSummary: RegexColumnSummary;
  modeDetails: RegexModeDetails;
  sampleMap: RegexSampleMap;
  selectedMode: RegexMode;
  replacementValue: string | null;
};

const describeReplacementValue = (value: string | null): string | null => {
  if (!value) {
    return null;
  }
  const parts = value.split('|||');
  const pattern = parts[0] ?? '';
  const flags = parts[1] ?? '';
  const replacement = parts[2] ?? '';
  if (!pattern.trim() && !replacement.trim()) {
    return null;
  }
  return `Pattern: /${pattern || '(empty)'}/${flags || '(none)'} Replacement: ${replacement || "''"}`;
};

export const RegexCleanupSection: React.FC<RegexCleanupSectionProps> = ({
  sourceId,
  hasReachableSource,
  columnsParameter,
  modeParameter,
  patternParameter,
  replacementParameter,
  renderMultiSelectField,
  renderParameterField,
  columnSummary,
  modeDetails,
  sampleMap,
  selectedMode,
  replacementValue,
}) => {
  if (!columnsParameter && !modeParameter && !patternParameter && !replacementParameter) {
    return null;
  }

  const selectedCount = columnSummary.selectedColumns.length;
  const recommendationSummary = summarizeRegexRecommendations(columnSummary.recommendedColumns);
  const digitsSummary = summarizeRegexWarnings(columnSummary.digitsCandidates);
  const lettersSummary = summarizeRegexWarnings(columnSummary.lettersCandidates);
  const whitespaceSummary = summarizeRegexWarnings(columnSummary.whitespaceCandidates);
  const nonTextSummary = summarizeRegexWarnings(columnSummary.nonTextSelected);
  const replacementDescription = describeReplacementValue(replacementValue);
  const samplePreviews = summarizeRegexSamples(sampleMap, columnSummary.selectedColumns, selectedMode, replacementValue);
  const hasDatasetContextWarning = !sourceId || !hasReachableSource;
  const isCustomMode = selectedMode === 'custom';

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Regex cleanup</h3>
      </div>
      <p className="canvas-modal__note">
        <strong>{modeDetails.label}</strong> mode. {modeDetails.guidance} Example: {modeDetails.example}.
      </p>
      {replacementDescription && (
        <p className="canvas-modal__note">{replacementDescription}</p>
      )}
      {columnSummary.autoDetectionActive ? (
        <p className="canvas-modal__note">
          No columns selected&mdash;text columns will be auto-detected when the node runs.
        </p>
      ) : (
        <p className="canvas-modal__note">
          Targeting <strong>{selectedCount}</strong> column{selectedCount === 1 ? '' : 's'} for regex cleanup.
        </p>
      )}
      {samplePreviews.length > 0 && (
        <p className="canvas-modal__note">
          Samples: {samplePreviews.map((preview) => `${preview.column} (${preview.values.join('; ')})`).join(' | ')}.
        </p>
      )}
      {digitsSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Columns containing digits: {digitsSummary.preview.join(', ')}
          {digitsSummary.remaining > 0 ? `, ... (${digitsSummary.remaining} more)` : ''}.
        </p>
      )}
      {lettersSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Columns containing letters: {lettersSummary.preview.join(', ')}
          {lettersSummary.remaining > 0 ? `, ... (${lettersSummary.remaining} more)` : ''}.
        </p>
      )}
      {whitespaceSummary.preview.length > 0 && (
        <p className="canvas-modal__note">
          Columns with repeated whitespace: {whitespaceSummary.preview.join(', ')}
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
            : ' Confirm regex cleanup is appropriate for these values.'}
        </p>
      )}
      {isCustomMode && !replacementDescription && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Custom regex mode is enabled, but no pattern or replacement details were detected. Provide both to avoid no-ops.
        </p>
      )}
      {hasDatasetContextWarning ? (
        !sourceId ? (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Select a dataset to inspect column samples before applying regex transformations.
          </p>
        ) : (
          <p className="canvas-modal__note canvas-modal__note--warning">
            Connect this node to an upstream output to review column insights.
          </p>
        )
      ) : null}
      {columnsParameter && (
        <div className="canvas-modal__subsection">
          <h4>Columns to clean</h4>
          {renderMultiSelectField(columnsParameter)}
        </div>
      )}
      {(modeParameter || patternParameter || replacementParameter) && (
        <div className="canvas-modal__parameter-list">
          {modeParameter ? renderParameterField(modeParameter) : null}
          {patternParameter ? renderParameterField(patternParameter) : null}
          {replacementParameter ? renderParameterField(replacementParameter) : null}
        </div>
      )}
    </section>
  );
};
