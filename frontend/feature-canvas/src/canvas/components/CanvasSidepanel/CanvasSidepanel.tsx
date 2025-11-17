import React from 'react';
import type { DatasetSourceSummary, FeaturePipelineResponse } from '../../../api';
import type { SaveFeedback } from '../../types/feedback';
import { formatRelativeTime, formatTimestamp } from '../../utils/time';

export type DatasetOption = {
  value: string;
  label: string;
  isOwned: boolean;
};

export type CanvasSidepanelProps = {
  activePipelineUpdatedAt: string | null;
  datasetErrorMessage: string | null;
  isDatasetLoading: boolean;
  hasDatasets: boolean;
  ownedDatasetsCount: number;
  canSelectDatasets: boolean;
  datasetOptions: DatasetOption[];
  selectedDataset: DatasetSourceSummary | null;
  activeSourceId: string | null;
  canClearCanvas: boolean;
  onDatasetSelection: (value: string) => void;
  onClearCanvas: () => void;
  onRefreshHistory: () => void;
  onSaveClick: () => void;
  isHistoryLoading: boolean;
  historyErrorMessage: string | null;
  historyItems: FeaturePipelineResponse[];
  activePipelineId: number | null;
  isDirty: boolean;
  isSaveDisabled: boolean;
  saveButtonLabel: string;
  saveFeedback: SaveFeedback | null;
  feedbackIcon: string;
  feedbackClass: string;
  onSelectHistory: (item: FeaturePipelineResponse) => void;
};

export const CanvasSidepanel: React.FC<CanvasSidepanelProps> = ({
  activePipelineUpdatedAt,
  datasetErrorMessage,
  isDatasetLoading,
  hasDatasets,
  ownedDatasetsCount,
  canSelectDatasets,
  datasetOptions,
  selectedDataset,
  activeSourceId,
  canClearCanvas,
  onDatasetSelection,
  onClearCanvas,
  onRefreshHistory,
  onSaveClick,
  isHistoryLoading,
  historyErrorMessage,
  historyItems,
  activePipelineId,
  isDirty,
  isSaveDisabled,
  saveButtonLabel,
  saveFeedback,
  feedbackIcon,
  feedbackClass,
  onSelectHistory,
}) => {
  const datasetMetaLabel = activeSourceId ? selectedDataset?.name ?? activeSourceId : null;

  return (
    <aside
      id="feature-canvas-sidepanel"
      className="canvas-sidepanel"
      role="complementary"
      aria-label="Canvas details"
    >
      <div className="canvas-sidepanel__content" role="group" aria-label="Canvas controls">
        <section className="canvas-sidepanel__section">
          <div className="canvas-sidepanel__section-heading">
            <h2>Dataset</h2>
            {activePipelineUpdatedAt && (
              <span className="canvas-sidepanel__meta">
                {formatRelativeTime(activePipelineUpdatedAt) ?? formatTimestamp(activePipelineUpdatedAt)}
              </span>
            )}
          </div>
          {datasetErrorMessage ? (
            <p className="text-danger">{datasetErrorMessage}</p>
          ) : ownedDatasetsCount ? (
            <div className="canvas-sidepanel__select-wrapper">
              <select
                className="canvas-sidepanel__select"
                value={activeSourceId ?? ''}
                onChange={(event) => onDatasetSelection(event.target.value)}
                disabled={isDatasetLoading || !canSelectDatasets}
                aria-label="Select dataset"
              >
                {datasetOptions.map((option) => (
                  <option key={option.value} value={option.value} disabled={!option.isOwned}>
                    {`${option.label}${option.isOwned ? '' : ' (locked)'}`}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <p className="text-muted">Dataset switching is limited to collections you own.</p>
          )}
          {isDatasetLoading && <p className="text-muted">Loading datasets…</p>}
          {!isDatasetLoading && !hasDatasets && !datasetErrorMessage && (
            <p className="text-muted">No datasets available. Send one from EDA to get started.</p>
          )}
          {selectedDataset?.description && (
            <p className="canvas-sidepanel__description">{selectedDataset.description}</p>
          )}
          {datasetMetaLabel && (
            <p className="canvas-sidepanel__meta">
              Working with <strong>{datasetMetaLabel}</strong>
            </p>
          )}
          {!canSelectDatasets && ownedDatasetsCount > 0 && (
            <p className="canvas-sidepanel__hint">Add more owned datasets to switch between them.</p>
          )}
        </section>

        <section className="canvas-sidepanel__section">
          <div className="canvas-sidepanel__section-action">
            <button
              type="button"
              className="canvas-sidepanel__action-button"
              onClick={onClearCanvas}
              disabled={!canClearCanvas}
              aria-label="Clean all nodes"
              title={canClearCanvas ? 'Remove all nodes and edges from the canvas' : 'Nothing to clean yet'}
            >
              <span aria-hidden="true">🧹</span>
              <span>Clean all nodes</span>
            </button>
          </div>
          <div className="canvas-sidepanel__section-heading">
            <h2>History</h2>
            <div className="canvas-sidepanel__section-controls">
              <button
                type="button"
                className="canvas-sidepanel__icon-button"
                onClick={onRefreshHistory}
                disabled={isHistoryLoading || !activeSourceId}
                aria-label="Refresh history"
                title="Refresh history"
              >
                ⟳
              </button>
              <button
                type="button"
                className="canvas-sidepanel__icon-button"
                onClick={onSaveClick}
                disabled={isSaveDisabled}
                aria-label={saveButtonLabel}
                title={saveButtonLabel}
              >
                💾
              </button>
            </div>
          </div>
          <div className="canvas-sidepanel__history">
            {isHistoryLoading ? (
              <p className="text-muted">Loading history…</p>
            ) : historyErrorMessage ? (
              <p className="text-danger">{historyErrorMessage}</p>
            ) : historyItems.length ? (
              historyItems.map((item) => {
                const relative = formatRelativeTime(item.updated_at);
                const timestampLabel = relative ?? formatTimestamp(item.updated_at);
                const matchesActive = item.id === activePipelineId;
                const disableSelection = matchesActive && !isDirty;

                return (
                  <button
                    key={item.id}
                    type="button"
                    className={`canvas-sidepanel__history-item${
                      matchesActive ? ' canvas-sidepanel__history-item--active' : ''
                    }`}
                    onClick={() => onSelectHistory(item)}
                    disabled={disableSelection}
                  >
                    <span className="canvas-sidepanel__history-title">
                      {item.name || `Pipeline #${item.id}`}
                    </span>
                    <span className="canvas-sidepanel__history-meta">{timestampLabel}</span>
                    {matchesActive && isDirty && (
                      <span className="canvas-sidepanel__history-badge">Unsaved edits</span>
                    )}
                  </button>
                );
              })
            ) : (
              <p className="text-muted">No saved revisions yet. Save a draft to populate history.</p>
            )}
          </div>
        </section>
      </div>

      {saveFeedback && (
        <p className={`canvas-sidepanel__feedback ${feedbackClass}`}>
          <span className="canvas-sidepanel__feedback-icon" aria-hidden="true">{feedbackIcon}</span>
          <span className="canvas-sidepanel__feedback-text">{saveFeedback.message}</span>
        </p>
      )}
    </aside>
  );
};
