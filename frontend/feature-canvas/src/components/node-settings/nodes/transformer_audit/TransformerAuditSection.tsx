import React, { useMemo } from 'react';
import type { TransformerAuditNodeSignal, TransformerAuditEntrySignal, TransformerSplitActivitySignal } from '../../../../api';

type TransformerAuditSectionProps = {
  signal: TransformerAuditNodeSignal | null;
  previewStatus: 'idle' | 'loading' | 'success' | 'error';
  hasSource: boolean;
  hasReachableSource: boolean;
  onRefreshPreview: () => void;
};

const ACTION_LABELS: Record<TransformerSplitActivitySignal['action'], string> = {
  fit_transform: 'Fit + transform',
  transform: 'Transform',
  not_applied: 'Not applied',
  not_available: '—',
};

const formatTimestamp = (value?: string | null): string => {
  if (!value) {
    return '—';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
};

const formatRowCount = (value?: number | null): string | null => {
  if (value === null || value === undefined || Number.isNaN(value) || value <= 0) {
    return null;
  }
  const rounded = Math.round(value);
  return `${rounded.toLocaleString()} row${rounded === 1 ? '' : 's'}`;
};

const getSplitActivity = (
  entry: TransformerAuditEntrySignal,
  split: 'train' | 'test' | 'validation',
): TransformerSplitActivitySignal | null => {
  if (!Array.isArray(entry.split_activity)) {
    return null;
  }
  const match = entry.split_activity.find(
    (activity: TransformerSplitActivitySignal) => activity && activity.split === split,
  );
  return match ?? null;
};

const renderSplitCell = (entry: TransformerAuditEntrySignal, split: 'train' | 'test' | 'validation') => {
  const activity = getSplitActivity(entry, split);
  if (!activity || activity.action === 'not_available') {
    return <span>—</span>;
  }
  const label = ACTION_LABELS[activity.action] ?? activity.action;
  const rowSummary = formatRowCount(activity.rows ?? null);

  return (
    <div className="canvas-cast__column-cell">
      <span>{label}</span>
      {rowSummary && <span className="canvas-cast__muted">{rowSummary}</span>}
    </div>
  );
};

const extraSplitNote = (entry: TransformerAuditEntrySignal): string | null => {
  if (!Array.isArray(entry.split_activity)) {
    return null;
  }
  const extras = entry.split_activity.filter(
    (activity: TransformerSplitActivitySignal) =>
      activity && !['train', 'test', 'validation'].includes(activity.split),
  );
  if (!extras.length) {
    return null;
  }
  const formatted = extras
    .map((activity: TransformerSplitActivitySignal) => {
      const label = activity.label || activity.split;
      const action = ACTION_LABELS[activity.action] ?? activity.action;
      const count = formatRowCount(activity.rows ?? null);
      return count ? `${label}: ${action} (${count})` : `${label}: ${action}`;
    })
    .join('; ');
  return formatted || null;
};

const getMethodLabel = (entry: TransformerAuditEntrySignal): string | null => {
  if (!entry.metadata || typeof entry.metadata !== 'object') {
    return null;
  }
  // For scalers: method_label or method
  if (entry.metadata.method_label) {
    return String(entry.metadata.method_label);
  }
  if (entry.metadata.method) {
    return String(entry.metadata.method);
  }
  return null;
};

export const TransformerAuditSection: React.FC<TransformerAuditSectionProps> = ({
  signal,
  previewStatus,
  hasSource,
  hasReachableSource,
  onRefreshPreview,
}) => {
  const entries = useMemo<TransformerAuditEntrySignal[]>(() => signal?.transformers ?? [], [signal?.transformers]);
  const totalTransformers = signal?.total_transformers ?? entries.length;
  const notes: string[] = Array.isArray(signal?.notes) ? (signal?.notes as string[]) : [];
  const isLoading = previewStatus === 'loading';
  const summaryText = totalTransformers
    ? `Tracking ${totalTransformers.toLocaleString()} transformer${totalTransformers === 1 ? '' : 's'}.`
    : null;
  const disableRefresh = isLoading || !hasSource || !hasReachableSource;

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <h3>Transformer activity</h3>
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={onRefreshPreview}
            disabled={disableRefresh}
          >
            {isLoading ? 'Refreshing…' : 'Refresh preview'}
          </button>
        </div>
      </div>
      <p className="canvas-modal__note">
        Review whether split-aware transformers (e.g. one-hot encoders) fit on training data and propagated to
        test/validation splits.
      </p>
      {summaryText && <p className="canvas-modal__note">{summaryText}</p>}
      {!hasSource && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Select a dataset to surface transformer activity.
        </p>
      )}
      {hasSource && !hasReachableSource && (
        <p className="canvas-modal__note canvas-modal__note--warning">
          Connect this node to an upstream branch to inspect transformer history.
        </p>
      )}
      {notes.length > 0 && (
        <ul className="canvas-modal__note-list">
          {notes.map((note: string, index: number) => (
            <li key={`transformer-audit-note-${index}`}>{note}</li>
          ))}
        </ul>
      )}
      {isLoading && <p className="canvas-modal__note">Loading transformer activity…</p>}
      {!isLoading && previewStatus === 'error' && hasSource && hasReachableSource && (
        <p className="canvas-modal__note canvas-modal__note--error">
          Unable to load transformer activity. Try refreshing the preview.
        </p>
      )}
      {!isLoading && hasSource && hasReachableSource && !entries.length && (
        <p className="canvas-modal__note">
          No transformer activity recorded yet. Run a preview after executing split-aware transformation nodes.
        </p>
      )}
      {!isLoading && entries.length > 0 && (
        <div className="canvas-cast__table-wrapper">
          <table className="canvas-cast__table">
            <thead>
              <tr>
                <th scope="col">Node</th>
                <th scope="col">Transformer</th>
                <th scope="col">Column</th>
                <th scope="col">Method</th>
                <th scope="col">Train</th>
                <th scope="col">Test</th>
                <th scope="col">Validation</th>
                <th scope="col">Updated</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((entry, index) => {
                const rowKey = entry.storage_key ?? `${entry.transformer_name}-${entry.column_name ?? 'all'}-${index}`;
                const extraNote = extraSplitNote(entry);
                const nodeLabel = entry.source_node_label || entry.source_node_id || '—';
                const updatedDisplay = formatTimestamp(entry.updated_at ?? entry.created_at ?? null);
                const columnDisplay = entry.column_name ?? '—';
                const methodLabel = getMethodLabel(entry);

                return (
                  <React.Fragment key={rowKey}>
                    <tr className="canvas-cast__row">
                      <th scope="row">
                        <div className="canvas-cast__column-cell">
                          <span className="canvas-cast__column-name">{nodeLabel}</span>
                        </div>
                      </th>
                      <td>{entry.transformer_name}</td>
                      <td>{columnDisplay}</td>
                      <td>{methodLabel || '—'}</td>
                      <td>{renderSplitCell(entry, 'train')}</td>
                      <td>{renderSplitCell(entry, 'test')}</td>
                      <td>{renderSplitCell(entry, 'validation')}</td>
                      <td>{updatedDisplay}</td>
                    </tr>
                    {extraNote && (
                      <tr className="canvas-cast__row canvas-cast__row--subtle">
                        <td colSpan={8}>
                          <span className="canvas-cast__muted">Additional splits: {extraNote}</span>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
};
