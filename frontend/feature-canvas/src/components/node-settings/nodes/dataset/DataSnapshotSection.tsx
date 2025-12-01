import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import {
  fetchPipelinePreviewRows,
  type PipelinePreviewResponse,
  type PipelinePreviewMetrics,
  type PipelinePreviewColumnStat,
  type PipelinePreviewRowStat,
  type FullExecutionSignal,
} from '../../../../api';

export type PreviewState = {
  status: 'idle' | 'loading' | 'success' | 'error';
  data: PipelinePreviewResponse | null;
  error: string | null;
};

type PreviewPanelProps = {
  preview: PipelinePreviewResponse;
  columns: string[];
  rows: Record<string, any>[];
  tableNote?: string | null;
  formatCellValue: (value: unknown) => string;
  formatMetricValue: (value?: number | null, precision?: number) => string;
  formatMissingPercentage: (value?: number | null) => string;
  formatNumericStat: (value?: number | null) => string;
  formatModeStat: (value?: string | number | null) => string;
  fullExecutionStepLabel?: string | null;
  fullExecutionWarnings?: string[];
};

type RowWindowStatus = 'idle' | 'loading' | 'ready' | 'error';

type RowWindowState = {
  status: RowWindowStatus;
  columns: string[];
  rows: Record<string, any>[];
  nextOffset: number | null;
  hasMore: boolean;
  totalRows: number | null;
  samplingAdjustments: string[];
  largeDataset: boolean;
  error: string | null;
};

const WINDOW_LIMIT = 100;
const ROW_STATS_DISPLAY_LIMIT = 12;

const SAMPLING_ADJUSTMENT_MESSAGES: Record<string, string> = {
  limited_full_sample_for_large_dataset: 'Preview limited because the dataset is large.',
  capped_sample_for_large_dataset: 'Requested sample size capped for a large dataset.',
  switched_to_first_last_for_large_dataset: 'Using first and last rows to summarize the large dataset.',
  default_offset: 'Offset fallback applied; starting from the beginning.',
  normalized_negative_offset: 'Negative offset normalized to zero.',
  default_limit: 'Using the default preview window size.',
  normalized_non_positive_limit: 'Requested preview size too small; using the minimum window.',
  capped_limit_for_sample: 'Preview window capped to stay within safe limits.',
};

type FullExecutionIndicatorTone =
  | 'succeeded'
  | 'deferred'
  | 'queued'
  | 'running'
  | 'failed'
  | 'skipped'
  | 'cancelled';

type FullExecutionSummary = {
  label: string;
  tone: FullExecutionIndicatorTone;
  details: string[];
  metadata: Array<{ label: string; value: string }>;
  warnings: string[];
  stepLabel: string | null;
  isActive: boolean;
};

const FULL_EXECUTION_STATUS_LABELS: Record<string, string> = {
  succeeded: 'Completed',
  deferred: 'Deferred',
  queued: 'Queued',
  running: 'In progress',
  failed: 'Failed',
  skipped: 'Skipped',
  cancelled: 'Cancelled',
};

const FULL_EXECUTION_WARNING_MESSAGES: Record<string, string> = {
  dataset_too_large: 'Full dataset run is executing in the background for large datasets.',
  memory_error: 'Full dataset execution failed because the server ran out of memory.',
  background_failure: 'Background job ended unexpectedly. Check server logs for details.',
};

const FULL_EXECUTION_JOB_STATUS_LABELS: Record<string, string> = {
  queued: 'Queued',
  running: 'Running',
  succeeded: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

const normaliseFullExecutionTone = (value: string | null | undefined): FullExecutionIndicatorTone => {
  switch (value) {
    case 'succeeded':
    case 'deferred':
    case 'queued':
    case 'running':
    case 'failed':
    case 'skipped':
    case 'cancelled':
      return value;
    default:
      return 'deferred';
  }
};

const resolveFullExecutionWarning = (value: string): string => {
  if (!value) {
    return '';
  }
  const mapped = FULL_EXECUTION_WARNING_MESSAGES[value];
  if (mapped) {
    return mapped;
  }
  const normalised = value.replace(/_/g, ' ').trim();
  if (!normalised) {
    return value;
  }
  return normalised.charAt(0).toUpperCase() + normalised.slice(1);
};

const resolveFullExecutionSummary = (
  signal: FullExecutionSignal | null | undefined,
  formatMetricValue: (value?: number | null, precision?: number) => string,
): FullExecutionSummary | null => {
  if (!signal) {
    return null;
  }

  const jobStatus = signal.job_status ?? null;
  const tone = normaliseFullExecutionTone(jobStatus ?? signal.status ?? null);
  const statusLabel = FULL_EXECUTION_STATUS_LABELS[jobStatus ?? signal.status ?? tone] ?? FULL_EXECUTION_STATUS_LABELS[tone];

  const processedRows =
    typeof signal.processed_rows === 'number' && signal.processed_rows >= 0
      ? formatMetricValue(signal.processed_rows)
      : null;
  const totalRows =
    typeof signal.total_rows === 'number' && signal.total_rows >= 0
      ? formatMetricValue(signal.total_rows)
      : null;

  let label = `Full dataset run: ${statusLabel}`;
  if (tone === 'running' && processedRows && totalRows) {
    label = `Full dataset run: In progress (${processedRows}/${totalRows})`;
  } else if (tone === 'running' && processedRows) {
    label = `Full dataset run: In progress (${processedRows} rows)`;
  } else if (tone === 'succeeded' && processedRows) {
    label = `Full dataset run: Completed (${processedRows} rows)`;
  } else if (tone === 'succeeded' && totalRows) {
    label = `Full dataset run: Completed (${totalRows} rows)`;
  }

  const detailParts: string[] = [];
  const metadata: Array<{ label: string; value: string }> = [];
  const jobStatusLabel = jobStatus ? FULL_EXECUTION_JOB_STATUS_LABELS[jobStatus] ?? jobStatus : null;

  if (jobStatus === 'running') {
    if (processedRows && totalRows) {
      detailParts.push(`Processed ${processedRows} of ${totalRows} rows.`);
    } else if (processedRows) {
      detailParts.push(`Processed ${processedRows} rows.`);
    } else if (totalRows) {
      detailParts.push(`Processing ${totalRows} rows.`);
    }
  } else {
    if (processedRows) {
      detailParts.push(`Processed ${processedRows} rows.`);
    } else if (totalRows) {
      detailParts.push(`Total rows ${totalRows}.`);
    }
  }

  if (signal.reason) {
    detailParts.push(signal.reason);
  }

  if (jobStatusLabel) {
    metadata.push({ label: 'Background job', value: jobStatusLabel });
  }
  if (signal.job_id) {
    metadata.push({ label: 'Background job ID', value: signal.job_id });
  }
  if (totalRows) {
    metadata.push({ label: 'Dataset rows', value: totalRows });
  }
  if (processedRows && tone !== 'queued' && tone !== 'skipped') {
    metadata.push({ label: 'Processed rows', value: processedRows });
  }
  if (typeof signal.poll_after_seconds === 'number' && signal.poll_after_seconds > 0) {
    metadata.push({ label: 'Next status check', value: `${signal.poll_after_seconds}s` });
  }
  if (signal.last_updated) {
    try {
      const formatted = new Date(signal.last_updated).toLocaleString();
      metadata.push({ label: 'Last updated', value: formatted });
    } catch (error) {
      metadata.push({ label: 'Last updated', value: String(signal.last_updated) });
    }
  }

  const warnings = Array.isArray(signal.warnings)
    ? signal.warnings.map(resolveFullExecutionWarning).map((warning) => warning.trim()).filter(Boolean)
    : [];

  let stepLabel: string | null = null;
  switch (tone) {
    case 'queued':
      stepLabel = signal.reason ? signal.reason : 'Full dataset run queued in background.';
      break;
    case 'running':
      stepLabel = signal.reason ? signal.reason : 'Full dataset run in progress.';
      break;
    case 'succeeded':
      if (processedRows) {
        const plural = signal.processed_rows === 1 ? '' : 's';
        stepLabel = `Full dataset run processed ${processedRows} row${plural}.`;
      } else if (totalRows) {
        stepLabel = `Full dataset run completed (${totalRows} rows).`;
      } else {
        stepLabel = 'Full dataset run completed.';
      }
      break;
    case 'failed':
      stepLabel = signal.reason ? `Full dataset run failed: ${signal.reason}` : 'Full dataset run failed.';
      break;
    case 'cancelled':
      stepLabel = 'Full dataset run cancelled.';
      break;
    case 'skipped':
      stepLabel = signal.reason ? `Full dataset run skipped: ${signal.reason}` : 'Full dataset run skipped.';
      break;
    case 'deferred':
      stepLabel = signal.reason ? signal.reason : 'Full dataset run deferred to background processing.';
      break;
    default:
      stepLabel = null;
      break;
  }

  return {
    label,
    tone,
    details: detailParts,
    metadata,
    warnings,
    stepLabel,
    isActive: jobStatus === 'queued' || jobStatus === 'running',
  };
};

const createInitialRowWindowState = (): RowWindowState => ({
  status: 'idle',
  columns: [],
  rows: [],
  nextOffset: null,
  hasMore: false,
  totalRows: null,
  samplingAdjustments: [],
  largeDataset: false,
  error: null,
});

const resolveAdjustmentMessage = (value: string): string => {
  if (!value) {
    return '';
  }
  if (value.startsWith('window_mode_fallback:')) {
    const requested = value.split(':', 2)[1] ?? '';
    return requested
      ? `Preview mode "${requested}" is not supported; falling back to head sampling.`
      : 'Preview mode fallback applied.';
  }
  const mapped = SAMPLING_ADJUSTMENT_MESSAGES[value];
  if (mapped) {
    return mapped;
  }
  const normalised = value.replace(/_/g, ' ').trim();
  if (!normalised) {
    return value;
  }
  return normalised.charAt(0).toUpperCase() + normalised.slice(1);
};

export const PreviewPanel: React.FC<PreviewPanelProps> = ({
  preview,
  columns,
  rows,
  tableNote,
  formatCellValue,
  formatMetricValue,
  formatMissingPercentage,
  formatNumericStat,
  formatModeStat,
  fullExecutionStepLabel,
  fullExecutionWarnings,
}) => {
  const metrics: PipelinePreviewMetrics | undefined = preview?.metrics;
  const columnStats: PipelinePreviewColumnStat[] = Array.isArray(preview?.column_stats)
    ? preview.column_stats
    : [];
  const rowStats: PipelinePreviewRowStat[] = Array.isArray(preview?.row_missing_stats)
    ? preview.row_missing_stats
    : [];
  const appliedSteps = Array.isArray(preview?.applied_steps) ? preview.applied_steps : [];
  let displayedSteps = appliedSteps;
  const displayStep = fullExecutionStepLabel || null;
  if (displayStep) {
    const matchIndex = appliedSteps.findIndex((step) =>
      typeof step === 'string' && step.toLowerCase().startsWith('full dataset run')
    );
    if (matchIndex >= 0) {
      displayedSteps = [...appliedSteps.slice(0, matchIndex), displayStep, ...appliedSteps.slice(matchIndex + 1)];
    } else if (!appliedSteps.includes(displayStep)) {
      displayedSteps = [...appliedSteps, displayStep];
    }
  }
  const sampleRows = Array.isArray(preview?.sample_rows) ? preview.sample_rows : [];
  const rowMissingMap = rowStats.reduce<Record<number, number>>((acc, stat) => {
    if (stat && typeof stat.index === 'number') {
      acc[stat.index] = typeof stat.missing_percentage === 'number' ? stat.missing_percentage : 0;
    }
    return acc;
  }, {});
  const rowStatsDisplay = sampleRows.slice(0, ROW_STATS_DISPLAY_LIMIT);

  return (
    <div className="canvas-preview">
      {metrics && (
        <div className="canvas-preview__metrics" role="list">
          <div className="canvas-preview__metric" role="listitem">
            <span className="canvas-preview__metric-label">Rows</span>
            <span className="canvas-preview__metric-value">{formatMetricValue(metrics.row_count)}</span>
          </div>
          <div className="canvas-preview__metric" role="listitem">
            <span className="canvas-preview__metric-label">Columns</span>
            <span className="canvas-preview__metric-value">{formatMetricValue(metrics.column_count)}</span>
          </div>
          <div className="canvas-preview__metric" role="listitem">
            <span className="canvas-preview__metric-label">Duplicate rows</span>
            <span className="canvas-preview__metric-value">{formatMetricValue(metrics.duplicate_rows)}</span>
          </div>
          <div className="canvas-preview__metric" role="listitem">
            <span className="canvas-preview__metric-label">Missing cells</span>
            <span className="canvas-preview__metric-value">{formatMetricValue(metrics.missing_cells)}</span>
          </div>
          <div className="canvas-preview__metric" role="listitem">
            <span className="canvas-preview__metric-label">Rows sampled</span>
            <span className="canvas-preview__metric-value">{formatMetricValue(metrics.preview_rows)}</span>
          </div>
          <div className="canvas-preview__metric" role="listitem">
            <span className="canvas-preview__metric-label">Dataset rows</span>
            <span className="canvas-preview__metric-value">{formatMetricValue(metrics.total_rows)}</span>
          </div>
          <div className="canvas-preview__metric" role="listitem">
            <span className="canvas-preview__metric-label">Sample request</span>
            <span className="canvas-preview__metric-value">
              {metrics.requested_sample_size === 0
                ? 'Full dataset'
                : formatMetricValue(metrics.requested_sample_size)}
            </span>
          </div>
        </div>
      )}

      {displayedSteps.length > 0 && (
        <div className="canvas-preview__steps">
          <h4>Applied steps</h4>
          <ol>
            {displayedSteps.map((step, index) => (
              <li key={`${step}-${index}`}>{step}</li>
            ))}
          </ol>
        </div>
      )}

      {columns.length > 0 && rows.length > 0 ? (
        <div className="canvas-preview__table-wrapper" role="region" aria-label="Sample rows">
          <table className="canvas-preview__table">
            <thead>
              <tr>
                {columns.map((column) => (
                  <th key={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={`row-${rowIndex}`}>
                  {columns.map((column) => (
                    <td key={`${rowIndex}-${column}`}>{formatCellValue(row?.[column])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          {tableNote ? <p className="canvas-modal__note">{tableNote}</p> : null}
        </div>
      ) : (
        <p className="canvas-modal__note">No sample rows available after applying the upstream steps.</p>
      )}

      {rowStats.length > 0 && rowStatsDisplay.length > 0 && (
        <div className="canvas-preview__row-stats">
          <h4>Row missingness</h4>
          <div className="canvas-preview__row-grid">
            {rowStatsDisplay.map((_, rowIndex) => {
              const value = rowMissingMap[rowIndex] ?? 0;
              const precision = value % 1 === 0 ? 0 : 1;
              return (
                <div key={`row-missing-${rowIndex}`} className="canvas-preview__row-chip">
                  <span>Row {rowIndex + 1}</span>
                  <strong>{formatMetricValue(value, precision)}%</strong>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {columnStats.length > 0 && (
        <div className="canvas-preview__column-stats">
          <h4>Column summary</h4>
          <div className="canvas-preview__column-grid">
            {columnStats.map((stat) => (
              <div key={stat.name} className="canvas-preview__column-card">
                <div className="canvas-preview__column-name">{stat.name}</div>
                {stat.dtype && <div className="canvas-preview__column-detail">Type: {stat.dtype}</div>}
                <div className="canvas-preview__column-detail">
                  Missing: {formatMissingPercentage(stat.missing_percentage)} ({formatMetricValue(stat.missing_count)})
                </div>
                {typeof stat.mean === 'number' && !Number.isNaN(stat.mean) && (
                  <div className="canvas-preview__column-detail">Mean: {formatNumericStat(stat.mean)}</div>
                )}
                {typeof stat.median === 'number' && !Number.isNaN(stat.median) && (
                  <div className="canvas-preview__column-detail">Median: {formatNumericStat(stat.median)}</div>
                )}
                {stat.mode !== undefined && stat.mode !== null && stat.mode !== '' && (
                  <div className="canvas-preview__column-detail">Mode: {formatModeStat(stat.mode)}</div>
                )}
                {typeof stat.distinct_count === 'number' && (
                  <div className="canvas-preview__column-detail">
                    Distinct values: {formatMetricValue(stat.distinct_count)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {fullExecutionWarnings && fullExecutionWarnings.length > 0 && (
        <div className="canvas-preview__full-execution-warnings">
          <strong>Full execution notes:</strong>
          <ul>
            {fullExecutionWarnings.map((warning, index) => (
              <li key={`${warning}-${index}`}>{warning}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

type DataSnapshotSectionProps = {
  previewState: PreviewState;
  datasetSourceId?: string | null;
  canTriggerPreview: boolean;
  onRefresh: () => void;
  formatCellValue: (value: unknown) => string;
  formatMetricValue: (value?: number | null, precision?: number) => string;
  formatMissingPercentage: (value?: number | null) => string;
  formatNumericStat: (value?: number | null) => string;
  formatModeStat: (value?: string | number | null) => string;
};

export const DataSnapshotSection: React.FC<DataSnapshotSectionProps> = ({
  previewState,
  datasetSourceId,
  canTriggerPreview,
  onRefresh,
  formatCellValue,
  formatMetricValue,
  formatMissingPercentage,
  formatNumericStat,
  formatModeStat,
}) => {
  const [rowWindow, setRowWindow] = useState<RowWindowState>(() => createInitialRowWindowState());
  const [fullExecutionSummary, setFullExecutionSummary] = useState<FullExecutionSummary | null>(null);
  const [expandedContext, setExpandedContext] = useState<boolean>(false);

  const isRefreshing = previewState.status === 'loading';
  const isButtonDisabled = isRefreshing || !canTriggerPreview;

  const baseColumns = useMemo<string[]>(() => {
    const rawColumns = previewState.data?.columns;
    if (!Array.isArray(rawColumns)) {
      return [];
    }
    return rawColumns.filter((column): column is string => typeof column === 'string');
  }, [previewState.data?.columns]);

  const basePreviewRows = useMemo<Record<string, any>[]>(() => {
    if (!Array.isArray(previewState.data?.sample_rows)) {
      return [];
    }
    // Limit initial display to WINDOW_LIMIT (100 rows) to improve performance
    // User can load more using "Load next 100 rows" button
    return previewState.data.sample_rows.slice(0, WINDOW_LIMIT);
  }, [previewState.data?.sample_rows]);

  const baseRowCount = basePreviewRows.length;
  const totalAvailableInPreview = Array.isArray(previewState.data?.sample_rows) 
    ? previewState.data.sample_rows.length 
    : 0;

  const metrics = previewState.data?.metrics;
  const baseHasMore = useMemo(() => {
    if (!metrics) {
      return false;
    }
    // Check if there are more rows in the cached preview data
    if (totalAvailableInPreview > baseRowCount) {
      return true;
    }
    // Check if there are more rows in the full dataset
    if (typeof metrics.total_rows === 'number' && metrics.total_rows > totalAvailableInPreview) {
      return true;
    }
    return false;
  }, [metrics, baseRowCount, totalAvailableInPreview]);

  useEffect(() => {
    if (previewState.status !== 'success' || !previewState.data) {
      setRowWindow((previous) => {
        if (previous.status === 'idle') {
          return previous;
        }
        return createInitialRowWindowState();
      });
      setFullExecutionSummary(null);
      return;
    }

    const summary = resolveFullExecutionSummary(
      previewState.data.signals?.full_execution,
      formatMetricValue,
    );
    setFullExecutionSummary(summary);

    setRowWindow({
      status: 'ready',
      columns: [...baseColumns],
      rows: [...basePreviewRows],
      nextOffset: baseRowCount,
      hasMore: baseHasMore,
      totalRows: typeof metrics?.total_rows === 'number' ? metrics.total_rows : null,
      samplingAdjustments: [],
      largeDataset: false,
      error: null,
    });
  }, [baseColumns, basePreviewRows, baseRowCount, baseHasMore, formatMetricValue, metrics?.total_rows, previewState.data, previewState.status]);

  // Poll for background job status updates
  useEffect(() => {
    if (!fullExecutionSummary || !fullExecutionSummary.isActive) {
      return;
    }

    const signal = previewState.data?.signals?.full_execution;
    if (!signal || !signal.job_id || !datasetSourceId) {
      return;
    }

    const pollInterval = typeof signal.poll_after_seconds === 'number' && signal.poll_after_seconds > 0
      ? signal.poll_after_seconds * 1000
      : 5000;

    const timer = setTimeout(() => {
      import('../../../../api')
        .then((api) => api.fetchFullExecutionStatus(datasetSourceId, signal.job_id!))
        .then((updatedSignal) => {
          const updatedSummary = resolveFullExecutionSummary(updatedSignal, formatMetricValue);
          setFullExecutionSummary(updatedSummary);
          
          // Update the preview state with the new signal
          if (previewState.data) {
            const updatedPreviewData = {
              ...previewState.data,
              signals: {
                ...previewState.data.signals,
                full_execution: updatedSignal,
              },
            };
            // Note: This assumes previewState can be updated externally
            // If not, you might need to add an onUpdateSignal callback prop
          }
        })
        .catch((error) => {
          console.error('Failed to poll background job status:', error);
        });
    }, pollInterval);

    return () => clearTimeout(timer);
  }, [fullExecutionSummary, datasetSourceId, formatMetricValue, previewState.data]);

  const visibleColumns = rowWindow.columns.length > 0 ? rowWindow.columns : baseColumns;
  const visibleRows = rowWindow.rows.length > 0 ? rowWindow.rows : basePreviewRows;

  const totalRows = rowWindow.totalRows ?? (typeof metrics?.total_rows === 'number' ? metrics.total_rows : null);

  const effectiveHasMore = rowWindow.status === 'idle' ? baseHasMore : rowWindow.hasMore;
  const isLoadingMore = rowWindow.status === 'loading';

  const loadedRowCount = visibleRows.length;

  const tableNote = useMemo(() => {
    if (!loadedRowCount) {
      return null;
    }

    const formattedLoaded = formatMetricValue(loadedRowCount);

    if (totalRows !== null && totalRows >= 0) {
      const formattedTotal = formatMetricValue(totalRows);
      const baseMessage = `Loaded ${formattedLoaded} of ${formattedTotal} rows.`;
      if (effectiveHasMore && loadedRowCount < totalRows) {
        return `${baseMessage} Load more rows to continue.`;
      }
      if (!effectiveHasMore && loadedRowCount < totalRows) {
        return `${baseMessage} Preview limited by backend settings.`;
      }
      return baseMessage;
    }

    return effectiveHasMore
      ? `Loaded ${formattedLoaded} rows. Additional rows available.`
      : `Loaded ${formattedLoaded} rows.`;
  }, [effectiveHasMore, formatMetricValue, loadedRowCount, totalRows]);

  const samplingAdjustmentMessages = useMemo(() => {
    const adjustments = rowWindow.status === 'idle' ? [] : rowWindow.samplingAdjustments;
    return adjustments
      .map(resolveAdjustmentMessage)
      .map((message) => message.trim())
      .filter(Boolean);
  }, [rowWindow.samplingAdjustments, rowWindow.status]);

  const showLargeDatasetNote = rowWindow.status === 'idle' ? false : rowWindow.largeDataset;

  const handleLoadMore = useCallback(() => {
    if (!datasetSourceId) {
      return;
    }
    if (!effectiveHasMore) {
      return;
    }
    if (rowWindow.status === 'loading') {
      return;
    }

    const currentRows = rowWindow.rows.length > 0 ? rowWindow.rows : basePreviewRows;
    const requestOffset = rowWindow.status === 'idle'
      ? currentRows.length
      : rowWindow.nextOffset ?? currentRows.length;

    // Check if we have cached rows available in the preview data
    const allPreviewRows = Array.isArray(previewState.data?.sample_rows) ? previewState.data.sample_rows : [];
    if (requestOffset < allPreviewRows.length) {
      // Load from cached preview data without making an API call
      const nextChunk = allPreviewRows.slice(requestOffset, requestOffset + WINDOW_LIMIT);
      const newRows = currentRows.concat(nextChunk);
      const newOffset = requestOffset + nextChunk.length;
      const stillHasMore = newOffset < allPreviewRows.length || 
        (typeof metrics?.total_rows === 'number' && metrics.total_rows > allPreviewRows.length);

      setRowWindow({
        status: 'ready',
        columns: rowWindow.columns.length > 0 ? rowWindow.columns : baseColumns,
        rows: newRows,
        nextOffset: stillHasMore ? newOffset : null,
        hasMore: stillHasMore,
        totalRows: typeof metrics?.total_rows === 'number' ? metrics.total_rows : null,
        samplingAdjustments: rowWindow.samplingAdjustments,
        largeDataset: rowWindow.largeDataset,
        error: null,
      });
      return;
    }

    // If we've exhausted cached rows, fetch from API
    setRowWindow((previous) => ({
      ...previous,
      status: 'loading',
      error: null,
    }));

    fetchPipelinePreviewRows(datasetSourceId, {
      offset: requestOffset,
      limit: WINDOW_LIMIT,
    })
      .then((response) => {
        setRowWindow((previous) => {
          const existingRows = previous.rows.length > 0 ? previous.rows : currentRows;

          const safeColumns = Array.isArray(response.columns) && response.columns.length > 0
            ? response.columns.map((column) => String(column))
            : previous.columns.length > 0
            ? previous.columns
            : baseColumns;

          const normalizedOffset = typeof response.offset === 'number' && response.offset >= 0 ? response.offset : 0;
          const returnedRows = Array.isArray(response.rows) ? response.rows : [];
          const prefix = normalizedOffset <= existingRows.length
            ? existingRows.slice(0, normalizedOffset)
            : existingRows.slice();
          const mergedRows = normalizedOffset <= existingRows.length
            ? prefix.concat(returnedRows)
            : existingRows.concat(returnedRows);

          const returnedCount = returnedRows.length;
          const hasMore = Boolean(response.has_more);
          const nextOffsetValue = typeof response.next_offset === 'number'
            ? response.next_offset
            : hasMore
            ? normalizedOffset + returnedCount
            : null;

          const responseTotal = typeof response.total_rows === 'number' ? response.total_rows : null;
          const fallbackTotal = !hasMore ? mergedRows.length : null;
          const totalRowsValue = responseTotal !== null
            ? responseTotal
            : previous.totalRows !== null
            ? previous.totalRows
            : fallbackTotal;

          return {
            status: 'ready',
            columns: [...safeColumns],
            rows: mergedRows,
            nextOffset: nextOffsetValue,
            hasMore,
            totalRows: totalRowsValue,
            samplingAdjustments: Array.isArray(response.sampling_adjustments)
              ? response.sampling_adjustments
              : [],
            largeDataset: Boolean(response.large_dataset),
            error: null,
          };
        });
      })
      .catch((error: any) => {
        setRowWindow((previous) => ({
          ...previous,
          status: 'error',
          error: error?.message ?? 'Unable to load additional rows.',
        }));
      });
  }, [baseColumns, basePreviewRows, datasetSourceId, effectiveHasMore, rowWindow, previewState.data?.sample_rows, metrics]);

  return (
    <section className="canvas-modal__section">
      <div className="canvas-modal__section-header">
        <div className="canvas-modal__section-title">
          <h3>Dataset snapshot</h3>
          {fullExecutionSummary && (
            <div
              className={`canvas-preview__full-execution-badge canvas-preview__full-execution--${fullExecutionSummary.tone}`}
              role="status"
              aria-live={fullExecutionSummary.isActive ? 'polite' : 'off'}
              title={fullExecutionSummary.details.length > 0 ? fullExecutionSummary.details.join(' ') : undefined}
            >
              {fullExecutionSummary.label}
            </div>
          )}
        </div>
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={onRefresh}
            disabled={isButtonDisabled}
          >
            {isRefreshing ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>
      </div>

      {fullExecutionSummary && (
        <div className="canvas-preview__full-execution-context">
          {fullExecutionSummary.details.length > 0 && (
            <ul className="canvas-preview__full-execution-details">
              {fullExecutionSummary.details.map((detail, index) => (
                <li key={`${detail}-${index}`}>{detail}</li>
              ))}
            </ul>
          )}
          {fullExecutionSummary.metadata.length > 0 && (
            <>
              <button
                type="button"
                className="canvas-preview__full-execution-toggle"
                onClick={() => setExpandedContext(!expandedContext)}
                aria-expanded={expandedContext}
                style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}
              >
                {expandedContext ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                {expandedContext ? 'Hide' : 'Show'} job details
              </button>
              {expandedContext && (
                <dl className="canvas-preview__full-execution-meta">
                  {fullExecutionSummary.metadata.map((item, index) => (
                    <div key={`${item.label}-${item.value}-${index}`} className="canvas-preview__full-execution-meta-item">
                      <dt>{item.label}</dt>
                      <dd>{item.value}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </>
          )}
        </div>
      )}

      {isRefreshing && <p className="canvas-modal__note">Generating preview…</p>}

      {previewState.status === 'error' && previewState.error && (
        <p className="canvas-modal__note canvas-modal__note--error">{previewState.error}</p>
      )}

      {previewState.status === 'success' && previewState.data && (
        <PreviewPanel
          preview={previewState.data}
          columns={visibleColumns}
          rows={visibleRows}
          tableNote={tableNote}
          formatCellValue={formatCellValue}
          formatMetricValue={formatMetricValue}
          formatMissingPercentage={formatMissingPercentage}
          formatNumericStat={formatNumericStat}
          formatModeStat={formatModeStat}
          fullExecutionStepLabel={fullExecutionSummary?.stepLabel ?? null}
          fullExecutionWarnings={fullExecutionSummary?.warnings ?? []}
        />
      )}

      {rowWindow.error && (
        <p className="canvas-modal__note canvas-modal__note--error">{rowWindow.error}</p>
      )}

      {showLargeDatasetNote && (
        <p className="canvas-modal__note">
          Large dataset detected. Preview windows are capped at {formatMetricValue(WINDOW_LIMIT)} rows to stay responsive.
        </p>
      )}

      {samplingAdjustmentMessages.length > 0 && (
        <div className="canvas-modal__note">
          <strong>Sampling adjustments:</strong>
          <ul>
            {samplingAdjustmentMessages.map((message, index) => (
              <li key={`${message}-${index}`}>{message}</li>
            ))}
          </ul>
        </div>
      )}

      {effectiveHasMore && datasetSourceId && (
        <div className="canvas-modal__section-actions">
          <button
            type="button"
            className="btn btn-outline-secondary"
            onClick={handleLoadMore}
            disabled={isLoadingMore}
          >
            {isLoadingMore ? 'Loading rows…' : `Load next ${formatMetricValue(WINDOW_LIMIT)} rows`}
          </button>
        </div>
      )}

      {!effectiveHasMore && !isRefreshing && totalRows !== null && loadedRowCount >= totalRows && loadedRowCount > 0 && (
        <p className="canvas-modal__note">All available rows have been loaded.</p>
      )}
    </section>
  );
};
