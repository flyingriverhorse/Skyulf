import { useMemo } from 'react';
import type { useNodePreview } from './useNodePreview';

type PreviewState = ReturnType<typeof useNodePreview>['previewState'];

type UsePreviewDataResult = {
  previewColumns: string[];
  previewColumnStats: any[];
  previewSampleRows: Record<string, any>[];
};

/**
 * Normalizes preview payload fragments (columns, stats, sample rows) so the modal can
 * consume tidy arrays without inlining the defensive guards everywhere.
 */
export const usePreviewData = (previewState: PreviewState): UsePreviewDataResult => {
  const previewColumns = useMemo(() => {
    const rawColumns = previewState.data?.columns;
    if (!Array.isArray(rawColumns)) {
      return [] as string[];
    }
    return rawColumns.filter((column): column is string => typeof column === 'string');
  }, [previewState.data?.columns]);

  const previewColumnStats = useMemo(() => {
    const stats = previewState.data?.column_stats;
    if (!Array.isArray(stats)) {
      return [] as any[];
    }
    return stats;
  }, [previewState.data?.column_stats]);

  const previewSampleRows = useMemo(() => {
    if (!Array.isArray(previewState.data?.sample_rows)) {
      return [] as Record<string, any>[];
    }
    return previewState.data.sample_rows;
  }, [previewState.data?.sample_rows]);

  return {
    previewColumns,
    previewColumnStats,
    previewSampleRows,
  };
};
