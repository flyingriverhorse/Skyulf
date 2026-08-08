import React, { useEffect, useRef, useState } from 'react';
import { FileText, Database, Columns, AlignJustify } from 'lucide-react';
import { DatasetService, DatasetApiError } from '../../core/api/datasets';
import { Dataset } from '../../core/types/api';
import { formatBytes } from '../../core/utils/format';
import { EmptyState, ErrorState, LoadingState, ModalShell } from '../shared';

interface DatasetPreviewModalProps {
  dataset: Dataset | null;
  isOpen: boolean;
  onClose: () => void;
}

interface ColumnProfile {
  name: string;
  dtype: string;
  missing_count: number;
  missing_percentage: number;
  distinct_count: number;
  numeric_summary?: {
    mean: number;
    std: number;
    minimum: number;
    maximum: number;
  } | undefined;
}

interface DatasetProfile {
  metrics: {
    row_count: number;
    column_count: number;
    missing_cells: number;
    missing_percentage: number;
  };
  columns: ColumnProfile[];
}

/** Shows a dataset preview with independent sample/profile recovery states. */
export const DatasetPreviewModal: React.FC<DatasetPreviewModalProps> = ({ dataset, isOpen, onClose }) => {
  const [sampleData, setSampleData] = useState<unknown[]>([]);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'sample' | 'stats'>('sample');
  const [sampleSize, setSampleSize] = useState(100);

  // Track in-flight request id so a slow/stale response doesn't overwrite
  // state from a newer request (e.g. rapid dataset switch or Load More).
  const requestIdRef = useRef(0);

  useEffect(() => {
    if (isOpen && dataset) {
      void fetchData(100);
    } else {
      setSampleData([]);
      setProfile(null);
      setSampleError(null);
      setProfileError(null);
      setActiveTab('sample');
      setSampleSize(100);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, dataset]);

  const fetchData = async (limit: number) => {
    if (!dataset) return;
    const myRequestId = ++requestIdRef.current;
    setLoading(true);
    setSampleError(null);
    setProfileError(null);
    try {
      const [sampleResult, profileResult] = await Promise.allSettled([
        DatasetService.getSample(dataset.id, limit),
        DatasetService.getProfile(dataset.id)
      ]);
      if (myRequestId !== requestIdRef.current) return;
      if (sampleResult.status === 'fulfilled') {
        setSampleData(Array.isArray(sampleResult.value) ? sampleResult.value : []);
      } else {
        console.error('Failed to fetch sample preview:', sampleResult.reason);
        setSampleData([]);
        setSampleError(formatPreviewError('sample', dataset.name, sampleResult.reason));
      }
      if (profileResult.status === 'fulfilled') {
        setProfile(profileResult.value);
      } else {
        console.error('Failed to fetch dataset profile:', profileResult.reason);
        setProfile(null);
        setProfileError(formatPreviewError('statistics', dataset.name, profileResult.reason));
      }
    } catch (err) {
      console.error('Failed to fetch data:', err);
    } finally {
      if (myRequestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  };

  const handleLoadMore = () => {
    const newSize = sampleSize + 500;
    setSampleSize(newSize);
    void fetchData(newSize);
  };

  if (!dataset) return null;

  const sampleLoaded = sampleError === null && sampleData.length > 0;
  const profileLoaded = profileError === null && profile !== null;
  const columns = sampleData.length > 0 ? Object.keys(sampleData[0] as object) : [];
  // Backend JSON can send an explicit null, which `?? undefined` normalizes so
  // "unknown" and a real 0 stay distinguishable without a crash on toLocaleString.
  const rowCount = profile?.metrics.row_count ?? dataset.rows ?? undefined;
  const colCount = profile?.metrics.column_count ?? dataset.columns ?? undefined;
  const sizeBytes = dataset.size_bytes ?? undefined;
  const unknownMetric = <span aria-label="Unknown">—</span>;

  const titleNode = (
    <div className="flex items-center gap-3">
      <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-blue-600 dark:text-blue-400">
        <Database size={24} />
      </div>
      <div>
        <span className="text-xl font-bold text-slate-900 dark:text-slate-100">{dataset.name}</span>
        <div className="flex items-center gap-3 text-sm text-slate-500 dark:text-slate-400 mt-1 font-normal">
          <span className="flex items-center gap-1">
            <FileText size={14} /> {dataset.format || 'CSV'}
          </span>
          <span>•</span>
          <span className="flex items-center gap-1">
            <AlignJustify size={14} /> {rowCount !== undefined ? rowCount.toLocaleString() : unknownMetric} rows
          </span>
          <span>•</span>
          <span className="flex items-center gap-1">
            <Columns size={14} /> {colCount !== undefined ? colCount : unknownMetric} columns
          </span>
          <span>•</span>
          <span>{sizeBytes !== undefined ? formatBytes(sizeBytes) : unknownMetric}</span>
        </div>
      </div>
    </div>
  );

  const footerNode = (
    <div className="flex justify-end">
      <button
        onClick={onClose}
        className="px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-md text-sm font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
      >
        Close
      </button>
    </div>
  );

  return (
    <ModalShell
      isOpen={isOpen}
      onClose={onClose}
      title={titleNode}
      size="5xl"
      footer={footerNode}
    >
      <div className="flex flex-col h-full">
        {/* Tabs */}
        <div className="flex border-b border-slate-200 dark:border-slate-700 px-6">
          <button
            onClick={() => { setActiveTab('sample'); }}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'sample'
                ? 'border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-400'
                : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200'
            }`}
          >
            Data Sample
          </button>
          <button
            onClick={() => { setActiveTab('stats'); }}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'stats'
                ? 'border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-400'
                : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200'
            }`}
          >
            Statistics
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-auto p-6">
          {loading ? (
            <LoadingState message={`Loading preview data for ${dataset.name}...`} />
          ) : activeTab === 'sample' ? (
            sampleError ? (
              <ErrorState error={sampleError} onRetry={() => { void fetchData(sampleSize); }} />
            ) : sampleLoaded ? (
              <div className="space-y-4">
                <div className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
                  <div className="overflow-x-auto max-h-[60vh]">
                    <table className="w-full text-sm text-left">
                      <thead className="text-xs text-slate-500 dark:text-slate-400 uppercase bg-slate-50 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 sticky top-0 z-10">
                        <tr>
                          {columns.map((col) => (
                            <th key={col} className="px-4 py-3 font-semibold whitespace-nowrap bg-slate-50 dark:bg-slate-800">
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 dark:divide-slate-700 bg-white dark:bg-slate-900">
                        {sampleData.map((row, i) => (
                          <tr key={i} className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                            {columns.map((col) => (
                              <td key={`${i}-${col}`} className="px-4 py-2 whitespace-nowrap text-slate-700 dark:text-slate-300">
                                {String((row as Record<string, unknown>)[col] ?? '')}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                <div className="flex justify-center items-center gap-4">
                  <button
                    onClick={handleLoadMore}
                    className="px-4 py-2 text-sm font-medium text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 border border-blue-200 dark:border-blue-800 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors"
                  >
                    Load More (+500 rows)
                  </button>
                  <span className="text-xs text-slate-500">
                    Showing first {sampleData.length} rows
                  </span>
                </div>
              </div>
            ) : columns.length === 0 ? (
              <EmptyState title="No data available to preview." description={`"${dataset.name}" did not return any columns.`} />
            ) : (
              <EmptyState title={`No preview data for "${dataset.name}".`} description="The sample request returned no rows." />
            )
          ) : (
            <div className="space-y-6">
              {profileError ? (
                <ErrorState error={profileError} onRetry={() => { void fetchData(sampleSize); }} />
              ) : profileLoaded ? (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <StatCard label="Total Rows" value={profile?.metrics.row_count !== undefined ? profile.metrics.row_count.toLocaleString() : undefined} />
                    <StatCard label="Total Columns" value={profile?.metrics.column_count} />
                    <StatCard label="Missing Cells" value={profile?.metrics.missing_cells !== undefined ? profile.metrics.missing_cells.toLocaleString() : undefined} />
                    <StatCard label="Missing %" value={profile?.metrics.missing_percentage !== undefined ? `${profile.metrics.missing_percentage}%` : undefined} />
                  </div>

                  <div className="border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden">
                    <table className="w-full text-sm text-left">
                      <thead className="text-xs text-slate-500 dark:text-slate-400 uppercase bg-slate-50 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
                        <tr>
                          <th className="px-4 py-3 font-semibold">Column</th>
                          <th className="px-4 py-3 font-semibold">Type</th>
                          <th className="px-4 py-3 font-semibold">Missing</th>
                          <th className="px-4 py-3 font-semibold">Unique</th>
                          <th className="px-4 py-3 font-semibold">Min</th>
                          <th className="px-4 py-3 font-semibold">Max</th>
                          <th className="px-4 py-3 font-semibold">Mean</th>
                          <th className="px-4 py-3 font-semibold">Std Dev</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-200 dark:divide-slate-700 bg-white dark:bg-slate-900">
                        {profile.columns.map((col) => (
                          <tr key={col.name} className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                            <td className="px-4 py-3 font-medium text-slate-900 dark:text-slate-100">{col.name}</td>
                            <td className="px-4 py-3 font-mono text-xs text-slate-500">{col.dtype}</td>
                            <td className="px-4 py-3 text-slate-600 dark:text-slate-300">
                              {col.missing_count} <span className="text-xs text-slate-400">({col.missing_percentage}%)</span>
                            </td>
                            <td className="px-4 py-3 text-slate-600 dark:text-slate-300">{col.distinct_count}</td>
                            <td className="px-4 py-3 text-slate-600 dark:text-slate-300 font-mono text-xs">
                              {col.numeric_summary?.minimum.toFixed(2) ?? '-'}
                            </td>
                            <td className="px-4 py-3 text-slate-600 dark:text-slate-300 font-mono text-xs">
                              {col.numeric_summary?.maximum.toFixed(2) ?? '-'}
                            </td>
                            <td className="px-4 py-3 text-slate-600 dark:text-slate-300 font-mono text-xs">
                              {col.numeric_summary?.mean.toFixed(2) ?? '-'}
                            </td>
                            <td className="px-4 py-3 text-slate-600 dark:text-slate-300 font-mono text-xs">
                              {col.numeric_summary?.std.toFixed(2) ?? '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              ) : (
                <EmptyState title={`No statistics for "${dataset.name}".`} description="The profile request returned no metrics." />
              )}
            </div>
          )}
        </div>
      </div>
    </ModalShell>
  );
};

const StatCard = ({ label, value }: { label: string; value: React.ReactNode }) => (
  <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700">
    <div className="text-xs text-slate-500 dark:text-slate-400 uppercase font-semibold">{label}</div>
    <div className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">{value ?? <span aria-label="Unknown">—</span>}</div>
  </div>
);

const formatPreviewError = (part: 'sample' | 'statistics', datasetName: string, error: unknown): string => {
  const datasetPart = part === 'sample' ? 'Sample preview' : 'Statistics';
  if (error instanceof DatasetApiError) {
    if (error.status === 404) {
      return `${datasetPart} for "${datasetName}" is unavailable because the source was deleted or moved.${error.message ? ` ${error.message}` : ''}`;
    }
    if (error.message) {
      return `${datasetPart} for "${datasetName}" could not be loaded: ${error.message}`;
    }
  }
  return `${datasetPart} for "${datasetName}" could not be loaded. Please try again.`;
};
