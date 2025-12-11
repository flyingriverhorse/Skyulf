import React, { useEffect, useState } from 'react';
import { X, FileText, Database, Columns, AlignJustify } from 'lucide-react';
import { DatasetService } from '../../core/api/datasets';
import { Dataset } from '../../core/types/api';
import { formatBytes } from '../../core/utils/format';

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
  };
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

export const DatasetPreviewModal: React.FC<DatasetPreviewModalProps> = ({ dataset, isOpen, onClose }) => {
  const [sampleData, setSampleData] = useState<any[]>([]);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'sample' | 'stats'>('sample');
  const [sampleSize, setSampleSize] = useState(100);

  useEffect(() => {
    if (isOpen && dataset) {
      fetchData(100);
    } else {
      setSampleData([]);
      setProfile(null);
      setError(null);
      setActiveTab('sample');
      setSampleSize(100);
    }
  }, [isOpen, dataset]);

  const fetchData = async (limit: number) => {
    if (!dataset) return;
    setLoading(true);
    setError(null);
    try {
      const [sample, profileData] = await Promise.all([
        DatasetService.getSample(dataset.id, limit),
        DatasetService.getProfile(dataset.id)
      ]);
      setSampleData(sample);
      setProfile(profileData);
    } catch (err) {
      console.error('Failed to fetch data:', err);
      setError('Failed to load dataset preview.');
    } finally {
      setLoading(false);
    }
  };

  const handleLoadMore = () => {
    const newSize = sampleSize + 500;
    setSampleSize(newSize);
    fetchData(newSize);
  };

  if (!isOpen || !dataset) return null;

  const columns = sampleData.length > 0 ? Object.keys(sampleData[0]) : [];
  const rowCount = profile?.metrics.row_count ?? dataset.rows ?? 0;
  const colCount = profile?.metrics.column_count ?? dataset.columns ?? 0;
  const sizeBytes = dataset.size_bytes ?? 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-white dark:bg-slate-900 rounded-xl shadow-2xl w-full max-w-5xl max-h-[90vh] flex flex-col border border-slate-200 dark:border-slate-700 animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-200 dark:border-slate-700">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-blue-600 dark:text-blue-400">
              <Database size={24} />
            </div>
            <div>
              <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100">{dataset.name}</h2>
              <div className="flex items-center gap-3 text-sm text-slate-500 dark:text-slate-400 mt-1">
                <span className="flex items-center gap-1">
                  <FileText size={14} /> {dataset.format || 'CSV'}
                </span>
                <span>•</span>
                <span className="flex items-center gap-1">
                  <AlignJustify size={14} /> {rowCount.toLocaleString()} rows
                </span>
                <span>•</span>
                <span className="flex items-center gap-1">
                  <Columns size={14} /> {colCount} columns
                </span>
                <span>•</span>
                <span>{formatBytes(sizeBytes)}</span>
              </div>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-full transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-slate-200 dark:border-slate-700 px-6">
          <button
            onClick={() => setActiveTab('sample')}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'sample'
                ? 'border-blue-600 text-blue-600 dark:text-blue-400 dark:border-blue-400'
                : 'border-transparent text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200'
            }`}
          >
            Data Sample
          </button>
          <button
            onClick={() => setActiveTab('stats')}
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
            <div className="flex flex-col items-center justify-center h-64 text-slate-500">
              <div className="w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4" />
              <p>Loading preview data...</p>
            </div>
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-64 text-red-500 bg-red-50 dark:bg-red-900/10 rounded-lg border border-red-100 dark:border-red-900/20">
              <p>{error}</p>
              <button 
                onClick={() => fetchData(sampleSize)}
                className="mt-4 px-4 py-2 bg-white dark:bg-slate-800 border border-red-200 dark:border-red-800 rounded-md text-sm font-medium hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
              >
                Retry
              </button>
            </div>
          ) : activeTab === 'sample' ? (
            sampleData.length === 0 ? (
              <div className="text-center py-12 text-slate-500">
                No data available to preview.
              </div>
            ) : columns.length === 0 ? (
              <div className="text-center py-12 text-slate-500">
                Data loaded but no columns found.
              </div>
            ) : (
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
                                {String(row[col] ?? '')}
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
            )
          ) : (
            <div className="space-y-6">
              {/* Overall Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <StatCard label="Total Rows" value={profile?.metrics.row_count.toLocaleString()} />
                <StatCard label="Total Columns" value={profile?.metrics.column_count} />
                <StatCard label="Missing Cells" value={profile?.metrics.missing_cells.toLocaleString()} />
                <StatCard label="Missing %" value={`${profile?.metrics.missing_percentage}%`} />
              </div>

              {/* Column Statistics Table */}
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
                    {profile?.columns.map((col) => (
                      <tr key={col.name} className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                        <td className="px-4 py-3 font-medium text-slate-900 dark:text-slate-100">{col.name}</td>
                        <td className="px-4 py-3 font-mono text-xs text-slate-500">{col.dtype}</td>
                        <td className="px-4 py-3 text-slate-600 dark:text-slate-300">
                          {col.missing_count} <span className="text-xs text-slate-400">({col.missing_percentage}%)</span>
                        </td>
                        <td className="px-4 py-3 text-slate-600 dark:text-slate-300">{col.distinct_count}</td>
                        <td className="px-4 py-3 text-slate-600 dark:text-slate-300 font-mono text-xs">
                          {col.numeric_summary?.minimum?.toFixed(2) ?? '-'}
                        </td>
                        <td className="px-4 py-3 text-slate-600 dark:text-slate-300 font-mono text-xs">
                          {col.numeric_summary?.maximum?.toFixed(2) ?? '-'}
                        </td>
                        <td className="px-4 py-3 text-slate-600 dark:text-slate-300 font-mono text-xs">
                          {col.numeric_summary?.mean?.toFixed(2) ?? '-'}
                        </td>
                        <td className="px-4 py-3 text-slate-600 dark:text-slate-300 font-mono text-xs">
                          {col.numeric_summary?.std?.toFixed(2) ?? '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 flex justify-end rounded-b-xl">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-md text-sm font-medium text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ label, value }: { label: string; value: string | number | undefined }) => (
  <div className="bg-slate-50 dark:bg-slate-800 p-4 rounded-lg border border-slate-200 dark:border-slate-700">
    <div className="text-xs text-slate-500 dark:text-slate-400 uppercase font-semibold">{label}</div>
    <div className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">{value ?? '-'}</div>
  </div>
);

