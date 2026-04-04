import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { DatasetService } from '../core/api/datasets';
import { Dataset } from '../core/types/api';
import { FileUpload } from '../modules/nodes/data/FileUpload';
import { Trash2, Play, FileText, Calendar, Database, Plus, Eye, Loader2, XCircle, BarChart2, Download, Search, Filter } from 'lucide-react';
import { formatBytes } from '../core/utils/format';
import { DatasetPreviewModal } from '../components/data/DatasetPreviewModal';
import { AddSourceModal } from '../components/data/AddSourceModal';
import { IngestionJobsModal } from '../components/data/IngestionJobsModal';
import { LoadingState, EmptyState } from '../components/shared';

export const DataSources: React.FC = () => {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [showAddSource, setShowAddSource] = useState(false);
  const [showIngestionJobs, setShowIngestionJobs] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [cancellingId, setCancellingId] = useState<string | null>(null);
  const [previewDataset, setPreviewDataset] = useState<Dataset | null>(null);
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<string>('all');
  const [filterFormat, setFilterFormat] = useState<string>('all');
  const [showFilters, setShowFilters] = useState(false);

  const filteredDatasets = datasets.filter(d => {
    const status = d.source_metadata?.ingestion_status?.status || 'completed';
    if (filterStatus !== 'all' && status !== filterStatus) return false;
    if (filterType !== 'all' && d.type !== filterType) return false;
    if (filterFormat !== 'all' && (d.format || 'csv') !== filterFormat) return false;
    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      const matchesName = d.name.toLowerCase().includes(q);
      const matchesId = (d.source_id || d.id).toLowerCase().includes(q);
      const matchesFormat = (d.format || 'csv').toLowerCase().includes(q);
      if (!matchesName && !matchesId && !matchesFormat) return false;
    }
    return true;
  });

  const activeFilterCount = (filterType !== 'all' ? 1 : 0) + (filterFormat !== 'all' ? 1 : 0);

  const fetchDatasets = async () => {
    // Don't set loading to true on subsequent polls to avoid flickering
    if (datasets.length === 0) setLoading(true);
    try {
      const data = await DatasetService.getAll();
      setDatasets(data);
    } catch (error) {
      console.error('Failed to fetch datasets:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchDatasets();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // Poll every 5 seconds if there are pending jobs
    const interval = setInterval(() => {
      const hasPending = datasets.some(d => 
        d.source_metadata?.ingestion_status?.status === 'pending' || 
        d.source_metadata?.ingestion_status?.status === 'processing'
      );
      if (hasPending) {
        void fetchDatasets();
      }
    }, 5000);
    return () => { clearInterval(interval); };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasets]); // Re-run effect when datasets change to update "hasPending" check

  const handleDelete = async (id: string) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) return;
    
    setDeletingId(id);
    try {
      await DatasetService.delete(id);
      setDatasets(datasets.filter(d => d.id !== id));
    } catch (error) {
      console.error('Failed to delete dataset:', error);
      alert('Failed to delete dataset');
    } finally {
      setDeletingId(null);
    }
  };

  const handleCancel = async (id: string) => {
    if (!window.confirm('Are you sure you want to cancel this ingestion job?')) return;
    
    setCancellingId(id);
    try {
      await DatasetService.cancelIngestion(id);
      fetchDatasets();
    } catch (error) {
      console.error('Failed to cancel ingestion:', error);
      alert('Failed to cancel ingestion');
    } finally {
      setCancellingId(null);
    }
  };

  const handleUseInCanvas = (id: string) => {
    navigate(`/canvas?source_id=${id}`);
  };

  const handleUploadComplete = () => {
    setShowUpload(false);
    fetchDatasets();
  };

  const handleSourceCreated = () => {
    setShowAddSource(false);
    fetchDatasets();
  };

  const getStatusBadge = (dataset: Dataset) => {
    const status = dataset.source_metadata?.ingestion_status?.status;
    if (!status || status === 'completed') return null;

    if (status === 'failed') {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400">
          Failed
        </span>
      );
    }

    if (status === 'cancelled') {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-slate-100 text-slate-800 dark:bg-slate-800 dark:text-slate-400">
          Cancelled
        </span>
      );
    }

    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400">
        <Loader2 size={12} className="animate-spin" />
        {status === 'processing' ? 'Processing...' : 'Pending...'}
      </span>
    );
  };

  return (
    <div className="p-8 space-y-6">
      <DatasetPreviewModal 
        dataset={previewDataset} 
        isOpen={!!previewDataset} 
        onClose={() => { setPreviewDataset(null); }}
      />

      <AddSourceModal
        isOpen={showAddSource}
        onClose={() => { setShowAddSource(false); }}
        onSuccess={handleSourceCreated}
      />

      <IngestionJobsModal
        isOpen={showIngestionJobs}
        onClose={() => { setShowIngestionJobs(false); }}
        datasets={datasets}
        onRefresh={fetchDatasets}
      />

      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">Data Sources</h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">Manage your uploaded datasets and use them in experiments.</p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={() => { setShowIngestionJobs(true); }}
            className="flex items-center gap-2 px-4 py-2 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-md shadow-sm hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-slate-700 dark:text-slate-200"
          >
            <Calendar size={18} />
            Ingestion Jobs
          </button>
          <button
            onClick={() => { setShowAddSource(true); }}
            className="flex items-center gap-2 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 border border-slate-300 dark:border-slate-600 px-4 py-2 rounded-md shadow-sm transition-all hover:bg-slate-50 dark:hover:bg-slate-700"
          >
            <Database size={18} />
            Add Source
          </button>
          <button
            onClick={() => { setShowUpload(!showUpload); }}
            className="flex items-center gap-2 text-white px-4 py-2 rounded-md shadow-sm transition-all hover:opacity-90"
            style={{ background: 'var(--main-gradient)' }}
          >
            <Plus size={18} />
            {showUpload ? 'Cancel Upload' : 'Upload File'}
          </button>
        </div>
      </div>

      {showUpload && (
        <div className="bg-white dark:bg-slate-800 p-6 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm animate-in fade-in slide-in-from-top-4">
          <FileUpload onUploadComplete={handleUploadComplete} onCancel={() => { setShowUpload(false); }} />
        </div>
      )}

      <div className="flex flex-col gap-3 mb-4">
        <div className="flex items-center gap-3">
          <div className="relative flex-1 max-w-sm">
            <Search size={16} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none z-10" />
            <input
              type="text"
              placeholder="Search datasets..."
              value={searchQuery}
              onChange={e => { setSearchQuery(e.target.value); }}
              className="w-full pl-8 pr-3 py-1.5 text-sm rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>
          <button
            onClick={() => { setShowFilters(!showFilters); }}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md border transition-colors ${
              showFilters || filterType !== 'all'
                ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300'
                : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-slate-700'
            }`}
          >
            <Filter size={14} />
            Filters
            {activeFilterCount > 0 && (
              <span className="ml-1 w-5 h-5 flex items-center justify-center rounded-full bg-blue-600 text-white text-xs">{activeFilterCount}</span>
            )}
          </button>
          {(searchQuery || filterStatus !== 'all' || filterType !== 'all' || filterFormat !== 'all') && (
            <button
              onClick={() => { setSearchQuery(''); setFilterStatus('all'); setFilterType('all'); setFilterFormat('all'); }}
              className="text-xs text-blue-600 dark:text-blue-400 hover:underline whitespace-nowrap"
            >
              Clear all
            </button>
          )}
        </div>

        {showFilters && (
          <div className="flex items-center gap-4 pl-1">
            <div className="flex items-center gap-2">
              <label className="text-xs text-slate-500 dark:text-slate-400 font-medium">Type</label>
              <select
                value={filterType}
                onChange={e => { setFilterType(e.target.value); }}
                className="px-2 py-1 text-sm rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="all">All types</option>
                {[...new Set(datasets.map(d => d.type))].map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-slate-500 dark:text-slate-400 font-medium">Format</label>
              <select
                value={filterFormat}
                onChange={e => { setFilterFormat(e.target.value); }}
                className="px-2 py-1 text-sm rounded-md border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value="all">All formats</option>
                {[...new Set(datasets.map(d => d.format || 'csv'))].map(f => (
                  <option key={f} value={f}>{f.toUpperCase()}</option>
                ))}
              </select>
            </div>
          </div>
        )}

        <div className="flex items-center gap-2 overflow-x-auto pb-1 scrollbar-hide">
          {['all', 'completed', 'processing', 'failed', 'cancelled'].map(status => (
            <button
              key={status}
              onClick={() => { setFilterStatus(status); }}
              className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors whitespace-nowrap ${
                filterStatus === status 
                  ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300' 
                  : 'bg-white dark:bg-slate-800 text-slate-600 border border-slate-200 dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700'
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </button>
          ))}
        </div>
      </div>

      <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-slate-500 dark:text-slate-400 uppercase bg-slate-50 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
              <tr>
                <th className="px-3 py-4 font-semibold">Dataset Name</th>
                <th className="px-3 py-4 font-semibold">Type</th>
                <th className="px-6 py-4 font-semibold">Format</th>
                <th className="px-6 py-4 font-semibold">Size</th>
                <th className="px-6 py-4 font-semibold">Created</th>
                <th className="px-6 py-4 font-semibold text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700 bg-white dark:bg-slate-900">
              {loading && datasets.length === 0 ? (
                <tr>
                  <td colSpan={6}>
                    <LoadingState message="Loading datasets..." />
                  </td>
                </tr>
              ) : filteredDatasets.length === 0 ? (
                <tr>
                  <td colSpan={6}>
                    {datasets.length === 0 ? (
                      <EmptyState
                        icon={<Database className="w-12 h-12 text-slate-300 dark:text-slate-600" />}
                        title="No datasets found"
                        description="Upload a dataset to get started with your analysis."
                      />
                    ) : (
                      <EmptyState title="No datasets match your search or filters" />
                    )}
                  </td>
                </tr>
              ) : (
                filteredDatasets.map((d) => (
                  <tr key={d.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                    <td className="px-3 py-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-blue-600 dark:text-blue-400">
                          <FileText size={18} />
                        </div>
                        <div>
                          <div className="font-medium text-slate-900 dark:text-slate-100 flex items-center gap-2">
                            {d.name}
                            {getStatusBadge(d)}
                          </div>
                          <div className="text-xs text-slate-500 dark:text-slate-400 font-mono mt-0.5" title="Dataset ID">
                            {d.source_id || d.id}
                          </div>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-4">
                      <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${
                        d.type === 's3' 
                          ? 'bg-orange-50 text-orange-700 border-orange-200 dark:bg-orange-900/20 dark:text-orange-400 dark:border-orange-800'
                          : d.type === 'file'
                          ? 'bg-slate-100 text-slate-700 border-slate-200 dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700'
                          : 'bg-indigo-50 text-indigo-700 border-indigo-200 dark:bg-indigo-900/20 dark:text-indigo-400 dark:border-indigo-800'
                      }`}>
                        {d.type === 'file' ? 'Upload' : d.type.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 uppercase border border-slate-200 dark:border-slate-700">
                        {d.format || 'CSV'}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-slate-600 dark:text-slate-300">
                      <div className="flex flex-col">
                        <span className="font-medium">{formatBytes(d.size_bytes || 0)}</span>
                        <span className="text-xs text-slate-400 dark:text-slate-500">
                          {d.rows?.toLocaleString() || '-'} rows • {d.columns || '-'} cols
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-slate-600 dark:text-slate-300">
                      <div className="flex items-center gap-2">
                        <Calendar size={14} className="text-slate-400 dark:text-slate-500" />
                        {new Date(d.created_at).toLocaleDateString()}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        {(!d.source_metadata?.ingestion_status?.status || d.source_metadata?.ingestion_status?.status === 'completed') && (
                          <>
                            <button
                              onClick={() => { setPreviewDataset(d); }}
                              className="p-2 text-slate-400 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-md transition-colors"
                              title="Preview Dataset"
                              aria-label="Preview dataset"
                            >
                              <Eye size={16} />
                            </button>
                            <button
                              onClick={() => { handleUseInCanvas(d.id); }}
                              className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-md transition-colors"
                              title="Use in Canvas"
                            >
                              <Play size={16} />
                              Canvas
                            </button>
                            <button
                              onClick={() => { navigate(`/eda?dataset_id=${d.id}`); }}
                              className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-purple-600 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/20 rounded-md transition-colors"
                              title="Analyze with EDA"
                            >
                              <BarChart2 size={16} />
                              EDA
                            </button>
                            <button
                              onClick={() => { void DatasetService.exportData(d.id, 'csv'); }}
                              className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-green-600 dark:text-green-400 hover:bg-green-50 dark:hover:bg-green-900/20 rounded-md transition-colors"
                              title="Download CSV"
                            >
                              <Download size={16} />
                              CSV
                            </button>
                          </>
                        )}
                        {(d.source_metadata?.ingestion_status?.status === 'pending' || d.source_metadata?.ingestion_status?.status === 'processing') && (
                          <button
                            onClick={() => { void handleCancel(d.id); }}
                            disabled={cancellingId === d.id}
                            className="p-2 text-slate-400 hover:text-orange-600 hover:bg-orange-50 dark:hover:bg-orange-900/20 rounded-md transition-colors disabled:opacity-50"
                            title="Cancel Ingestion"
                            aria-label="Cancel ingestion"
                          >
                            {cancellingId === d.id ? (
                              <div className="w-4 h-4 border-2 border-orange-600 border-t-transparent rounded-full animate-spin" />
                            ) : (
                              <XCircle size={16} />
                            )}
                          </button>
                        )}
                        <button
                          onClick={() => { void handleDelete(d.id); }}
                          disabled={deletingId === d.id}
                          className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md transition-colors disabled:opacity-50"
                          title="Delete Dataset"
                          aria-label="Delete dataset"
                        >
                          {deletingId === d.id ? (
                            <div className="w-4 h-4 border-2 border-red-600 border-t-transparent rounded-full animate-spin" />
                          ) : (
                            <Trash2 size={16} />
                          )}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
