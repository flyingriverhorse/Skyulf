import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { DatasetService } from '../core/api/datasets';
import { Dataset } from '../core/types/api';
import { FileUpload } from '../modules/nodes/data/FileUpload';
import { Trash2, Play, FileText, Calendar, Database, Plus, Eye } from 'lucide-react';
import { formatBytes } from '../core/utils/format';
import { DatasetPreviewModal } from '../components/data/DatasetPreviewModal';

export const DataSources: React.FC = () => {
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [previewDataset, setPreviewDataset] = useState<Dataset | null>(null);

  const fetchDatasets = async () => {
    setLoading(true);
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
    fetchDatasets();
  }, []);

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

  const handleUseInCanvas = (id: string) => {
    navigate(`/canvas?source_id=${id}`);
  };

  const handleUploadComplete = () => {
    setShowUpload(false);
    fetchDatasets();
  };

  return (
    <div className="p-8 space-y-6 max-w-7xl mx-auto">
      <DatasetPreviewModal 
        dataset={previewDataset} 
        isOpen={!!previewDataset} 
        onClose={() => setPreviewDataset(null)} 
      />

      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-100">Data Sources</h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">Manage your uploaded datasets and use them in experiments.</p>
        </div>
        <button
          onClick={() => setShowUpload(!showUpload)}
          className="flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
        >
          <Plus size={18} />
          {showUpload ? 'Cancel Upload' : 'Upload New Dataset'}
        </button>
      </div>

      {showUpload && (
        <div className="bg-white dark:bg-slate-800 p-6 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm animate-in fade-in slide-in-from-top-4">
          <FileUpload onUploadComplete={handleUploadComplete} onCancel={() => setShowUpload(false)} />
        </div>
      )}

      <div className="bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700 shadow-sm overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-slate-500 dark:text-slate-400 uppercase bg-slate-50 dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700">
              <tr>
                <th className="px-6 py-4 font-semibold">Dataset Name</th>
                <th className="px-6 py-4 font-semibold">Format</th>
                <th className="px-6 py-4 font-semibold">Size</th>
                <th className="px-6 py-4 font-semibold">Created</th>
                <th className="px-6 py-4 font-semibold text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-700 bg-white dark:bg-slate-900">
              {loading ? (
                <tr>
                  <td colSpan={5} className="px-6 py-8 text-center text-slate-500 dark:text-slate-400">
                    <div className="flex items-center justify-center gap-2">
                      <div className="animate-spin w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full" />
                      Loading datasets...
                    </div>
                  </td>
                </tr>
              ) : datasets.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-6 py-12 text-center text-slate-500 dark:text-slate-400">
                    <Database className="w-12 h-12 mx-auto text-slate-300 dark:text-slate-600 mb-3" />
                    <p className="text-lg font-medium text-slate-900 dark:text-slate-100">No datasets found</p>
                    <p className="text-sm">Upload a dataset to get started with your analysis.</p>
                  </td>
                </tr>
              ) : (
                datasets.map((d) => (
                  <tr key={d.id} className="hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 bg-blue-50 dark:bg-blue-900/20 rounded text-blue-600 dark:text-blue-400">
                          <FileText size={18} />
                        </div>
                        <div>
                          <div className="font-medium text-slate-900 dark:text-slate-100">{d.name}</div>
                          <div className="text-xs text-slate-500 dark:text-slate-400 font-mono mt-0.5">{d.id}</div>
                        </div>
                      </div>
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
                        <button
                          onClick={() => setPreviewDataset(d)}
                          className="p-2 text-slate-400 hover:text-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-md transition-colors"
                          title="Preview Dataset"
                        >
                          <Eye size={16} />
                        </button>
                        <button
                          onClick={() => handleUseInCanvas(d.id)}
                          className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-md transition-colors"
                          title="Use in Canvas"
                        >
                          <Play size={16} />
                          Canvas
                        </button>
                        <button
                          onClick={() => handleDelete(d.id)}
                          disabled={deletingId === d.id}
                          className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md transition-colors disabled:opacity-50"
                          title="Delete Dataset"
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
