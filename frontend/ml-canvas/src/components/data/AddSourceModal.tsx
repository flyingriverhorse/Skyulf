import React, { useState } from 'react';
import { X, Database, Globe, Check } from 'lucide-react';
import { DatasetService } from '../../core/api/datasets';
import { DataSourceCreate } from '../../core/types/api';

interface AddSourceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (jobId: string) => void;
}

export const AddSourceModal: React.FC<AddSourceModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [type, setType] = useState<'database' | 'api'>('database');
  const [name, setName] = useState('');
  const [connectionString, setConnectionString] = useState('');
  const [query, setQuery] = useState('');
  const [apiUrl, setApiUrl] = useState('');
  const [method, setMethod] = useState('GET');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const config: Record<string, string> = {};
      if (type === 'database') {
        config.connection_string = connectionString;
        config.query = query;
      } else {
        config.url = apiUrl;
        config.method = method;
      }

      const payload: DataSourceCreate = {
        name,
        type,
        config,
        description: `Imported from ${type}`
      };

      const response = await DatasetService.createSource(payload);
      onSuccess(response.job_id);
      onClose();
    } catch (err: unknown) {
      setError((err as Error).message || 'Failed to create source');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow-xl w-full max-w-md overflow-hidden border border-slate-200 dark:border-slate-700">
        <div className="flex justify-between items-center p-4 border-b border-slate-200 dark:border-slate-700">
          <h3 className="font-semibold text-lg text-slate-900 dark:text-slate-100">Add Data Source</h3>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-700 dark:hover:text-slate-300">
            <X size={20} />
          </button>
        </div>

        <div className="p-4">
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => { setType('database'); }}
              className={`flex-1 py-2 px-4 rounded-md flex items-center justify-center gap-2 border ${
                type === 'database'
                  ? 'bg-blue-50 border-blue-500 text-blue-700 dark:bg-blue-900/20 dark:border-blue-400 dark:text-blue-400'
                  : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400'
              }`}
            >
              <Database size={16} /> Database
            </button>
            <button
              onClick={() => { setType('api'); }}
              className={`flex-1 py-2 px-4 rounded-md flex items-center justify-center gap-2 border ${
                type === 'api'
                  ? 'bg-blue-50 border-blue-500 text-blue-700 dark:bg-blue-900/20 dark:border-blue-400 dark:text-blue-400'
                  : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400'
              }`}
            >
              <Globe size={16} /> API
            </button>
          </div>

          <form onSubmit={(e) => { void handleSubmit(e); }} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Name</label>
              <input
                type="text"
                required
                value={name}
                onChange={(e) => { setName(e.target.value); }}
                className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500"
                placeholder="My Dataset"
              />
            </div>

            {type === 'database' ? (
              <>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Connection String</label>
                  <input
                    type="text"
                    required
                    value={connectionString}
                    onChange={(e) => { setConnectionString(e.target.value); }}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500"
                    placeholder="postgresql://user:pass@localhost:5432/db"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">SQL Query</label>
                  <textarea
                    required
                    value={query}
                    onChange={(e) => { setQuery(e.target.value); }}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500 h-24"
                    placeholder="SELECT * FROM users"
                  />
                </div>
              </>
            ) : (
              <>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">API URL</label>
                  <input
                    type="url"
                    required
                    value={apiUrl}
                    onChange={(e) => { setApiUrl(e.target.value); }}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500"
                    placeholder="https://api.example.com/data"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Method</label>
                  <select
                    value={method}
                    onChange={(e) => { setMethod(e.target.value); }}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="GET">GET</option>
                    <option value="POST">POST</option>
                  </select>
                </div>
              </>
            )}

            {error && (
              <div className="p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm rounded-md">
                {error}
              </div>
            )}

            <div className="flex justify-end gap-3 mt-6">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 rounded-md"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={loading}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md flex items-center gap-2 disabled:opacity-50"
              >
                {loading ? 'Creating...' : <><Check size={16} /> Create Source</>}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};
