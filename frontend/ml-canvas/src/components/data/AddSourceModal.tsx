import React, { useState } from 'react';
import { X, Database, Globe, Check, Cloud, ChevronDown, ChevronRight } from 'lucide-react';
import { DatasetService } from '../../core/api/datasets';
import { DataSourceCreate } from '../../core/types/api';

interface AddSourceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (jobId: string) => void;
}

export const AddSourceModal: React.FC<AddSourceModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [type, setType] = useState<'database' | 'api' | 's3'>('database');
  const [name, setName] = useState('');
  const [connectionString, setConnectionString] = useState('');
  const [query, setQuery] = useState('');
  const [apiUrl, setApiUrl] = useState('');
  const [method, setMethod] = useState('GET');
  const [s3Path, setS3Path] = useState('');
  const [showCredentials, setShowCredentials] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const config: Record<string, any> = {};
      if (type === 'database') {
        config.connection_string = connectionString;
        config.query = query;
      } else if (type === 's3') {
        config.path = s3Path;
        
        // Extract credentials from form
        const form = e.target as HTMLFormElement;
        const accessKey = (form.elements.namedItem('aws_access_key_id') as HTMLInputElement)?.value;
        const secretKey = (form.elements.namedItem('aws_secret_access_key') as HTMLInputElement)?.value;
        const region = (form.elements.namedItem('region_name') as HTMLInputElement)?.value;
        
        if (accessKey && secretKey) {
          config.storage_options = {
            key: accessKey,
            secret: secretKey,
            client_kwargs: region ? { region_name: region } : {}
          };
        }
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm p-4">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow-xl w-full max-w-md flex flex-col max-h-[90vh] border border-slate-200 dark:border-slate-700">
        <div className="flex justify-between items-center p-4 border-b border-slate-200 dark:border-slate-700 shrink-0">
          <h3 className="font-semibold text-lg text-slate-900 dark:text-slate-100">Add Data Source</h3>
          <button onClick={onClose} className="text-slate-500 hover:text-slate-700 dark:hover:text-slate-300">
            <X size={20} />
          </button>
        </div>

        <div className="p-4 overflow-y-auto">
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
              onClick={() => { setType('s3'); }}
              className={`flex-1 py-2 px-4 rounded-md flex items-center justify-center gap-2 border ${
                type === 's3'
                  ? 'bg-blue-50 border-blue-500 text-blue-700 dark:bg-blue-900/20 dark:border-blue-400 dark:text-blue-400'
                  : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400'
              }`}
            >
              <Cloud size={16} /> S3
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

            {type === 'database' && (
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
            )}

            {type === 's3' && (
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">S3 Path</label>
                  <input
                    type="text"
                    required
                    value={s3Path}
                    onChange={(e) => { setS3Path(e.target.value); }}
                    className="w-full px-3 py-2 border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500"
                    placeholder="s3://my-bucket/path/to/data.parquet"
                  />
                </div>
                
                <div className="border border-slate-200 dark:border-slate-700 rounded-md overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setShowCredentials(!showCredentials)}
                    className="w-full flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                  >
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300">Credentials (Optional)</h4>
                    </div>
                    {showCredentials ? <ChevronDown size={16} className="text-slate-500" /> : <ChevronRight size={16} className="text-slate-500" />}
                  </button>
                  
                  {showCredentials && (
                    <div className="p-3 bg-slate-50 dark:bg-slate-800/50 border-t border-slate-200 dark:border-slate-700">
                      <p className="text-xs text-slate-500 mb-3">
                        If your bucket is private, provide credentials here. They will be stored securely.
                        Leave blank if using backend's IAM role.
                      </p>
                      
                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Access Key ID</label>
                          <input
                            type="text"
                            name="aws_access_key_id"
                            className="w-full px-2 py-1.5 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"
                            placeholder="AKIA..."
                          />
                        </div>
                        <div>
                          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Secret Access Key</label>
                          <input
                            type="password"
                            name="aws_secret_access_key"
                            className="w-full px-2 py-1.5 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"
                            placeholder="Secret..."
                          />
                        </div>
                        <div className="col-span-2">
                          <label className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Region</label>
                          <input
                            type="text"
                            name="region_name"
                            className="w-full px-2 py-1.5 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"
                            placeholder="us-east-1"
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {type === 'api' && (
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
