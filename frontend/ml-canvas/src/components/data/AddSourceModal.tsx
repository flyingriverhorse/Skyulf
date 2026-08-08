import React, { useState } from 'react';
import { Check, Cloud, ChevronDown, ChevronRight } from 'lucide-react';
import { useCreateDataSource } from '../../core/hooks/useDatasets';
import { DataSourceCreate } from '../../core/types/api';
import { ModalShell } from '../shared';

interface AddSourceModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (jobId: string) => void;
}

export const AddSourceModal: React.FC<AddSourceModalProps> = ({ isOpen, onClose, onSuccess }) => {
  const [type, setType] = useState<'s3'>('s3');
  const [name, setName] = useState('');
  const [s3Path, setS3Path] = useState('');
  const [showCredentials, setShowCredentials] = useState(false);
  const [accessKeyId, setAccessKeyId] = useState('');
  const [secretAccessKey, setSecretAccessKey] = useState('');
  const [regionName, setRegionName] = useState('');
  const [error, setError] = useState<string | null>(null);
  // Tracks whether the user has attempted a submit, so required-field
  // errors only appear after a real attempt rather than as the user is
  // still typing their first character.
  const [submitAttempted, setSubmitAttempted] = useState(false);

  // Mutation owns the loading state + invalidates the dataset list cache on success.
  const createMutation = useCreateDataSource();
  const loading = createMutation.isPending;

  const nameMissing = submitAttempted && name.trim() === '';
  const s3PathMissing = submitAttempted && type === 's3' && s3Path.trim() === '';

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSubmitAttempted(true);

    if (name.trim() === '' || (type === 's3' && s3Path.trim() === '')) {
      return;
    }

    try {
      const config: Record<string, unknown> = {};
      if (type === 's3') {
        config.path = s3Path;

        if (accessKeyId && secretAccessKey) {
          config.storage_options = {
            aws_access_key_id: accessKeyId,
            aws_secret_access_key: secretAccessKey,
            region: regionName || undefined
          };
        }
      }

      const payload: DataSourceCreate = {
        name,
        type,
        config,
        description: `Imported from ${type}`
      };

      const response = await createMutation.mutateAsync(payload);
      onSuccess(response.job_id);
      onClose();
    } catch (err: unknown) {
      setError((err as Error).message || 'Failed to create source');
    }
  };

  return (
    <ModalShell isOpen={isOpen} onClose={onClose} title="Add Data Source" size="md">
      <div className="p-4">
          <p className="text-xs text-slate-500 dark:text-slate-400 mb-4">
            Connects a remote source that Skyulf re-reads on demand. Amazon S3 is the only
            connected source type today — for a local file, use{' '}
            <span className="font-medium text-slate-700 dark:text-slate-300">Upload File</span> on
            the Data Sources page instead.
          </p>
          <div className="flex gap-2 mb-6">
            <button
              onClick={() => { setType('s3'); }}
              aria-pressed={type === 's3'}
              className={`flex-1 py-2 px-4 rounded-md flex items-center justify-center gap-2 border ${
                type === 's3'
                  ? 'bg-blue-50 border-blue-500 text-blue-700 dark:bg-blue-900/20 dark:border-blue-400 dark:text-blue-400'
                  : 'border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-400'
              }`}
            >
              <Cloud size={16} /> S3
            </button>
          </div>

          <form onSubmit={(e) => { void handleSubmit(e); }} className="space-y-4" noValidate>
            <div>
              <label htmlFor="add-source-name" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                Name
              </label>
              <input
                id="add-source-name"
                type="text"
                required
                value={name}
                onChange={(e) => { setName(e.target.value); }}
                aria-invalid={nameMissing}
                aria-describedby={nameMissing ? 'add-source-name-error' : undefined}
                className={`w-full px-3 py-2 border rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500 ${
                  nameMissing
                    ? 'border-red-400 dark:border-red-600'
                    : 'border-slate-300 dark:border-slate-600'
                }`}
                placeholder="My Dataset"
              />
              {nameMissing && (
                <p id="add-source-name-error" role="alert" className="mt-1 text-xs text-red-600 dark:text-red-400">
                  Name is required.
                </p>
              )}
            </div>

            {type === 's3' && (
              <div className="space-y-4">
                <div>
                  <label htmlFor="add-source-s3-path" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                    S3 Path
                  </label>
                  <input
                    id="add-source-s3-path"
                    type="text"
                    required
                    value={s3Path}
                    onChange={(e) => { setS3Path(e.target.value); }}
                    aria-invalid={s3PathMissing}
                    aria-describedby={s3PathMissing ? 'add-source-s3-path-error' : undefined}
                    className={`w-full px-3 py-2 border rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:ring-2 focus:ring-blue-500 ${
                      s3PathMissing
                        ? 'border-red-400 dark:border-red-600'
                        : 'border-slate-300 dark:border-slate-600'
                    }`}
                    placeholder="s3://my-bucket/path/to/data.parquet"
                  />
                  {s3PathMissing && (
                    <p id="add-source-s3-path-error" role="alert" className="mt-1 text-xs text-red-600 dark:text-red-400">
                      S3 Path is required.
                    </p>
                  )}
                </div>

                <div className="border border-slate-200 dark:border-slate-700 rounded-md overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setShowCredentials(!showCredentials)}
                    aria-expanded={showCredentials}
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
                        Leave blank if using backend&apos;s IAM role.
                      </p>

                      <div className="grid grid-cols-2 gap-3">
                        <div>
                          <label htmlFor="add-source-access-key" className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                            Access Key ID
                          </label>
                          <input
                            id="add-source-access-key"
                            type="text"
                            name="aws_access_key_id"
                            value={accessKeyId}
                            onChange={(e) => { setAccessKeyId(e.target.value); }}
                            className="w-full px-2 py-1.5 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"
                            placeholder="AKIA..."
                          />
                        </div>
                        <div>
                          <label htmlFor="add-source-secret-key" className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                            Secret Access Key
                          </label>
                          <input
                            id="add-source-secret-key"
                            type="password"
                            name="aws_secret_access_key"
                            value={secretAccessKey}
                            onChange={(e) => { setSecretAccessKey(e.target.value); }}
                            className="w-full px-2 py-1.5 text-sm border border-slate-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100"
                            placeholder="Secret..."
                          />
                        </div>
                        <div className="col-span-2">
                          <label htmlFor="add-source-region" className="block text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">
                            Region
                          </label>
                          <input
                            id="add-source-region"
                            type="text"
                            name="region_name"
                            value={regionName}
                            onChange={(e) => { setRegionName(e.target.value); }}
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

            {error && (
              <div role="alert" className="p-3 bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm rounded-md">
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
    </ModalShell>
  );
};
