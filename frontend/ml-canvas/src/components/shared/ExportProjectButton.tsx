import { useId, useState } from 'react';
import { Download, Loader2 } from 'lucide-react';
import { deploymentApi, type ProjectExportMode } from '../../core/api/deployment';

interface ExportProjectButtonProps {
  jobId: string;
  modelName: string;
  version: string | number;
}

/** Download the selected fitted version and release the temporary object URL. */
function downloadProject(archive: Blob, jobId: string): void {
  const url = URL.createObjectURL(archive);
  const link = document.createElement('a');
  link.href = url;
  link.download = `kayra_model_${jobId.replace(/[^a-zA-Z0-9_-]/g, '_')}.zip`;
  document.body.appendChild(link);
  try {
    link.click();
  } finally {
    link.remove();
    // Let the browser start the download before releasing its backing object.
    window.setTimeout(() => { URL.revokeObjectURL(url); }, 0);
  }
}

/** Export one completed registry version without changing its deployment. */
export function ExportProjectButton({ jobId, modelName, version }: ExportProjectButtonProps) {
  const [mode, setMode] = useState<ProjectExportMode>('both');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const errorId = useId();
  const identity = `${modelName} v${version}`;

  const exportProject = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      downloadProject(await deploymentApi.exportProject(jobId, mode), jobId);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Unable to export this project. Please try again.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mt-3 space-y-2 text-left" role="group" aria-label={`Project export for ${identity}`}>
      <div className="flex flex-wrap items-center justify-end gap-2">
        <select
          aria-label={`Project format for ${identity}`}
          value={mode}
          onChange={(event) => { setMode(event.target.value as ProjectExportMode); }}
          disabled={busy}
          className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-700 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-200"
        >
          <option value="both">API + batch</option>
          <option value="api">API only</option>
          <option value="batch">Batch only</option>
        </select>
        <button
          type="button"
          onClick={() => { void exportProject(); }}
          disabled={busy}
          aria-label={`Export project for ${identity}`}
          aria-busy={busy}
          aria-describedby={error ? errorId : undefined}
          className="inline-flex items-center gap-1.5 rounded border border-slate-300 px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:bg-slate-50 focus:ring-2 focus:ring-blue-500 disabled:opacity-50 dark:border-slate-600 dark:text-slate-200 dark:hover:bg-slate-700"
        >
          {busy ? <Loader2 size={14} className="animate-spin" aria-hidden="true" /> : <Download size={14} aria-hidden="true" />}
          {busy ? 'Exporting…' : 'Export project'}
        </button>
      </div>
      {error && (
        <p id={errorId} role="alert" className="max-w-xs whitespace-normal text-xs font-normal text-red-600 dark:text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}
