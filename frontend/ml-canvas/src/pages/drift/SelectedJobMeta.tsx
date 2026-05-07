import React, { useEffect, useState } from 'react';
import { Check, Pencil, X } from 'lucide-react';
import type { DriftJobOption } from '../../core/api/monitoring';

interface SelectedJobMetaProps {
    job: DriftJobOption;
    onUpdateDescription: (description: string) => Promise<void> | void;
}

/**
 * Metadata badges (model type, target, feature/row counts, best metric) plus
 * inline editor for the job description. Hides itself when no job is selected.
 */
export const SelectedJobMeta: React.FC<SelectedJobMetaProps> = ({ job, onUpdateDescription }) => {
    const [editing, setEditing] = useState(false);
    const [draft, setDraft] = useState('');

    // Cancel any in-progress edit when a different job is selected.
    useEffect(() => {
        setEditing(false);
    }, [job.job_id]);

    const handleSave = async () => {
        await onUpdateDescription(draft);
        setEditing(false);
    };

    return (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 pb-3 text-xs text-gray-500 dark:text-slate-400">
            {job.model_type && (
                <span className="px-2 py-0.5 rounded bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-300 font-medium">
                    {job.model_type}
                </span>
            )}
            {job.target_column && (
                <span>
                    Target:{' '}
                    <span className="font-medium text-slate-600 dark:text-slate-300">{job.target_column}</span>
                </span>
            )}
            {job.n_features != null && <span>{job.n_features} features</span>}
            {job.n_rows != null && <span>{job.n_rows.toLocaleString()} rows</span>}
            {job.best_metric && (
                <span className="px-2 py-0.5 rounded bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-300 font-medium tabular-nums">
                    {job.best_metric}
                </span>
            )}
            <span className="border-l border-gray-200 dark:border-slate-700 h-3" />
            {editing ? (
                <div className="flex items-center gap-1.5 flex-1 min-w-0">
                    <input
                        // eslint-disable-next-line jsx-a11y/no-autofocus -- focus the description field when entering edit mode
                        autoFocus
                        value={draft}
                        onChange={e => setDraft(e.target.value)}
                        onKeyDown={e => {
                            if (e.key === 'Enter') void handleSave();
                            if (e.key === 'Escape') setEditing(false);
                        }}
                        placeholder="Add a description..."
                        className="flex-1 min-w-0 px-2 py-0.5 text-xs border border-gray-200 dark:border-slate-600 rounded bg-white dark:bg-slate-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                    <button onClick={() => void handleSave()} className="text-green-500 hover:text-green-600">
                        <Check size={13} />
                    </button>
                    <button onClick={() => setEditing(false)} className="text-gray-400 hover:text-red-500">
                        <X size={13} />
                    </button>
                </div>
            ) : (
                <button
                    onClick={() => {
                        setDraft(job.description || '');
                        setEditing(true);
                    }}
                    className="flex items-center gap-1 hover:text-blue-500 transition-colors group"
                >
                    <Pencil size={11} className="opacity-50 group-hover:opacity-100" />
                    <span className={job.description ? '' : 'italic text-gray-400'}>
                        {job.description || 'Add description...'}
                    </span>
                </button>
            )}
        </div>
    );
};
