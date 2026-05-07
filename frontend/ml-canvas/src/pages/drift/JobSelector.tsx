import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, Database, Search } from 'lucide-react';
import type { DriftJobOption } from '../../core/api/monitoring';

interface JobSelectorProps {
    jobs: DriftJobOption[];
    selectedJob: string;
    onSelect: (jobId: string) => void;
}

/**
 * Combobox that lists training jobs grouped by date. Acts as the "reference
 * model" picker for drift comparison. Self-contained: owns its own search
 * input, open/close state, and outside-click handler.
 */
export const JobSelector: React.FC<JobSelectorProps> = ({ jobs, selectedJob, onSelect }) => {
    const [open, setOpen] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const dropdownRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
                setOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const filteredJobs = useMemo(() => {
        if (!searchTerm) return jobs;
        const lower = searchTerm.toLowerCase();
        return jobs.filter(
            j =>
                j.dataset_name.toLowerCase().includes(lower) ||
                j.job_id.toLowerCase().includes(lower) ||
                j.description?.toLowerCase().includes(lower) ||
                j.model_type?.toLowerCase().includes(lower) ||
                j.target_column?.toLowerCase().includes(lower),
        );
    }, [searchTerm, jobs]);

    const groupedJobs = useMemo(() => {
        const groups: Record<string, DriftJobOption[]> = {};
        for (const job of filteredJobs) {
            const date = job.created_at?.split(' ')[0] || 'Unknown';
            if (!groups[date]) groups[date] = [];
            groups[date].push(job);
        }
        return Object.entries(groups).sort(([a], [b]) => b.localeCompare(a));
    }, [filteredJobs]);

    const selectedJobData = useMemo(
        () => jobs.find(j => j.job_id === selectedJob),
        [jobs, selectedJob],
    );

    return (
        <div ref={dropdownRef} className="relative flex-1 min-w-0">
            <button
                type="button"
                onClick={() => {
                    setOpen(!open);
                    setSearchTerm('');
                }}
                className={`w-full flex items-center gap-2 px-3 py-2.5 border rounded-md text-sm transition-colors ${
                    selectedJob
                        ? 'border-blue-300 dark:border-blue-700 bg-blue-50/50 dark:bg-blue-900/20'
                        : 'border-gray-200 dark:border-slate-600 hover:border-gray-300 dark:hover:border-slate-500'
                }`}
            >
                <Database size={15} className="shrink-0 text-gray-400" />
                <span className={`truncate ${selectedJob ? 'text-slate-800 dark:text-slate-200' : 'text-gray-400'}`}>
                    {selectedJobData
                        ? `${selectedJobData.dataset_name}  (${selectedJobData.job_id.slice(0, 8)})`
                        : 'Select reference model...'}
                </span>
                <ChevronDown
                    size={14}
                    className={`ml-auto shrink-0 text-gray-400 transition-transform ${open ? 'rotate-180' : ''}`}
                />
            </button>

            {open && (
                <div className="absolute z-50 left-0 right-0 mt-1 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-600 rounded-md shadow-lg overflow-hidden">
                    <div className="p-2 border-b border-gray-100 dark:border-slate-700">
                        <div className="relative">
                            <Search
                                className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none z-10"
                                size={14}
                            />
                            <input
                                type="text"
                                // eslint-disable-next-line jsx-a11y/no-autofocus -- focus the search field on dropdown open
                                autoFocus
                                placeholder="Search jobs..."
                                value={searchTerm}
                                onChange={e => setSearchTerm(e.target.value)}
                                className="w-full pl-8 pr-3 py-1.5 text-sm border border-gray-200 dark:border-slate-600 rounded bg-gray-50 dark:bg-slate-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                            />
                        </div>
                    </div>
                    <div className="max-h-72 overflow-y-auto">
                        {filteredJobs.length === 0 ? (
                            <div className="p-4 text-center text-gray-400 text-sm">No jobs found</div>
                        ) : (
                            groupedJobs.map(([date, dateJobs]) => (
                                <div key={date}>
                                    <div className="sticky top-0 px-3 py-1.5 text-[11px] font-semibold text-gray-400 dark:text-slate-500 bg-gray-50 dark:bg-slate-900/80 uppercase tracking-wider border-b border-gray-100 dark:border-slate-700">
                                        {date}
                                    </div>
                                    {dateJobs.map(job => (
                                        <button
                                            key={job.job_id}
                                            type="button"
                                            onClick={() => {
                                                onSelect(job.job_id);
                                                setOpen(false);
                                            }}
                                            className={`w-full text-left px-3 py-2.5 text-sm transition-colors ${
                                                selectedJob === job.job_id
                                                    ? 'bg-blue-50 dark:bg-blue-900/30'
                                                    : 'hover:bg-gray-50 dark:hover:bg-slate-700/50'
                                            }`}
                                        >
                                            <div className="flex items-center gap-2">
                                                <span className="font-medium truncate">{job.dataset_name}</span>
                                                <span className="text-[11px] text-gray-400 font-mono shrink-0">
                                                    {job.job_id.slice(0, 8)}
                                                </span>
                                                {job.model_type && (
                                                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-100 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-300 shrink-0">
                                                        {job.model_type}
                                                    </span>
                                                )}
                                                {job.best_metric && (
                                                    <span
                                                        className="text-[10px] px-1.5 py-0.5 rounded bg-green-100 dark:bg-green-900/40 text-green-600 dark:text-green-300 shrink-0 ml-auto tabular-nums truncate max-w-[140px]"
                                                        title={job.best_metric}
                                                    >
                                                        {job.best_metric.split(' | ')[0]}
                                                    </span>
                                                )}
                                            </div>
                                            {(job.target_column || job.description) && (
                                                <div className="flex items-center gap-2 mt-0.5 text-[11px] text-gray-400 dark:text-slate-500">
                                                    {job.target_column && <span>target: {job.target_column}</span>}
                                                    {job.description && (
                                                        <span className="truncate italic">— {job.description}</span>
                                                    )}
                                                </div>
                                            )}
                                        </button>
                                    ))}
                                </div>
                            ))
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};
