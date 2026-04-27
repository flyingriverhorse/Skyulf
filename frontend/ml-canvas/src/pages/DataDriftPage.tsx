import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { monitoringApi, DriftReport, DriftJobOption, DriftHistoryEntry, DriftThresholds } from '../core/api/monitoring';
import { Loader2, Upload, AlertTriangle, CheckCircle, XCircle, Lightbulb, RefreshCw, Search, Database, ChevronDown, ChevronUp, BarChart2, Info, FileUp, X, Pencil, Check, ShieldAlert, Shield, TrendingUp, Filter, ArrowUpDown, ArrowUp, ArrowDown, Columns, Activity, Target, Gauge, Download, Settings } from 'lucide-react';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';

const MetricTooltip = ({ label, tooltip, icon }: { label: string, tooltip: string, icon?: React.ReactNode }) => (
    <div className="group relative flex items-center gap-1 cursor-help">
        <span className="border-b border-dotted border-gray-400">{label}</span>
        {icon && <span className="text-gray-400">{icon}</span>}
        <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-64 p-2 bg-slate-900 text-white text-xs rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 text-center pointer-events-none">
            {tooltip}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 -mb-1 border-4 border-transparent border-b-slate-900"></div>
        </div>
    </div>
);

const Sparkline = ({ values, width = 64, height = 20 }: { values: number[], width?: number, height?: number }) => {
    if (values.length < 2) return <span className="text-[10px] text-gray-400">—</span>;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min || 1;
    const points = values.map((v, i) => {
        const x = (i / (values.length - 1)) * width;
        const y = height - ((v - min) / range) * (height - 4) - 2;
        return `${x},${y}`;
    }).join(' ');
    const last = values[values.length - 1] ?? 0;
    const color = last > 0.2 ? '#ef4444' : last > 0.1 ? '#f59e0b' : '#22c55e';
    return (
        <svg width={width} height={height} className="inline-block">
            <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" />
            <circle cx={(values.length - 1) / (values.length - 1) * width} cy={height - ((last - min) / range) * (height - 4) - 2} r="2" fill={color} />
        </svg>
    );
};

export const DataDriftPage: React.FC = () => {
    const [jobs, setJobs] = useState<DriftJobOption[]>([]);
    const [filteredJobs, setFilteredJobs] = useState<DriftJobOption[]>([]);
    const [selectedJob, setSelectedJob] = useState<string>('');
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [report, setReport] = useState<DriftReport | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [refreshing, setRefreshing] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [expandedRows, setExpandedRows] = useState<Record<string, boolean>>({});
    const [jobDropdownOpen, setJobDropdownOpen] = useState(false);
    const dropdownRef = useRef<HTMLDivElement>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [editingDescription, setEditingDescription] = useState(false);
    const [descriptionDraft, setDescriptionDraft] = useState('');
    const [driftHistory, setDriftHistory] = useState<DriftHistoryEntry[]>([]);
    const [showOnlyDrifted, setShowOnlyDrifted] = useState(false);
    const [sortConfig, setSortConfig] = useState<{ key: string; dir: 'asc' | 'desc' } | null>(null);
    const [showThresholds, setShowThresholds] = useState(false);
    const [thresholds, setThresholds] = useState<DriftThresholds>({ psi: 0.2, ks: 0.05, wasserstein: 0.1, kl: 0.1 });

    const fetchJobs = useCallback(async () => {
        setRefreshing(true);
        try {
            const data = await monitoringApi.getJobs();
            setJobs(data);
            setFilteredJobs(data);
        } catch (err) {
            console.error("Failed to fetch jobs", err);
        } finally {
            setRefreshing(false);
        }
    }, []);

    useEffect(() => {
        fetchJobs();
    }, [fetchJobs]);

    useEffect(() => {
        if (searchTerm) {
            const lower = searchTerm.toLowerCase();
            setFilteredJobs(jobs.filter(j => 
                j.dataset_name.toLowerCase().includes(lower) || 
                j.job_id.toLowerCase().includes(lower) ||
                j.description?.toLowerCase().includes(lower) ||
                j.model_type?.toLowerCase().includes(lower) ||
                j.target_column?.toLowerCase().includes(lower)
            ));
        } else {
            setFilteredJobs(jobs);
        }
    }, [searchTerm, jobs]);

    useEffect(() => {
        const handleClickOutside = (e: MouseEvent) => {
            if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
                setJobDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    useEffect(() => {
        const handleEsc = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setExpandedRows({});
        };
        document.addEventListener('keydown', handleEsc);
        return () => document.removeEventListener('keydown', handleEsc);
    }, []);

    const selectedJobData = useMemo(() => jobs.find(j => j.job_id === selectedJob), [jobs, selectedJob]);

    useEffect(() => { setEditingDescription(false); }, [selectedJob]);

    useEffect(() => {
        if (selectedJob) {
            monitoringApi.getDriftHistory(selectedJob).then(setDriftHistory).catch(() => setDriftHistory([]));
        } else {
            setDriftHistory([]);
        }
    }, [selectedJob]);

    const handleSaveDescription = async () => {
        if (!selectedJob) return;
        try {
            await monitoringApi.updateJobDescription(selectedJob, descriptionDraft);
            setJobs(prev => prev.map(j => j.job_id === selectedJob ? { ...j, description: descriptionDraft } : j));
            setEditingDescription(false);
        } catch (err) {
            console.error("Failed to save description", err);
        }
    };

    const groupedJobs = useMemo(() => {
        const groups: Record<string, DriftJobOption[]> = {};
        for (const job of filteredJobs) {
            const date = job.created_at?.split(' ')[0] || 'Unknown';
            if (!groups[date]) groups[date] = [];
            groups[date].push(job);
        }
        return Object.entries(groups).sort(([a], [b]) => b.localeCompare(a));
    }, [filteredJobs]);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const handleCalculate = async () => {
        if (!selectedJob || !file) {
            setError("Please select a Reference Job and upload Current Data.");
            return;
        }
        setLoading(true);
        setError(null);
        setExpandedRows({});
        try {
            // Find dataset name from selected job
            const job = jobs.find(j => j.job_id === selectedJob);
            const result = await monitoringApi.calculateDrift(selectedJob, file, job?.dataset_name, thresholds);
            setReport(result);
            // Refresh drift history
            monitoringApi.getDriftHistory(selectedJob).then(setDriftHistory).catch(() => {});
        } catch (err: unknown) {
            const detail =
                err && typeof err === 'object' && 'response' in err
                    ? ((err as { response?: { data?: { detail?: string } } }).response?.data?.detail)
                    : undefined;
            setError(detail || "Failed to calculate drift.");
        } finally {
            setLoading(false);
        }
    };

    const toggleRow = (col: string) => {
        setExpandedRows(prev => ({ ...prev, [col]: !prev[col] }));
    };

    const handleSort = (key: string) => {
        setSortConfig(prev => {
            if (prev?.key === key) {
                if (prev.dir === 'asc') return { key, dir: 'desc' };
                return null; // third click clears
            }
            return { key, dir: 'asc' };
        });
    };

    const renderSortIcon = (col: string) => {
        if (sortConfig?.key !== col) return <ArrowUpDown size={11} className="opacity-30" />;
        return sortConfig.dir === 'asc' ? <ArrowUp size={11} /> : <ArrowDown size={11} />;
    };

    // #15 — Client-side threshold re-evaluation
    const evaluatedReport = useMemo(() => {
        if (!report) return null;
        const t = thresholds;
        const newDrifts: Record<string, typeof report.column_drifts[string]> = {};
        let driftedCount = 0;
        for (const [colName, col] of Object.entries(report.column_drifts)) {
            const newMetrics = col.metrics.map(m => {
                let hasDrift = m.has_drift;
                if (m.metric === 'psi' && t.psi != null) hasDrift = m.value > t.psi;
                if (m.metric === 'ks_test_p_value' && t.ks != null) hasDrift = m.value < t.ks;
                if (m.metric === 'wasserstein_distance' && t.wasserstein != null) hasDrift = m.value > t.wasserstein;
                if (m.metric === 'kl_divergence' && t.kl != null) hasDrift = m.value > t.kl;
                return { ...m, has_drift: hasDrift };
            });
            const drifted = newMetrics.some(m => m.has_drift);
            if (drifted) driftedCount++;
            newDrifts[colName] = { ...col, metrics: newMetrics, drift_detected: drifted };
        }
        return { ...report, column_drifts: newDrifts, drifted_columns_count: driftedCount };
    }, [report, thresholds]);

    // #14 — Build per-column PSI sparkline data from drift history
    const columnSparklines = useMemo(() => {
        if (driftHistory.length < 2) return {};
        const reversed = [...driftHistory].reverse(); // oldest first
        const result: Record<string, number[]> = {};
        for (const entry of reversed) {
            if (!entry.summary) continue;
            for (const [col, data] of Object.entries(entry.summary)) {
                if (!result[col]) result[col] = [];
                result[col].push(data.psi ?? 0);
            }
        }
        return result;
    }, [driftHistory]);

    const exportCSV = useCallback(() => {
        if (!evaluatedReport) return;
        const fi = evaluatedReport.feature_importances;
        const headers = ['Column', 'Status', 'Wasserstein', 'PSI', 'KL Divergence', 'KS P-Value', ...(fi ? ['Importance', 'Risk'] : [])];
        const rows = Object.values(evaluatedReport.column_drifts).map(col => {
            const get = (m: string) => col.metrics.find(x => x.metric === m)?.value?.toFixed(6) ?? '';
            const importance = fi?.[col.column];
            const rank = fi ? Object.values(fi).filter(v => v > (importance ?? 0)).length + 1 : null;
            const risk = col.drift_detected && rank != null ? (rank <= 5 ? 'High' : rank <= 15 ? 'Medium' : 'Low') : (fi ? 'Low' : '');
            return [
                col.column,
                col.drift_detected ? 'Drifted' : 'Stable',
                get('wasserstein_distance'),
                get('psi'),
                get('kl_divergence'),
                get('ks_test_p_value'),
                ...(fi ? [importance?.toFixed(6) ?? '', risk] : []),
            ];
        });
        const csv = [headers, ...rows].map(r => r.map(c => `"${c}"`).join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `drift_report_${selectedJobData?.dataset_name ?? 'export'}_${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    }, [evaluatedReport, selectedJobData]);

    return (
        <div className="p-6 w-full text-slate-900 dark:text-slate-100">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold">Data Drift Analysis</h1>
            </div>
            
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow mb-6 border dark:border-slate-700">
                {/* Compact toolbar */}
                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 p-4">

                    {/* Job combobox */}
                    <div ref={dropdownRef} className="relative flex-1 min-w-0">
                        <button
                            type="button"
                            onClick={() => { setJobDropdownOpen(!jobDropdownOpen); setSearchTerm(''); }}
                            className={`w-full flex items-center gap-2 px-3 py-2.5 border rounded-md text-sm transition-colors ${
                                selectedJob
                                    ? 'border-blue-300 dark:border-blue-700 bg-blue-50/50 dark:bg-blue-900/20'
                                    : 'border-gray-200 dark:border-slate-600 hover:border-gray-300 dark:hover:border-slate-500'
                            }`}
                        >
                            <Database size={15} className="shrink-0 text-gray-400" />
                            <span className={`truncate ${selectedJob ? 'text-slate-800 dark:text-slate-200' : 'text-gray-400'}`}>
                                {selectedJobData ? `${selectedJobData.dataset_name}  (${selectedJobData.job_id.slice(0, 8)})` : 'Select reference model...'}
                            </span>
                            <ChevronDown size={14} className={`ml-auto shrink-0 text-gray-400 transition-transform ${jobDropdownOpen ? 'rotate-180' : ''}`} />
                        </button>

                        {jobDropdownOpen && (
                            <div className="absolute z-50 left-0 right-0 mt-1 bg-white dark:bg-slate-800 border border-gray-200 dark:border-slate-600 rounded-md shadow-lg overflow-hidden">
                                <div className="p-2 border-b border-gray-100 dark:border-slate-700">
                                    <div className="relative">
                                        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none z-10" size={14} />
                                        <input
                                            type="text"
                                            // eslint-disable-next-line jsx-a11y/no-autofocus -- focus the search field on dropdown open
                                            autoFocus
                                            placeholder="Search jobs..."
                                            value={searchTerm}
                                            onChange={(e) => setSearchTerm(e.target.value)}
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
                                                        onClick={() => { setSelectedJob(job.job_id); setJobDropdownOpen(false); }}
                                                        className={`w-full text-left px-3 py-2.5 text-sm transition-colors ${
                                                            selectedJob === job.job_id
                                                                ? 'bg-blue-50 dark:bg-blue-900/30'
                                                                : 'hover:bg-gray-50 dark:hover:bg-slate-700/50'
                                                        }`}
                                                    >
                                                        <div className="flex items-center gap-2">
                                                            <span className="font-medium truncate">{job.dataset_name}</span>
                                                            <span className="text-[11px] text-gray-400 font-mono shrink-0">{job.job_id.slice(0, 8)}</span>
                                                            {job.model_type && (
                                                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-indigo-100 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-300 shrink-0">{job.model_type}</span>
                                                            )}
                                                            {job.best_metric && (
                                                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-100 dark:bg-green-900/40 text-green-600 dark:text-green-300 shrink-0 ml-auto tabular-nums truncate max-w-[140px]" title={job.best_metric}>
                                                                    {job.best_metric.split(' | ')[0]}
                                                                </span>
                                                            )}
                                                        </div>
                                                        {(job.target_column || job.description) && (
                                                            <div className="flex items-center gap-2 mt-0.5 text-[11px] text-gray-400 dark:text-slate-500">
                                                                {job.target_column && <span>target: {job.target_column}</span>}
                                                                {job.description && <span className="truncate italic">— {job.description}</span>}
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

                    {/* File picker */}
                    <div className="flex items-center gap-2 shrink-0">
                        <input ref={fileInputRef} type="file" className="hidden" onChange={handleFileChange} accept=".csv,.parquet" />
                        {file ? (
                            <div className="flex items-center gap-1.5 px-3 py-2.5 border border-gray-200 dark:border-slate-600 rounded-md text-sm bg-gray-50 dark:bg-slate-900">
                                <FileUp size={14} className="text-green-500 shrink-0" />
                                <span className="text-slate-700 dark:text-slate-300 truncate max-w-[180px]">{file.name}</span>
                                <button type="button" onClick={() => setFile(null)} className="text-gray-400 hover:text-red-500 transition-colors ml-1">
                                    <X size={14} />
                                </button>
                            </div>
                        ) : (
                            <button
                                type="button"
                                onClick={() => fileInputRef.current?.click()}
                                className="flex items-center gap-2 px-3 py-2.5 border border-dashed border-gray-300 dark:border-slate-600 rounded-md text-sm text-gray-500 dark:text-gray-400 hover:border-blue-400 hover:text-blue-500 dark:hover:border-blue-500 transition-colors"
                            >
                                <Upload size={15} />
                                Upload CSV / Parquet
                            </button>
                        )}
                    </div>

                    {/* Run button */}
                    <button
                        onClick={handleCalculate}
                        disabled={loading || !selectedJob || !file}
                        className="flex items-center justify-center gap-2 px-5 py-2.5 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
                    >
                        {loading ? <Loader2 className="animate-spin" size={16} /> : <BarChart2 size={16} />}
                        {loading ? 'Analyzing...' : 'Run Analysis'}
                    </button>

                    {/* Refresh */}
                    <button
                        onClick={fetchJobs}
                        className="p-2.5 rounded-md hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-400 transition-colors shrink-0"
                        title="Refresh jobs"
                    >
                        <RefreshCw size={16} className={refreshing ? "animate-spin" : ""} />
                    </button>

                    {/* Thresholds toggle */}
                    <button
                        onClick={() => setShowThresholds(p => !p)}
                        className={`p-2.5 rounded-md transition-colors shrink-0 ${showThresholds ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400' : 'hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-400'}`}
                        title="Drift thresholds"
                    >
                        <Settings size={16} />
                    </button>
                </div>

                {/* Threshold settings panel */}
                {showThresholds && (
                    <div className="px-4 pb-3 flex flex-wrap items-center gap-4 border-t border-gray-100 dark:border-slate-700 pt-3">
                        <span className="text-xs font-medium text-gray-500 dark:text-slate-400">Thresholds:</span>
                        {([
                            { key: 'psi' as const, label: 'PSI', step: 0.05, min: 0.01, max: 1 },
                            { key: 'ks' as const, label: 'KS p-value', step: 0.01, min: 0.001, max: 0.2 },
                            { key: 'wasserstein' as const, label: 'Wasserstein', step: 0.05, min: 0.01, max: 1 },
                            { key: 'kl' as const, label: 'KL Div', step: 0.05, min: 0.01, max: 1 },
                        ]).map(({ key, label, step, min, max }) => (
                            <label key={key} className="flex items-center gap-1.5 text-xs text-gray-600 dark:text-slate-400">
                                <span className="font-medium">{label}</span>
                                <input
                                    type="number"
                                    value={thresholds[key] ?? ''}
                                    onChange={e => setThresholds(prev => ({ ...prev, [key]: parseFloat(e.target.value) || undefined }))}
                                    step={step}
                                    min={min}
                                    max={max}
                                    className="w-20 px-2 py-1 text-xs border border-gray-200 dark:border-slate-600 rounded bg-white dark:bg-slate-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500 tabular-nums"
                                />
                            </label>
                        ))}
                        <button
                            onClick={() => setThresholds({ psi: 0.2, ks: 0.05, wasserstein: 0.1, kl: 0.1 })}
                            className="text-[11px] text-gray-400 hover:text-gray-600 dark:hover:text-slate-300 transition-colors"
                        >
                            Reset defaults
                        </button>
                    </div>
                )}

                {/* Selected job metadata + description */}
                {selectedJobData && (
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 pb-3 text-xs text-gray-500 dark:text-slate-400">
                        {selectedJobData.model_type && (
                            <span className="px-2 py-0.5 rounded bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-300 font-medium">{selectedJobData.model_type}</span>
                        )}
                        {selectedJobData.target_column && (
                            <span>Target: <span className="font-medium text-slate-600 dark:text-slate-300">{selectedJobData.target_column}</span></span>
                        )}
                        {selectedJobData.n_features != null && (
                            <span>{selectedJobData.n_features} features</span>
                        )}
                        {selectedJobData.n_rows != null && (
                            <span>{selectedJobData.n_rows.toLocaleString()} rows</span>
                        )}
                        {selectedJobData.best_metric && (
                            <span className="px-2 py-0.5 rounded bg-green-50 dark:bg-green-900/30 text-green-600 dark:text-green-300 font-medium tabular-nums">{selectedJobData.best_metric}</span>
                        )}
                        <span className="border-l border-gray-200 dark:border-slate-700 h-3" />
                        {editingDescription ? (
                            <div className="flex items-center gap-1.5 flex-1 min-w-0">
                                <input
                                    // eslint-disable-next-line jsx-a11y/no-autofocus -- focus the description field when entering edit mode
                                    autoFocus
                                    value={descriptionDraft}
                                    onChange={e => setDescriptionDraft(e.target.value)}
                                    onKeyDown={e => { if (e.key === 'Enter') handleSaveDescription(); if (e.key === 'Escape') setEditingDescription(false); }}
                                    placeholder="Add a description..."
                                    className="flex-1 min-w-0 px-2 py-0.5 text-xs border border-gray-200 dark:border-slate-600 rounded bg-white dark:bg-slate-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                                />
                                <button onClick={handleSaveDescription} className="text-green-500 hover:text-green-600"><Check size={13} /></button>
                                <button onClick={() => setEditingDescription(false)} className="text-gray-400 hover:text-red-500"><X size={13} /></button>
                            </div>
                        ) : (
                            <button
                                onClick={() => { setDescriptionDraft(selectedJobData.description || ''); setEditingDescription(true); }}
                                className="flex items-center gap-1 hover:text-blue-500 transition-colors group"
                            >
                                <Pencil size={11} className="opacity-50 group-hover:opacity-100" />
                                <span className={selectedJobData.description ? '' : 'italic text-gray-400'}>
                                    {selectedJobData.description || 'Add description...'}
                                </span>
                            </button>
                        )}
                    </div>
                )}

                {error && (
                    <div className="mx-4 mb-4 p-3 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-200 rounded-md flex items-center gap-2 border border-red-200 dark:border-red-800 text-sm">
                        <AlertTriangle size={16} className="shrink-0" />
                        {error}
                    </div>
                )}
            </div>

            {!report && !loading && (
                <div className="bg-white dark:bg-slate-800 rounded-lg shadow border dark:border-slate-700 p-12 text-center">
                    <div className="flex flex-col items-center gap-3 text-gray-400 dark:text-slate-500">
                        <Activity size={48} strokeWidth={1.2} />
                        <h2 className="text-lg font-medium text-gray-500 dark:text-slate-400">No Drift Report Yet</h2>
                        <p className="text-sm max-w-md">
                            Select a reference model from the dropdown above, upload current production data (CSV or Parquet), and click <strong>Run Analysis</strong> to compare distributions.
                        </p>
                    </div>
                </div>
            )}

            {evaluatedReport && (() => {
                const allCols = Object.values(evaluatedReport.column_drifts);
                const totalCols = allCols.length;
                const driftedCount = evaluatedReport.drifted_columns_count;
                const psiValues = allCols.map(c => c.metrics.find(m => m.metric === 'psi')?.value).filter((v): v is number => v != null);
                const avgPsi = psiValues.length > 0 ? psiValues.reduce((a, b) => a + b, 0) / psiValues.length : 0;
                const mostDrifted = [...allCols].sort((a, b) => {
                    const pa = a.metrics.find(m => m.metric === 'psi')?.value ?? 0;
                    const pb = b.metrics.find(m => m.metric === 'psi')?.value ?? 0;
                    return pb - pa;
                })[0];

                const hasSparklines = Object.keys(columnSparklines).length > 0;

                return (
                <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow border dark:border-slate-700">
                    {/* Summary Cards */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <div className="bg-gray-50 dark:bg-slate-900/50 rounded-lg p-4 border dark:border-slate-700">
                            <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-slate-400 mb-1"><Columns size={13} /> Total Columns</div>
                            <div className="text-2xl font-bold tabular-nums">{totalCols}</div>
                            <div className="text-[11px] text-gray-400 mt-0.5">Ref: {evaluatedReport.reference_rows.toLocaleString()} rows | Cur: {evaluatedReport.current_rows.toLocaleString()} rows</div>
                        </div>
                        <div className={`rounded-lg p-4 border ${driftedCount > 0 ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'}`}>
                            <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-slate-400 mb-1">{driftedCount > 0 ? <AlertTriangle size={13} /> : <CheckCircle size={13} />} Drifted</div>
                            <div className="text-2xl font-bold tabular-nums">{driftedCount} <span className="text-sm font-normal text-gray-400">/ {totalCols}</span></div>
                            <div className="text-[11px] text-gray-400 mt-0.5">{totalCols > 0 ? Math.round((driftedCount / totalCols) * 100) : 0}% of features</div>
                        </div>
                        <div className="bg-gray-50 dark:bg-slate-900/50 rounded-lg p-4 border dark:border-slate-700">
                            <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-slate-400 mb-1"><Gauge size={13} /> Avg PSI</div>
                            <div className={`text-2xl font-bold tabular-nums ${avgPsi > 0.2 ? 'text-red-600 dark:text-red-400' : avgPsi > 0.1 ? 'text-amber-600 dark:text-amber-400' : ''}`}>{avgPsi.toFixed(4)}</div>
                            <div className="text-[11px] text-gray-400 mt-0.5">{avgPsi < 0.1 ? 'Stable' : avgPsi < 0.2 ? 'Minor drift' : 'Significant drift'}</div>
                        </div>
                        <div className="bg-gray-50 dark:bg-slate-900/50 rounded-lg p-4 border dark:border-slate-700">
                            <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-slate-400 mb-1"><Target size={13} /> Most Drifted</div>
                            <div className="text-lg font-bold truncate" title={mostDrifted?.column}>{mostDrifted?.column ?? '—'}</div>
                            <div className="text-[11px] text-gray-400 mt-0.5">PSI: {(mostDrifted?.metrics.find(m => m.metric === 'psi')?.value ?? 0).toFixed(4)}</div>
                        </div>
                    </div>

                    {/* Schema Drift Section */}
                    {(() => {
                        const missing = evaluatedReport.missing_columns.filter(c => c.trim() !== '');
                        const added = evaluatedReport.new_columns.filter(c => c.trim() !== '');
                        if (missing.length === 0 && added.length === 0) return null;
                        return (
                        <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                            {missing.length > 0 && (
                                <details className="group bg-red-50 dark:bg-red-900/20 rounded border border-red-200 dark:border-red-800">
                                    <summary className="font-semibold text-red-800 dark:text-red-200 p-4 flex items-center gap-2 cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden">
                                        <ChevronDown size={14} className="transition-transform group-open:rotate-180 shrink-0" />
                                        <AlertTriangle size={16} className="shrink-0" /> Missing Columns
                                        <span className="ml-auto text-xs font-normal bg-red-200 dark:bg-red-800 px-2 py-0.5 rounded-full">{missing.length}</span>
                                    </summary>
                                    <div className="px-4 pb-4">
                                        <p className="text-[11px] text-red-600/70 dark:text-red-300/60 mb-2">Columns present in reference (training) data but missing from your uploaded file.</p>
                                        <ul className="list-disc list-inside text-sm text-red-700 dark:text-red-300">
                                            {missing.map(col => <li key={col}>{col}</li>)}
                                        </ul>
                                    </div>
                                </details>
                            )}
                            {added.length > 0 && (
                                <details className="group bg-blue-50 dark:bg-blue-900/20 rounded border border-blue-200 dark:border-blue-800">
                                    <summary className="font-semibold text-blue-800 dark:text-blue-200 p-4 flex items-center gap-2 cursor-pointer select-none list-none [&::-webkit-details-marker]:hidden">
                                        <ChevronDown size={14} className="transition-transform group-open:rotate-180 shrink-0" />
                                        <Info size={16} className="shrink-0" /> Extra Columns
                                        <span className="ml-auto text-xs font-normal bg-blue-200 dark:bg-blue-800 px-2 py-0.5 rounded-full">{added.length}</span>
                                    </summary>
                                    <div className="px-4 pb-4">
                                        <p className="text-[11px] text-blue-600/70 dark:text-blue-300/60 mb-2">Columns in your uploaded file that were not in the training data (e.g. target column, IDs, or dropped features). These are ignored during drift analysis.</p>
                                        <ul className="list-disc list-inside text-sm text-blue-700 dark:text-blue-300">
                                            {added.map(col => <li key={col}>{col}</li>)}
                                        </ul>
                                    </div>
                                </details>
                            )}
                        </div>
                        );
                    })()}

                    {/* Filter bar */}
                    <div className="flex items-center gap-3 mb-4">
                        <button
                            onClick={() => setShowOnlyDrifted(prev => !prev)}
                            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors border ${
                                showOnlyDrifted
                                    ? 'bg-red-50 dark:bg-red-900/30 border-red-200 dark:border-red-700 text-red-700 dark:text-red-300'
                                    : 'border-gray-200 dark:border-slate-600 text-gray-500 dark:text-slate-400 hover:bg-gray-50 dark:hover:bg-slate-700'
                            }`}
                        >
                            <Filter size={12} />
                            {showOnlyDrifted ? 'Showing drifted only' : 'Show only drifted'}
                        </button>
                        {sortConfig && (
                            <button
                                onClick={() => setSortConfig(null)}
                                className="flex items-center gap-1 px-2 py-1.5 rounded-md text-xs text-gray-400 hover:text-gray-600 dark:hover:text-slate-300 transition-colors"
                            >
                                <X size={11} /> Clear sort
                            </button>
                        )}
                        <div className="ml-auto">
                            <button
                                onClick={exportCSV}
                                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border border-gray-200 dark:border-slate-600 text-gray-500 dark:text-slate-400 hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors"
                                title="Download drift report as CSV"
                            >
                                <Download size={12} /> Export CSV
                            </button>
                        </div>
                    </div>

                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200 dark:divide-slate-700">
                            <thead className="bg-gray-50 dark:bg-slate-900">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer select-none hover:text-gray-700 dark:hover:text-slate-300" onClick={() => handleSort('column')}>
                                        <span className="flex items-center gap-1">Column {renderSortIcon('column')}</span>
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer select-none hover:text-gray-700 dark:hover:text-slate-300" onClick={() => handleSort('status')}>
                                        <span className="flex items-center gap-1">Status {renderSortIcon('status')}</span>
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer select-none hover:text-gray-700 dark:hover:text-slate-300" onClick={() => handleSort('wasserstein')}>
                                        <span className="flex items-center gap-1">
                                            <MetricTooltip 
                                                label="Wasserstein" 
                                                tooltip="Measures distance between distributions. Lower is better. < 0.1 usually means stable." 
                                                icon={<Info className="w-3 h-3 text-slate-400" />}
                                            />
                                            {renderSortIcon('wasserstein')}
                                        </span>
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer select-none hover:text-gray-700 dark:hover:text-slate-300" onClick={() => handleSort('psi')}>
                                        <span className="flex items-center gap-1">
                                            <MetricTooltip 
                                                label="PSI" 
                                                tooltip="Population Stability Index. < 0.1: Stable, < 0.2: Minor Drift, > 0.2: Significant Drift." 
                                                icon={<Info className="w-3 h-3 text-slate-400" />}
                                            />
                                            {renderSortIcon('psi')}
                                        </span>
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer select-none hover:text-gray-700 dark:hover:text-slate-300" onClick={() => handleSort('kl')}>
                                        <span className="flex items-center gap-1">
                                            <MetricTooltip 
                                                label="KL Div" 
                                                tooltip="Kullback-Leibler Divergence. Measures how one probability distribution diverts from a second." 
                                                icon={<Info className="w-3 h-3 text-slate-400" />}
                                            />
                                            {renderSortIcon('kl')}
                                        </span>
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer select-none hover:text-gray-700 dark:hover:text-slate-300" onClick={() => handleSort('ks')}>
                                        <span className="flex items-center gap-1">
                                            <MetricTooltip 
                                                label="KS P-Value" 
                                                tooltip="Kolmogorov-Smirnov Test. p-value < 0.05 indicates the distributions are significantly different." 
                                                icon={<Info className="w-3 h-3 text-slate-400" />}
                                            />
                                            {renderSortIcon('ks')}
                                        </span>
                                    </th>
                                    {evaluatedReport.feature_importances && (
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider cursor-pointer select-none hover:text-gray-700 dark:hover:text-slate-300" onClick={() => handleSort('risk')}>
                                            <span className="flex items-center gap-1">
                                                <MetricTooltip 
                                                    label="Risk" 
                                                    tooltip="Combines drift status with feature importance. High = drifted + important feature. Helps prioritize which drifts to investigate." 
                                                    icon={<Info className="w-3 h-3 text-slate-400" />}
                                                />
                                                {renderSortIcon('risk')}
                                            </span>
                                        </th>
                                    )}
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                                    {hasSparklines && (
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                            <span className="flex items-center gap-1"><TrendingUp size={11} /> Trend</span>
                                        </th>
                                    )}
                                </tr>
                            </thead>
                            <tbody className="bg-white dark:bg-slate-800 divide-y divide-gray-200 dark:divide-slate-700">
                                {(() => {
                                    const fi = evaluatedReport.feature_importances;
                                    const maxImportance = fi ? Math.max(...Object.values(fi)) : 0;

                                    let rows = Object.values(evaluatedReport.column_drifts);

                                    // Filter
                                    if (showOnlyDrifted) {
                                        rows = rows.filter(c => c.drift_detected);
                                    }

                                    // Sort
                                    if (sortConfig) {
                                        rows = [...rows].sort((a, b) => {
                                            const getMetric = (col: typeof a, metric: string) => col.metrics.find(m => m.metric === metric)?.value ?? 0;
                                            let cmp = 0;
                                            switch (sortConfig.key) {
                                                case 'column': cmp = a.column.localeCompare(b.column); break;
                                                case 'status': cmp = (a.drift_detected ? 1 : 0) - (b.drift_detected ? 1 : 0); break;
                                                case 'wasserstein': cmp = getMetric(a, 'wasserstein_distance') - getMetric(b, 'wasserstein_distance'); break;
                                                case 'psi': cmp = getMetric(a, 'psi') - getMetric(b, 'psi'); break;
                                                case 'kl': cmp = getMetric(a, 'kl_divergence') - getMetric(b, 'kl_divergence'); break;
                                                case 'ks': cmp = getMetric(a, 'ks_test_p_value') - getMetric(b, 'ks_test_p_value'); break;
                                                case 'risk': cmp = (fi?.[a.column] ?? 0) - (fi?.[b.column] ?? 0); break;
                                            }
                                            return sortConfig.dir === 'asc' ? cmp : -cmp;
                                        });
                                    }

                                    if (rows.length === 0 && showOnlyDrifted) {
                                        return (
                                            <tr>
                                                <td colSpan={(fi ? 8 : 7) + (hasSparklines ? 1 : 0)} className="px-6 py-8 text-center text-gray-400 dark:text-slate-500 text-sm">
                                                    <CheckCircle size={20} className="inline mr-2 text-green-500" />
                                                    No drifted columns found — all features are stable.
                                                </td>
                                            </tr>
                                        );
                                    }

                                    return rows.map((col) => {
                                    const wasserstein = col.metrics.find(m => m.metric === 'wasserstein_distance');
                                    const psi = col.metrics.find(m => m.metric === 'psi');
                                    const kl = col.metrics.find(m => m.metric === 'kl_divergence');
                                    const ks = col.metrics.find(m => m.metric === 'ks_test_p_value');
                                    const isExpanded = expandedRows[col.column];

                                    // Feature importance risk
                                    const importance = fi?.[col.column];
                                    const importanceRank = fi ? Object.values(fi).filter(v => v > (importance ?? 0)).length + 1 : null;
                                    const isHighRisk = col.drift_detected && importanceRank != null && importanceRank <= 5;
                                    const isMediumRisk = col.drift_detected && importanceRank != null && importanceRank <= 15 && !isHighRisk;
                                    
                                    return (
                                        <React.Fragment key={col.column}>
                                            <tr className={col.drift_detected ? "bg-red-50 dark:bg-red-900/10" : ""}>
                                                <td className="px-6 py-4 whitespace-nowrap font-medium">{col.column}</td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    {col.drift_detected ? (
                                                        <span className="text-red-600 dark:text-red-400 flex items-center gap-1"><XCircle size={16} /> Drifted</span>
                                                    ) : (
                                                        <span className="text-green-600 dark:text-green-400 flex items-center gap-1"><CheckCircle size={16} /> Stable</span>
                                                    )}
                                                </td>
                                                <td className={`px-6 py-4 whitespace-nowrap tabular-nums ${wasserstein?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''}`}>
                                                    {wasserstein?.value?.toFixed(4) ?? '—'}
                                                </td>
                                                <td className={`px-6 py-4 whitespace-nowrap tabular-nums ${psi?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''}`}>
                                                    {psi?.value?.toFixed(4) ?? '—'}
                                                </td>
                                                <td className={`px-6 py-4 whitespace-nowrap tabular-nums ${kl?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''}`}>
                                                    {kl?.value?.toFixed(4) ?? '—'}
                                                </td>
                                                <td className={`px-6 py-4 whitespace-nowrap tabular-nums ${ks?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''}`}>
                                                    {ks?.value?.toFixed(4) ?? '—'}
                                                </td>
                                                {evaluatedReport.feature_importances && (
                                                    <td className="px-6 py-4 whitespace-nowrap">
                                                        {importance != null ? (
                                                            <div className="flex items-center gap-1.5">
                                                                {isHighRisk ? (
                                                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-red-100 dark:bg-red-900/40 text-red-700 dark:text-red-300">
                                                                        <ShieldAlert size={12} /> High
                                                                    </span>
                                                                ) : isMediumRisk ? (
                                                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300">
                                                                        <ShieldAlert size={12} /> Medium
                                                                    </span>
                                                                ) : (
                                                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-gray-100 dark:bg-slate-700 text-gray-500 dark:text-slate-400">
                                                                        <Shield size={12} /> Low
                                                                    </span>
                                                                )}
                                                                <div className="w-12 h-1.5 bg-gray-200 dark:bg-slate-700 rounded-full overflow-hidden" title={`Importance: ${(importance * 100 / maxImportance).toFixed(0)}%`}>
                                                                    <div className="h-full bg-indigo-500 rounded-full" style={{ width: `${maxImportance > 0 ? (importance / maxImportance) * 100 : 0}%` }} />
                                                                </div>
                                                            </div>
                                                        ) : (
                                                            <span className="text-xs text-gray-400">—</span>
                                                        )}
                                                    </td>
                                                )}
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <button 
                                                        onClick={() => toggleRow(col.column)}
                                                        className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 flex items-center gap-1 text-sm"
                                                    >
                                                        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                                        {isExpanded ? "Hide" : "Details"}
                                                    </button>
                                                </td>
                                                {hasSparklines && (
                                                    <td className="px-6 py-4 whitespace-nowrap">
                                                        <Sparkline values={columnSparklines[col.column] ?? []} />
                                                    </td>
                                                )}
                                            </tr>
                                            {isExpanded && (
                                                <tr>
                                                    <td colSpan={(evaluatedReport.feature_importances ? 8 : 7) + (hasSparklines ? 1 : 0)} className="px-6 py-4 bg-gray-50 dark:bg-slate-900/50">
                                                        <div className="flex flex-col gap-4">
                                                            {col.suggestions && col.suggestions.length > 0 && (
                                                                <div className="bg-yellow-50 dark:bg-yellow-900/20 p-3 rounded border border-yellow-200 dark:border-yellow-800">
                                                                    <div className="flex items-start gap-2">
                                                                        <Lightbulb size={16} className="mt-0.5 shrink-0 text-yellow-600 dark:text-yellow-400" />
                                                                        <ul className="list-disc list-inside text-sm text-yellow-800 dark:text-yellow-200">
                                                                            {col.suggestions.map((s, i) => (
                                                                                <li key={i}>{s}</li>
                                                                            ))}
                                                                        </ul>
                                                                    </div>
                                                                </div>
                                                            )}
                                                            
                                                            {col.distribution && (
                                                                <div className="h-[350px] w-full bg-white dark:bg-slate-800 p-6 rounded border dark:border-slate-700 shadow-sm">
                                                                    <h4 className="text-sm font-semibold mb-6 flex items-center gap-2 text-slate-700 dark:text-slate-300">
                                                                        <BarChart2 size={16} /> Distribution Comparison
                                                                    </h4>
                                                                    <ResponsiveContainer width="100%" height="85%">
                                                                        <BarChart data={col.distribution.bins} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                                                            <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-slate-700" vertical={false} />
                                                                            <XAxis 
                                                                                dataKey={(bin: { bin_start: number }) => `${bin.bin_start.toFixed(2)}`} 
                                                                                className="text-xs"
                                                                                tick={{ fill: '#64748b' }}
                                                                                tickLine={false}
                                                                                axisLine={{ stroke: '#cbd5e1' }}
                                                                            />
                                                                            <YAxis 
                                                                                className="text-xs" 
                                                                                tick={{ fill: '#64748b' }} 
                                                                                tickLine={false}
                                                                                axisLine={false}
                                                                            />
                                                                            <RechartsTooltip 
                                                                                cursor={{ fill: 'rgba(0,0,0,0.05)' }}
                                                                                contentStyle={{ 
                                                                                    backgroundColor: '#1e293b', 
                                                                                    borderColor: '#334155', 
                                                                                    color: '#f8fafc',
                                                                                    borderRadius: '6px',
                                                                                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                                                                                }}
                                                                                itemStyle={{ color: '#f8fafc' }}
                                                                                labelStyle={{ color: '#94a3b8', marginBottom: '0.5rem' }}
                                                                            />
                                                                            <Legend 
                                                                                verticalAlign="top" 
                                                                                height={36}
                                                                                iconType="circle"
                                                                            />
                                                                            <Bar 
                                                                                dataKey="reference_count" 
                                                                                name="Reference (Training)" 
                                                                                fill="#94a3b8" 
                                                                                radius={[4, 4, 0, 0]} 
                                                                                barSize={30}
                                                                            />
                                                                            <Bar 
                                                                                dataKey="current_count" 
                                                                                name="Current (Production)" 
                                                                                fill="#3b82f6" 
                                                                                radius={[4, 4, 0, 0]} 
                                                                                barSize={30}
                                                                            />
                                                                        </BarChart>
                                                                    </ResponsiveContainer>
                                                                </div>
                                                            )}
                                                        </div>
                                                    </td>
                                                </tr>
                                            )}
                                        </React.Fragment>
                                    );
                                    });
                                })()}
                            </tbody>
                        </table>
                    </div>
                </div>
                );
            })()}

            {driftHistory.length > 1 && (
                <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow border dark:border-slate-700 mt-6">
                    <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <TrendingUp size={18} /> Drift History
                        <span className="text-xs font-normal text-gray-400 ml-1">({driftHistory.length} checks)</span>
                    </h2>
                    <div className="h-[280px]">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart
                                data={[...driftHistory].reverse().map(h => ({
                                    date: h.created_at?.split('T')[0] ?? '',
                                    drifted: h.drifted_columns_count ?? 0,
                                    total: h.total_columns ?? 0,
                                    pct: h.total_columns ? Math.round(((h.drifted_columns_count ?? 0) / h.total_columns) * 100) : 0,
                                }))}
                                margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
                            >
                                <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-slate-700" vertical={false} />
                                <XAxis dataKey="date" tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={{ stroke: '#cbd5e1' }} />
                                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} tickLine={false} axisLine={false} />
                                <RechartsTooltip
                                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '6px' }}
                                    itemStyle={{ color: '#f8fafc' }}
                                    labelStyle={{ color: '#94a3b8' }}
                                    formatter={(value: number, name: string) => {
                                        if (name === 'Drifted Columns') return [value, name];
                                        if (name === 'Drift %') return [`${value}%`, name];
                                        return [value, name];
                                    }}
                                />
                                <Legend verticalAlign="top" height={36} iconType="circle" />
                                <Line type="monotone" dataKey="drifted" name="Drifted Columns" stroke="#ef4444" strokeWidth={2} dot={{ r: 3, fill: '#ef4444' }} />
                                <Line type="monotone" dataKey="pct" name="Drift %" stroke="#f59e0b" strokeWidth={2} dot={{ r: 3, fill: '#f59e0b' }} strokeDasharray="5 5" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}
        </div>
    );
};
