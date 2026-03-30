import React, { useState, useEffect, useCallback, useRef } from 'react';
import { monitoringApi, DriftReport, DriftJobOption } from '../core/api/monitoring';
import { Loader2, Upload, AlertTriangle, CheckCircle, XCircle, Lightbulb, RefreshCw, Search, Database, ChevronDown, ChevronUp, BarChart2, Info, FileUp, X, Pencil, Check } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';

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

    const selectedJobData = jobs.find(j => j.job_id === selectedJob);

    useEffect(() => { setEditingDescription(false); }, [selectedJob]);

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

    const groupedJobs = (() => {
        const groups: Record<string, DriftJobOption[]> = {};
        for (const job of filteredJobs) {
            const date = job.created_at?.split(' ')[0] || 'Unknown';
            if (!groups[date]) groups[date] = [];
            groups[date].push(job);
        }
        return Object.entries(groups).sort(([a], [b]) => b.localeCompare(a));
    })();

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
            const result = await monitoringApi.calculateDrift(selectedJob, file, job?.dataset_name);
            setReport(result);
        } catch (err: any) {
            setError(err.response?.data?.detail || "Failed to calculate drift.");
        } finally {
            setLoading(false);
        }
    };

    const toggleRow = (col: string) => {
        setExpandedRows(prev => ({ ...prev, [col]: !prev[col] }));
    };

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
                                        <Search className="absolute left-2.5 top-2 text-gray-400" size={14} />
                                        <input
                                            type="text"
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
                                                                <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-100 dark:bg-green-900/40 text-green-600 dark:text-green-300 shrink-0 ml-auto tabular-nums">{job.best_metric}</span>
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
                </div>

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
                        <span className="border-l border-gray-200 dark:border-slate-700 h-3" />
                        {editingDescription ? (
                            <div className="flex items-center gap-1.5 flex-1 min-w-0">
                                <input
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

            {report && (
                <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow border dark:border-slate-700">
                    <div className="flex justify-between items-center mb-4">
                        <h2 className="text-xl font-semibold">Drift Report</h2>
                        <div className="text-sm text-gray-500 dark:text-gray-400">
                            Reference Rows: {report.reference_rows} | Current Rows: {report.current_rows}
                        </div>
                    </div>
                    
                    <div className="mb-6">
                        <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full ${
                            report.drifted_columns_count > 0 
                                ? 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200' 
                                : 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200'
                        }`}>
                            {report.drifted_columns_count > 0 ? <AlertTriangle size={20} /> : <CheckCircle size={20} />}
                            <span className="font-medium">
                                {report.drifted_columns_count} columns drifted
                            </span>
                        </div>
                    </div>

                    {/* Schema Drift Section */}
                    {(report.missing_columns.length > 0 || report.new_columns.length > 0) && (
                        <div className="mb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                            {report.missing_columns.length > 0 && (
                                <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded border border-red-200 dark:border-red-800">
                                    <h3 className="font-semibold text-red-800 dark:text-red-200 mb-2 flex items-center gap-2">
                                        <AlertTriangle size={16} /> Missing Columns
                                    </h3>
                                    <ul className="list-disc list-inside text-sm text-red-700 dark:text-red-300">
                                        {report.missing_columns.map(col => <li key={col}>{col}</li>)}
                                    </ul>
                                </div>
                            )}
                            {report.new_columns.length > 0 && (
                                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded border border-blue-200 dark:border-blue-800">
                                    <h3 className="font-semibold text-blue-800 dark:text-blue-200 mb-2 flex items-center gap-2">
                                        <AlertTriangle size={16} /> New Columns
                                    </h3>
                                    <ul className="list-disc list-inside text-sm text-blue-700 dark:text-blue-300">
                                        {report.new_columns.map(col => <li key={col}>{col}</li>)}
                                    </ul>
                                </div>
                            )}
                        </div>
                    )}

                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200 dark:divide-slate-700">
                            <thead className="bg-gray-50 dark:bg-slate-900">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Column</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Status</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                        <MetricTooltip 
                                            label="Wasserstein" 
                                            tooltip="Measures distance between distributions. Lower is better. < 0.1 usually means stable." 
                                            icon={<Info className="w-3 h-3 text-slate-400" />}
                                        />
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                        <MetricTooltip 
                                            label="PSI" 
                                            tooltip="Population Stability Index. < 0.1: Stable, < 0.2: Minor Drift, > 0.2: Significant Drift." 
                                            icon={<Info className="w-3 h-3 text-slate-400" />}
                                        />
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                        <MetricTooltip 
                                            label="KL Div" 
                                            tooltip="Kullback-Leibler Divergence. Measures how one probability distribution diverts from a second." 
                                            icon={<Info className="w-3 h-3 text-slate-400" />}
                                        />
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                                        <MetricTooltip 
                                            label="KS P-Value" 
                                            tooltip="Kolmogorov-Smirnov Test. p-value < 0.05 indicates the distributions are significantly different." 
                                            icon={<Info className="w-3 h-3 text-slate-400" />}
                                        />
                                    </th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Actions</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white dark:bg-slate-800 divide-y divide-gray-200 dark:divide-slate-700">
                                {Object.values(report.column_drifts).map((col) => {
                                    const wasserstein = col.metrics.find(m => m.metric === 'wasserstein_distance');
                                    const psi = col.metrics.find(m => m.metric === 'psi');
                                    const kl = col.metrics.find(m => m.metric === 'kl_divergence');
                                    const ks = col.metrics.find(m => m.metric === 'ks_test_p_value');
                                    const isExpanded = expandedRows[col.column];
                                    
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
                                                <td className={`px-6 py-4 whitespace-nowrap ${wasserstein?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''}`}>
                                                    {wasserstein?.value.toFixed(4)}
                                                </td>
                                                <td className={`px-6 py-4 whitespace-nowrap ${psi?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''}`}>
                                                    {psi?.value.toFixed(4)}
                                                </td>
                                                <td className={`px-6 py-4 whitespace-nowrap ${kl?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''}`}>
                                                    {kl?.value.toFixed(4)}
                                                </td>
                                                <td className={`px-6 py-4 whitespace-nowrap ${ks?.has_drift ? 'text-red-600 dark:text-red-400 font-bold' : ''}`}>
                                                    {ks?.value.toFixed(4)}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <button 
                                                        onClick={() => toggleRow(col.column)}
                                                        className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 flex items-center gap-1 text-sm"
                                                    >
                                                        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                                                        {isExpanded ? "Hide" : "Details"}
                                                    </button>
                                                </td>
                                            </tr>
                                            {isExpanded && (
                                                <tr>
                                                    <td colSpan={7} className="px-6 py-4 bg-gray-50 dark:bg-slate-900/50">
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
                                                                                dataKey={(bin: any) => `${bin.bin_start.toFixed(2)}`} 
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
                                })}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
};
