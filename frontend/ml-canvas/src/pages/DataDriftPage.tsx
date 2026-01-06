import React, { useState, useEffect, useCallback } from 'react';
import { monitoringApi, DriftReport, DriftJobOption } from '../core/api/monitoring';
import { Loader2, Upload, AlertTriangle, CheckCircle, XCircle, Lightbulb, RefreshCw, Search, Database, Calendar, ChevronDown, ChevronUp, BarChart2, Info } from 'lucide-react';
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
                j.job_id.toLowerCase().includes(lower)
            ));
        } else {
            setFilteredJobs(jobs);
        }
    }, [searchTerm, jobs]);

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
                <button 
                    onClick={fetchJobs} 
                    className="p-2 rounded hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-600 dark:text-gray-300 transition-colors"
                    title="Refresh Jobs"
                >
                    <RefreshCw size={20} className={refreshing ? "animate-spin" : ""} />
                </button>
            </div>
            
            <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow mb-6 border dark:border-slate-700">
                <div className="flex flex-col gap-8">
                    {/* Section 1: Reference Job Selection (Full Width) */}
                    <div className="flex flex-col h-[400px]">
                        <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
                            <span className="bg-blue-100 text-blue-800 text-xs font-bold px-2 py-1 rounded-full">STEP 1</span>
                            Select Reference Model (Training Job)
                        </h2>
                        
                        <div className="relative mb-4">
                            <Search className="absolute left-3 top-2.5 text-gray-400" size={16} />
                            <input 
                                type="text" 
                                placeholder="Search by Dataset Name or Job ID..." 
                                value={searchTerm}
                                onChange={(e) => setSearchTerm(e.target.value)}
                                className="w-full pl-9 pr-3 py-2 border rounded text-sm focus:ring-2 focus:ring-blue-500 dark:bg-slate-700 dark:border-slate-600 dark:text-white shadow-sm"
                            />
                        </div>

                        <div className="flex-1 overflow-y-auto border rounded-lg dark:border-slate-600 bg-gray-50 dark:bg-slate-900 shadow-inner">
                            {filteredJobs.length === 0 ? (
                                <div className="p-8 text-center text-gray-500 text-sm flex flex-col items-center gap-2">
                                    <Database size={32} className="text-gray-300" />
                                    <p>No training jobs found.</p>
                                    <p className="text-xs">Run a training pipeline to generate reference data.</p>
                                </div>
                            ) : (
                                <div className="divide-y divide-gray-200 dark:divide-slate-700">
                                    {filteredJobs.map(job => (
                                        <div 
                                            key={job.job_id}
                                            onClick={() => setSelectedJob(job.job_id)}
                                            className={`p-4 cursor-pointer transition-all hover:bg-blue-50 dark:hover:bg-slate-800/50 ${
                                                selectedJob === job.job_id 
                                                    ? 'bg-blue-50 dark:bg-blue-900/20 border-l-4 border-blue-500 pl-3' 
                                                    : 'border-l-4 border-transparent pl-4'
                                            }`}
                                        >
                                            <div className="flex justify-between items-start mb-1">
                                                <div className="flex items-center gap-2">
                                                    <span className="font-semibold text-slate-800 dark:text-slate-200 text-base">
                                                        {job.dataset_name}
                                                    </span>
                                                    <span className="px-2 py-0.5 bg-gray-200 dark:bg-slate-700 text-gray-600 dark:text-gray-300 text-[10px] rounded-full uppercase tracking-wider font-bold">
                                                        Training
                                                    </span>
                                                </div>
                                                {job.created_at && (
                                                    <div className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400 bg-white dark:bg-slate-800 px-2 py-1 rounded border dark:border-slate-700">
                                                        <Calendar size={12} />
                                                        {job.created_at}
                                                    </div>
                                                )}
                                            </div>
                                            
                                            <div className="grid grid-cols-2 gap-4 mt-2">
                                                <div className="text-xs text-gray-500 dark:text-gray-400 font-mono flex items-center gap-1">
                                                    <span className="font-semibold text-gray-400">ID:</span> {job.job_id}
                                                </div>
                                                <div className="text-xs text-gray-500 dark:text-gray-400 font-mono flex items-center gap-1 truncate" title={job.filename}>
                                                    <span className="font-semibold text-gray-400">File:</span> {job.filename}
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Section 2: File Upload & Action */}
                    <div>
                        <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-4 flex items-center gap-2">
                            <span className="bg-blue-100 text-blue-800 text-xs font-bold px-2 py-1 rounded-full">STEP 2</span>
                            Upload Production Data & Analyze
                        </h2>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <div className="md:col-span-2 border-2 border-dashed border-gray-300 dark:border-slate-600 rounded-lg p-8 flex flex-col items-center justify-center bg-gray-50 dark:bg-slate-900/50 hover:bg-gray-100 dark:hover:bg-slate-800 transition-colors h-[160px]">
                                <Upload size={32} className="text-gray-400 mb-3" />
                                <label className="cursor-pointer">
                                    <span className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 text-sm font-medium transition-colors shadow-sm">
                                        Select Production File
                                    </span>
                                    <input type="file" className="hidden" onChange={handleFileChange} accept=".csv,.parquet" />
                                </label>
                                <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 font-medium">
                                    {file ? file.name : "No file selected"}
                                </p>
                                <p className="text-xs text-gray-400 mt-1">Supports CSV and Parquet</p>
                            </div>

                            <div className="flex flex-col justify-center">
                                <button 
                                    onClick={handleCalculate} 
                                    disabled={loading || !selectedJob || !file}
                                    className="w-full h-[160px] bg-gradient-to-br from-blue-600 to-blue-700 text-white rounded-lg hover:from-blue-700 hover:to-blue-800 disabled:opacity-50 disabled:cursor-not-allowed flex flex-col items-center justify-center gap-3 font-medium shadow-lg shadow-blue-500/20 transition-all"
                                >
                                    {loading ? (
                                        <>
                                            <Loader2 className="animate-spin" size={32} />
                                            <span className="text-lg">Calculating...</span>
                                        </>
                                    ) : (
                                        <>
                                            <AlertTriangle size={32} />
                                            <span className="text-lg">Run Analysis</span>
                                            <span className="text-xs opacity-80 font-normal">Compare Reference vs Current</span>
                                        </>
                                    )}
                                </button>
                            </div>
                        </div>
                        
                        {error && (
                            <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-200 rounded-lg flex items-start gap-3 border border-red-200 dark:border-red-800 animate-in fade-in slide-in-from-top-2">
                                <AlertTriangle size={20} className="shrink-0 mt-0.5" />
                                <span className="text-sm font-medium">{error}</span>
                            </div>
                        )}
                    </div>
                </div>
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
