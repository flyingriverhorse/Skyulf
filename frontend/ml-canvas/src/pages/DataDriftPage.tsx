import React, { useMemo, useState } from 'react';
import { AlertTriangle, BarChart2, Loader2, RefreshCw, Settings } from 'lucide-react';
import type { DriftThresholds } from '../core/api/monitoring';
import { DriftFiltersBar } from './drift/DriftFiltersBar';
import { DriftHistoryChart } from './drift/DriftHistoryChart';
import { DriftTable } from './drift/DriftTable';
import { EmptyState } from './drift/EmptyState';
import { FileUploader } from './drift/FileUploader';
import { JobSelector } from './drift/JobSelector';
import { SchemaDriftPanel } from './drift/SchemaDriftPanel';
import { SelectedJobMeta } from './drift/SelectedJobMeta';
import { SummaryCards } from './drift/SummaryCards';
import { ThresholdsPanel } from './drift/ThresholdsPanel';
import { useDriftHistory } from './drift/_hooks/useDriftHistory';
import { useDriftJobs } from './drift/_hooks/useDriftJobs';
import { useDriftReport } from './drift/_hooks/useDriftReport';
import { useSortConfig } from './drift/_hooks/useSortConfig';
import { exportDriftReportCSV } from './drift/_utils/csvExport';

const DEFAULT_THRESHOLDS: DriftThresholds = { psi: 0.2, ks: 0.05, wasserstein: 0.1, kl: 0.1 };

/**
 * Top-level page for data-drift analysis. Owns the user's selections (job,
 * file, thresholds, drifted-only filter) and composes the toolbar, summary,
 * per-feature table, and history chart from focused sub-modules under
 * `pages/drift/`.
 */
export const DataDriftPage: React.FC = () => {
    // User selections
    const [selectedJob, setSelectedJob] = useState<string>('');
    const [file, setFile] = useState<File | null>(null);
    const [thresholds, setThresholds] = useState<DriftThresholds>(DEFAULT_THRESHOLDS);
    const [showThresholds, setShowThresholds] = useState(false);
    const [showOnlyDrifted, setShowOnlyDrifted] = useState(false);

    // Data sources
    const { jobs, refreshing, refresh, updateJobDescription } = useDriftJobs();
    const { evaluatedReport, loading, error, setError, calculate } = useDriftReport(thresholds);
    const { driftHistory, columnSparklines, refreshHistory } = useDriftHistory(selectedJob);
    const { sortConfig, handleSort, clearSort } = useSortConfig();

    const selectedJobData = useMemo(
        () => jobs.find(j => j.job_id === selectedJob),
        [jobs, selectedJob],
    );

    const handleCalculate = async () => {
        if (!selectedJob || !file) {
            setError('Please select a Reference Job and upload Current Data.');
            return;
        }
        const result = await calculate({ selectedJob, file, job: selectedJobData, thresholds });
        if (result) refreshHistory();
    };

    const handleExport = () => {
        if (evaluatedReport) exportDriftReportCSV(evaluatedReport, selectedJobData?.dataset_name);
    };

    return (
        <div className="p-6 w-full text-slate-900 dark:text-slate-100">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-2xl font-bold">Data Drift Analysis</h1>
            </div>

            <div className="bg-white dark:bg-slate-800 rounded-lg shadow mb-6 border dark:border-slate-700">
                {/* Toolbar */}
                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 p-4">
                    <JobSelector jobs={jobs} selectedJob={selectedJob} onSelect={setSelectedJob} />
                    <FileUploader file={file} onFileChange={setFile} />
                    <button
                        onClick={() => void handleCalculate()}
                        disabled={loading || !selectedJob || !file}
                        className="flex items-center justify-center gap-2 px-5 py-2.5 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors shrink-0"
                    >
                        {loading ? <Loader2 className="animate-spin" size={16} /> : <BarChart2 size={16} />}
                        {loading ? 'Analyzing...' : 'Run Analysis'}
                    </button>
                    <button
                        onClick={() => void refresh()}
                        className="p-2.5 rounded-md hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-400 transition-colors shrink-0"
                        title="Refresh jobs"
                    >
                        <RefreshCw size={16} className={refreshing ? 'animate-spin' : ''} />
                    </button>
                    <button
                        onClick={() => setShowThresholds(p => !p)}
                        className={`p-2.5 rounded-md transition-colors shrink-0 ${
                            showThresholds
                                ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400'
                                : 'hover:bg-gray-100 dark:hover:bg-slate-700 text-gray-400'
                        }`}
                        title="Drift thresholds"
                    >
                        <Settings size={16} />
                    </button>
                </div>

                {showThresholds && <ThresholdsPanel thresholds={thresholds} onChange={setThresholds} />}

                {selectedJobData && (
                    <SelectedJobMeta
                        job={selectedJobData}
                        onUpdateDescription={desc => updateJobDescription(selectedJobData.job_id, desc)}
                    />
                )}

                {error && (
                    <div className="mx-4 mb-4 p-3 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-200 rounded-md flex items-center gap-2 border border-red-200 dark:border-red-800 text-sm">
                        <AlertTriangle size={16} className="shrink-0" />
                        {error}
                    </div>
                )}
            </div>

            {!evaluatedReport && !loading && <EmptyState />}

            {evaluatedReport && (
                <div className="bg-white dark:bg-slate-800 p-6 rounded-lg shadow border dark:border-slate-700">
                    <SummaryCards report={evaluatedReport} />
                    <SchemaDriftPanel
                        missingColumns={evaluatedReport.missing_columns}
                        newColumns={evaluatedReport.new_columns}
                    />
                    <DriftFiltersBar
                        showOnlyDrifted={showOnlyDrifted}
                        onToggleDrifted={() => setShowOnlyDrifted(p => !p)}
                        sortConfig={sortConfig}
                        onClearSort={clearSort}
                        onExport={handleExport}
                    />
                    <DriftTable
                        report={evaluatedReport}
                        showOnlyDrifted={showOnlyDrifted}
                        sortConfig={sortConfig}
                        onSort={handleSort}
                        columnSparklines={columnSparklines}
                    />
                </div>
            )}

            <DriftHistoryChart history={driftHistory} />
        </div>
    );
};
