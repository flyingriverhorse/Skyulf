import { Loader2, RefreshCw, List, Play, HelpCircle, Target } from 'lucide-react';
import type { EdaPageModel } from './useEdaPageController';

export function EdaPageHeader(props: EdaPageModel) {
  return <>
    {/* Top Navigation Bar */}
    {/* Fixed 64px height only once the three control groups fit side by side
          (≥1024px). Below that the header wraps and grows instead of clipping
          Analyze/History off-screen. */}
    <header className="flex-none bg-white dark:bg-slate-900 border-b border-gray-200 dark:border-gray-800 px-4 py-2 flex flex-wrap items-center justify-between gap-3 z-20 shadow-sm lg:h-16 lg:flex-nowrap lg:py-0 lg:gap-4">

      <DatasetSelection {...props} />

      <AnalysisSettings {...props} />

      <HistoryActions {...props} />
    </header>
  </>;
}

function DatasetSelection(props: EdaPageModel) {
  const { report, selectionUnavailable, selectedDataset, searchParams, setSearchParams, datasetOptions } = props;
  return <>
    {/* Left: Title & Dataset */}
    <div className="flex items-center gap-4 lg:gap-6">
      <div>
        <h1 className="text-lg font-bold text-gray-900 dark:text-white leading-tight">Exploratory Analysis</h1>
        {report && report.created_at && (
          <p className="text-[10px] text-gray-500 dark:text-gray-400">
            Last analyzed: {new Date(report.created_at).toLocaleString()}
          </p>
        )}
      </div>

      <div className="h-8 w-px bg-gray-200 dark:bg-gray-700 mx-2"></div>

      <div className="flex flex-col">
        <label htmlFor="eda-toolbar-dataset" className="text-[10px] font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">Dataset</label>
        <select
          id="eda-toolbar-dataset"
          value={selectionUnavailable ? '' : selectedDataset || ''}
          onChange={(e) => {
            const next = new URLSearchParams(searchParams);
            next.set('dataset_id', e.target.value);
            setSearchParams(next);
          }}
          className="block w-48 text-sm font-medium bg-transparent border-none p-0 focus:ring-0 text-gray-900 dark:text-white cursor-pointer hover:text-blue-600"
        >
          <option value="" disabled>Select a dataset</option>
          {datasetOptions.map((option) => (
            <option key={option.value} value={option.value}>{option.label}</option>
          ))}
        </select>
      </div>
    </div>
  </>;
}

function AnalysisSettings(props: EdaPageModel) {
  const { taskType, setTaskType, report } = props;
  return <>
    {/* Center: Controls */}
    <div className="flex flex-wrap items-center gap-3 bg-gray-50 dark:bg-gray-800/50 p-1.5 rounded-lg border border-gray-200 dark:border-gray-700 lg:flex-nowrap lg:gap-4">
      <TargetSelection {...props} />

      <div className="w-px h-8 bg-gray-200 dark:bg-gray-700"></div>

      <div className="flex flex-col px-2">
        <div className="flex items-center gap-1">
          <label htmlFor="eda-toolbar-task-type" className="text-[10px] font-medium text-gray-400 uppercase tracking-wider">Task Type</label>
          <div className="group relative">
            <HelpCircle className="w-3 h-3 text-gray-400 cursor-help" />
            <div className="absolute bottom-full mb-2 w-56 p-2 bg-slate-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none left-1/2 -translate-x-1/2">
              Force Classification or Regression.
            </div>
          </div>
        </div>
        <select
          id="eda-toolbar-task-type"
          value={taskType}
          onChange={(e) => setTaskType(e.target.value)}
          disabled={!report || !report.profile_data}
          className="block w-28 text-sm bg-transparent border-none p-0 focus:ring-0 text-gray-700 dark:text-gray-200 cursor-pointer disabled:opacity-50"
        >
          <option value="">Auto</option>
          <option value="Classification">Classification</option>
          <option value="Regression">Regression</option>
        </select>
      </div>

      <RunAnalysisButton {...props} />
    </div>
  </>;
}

function TargetSelection(props: EdaPageModel) {
  const { targetCol, setTargetCol, report, excludedColsDraft } = props;
  return <>
    <div className="flex flex-col px-2">
      <label htmlFor="eda-toolbar-target-column" className="text-[10px] font-medium text-gray-400 uppercase tracking-wider">Target Column</label>
      <select
        id="eda-toolbar-target-column"
        value={targetCol}
        onChange={(e) => setTargetCol(e.target.value)}
        disabled={!report || !report.profile_data}
        className="block w-36 text-sm bg-transparent border-none p-0 focus:ring-0 text-gray-700 dark:text-gray-200 cursor-pointer disabled:opacity-50"
      >
        <option value="">None</option>
        {report && report.profile_data && Object.keys(report.profile_data.columns)
          .filter((col) => !excludedColsDraft.includes(col))
          .map(col => (
            <option key={col} value={col}>{col}</option>
          ))}
      </select>
    </div>
  </>;
}

function RunAnalysisButton(props: EdaPageModel) {
  const { selectedDataset, runAnalysis, analyzing, existingReport } = props;
  return <>
    <button
      onClick={() => selectedDataset && runAnalysis()}
      disabled={!selectedDataset || analyzing}
      className={`ml-2 px-3 py-1.5 rounded text-sm font-medium transition-colors flex items-center shadow-sm ${existingReport
          ? 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-200 border border-gray-300 dark:border-gray-600 hover:bg-gray-50'
          : 'action-primary border border-transparent'
        }`}
    >
      {analyzing ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
      {existingReport ? 'Re-Run' : 'Analyze'}
    </button>
  </>;
}

function HistoryActions(props: EdaPageModel) {
  const { existingReport, report, loadSpecificReport, isLoadingReport, setShowHistoryModal } = props;
  return <>
    {/* Right: History & Actions */}
    <div className="flex items-center gap-3">
      {existingReport && report && report.id !== existingReport.id && (
        <button
          onClick={() => loadSpecificReport(existingReport.id)}
          disabled={isLoadingReport}
          className="flex items-center px-3 py-1.5 text-xs bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400 border border-green-200 dark:border-green-800 rounded-md hover:bg-green-100 dark:hover:bg-green-900/40 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          title={`Load existing from ${new Date(existingReport.created_at).toLocaleString()}`}
        >
          {isLoadingReport ? <Loader2 className="w-3 h-3 mr-1 animate-spin" /> : <RefreshCw className="w-3 h-3 mr-1" />}
          Load Saved
        </button>
      )}

      <button
        onClick={() => setShowHistoryModal(true)}
        className="flex items-center px-3 py-1.5 text-sm font-medium text-gray-600 bg-gray-50 dark:bg-gray-800 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
      >
        <List className="w-4 h-4 mr-2" />
        History
      </button>
    </div>
  </>;
}

export function RecentTargets(props: EdaPageModel) {
  const { history, report, loadSpecificReport, isLoadingReport } = props;
  return <>
    {/* Recent Targets Bar */}
    {history.length > 0 && (
      <div className="flex-none bg-white dark:bg-slate-950 border-b border-gray-100 dark:border-gray-800 px-4 py-1.5 flex items-center gap-3 overflow-x-auto z-10 shadow-[0_2px_3px_-1px_rgba(0,0,0,0.02)]">
        <span className="text-[10px] uppercase font-bold text-gray-400 tracking-wider whitespace-nowrap">Recent Targets:</span>
        <div className="flex gap-2">
          {Array.from(new Set(history.filter(h => h.target_col && h.status === 'COMPLETED').map(h => h.target_col))).slice(0, 8).map(target => (
            <button
              key={target}
              onClick={() => {
                const match = history.find(h => h.target_col === target && h.status === 'COMPLETED');
                if (match) loadSpecificReport(match.id);
              }}
              disabled={isLoadingReport}
              className={`px-2 py-0.5 text-xs rounded-full border transition-colors flex items-center disabled:opacity-50 disabled:cursor-not-allowed ${report?.profile_data?.target_col === target
                  ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-600 dark:text-blue-400 border-blue-200 dark:border-blue-800 font-medium'
                  : 'bg-white dark:bg-gray-800 text-gray-500 dark:text-gray-400 border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 hover:text-gray-700 dark:hover:text-gray-200'
                }`}
            >
              <Target className="w-3 h-3 mr-1" />
              {target}
            </button>
          ))}
        </div>
      </div>
    )}
  </>;
}
