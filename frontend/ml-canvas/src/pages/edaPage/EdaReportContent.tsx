import { Loader2, RefreshCw, AlertCircle, BarChart2, Play, HelpCircle } from 'lucide-react';
import { LoadingState, ErrorState } from '../../components/shared';
import { EdaProfileContent } from './EdaProfileContent';
import type { EdaPageModel } from './useEdaPageController';

export function EdaReportContent(props: EdaPageModel) {
  const { loading, report, error, reportQuery, runAnalysis } = props;
  if (loading && !report) {
    return <LoadingState message="Analyzing dataset..." />;
  }

  if (error) {
    return (
      <ErrorState
        error={error}
        onRetry={() => reportQuery.refetch()}
      />
    );
  }

  if (!report) return <AnalysisSetup {...props} />;

  if (report.status === 'PENDING') {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-gray-500 dark:text-gray-400">
        <Loader2 className="w-16 h-16 mb-4 animate-spin text-blue-500" />
        <p>Analysis in progress...</p>
        <p className="text-sm text-gray-400">This may take a few moments.</p>
      </div>
    );
  }

  if (report.status === 'FAILED') {
    return (
      <div className="flex flex-col items-center justify-center h-64 text-red-500">
        <AlertCircle className="w-16 h-16 mb-4" />
        <p>Analysis Failed</p>
        <p className="text-sm text-gray-600 mt-2">{report.error_message}</p>
        <button
          onClick={() => runAnalysis()}
          className="mt-4 flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200"
        >
          <RefreshCw className="w-4 h-4 mr-2" />
          Retry
        </button>
      </div>
    );
  }

  const profile = props.profileForUi;
  if (!profile) return <div>No profile data</div>;

  return <EdaProfileContent {...props} report={report} profile={profile} />;
}

function AnalysisSetup(props: EdaPageModel) {
  const { targetCol, setTargetCol, taskType, setTaskType, existingReport, loadSpecificReport, isLoadingReport, runAnalysis, analyzing } = props;
  return <>
    <div className="flex flex-col items-center justify-center h-64 text-gray-500 dark:text-gray-400">
      <BarChart2 className="w-16 h-16 mb-4 opacity-20" />
      <p className="mb-4">No analysis found for this dataset.</p>

      <div className="flex flex-col items-center space-y-4">
        <div className="w-64 space-y-2">
          <label htmlFor="eda-setup-target-column" className="sr-only">Target Column (Optional)</label>
          <input
            id="eda-setup-target-column"
            type="text"
            value={targetCol}
            onChange={(e) => setTargetCol(e.target.value)}
            placeholder="Target Column (Optional)"
            className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
          />

          <div className="flex items-center space-x-2">
            <label htmlFor="eda-setup-task-type" className="sr-only">Task Type</label>
            <select
              id="eda-setup-task-type"
              value={taskType}
              onChange={(e) => setTaskType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
            >
              <option value="">Auto-Detect Task</option>
              <option value="Classification">Classification</option>
              <option value="Regression">Regression</option>
            </select>
            <div className="group relative flex items-center">
              <HelpCircle className="w-4 h-4 text-gray-400 cursor-help" />
              <div className="absolute left-full ml-2 w-64 p-2 bg-gray-800 text-white text-xs rounded shadow-lg opacity-0 group-hover:opacity-100 transition-opacity z-50 pointer-events-none">
                Force a specific task type. Useful for ID columns (force Classification) or numeric categories (force Regression).
              </div>
            </div>
          </div>
        </div>
        <div className="flex gap-2">
          {existingReport && (
            <button
              onClick={() => loadSpecificReport(existingReport.id)}
              disabled={isLoadingReport}
              className="flex items-center px-4 py-2 action-secondary rounded-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoadingReport ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <RefreshCw className="w-4 h-4 mr-2" />}
              Load Existing ({new Date(existingReport.created_at).toLocaleDateString()})
            </button>
          )}
          <button
            onClick={() => runAnalysis()}
            disabled={analyzing}
            className={`flex items-center px-4 py-2 action-primary rounded-md disabled:opacity-50`}
          >
            {analyzing ? <Loader2 className="w-4 h-4 mr-2 animate-spin" /> : <Play className="w-4 h-4 mr-2" />}
            {existingReport ? 'Run New Analysis' : 'Run Analysis'}
          </button>
        </div>
      </div>
    </div>
  </>;
}
