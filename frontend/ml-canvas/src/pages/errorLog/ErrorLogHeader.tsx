import React from 'react';
import { AlertTriangle, RefreshCw, Trash2, Download } from 'lucide-react';
import { exportCsv } from './errorLogFormatting';
import type { ErrorLogPageState } from './useErrorLogPage';

type ErrorLogHeaderProps = Pick<ErrorLogPageState,
  | 'load'
  | 'loading'
  | 'pipelineLogs'
  | 'handleClearPipelineLogs'
  | 'events'
  | 'handleClear'
  | 'clearing'
>;

export const ErrorLogHeader: React.FC<ErrorLogHeaderProps> = ({
  load,
  loading,
  pipelineLogs,
  handleClearPipelineLogs,
  events,
  handleClear,
  clearing,
}) => (
  <>
    {/* Header */}
    <div className="flex items-center justify-between mb-6">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
          <AlertTriangle size={20} className="text-red-500" />
        </div>
        <div>
          <h1 className="text-xl font-bold text-slate-800 dark:text-slate-100">Error Log</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400">In-house tracker — all unhandled 5xx and pipeline failures</p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <button
          onClick={load}
          className="flex items-center gap-2 px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-slate-600 dark:text-slate-300"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
        {pipelineLogs.length > 0 && (
          <button
            onClick={() => void handleClearPipelineLogs()}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg hover:bg-amber-100 dark:hover:bg-amber-900/40 transition-colors text-amber-600 dark:text-amber-400"
          >
            <Trash2 size={14} />
            Clear pipeline
          </button>
        )}
        {events.length > 0 && (
          <button
            onClick={handleClear}
            disabled={clearing}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/40 transition-colors text-red-600 dark:text-red-400"
          >
            <Trash2 size={14} />
            Clear HTTP
          </button>
        )}
        {events.length > 0 && (
          <button
            onClick={() => exportCsv(events)}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-slate-600 dark:text-slate-300"
          >
            <Download size={14} />
            Export CSV
          </button>
        )}
      </div>
    </div>

  </>
);

type ErrorLogTabsProps = Pick<ErrorLogPageState,
  | 'view'
  | 'setView'
  | 'grouped'
  | 'pipelineIssues'
  | 'events'
  | 'pipelineLogs'
>;

export const ErrorLogTabs: React.FC<ErrorLogTabsProps> = ({
  view,
  setView,
  grouped,
  pipelineIssues,
  events,
  pipelineLogs,
}) => (
  <>
    {/* View toggle */}
    <div className="flex items-center gap-1 mb-4">
      {(['events', 'issues'] as const).map(v => (
        <button
          key={v}
          onClick={() => setView(v)}
          className={`flex items-center gap-1.5 px-4 py-1.5 text-sm font-medium rounded-lg border transition-colors capitalize ${view === v
              ? 'bg-slate-800 dark:bg-slate-200 text-white dark:text-slate-900 border-slate-800 dark:border-slate-200'
              : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50'
            }`}
        >
          {v === 'issues' ? `Issues (${grouped.length + pipelineIssues.length})` : `Events (${events.length + pipelineLogs.length})`}
        </button>
      ))}
    </div>

  </>
);
