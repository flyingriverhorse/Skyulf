import React, { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { ErrorEvent, ErrorSeverity, PipelineRunLog } from '../../core/api/monitoring';
import type { OperationalTimeRange } from '../../core/utils/operationalContext';
import { CopyDiagnosticId, ErrorResourceLink } from './ErrorDiagnostics';
import { statusColor, severityBadgeClass, SEVERITY_LABELS, relativeTime, clockTime } from './errorLogFormatting';

export const ErrorRow: React.FC<{
  event: ErrorEvent;
  onExpand: (e: ErrorEvent) => void;
  onResolve: (e: ErrorEvent) => void;
  origin: string;
  timeRange: OperationalTimeRange;
  filters: Record<string, string>;
}> = ({ event, onExpand, onResolve, origin, timeRange, filters }) => {
  const [expanded, setExpanded] = useState(false);
  const isResolved = !!event.resolved_at;

  return (
    <>
      <tr
        className={`border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors cursor-pointer ${isResolved ? 'opacity-50' : ''}`}
        onClick={() => setExpanded(x => !x)}
      >
        <td className="px-4 py-3 w-8 text-slate-400">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </td>
        <td className="px-4 py-3">
          <span className={`text-xs px-2 py-0.5 rounded font-semibold whitespace-nowrap ${severityBadgeClass(event.severity)}`}>
            {SEVERITY_LABELS[event.severity]}
          </span>
        </td>
        <td className="px-4 py-3">
          <span className={`text-xs font-mono px-2 py-0.5 rounded font-semibold ${statusColor(event.status_code)}`}>
            {event.status_code}
          </span>
        </td>
        <td className={`px-4 py-3 font-mono text-sm text-slate-700 dark:text-slate-300 max-w-[200px] truncate ${isResolved ? 'line-through' : ''}`}>
          {event.error_type}
        </td>
        <td className="px-4 py-3 text-sm text-slate-600 dark:text-slate-400 max-w-[300px] truncate">
          {event.message}
        </td>
        <td className="px-4 py-3 text-xs" onClick={e => e.stopPropagation()}>
          <ErrorResourceLink jobId={event.job_id} origin={origin} timeRange={timeRange} filters={filters} />
        </td>
        <td className="px-4 py-3 text-xs text-slate-400 whitespace-nowrap">
          {relativeTime(event.created_at)}
        </td>
        <td className="px-4 py-3 text-right">
          <div className="flex items-center justify-end gap-2">
            <ResolutionButton event={event} onResolve={onResolve} />
            <button
              className="text-xs text-blue-500 hover:underline"
              onClick={e => { e.stopPropagation(); onExpand(event); }}
            >
              Traceback
            </button>
          </div>
        </td>
      </tr>
      {expanded && (
        <ErrorDetails event={event} origin={origin} timeRange={timeRange} filters={filters} />
      )}
    </>
  );
};

/** Maps the pipeline log's own `error`/`warning`/`info` taxonomy to the unified severity. */
const PIPELINE_LEVEL_TO_SEVERITY: Record<string, ErrorSeverity> = {
  error: 'critical',
  warning: 'warning',
  info: 'info',
};

export const PipelineRow: React.FC<{
  log: PipelineRunLog;
  origin: string;
  timeRange: OperationalTimeRange;
  filters: Record<string, string>;
}> = ({ log, origin, timeRange, filters }) => {
  const [expanded, setExpanded] = useState(false);
  const isError = log.level === 'error';
  const severity = PIPELINE_LEVEL_TO_SEVERITY[log.level] ?? 'info';
  return (
    <>
      <tr
        className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors cursor-pointer"
        onClick={() => setExpanded(x => !x)}
      >
        <td className="px-4 py-3 w-8 text-slate-400">
          {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </td>
        <td className="px-4 py-3">
          <span className={`text-xs px-2 py-0.5 rounded font-semibold whitespace-nowrap ${severityBadgeClass(severity)}`}>
            {SEVERITY_LABELS[severity]}
          </span>
        </td>
        <td className="px-4 py-3">
          <span className={`text-xs font-mono px-2 py-0.5 rounded font-semibold ${isError
              ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
              : 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400'
            }`}>
            {isError ? 'FAIL' : 'WARN'}
          </span>
        </td>
        <td className="px-4 py-3 font-mono text-sm text-slate-700 dark:text-slate-300 max-w-[200px] truncate">
          {log.node_type ?? 'pipeline'}
        </td>
        <td className="px-4 py-3 text-sm text-slate-600 dark:text-slate-400 max-w-[300px] truncate">
          {log.message}
        </td>
        <td className="px-4 py-3 text-xs" onClick={e => e.stopPropagation()}>
          <ErrorResourceLink
            nodeId={log.node_id}
            pipelineId={log.pipeline_id}
            origin={origin}
            timeRange={timeRange}
            filters={filters}
          />
        </td>
        <td className="px-4 py-3 text-xs text-slate-400 whitespace-nowrap">
          {log.run_at ? (
            <span title={relativeTime(log.run_at)}>{clockTime(log.run_at)}</span>
          ) : '\u2014'}
        </td>
        <td className="px-4 py-3 text-right">
          <span className="text-[10px] px-2 py-0.5 rounded bg-slate-100 dark:bg-slate-700 text-slate-500 dark:text-slate-400">
            pipeline
          </span>
        </td>
      </tr>
      {expanded && (
        <PipelineDetails log={log} origin={origin} timeRange={timeRange} filters={filters} />
      )}
    </>
  );
};

const ErrorDetails: React.FC<{ event: ErrorEvent; origin: string; timeRange: OperationalTimeRange; filters: Record<string, string>; }> = ({ event, origin, timeRange, filters }) => {
  const isResolved = !!event.resolved_at;
  return (
    <tr className="bg-slate-50 dark:bg-slate-800/40">
      <td colSpan={8} className="px-6 py-4">
        <div className="grid grid-cols-2 gap-4 text-xs mb-3 text-slate-500 dark:text-slate-400">
          <span className="flex items-center gap-2">
            <strong>Route:</strong> {event.route || '—'}
          </span>
          <span><strong>Time:</strong> {new Date(event.created_at).toLocaleString()}</span>
          <span className="flex items-center gap-2">
            <strong>Diagnostic ID:</strong> <CopyDiagnosticId id={event.id} label="diagnostic ID" />
          </span>
          <span className="flex items-center gap-2">
            <strong>Target:</strong>{' '}
            <ErrorResourceLink jobId={event.job_id} origin={origin} timeRange={timeRange} filters={filters} />
          </span>
          {isResolved && event.resolved_at && (
            <span className="text-green-600 dark:text-green-400"><strong>Resolved:</strong> {new Date(event.resolved_at).toLocaleString()}</span>
          )}
        </div>
        {event.traceback ? (
          <pre className="text-xs font-mono bg-slate-900 text-slate-200 rounded-lg p-4 overflow-auto max-h-48 whitespace-pre-wrap leading-relaxed">
            {event.traceback.slice(0, 800)}{event.traceback.length > 800 ? '\n…(click Traceback for full output)' : ''}
          </pre>
        ) : (
          <p className="text-xs text-slate-400 italic">No traceback recorded.</p>
        )}
      </td>
    </tr>
  );
};

const PipelineDetails: React.FC<{ log: PipelineRunLog; origin: string; timeRange: OperationalTimeRange; filters: Record<string, string>; }> = ({ log, origin, timeRange, filters }) => (
  <tr className="bg-slate-50 dark:bg-slate-800/40">
    <td colSpan={8} className="px-6 py-4">
      <div className="grid grid-cols-2 gap-4 text-xs mb-3 text-slate-500 dark:text-slate-400">
        {log.node_id && <span><strong>Node ID:</strong> {log.node_id}</span>}
        {log.pipeline_id && <span><strong>Pipeline:</strong> {log.pipeline_id}</span>}
        {log.run_at && <span><strong>Time:</strong> {new Date(log.run_at).toLocaleString()}</span>}
        <span className="flex items-center gap-2">
          <strong>Diagnostic ID:</strong> <CopyDiagnosticId id={log.id} label="diagnostic ID" />
        </span>
        <span className="flex items-center gap-2">
          <strong>Target:</strong>{' '}
          <ErrorResourceLink
            nodeId={log.node_id}
            pipelineId={log.pipeline_id}
            origin={origin}
            timeRange={timeRange}
            filters={filters}
          />
        </span>
      </div>
      <pre className="text-xs font-mono bg-slate-900 text-slate-200 rounded-lg p-4 overflow-auto max-h-48 whitespace-pre-wrap leading-relaxed">
        {log.message}
      </pre>
    </td>
  </tr>
);

const ResolutionButton: React.FC<{ event: ErrorEvent; onResolve: (event: ErrorEvent) => void }> = ({ event, onResolve }) => {
  const isResolved = !!event.resolved_at;
  return (
    <button
      className={`text-xs font-medium px-2 py-0.5 rounded border transition-colors ${isResolved
          ? 'border-green-300 dark:border-green-700 text-green-600 dark:text-green-400 hover:bg-green-50 dark:hover:bg-green-900/20'
          : 'border-slate-200 dark:border-slate-600 text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700'
        }`}
      title={isResolved ? 'Reopen' : 'Mark resolved'}
      onClick={e => { e.stopPropagation(); onResolve(event); }}
    >
      {isResolved ? '↩ Reopen' : '✓ Resolve'}
    </button>
  );
};
