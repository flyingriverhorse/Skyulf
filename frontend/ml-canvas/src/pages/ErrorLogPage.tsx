import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import {
  AlertTriangle, Bug, RefreshCw, Search, Trash2,
  ChevronDown, ChevronRight, X, Clock, Route, Server, Download, Copy, Check,
} from 'lucide-react';
import {
  monitoringApi, ErrorEvent, ErrorSeverity, GroupedIssue, PipelineRunLog,
} from '../core/api/monitoring';
import { LoadingState, EmptyState, RecordLink, NodeInspectorLink, useConfirm } from '../components/shared';
import type { OperationalTimeRange } from '../core/utils/operationalContext';
import { toast } from '../core/toast';

// ─── helpers ────────────────────────────────────────────────────────────────

type TimeRange = '1h' | '6h' | '24h' | '7d' | 'all';

const TIME_RANGES: { label: string; value: TimeRange }[] = [
  { label: '1h',  value: '1h'  },
  { label: '6h',  value: '6h'  },
  { label: '24h', value: '24h' },
  { label: '7d',  value: '7d'  },
  { label: 'All', value: 'all' },
];

function sinceIso(range: TimeRange): string | undefined {
  if (range === 'all') return undefined;
  const ms = { '1h': 3600_000, '6h': 21_600_000, '24h': 86_400_000, '7d': 604_800_000 }[range];
  return new Date(Date.now() - ms).toISOString();
}

/** Maps the unified severity facet to the pipeline log's own `error`/`warning`/`info` taxonomy. */
const SEVERITY_TO_PIPELINE_LEVEL: Record<ErrorSeverity, string> = {
  critical: 'error',
  warning: 'warning',
  info: 'info',
};

const SEVERITY_LABELS: Record<ErrorSeverity, string> = {
  critical: 'Critical',
  warning: 'Warning',
  info: 'Info',
};

function severityBadgeClass(severity: ErrorSeverity): string {
  if (severity === 'critical') return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
  if (severity === 'warning') return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400';
  return 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300';
}

function exportCsv(rows: ErrorEvent[]): void {
  const header = ['id', 'severity', 'status_code', 'error_type', 'message', 'route', 'job_id', 'created_at'];
  const escape = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const lines = [header.join(','), ...rows.map(r =>
    [r.id, r.severity, r.status_code, r.error_type, r.message, r.route, r.job_id ?? '', r.created_at].map(escape).join(',')
  )];
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `error-log-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function statusColor(code: number): string {
  if (code >= 500) return 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400';
  if (code >= 400) return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400';
  return 'bg-slate-100 text-slate-600 dark:bg-slate-700 dark:text-slate-300';
}

function relativeTime(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const s = Math.floor(diff / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

/** Format a naive server datetime string ("YYYY-MM-DDTHH:MM:SS") to HH:MM. */
function clockTime(iso: string): string {
  return iso.slice(11, 16);
}

/** Local-time ISO prefix "YYYY-MM-DDTHH" for bucket matching. */
function localHourPrefix(d: Date): string {
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}`;
}

// ─── Copyable diagnostic ID ─────────────────────────────────────────────────

/** Copies a stable diagnostic identifier so an investigator can paste it into a ticket. */
const CopyDiagnosticId: React.FC<{ id: string | number; label: string }> = ({ id, label }) => {
  const [copied, setCopied] = useState(false);
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(String(id));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard denied or unavailable — the id remains visible/selectable.
    }
  }, [id]);
  return (
    <button
      type="button"
      onClick={() => void handleCopy()}
      className="inline-flex items-center gap-1 text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200"
      aria-label={copied ? 'Diagnostic ID copied' : `Copy ${label}`}
      title={copied ? 'Copied' : `Copy ${label}`}
    >
      {copied ? <Check size={12} /> : <Copy size={12} />}
      <span className="font-mono">{id}</span>
    </button>
  );
};

/** Contextual View action for the resource an error/pipeline event identifies, or an explicit "no target" note. */
const ErrorResourceLink: React.FC<{
  jobId?: string | null | undefined;
  nodeId?: string | null | undefined;
  pipelineId?: string | null | undefined;
  origin: string;
  timeRange: OperationalTimeRange;
  filters: Record<string, string>;
}> = ({ jobId, nodeId, pipelineId, origin, timeRange, filters }) => {
  if (jobId) {
    return (
      <RecordLink
        recordRef={{ kind: 'job', jobId }}
        origin={origin}
        timeRange={timeRange}
        filters={filters}
      />
    );
  }
  if (nodeId) {
    return (
      <NodeInspectorLink
        nodeId={nodeId}
        pipelineId={pipelineId ?? null}
        origin={origin}
        filters={filters}
      />
    );
  }
  if (pipelineId) {
    return (
      <RecordLink
        recordRef={{ kind: 'pipeline', pipelineId }}
        origin={origin}
        timeRange={timeRange}
        filters={filters}
      />
    );
  }
  return <span className="text-xs text-slate-400 italic">No target available</span>;
};

// ─── Traceback modal ─────────────────────────────────────────────────────────

const TracebackModal: React.FC<{
  event: ErrorEvent;
  onClose: () => void;
  origin: string;
  timeRange: OperationalTimeRange;
  filters: Record<string, string>;
}> = ({ event, onClose, origin, timeRange, filters }) => (
  <>
    {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events,jsx-a11y/no-static-element-interactions -- backdrop dismiss zone */}
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      {/* eslint-disable-next-line jsx-a11y/click-events-have-key-events,jsx-a11y/no-static-element-interactions -- modal panel stopPropagation */}
      <div
        className="relative bg-slate-900 text-slate-100 rounded-xl shadow-2xl w-full max-w-4xl max-h-[80vh] flex flex-col"
        onClick={e => e.stopPropagation()}
      >
      {/* header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-700">
        <div className="flex items-center gap-3">
          <Bug size={18} className="text-red-400" />
          <span className="font-semibold text-sm">{event.error_type}</span>
          <span className={`text-xs px-2 py-0.5 rounded font-mono ${statusColor(event.status_code)}`}>
            {event.status_code}
          </span>
          <span className={`text-xs px-2 py-0.5 rounded font-semibold ${severityBadgeClass(event.severity)}`}>
            {SEVERITY_LABELS[event.severity]}
          </span>
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
          <X size={18} />
        </button>
      </div>

      {/* meta */}
      <div className="flex flex-wrap items-center gap-4 px-5 py-3 border-b border-slate-700 text-xs text-slate-400">
        <span className="flex items-center gap-1"><Route size={12} />{event.route || '—'}</span>
        <span className="flex items-center gap-1"><Clock size={12} />{new Date(event.created_at).toLocaleString()}</span>
        {event.job_id && <span className="flex items-center gap-1"><Server size={12} />job: {event.job_id}</span>}
        <CopyDiagnosticId id={event.id} label="diagnostic ID" />
        <ErrorResourceLink jobId={event.job_id} origin={origin} timeRange={timeRange} filters={filters} />
      </div>

      {/* message */}
      <div className="px-5 py-3 border-b border-slate-700">
        <p className="text-sm text-slate-200">{event.message}</p>
      </div>

      {/* traceback */}
      <div className="flex-1 overflow-auto px-5 py-4">
        {event.traceback ? (
          <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap leading-relaxed">
            {event.traceback}
          </pre>
        ) : (
          <p className="text-xs text-slate-500 italic">No traceback recorded.</p>
        )}
      </div>
    </div>
  </div>
  </>
);

// ─── Row ─────────────────────────────────────────────────────────────────────

const ErrorRow: React.FC<{
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
            <button
              className={`text-xs font-medium px-2 py-0.5 rounded border transition-colors ${
                isResolved
                  ? 'border-green-300 dark:border-green-700 text-green-600 dark:text-green-400 hover:bg-green-50 dark:hover:bg-green-900/20'
                  : 'border-slate-200 dark:border-slate-600 text-slate-500 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-700'
              }`}
              title={isResolved ? 'Reopen' : 'Mark resolved'}
              onClick={e => { e.stopPropagation(); onResolve(event); }}
            >
              {isResolved ? '↩ Reopen' : '✓ Resolve'}
            </button>
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
      )}
    </>
  );
};

// ─── Pipeline event row ─────────────────────────────────────────────────────

/** Maps the pipeline log's own `error`/`warning`/`info` taxonomy to the unified severity. */
const PIPELINE_LEVEL_TO_SEVERITY: Record<string, ErrorSeverity> = {
  error: 'critical',
  warning: 'warning',
  info: 'info',
};

const PipelineRow: React.FC<{
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
          <span className={`text-xs font-mono px-2 py-0.5 rounded font-semibold ${
            isError
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
      )}
    </>
  );
};

// ─── Grouped issue row ──────────────────────────────────────────────────────

const GroupedIssueRow: React.FC<{ issue: GroupedIssue; onViewSample: (id: number) => void }> = ({ issue, onViewSample }) => (
  <tr className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
    <td className="px-4 py-3">
      <span className="inline-flex items-center justify-center min-w-[1.75rem] h-6 px-1.5 rounded-full bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 text-xs font-bold">
        {issue.count}
      </span>
    </td>
    <td className="px-4 py-3 font-mono text-sm text-slate-700 dark:text-slate-300 max-w-[200px] truncate">
      {issue.error_type}
    </td>
    <td className="px-4 py-3 text-xs text-slate-500 dark:text-slate-400 font-mono max-w-[220px] truncate">
      {issue.route || '—'}
    </td>
    <td className="px-4 py-3 text-xs text-slate-400 whitespace-nowrap">
      {relativeTime(issue.last_seen)}
    </td>
    <td className="px-4 py-3 text-xs text-slate-400 whitespace-nowrap">
      {relativeTime(issue.first_seen)}
    </td>
    <td className="px-4 py-3 text-right">
      <button
        className="text-xs text-blue-500 hover:underline"
        onClick={() => onViewSample(issue.sample_id)}
      >
        View sample
      </button>
    </td>
  </tr>
);

// ─── Pipeline grouped issue row ───────────────────────────────────────────

interface PipelineIssue {
  node_type: string;
  count: number;
  last_seen: string;
  first_seen: string;
}

const PipelineIssueRow: React.FC<{ issue: PipelineIssue }> = ({ issue }) => (
  <tr className="border-b border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
    <td className="px-4 py-3">
      <span className="inline-flex items-center justify-center min-w-[1.75rem] h-6 px-1.5 rounded-full bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400 text-xs font-bold">
        {issue.count}
      </span>
    </td>
    <td className="px-4 py-3 font-mono text-sm text-slate-700 dark:text-slate-300 max-w-[200px] truncate">
      {issue.node_type}
    </td>
    <td className="px-4 py-3 text-xs text-amber-600 dark:text-amber-400 font-medium">
      pipeline
    </td>
    <td className="px-4 py-3 text-xs text-slate-400 whitespace-nowrap">
      {issue.last_seen ? relativeTime(issue.last_seen) : '\u2014'}
    </td>
    <td className="px-4 py-3 text-xs text-slate-400 whitespace-nowrap">
      {issue.first_seen ? relativeTime(issue.first_seen) : '\u2014'}
    </td>
    <td className="px-4 py-3" />
  </tr>
);

// ─── Facet select ───────────────────────────────────────────────────────────

const FacetSelect: React.FC<{
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  optionLabel?: (v: string) => string;
}> = ({ id, label, value, onChange, options, optionLabel }) => (
  <div className="flex flex-col gap-1">
    <label htmlFor={id} className="sr-only">{label}</label>
    <select
      id={id}
      value={value}
      onChange={e => onChange(e.target.value)}
      className="px-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-300 focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      <option value="">{label}</option>
      {options.map(o => (
        <option key={o} value={o}>{optionLabel ? optionLabel(o) : o}</option>
      ))}
    </select>
  </div>
);

// ─── Page ────────────────────────────────────────────────────────────────────

export const ErrorLogPage: React.FC = () => {
  const [events, setEvents] = useState<ErrorEvent[]>([]);
  const [eventsTotal, setEventsTotal] = useState(0);
  const [eventsTotalUnfiltered, setEventsTotalUnfiltered] = useState(0);
  const [errorFacets, setErrorFacets] = useState<{ severities: ErrorSeverity[]; error_types: string[]; job_ids: string[] }>(
    { severities: [], error_types: [], job_ids: [] },
  );
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');
  const [showResolved, setShowResolved] = useState(false);
  const [severityFilter, setSeverityFilter] = useState<'' | ErrorSeverity>('');
  const [errorTypeFilter, setErrorTypeFilter] = useState('');
  const [jobIdFilter, setJobIdFilter] = useState('');
  const [nodeIdFilter, setNodeIdFilter] = useState('');
  const [modal, setModal] = useState<ErrorEvent | null>(null);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<{ hour: string; count: number }[]>([]);
  const [view, setView] = useState<'events' | 'issues'>('events');
  const [grouped, setGrouped] = useState<GroupedIssue[]>([]);

  // --- Backend-persisted pipeline run log ---
  const [pipelineLogs, setPipelineLogs] = useState<PipelineRunLog[]>([]);
  const [pipelineFacets, setPipelineFacets] = useState<{ node_types: string[]; node_ids: string[] }>(
    { node_types: [], node_ids: [] },
  );

  // Operational context this view hands to every contextual RecordLink, so a
  // followed link and its return preserve the active time/facet scope.
  const operationalTimeRange = (timeRange === 'all' ? 'all' : timeRange) as OperationalTimeRange;
  const linkFilters = useMemo(() => {
    const f: Record<string, string> = {};
    if (showResolved) f.showResolved = 'true';
    if (severityFilter) f.severity = severityFilter;
    if (errorTypeFilter) f.errorType = errorTypeFilter;
    if (jobIdFilter) f.jobId = jobIdFilter;
    if (nodeIdFilter) f.nodeId = nodeIdFilter;
    if (search) f.q = search;
    return f;
  }, [showResolved, severityFilter, errorTypeFilter, jobIdFilter, nodeIdFilter, search]);

  const fetchPipelineLogs = useCallback(async () => {
    try {
      const resp = await monitoringApi.getPipelineLogs(200, sinceIso(timeRange), undefined, {
        ...(search ? { q: search } : {}),
        ...(severityFilter ? { level: SEVERITY_TO_PIPELINE_LEVEL[severityFilter] } : {}),
        ...(nodeIdFilter ? { nodeId: nodeIdFilter } : {}),
      });
      setPipelineLogs(resp.entries);
      setPipelineFacets({ node_types: resp.facets.node_types, node_ids: resp.facets.node_ids });
    } catch (err) {
      // backend may be unavailable; log for diagnostics but don't block the page
      console.debug('[error-log] failed to fetch pipeline logs', err);
    }
  }, [timeRange, search, severityFilter, nodeIdFilter]);

  const handleClearPipelineLogs = useCallback(async () => {
    try {
      await monitoringApi.clearPipelineLogs();
      setPipelineLogs([]);
    } catch (err) {
      console.error('Failed to clear pipeline logs', err);
      toast.error('Failed to clear pipeline logs', 'Please try again.');
    }
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [data, tl, grp] = await Promise.all([
        monitoringApi.getErrors(500, sinceIso(timeRange), showResolved, {
          ...(search ? { q: search } : {}),
          ...(severityFilter ? { severity: severityFilter } : {}),
          ...(errorTypeFilter ? { errorType: errorTypeFilter } : {}),
          ...(jobIdFilter ? { jobId: jobIdFilter } : {}),
        }),
        monitoringApi.getTimeline(24),
        monitoringApi.getGrouped(),
      ]);
      setEvents(data.entries);
      setEventsTotal(data.total);
      setEventsTotalUnfiltered(data.total_unfiltered);
      setErrorFacets(data.facets);
      setTimeline(tl);
      setGrouped(grp);
      // also refresh pipeline logs so they appear in Events tab
      void fetchPipelineLogs();
    } catch {
      setError('Could not reach the backend. Is the server running?');
    } finally {
      setLoading(false);
    }
  }, [timeRange, showResolved, severityFilter, errorTypeFilter, jobIdFilter, search, fetchPipelineLogs]);

  useEffect(() => { load(); }, [load]);

  const confirm = useConfirm();

  const handleClear = async () => {
    const ok = await confirm({
      title: 'Delete all error events?',
      message: `Delete all ${events.length} error events? This cannot be undone.`,
      confirmLabel: 'Delete all',
      variant: 'danger',
    });
    if (!ok) return;
    setClearing(true);
    try {
      await monitoringApi.clearErrors();
      setEvents([]);
    } finally {
      setClearing(false);
    }
  };

  const handleViewSample = async (id: number) => {
    try {
      const ev = await monitoringApi.getError(id);
      setModal(ev);
    } catch (err) {
      console.error('Failed to load error sample', err);
      toast.error('Failed to load error sample', 'Please try again.');
    }
  };

  const handleResolve = async (ev: ErrorEvent) => {
    const updated = ev.resolved_at
      ? await monitoringApi.unresolveError(ev.id)
      : await monitoringApi.resolveError(ev.id);
    setEvents(prev => prev.map(e => e.id === updated.id ? updated : e));
  };

  // Filters (time range, resolved state, severity, error type, job/node id, and
  // the generic search box) are all applied server-side across the full stored
  // history — see `monitoringApi.getErrors`/`getPipelineLogs` — so these lists
  // are already the matching set, not a client-side narrowing of one page.
  const filtered = events;
  const filteredPipeline = pipelineLogs;
  const hasActiveFilters =
    !!search || !!severityFilter || !!errorTypeFilter || !!jobIdFilter || !!nodeIdFilter;

  // summary stats
  const total500 = events.filter(e => e.status_code >= 500).length;

  const pipelineIssues = useMemo<PipelineIssue[]>(() => {
    const map = new Map<string, PipelineIssue>();
    pipelineLogs.filter(l => l.level === 'error').forEach(l => {
      const key = l.node_type ?? 'unknown';
      const entry = map.get(key);
      if (!entry) {
        map.set(key, { node_type: key, count: 1, last_seen: l.run_at ?? '', first_seen: l.run_at ?? '' });
      } else {
        entry.count++;
        if (l.run_at) {
          if (l.run_at > entry.last_seen) entry.last_seen = l.run_at;
          if (l.run_at < entry.first_seen) entry.first_seen = l.run_at;
        }
      }
    });
    return Array.from(map.values()).sort((a, b) => b.count - a.count);
  }, [pipelineLogs]);

  const mergedTimeline = useMemo(() => {
    // Always build 24 local-time buckets; merge BOTH the HTTP timeline (UTC strings,
    // converted to local) and pipeline logs (local naive strings) into them.
    const buckets = Array.from({ length: 24 }, (_, i) => {
      const d = new Date();
      d.setMinutes(0, 0, 0);
      d.setHours(d.getHours() - (23 - i));
      return { hour: localHourPrefix(d) + ':00:00', _key: localHourPrefix(d), count: 0 };
    });
    const byKey = new Map(buckets.map(b => [b._key, b]));
    // HTTP timeline: hour is "YYYY-MM-DDTHH:00" in UTC.
    timeline.forEach(t => {
      const utcStr = t.hour.length >= 13 ? t.hour.slice(0, 13) + ':00:00Z' : t.hour;
      const d = new Date(utcStr);
      if (Number.isNaN(d.getTime())) return;
      const key = localHourPrefix(d);
      const b = byKey.get(key);
      if (b) b.count += t.count;
    });
    // Pipeline logs: run_at is "YYYY-MM-DDTHH:MM:SS" (server local time, no Z).
    pipelineLogs.forEach(l => {
      if (!l.run_at) return;
      const key = l.run_at.replace(' ', 'T').slice(0, 13);
      const b = byKey.get(key);
      if (b) b.count += 1;
    });
    return buckets;
  }, [timeline, pipelineLogs]);

  return (
    <div className="p-6 max-w-7xl mx-auto">
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
              onClick={() => exportCsv(filtered.length ? filtered : events)}
              className="flex items-center gap-2 px-3 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-700 transition-colors text-slate-600 dark:text-slate-300"
            >
              <Download size={14} />
              Export CSV
            </button>
          )}
        </div>
      </div>

      {/* Stats */}
      {!loading && (events.length > 0 || pipelineLogs.length > 0) && (
        <div className="grid grid-cols-3 gap-4 mb-6">
          {[
            { label: 'HTTP events', value: events.length, color: 'text-slate-700 dark:text-slate-200' },
            { label: 'Server errors (5xx)', value: total500, color: 'text-red-600 dark:text-red-400' },
            { label: 'Pipeline failures', value: pipelineLogs.filter(l => l.level === 'error').length, color: 'text-amber-600 dark:text-amber-400' },
          ].map(s => (
            <div key={s.label} className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
              <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">{s.label}</p>
              <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Timeline chart */}
      {!loading && (events.length > 0 || pipelineLogs.length > 0) && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 mb-6">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-3">Errors per hour — last 24 h</p>
          <ResponsiveContainer width="100%" height={80}>
            <BarChart data={mergedTimeline} margin={{ top: 0, right: 0, left: -30, bottom: 0 }}>
              <XAxis
                dataKey="hour"
                tickFormatter={h => h.slice(11, 16)}
                tick={{ fontSize: 10, fill: '#94a3b8' }}
                interval="preserveStartEnd"
                axisLine={false}
                tickLine={false}
              />
              <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <Tooltip
                formatter={(v: number) => [v, 'errors']}
                labelFormatter={l => `Hour starting ${l}`}
                contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }}
              />
              <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                {mergedTimeline.map((t, i) => (
                  <Cell key={i} fill={t.count > 0 ? '#ef4444' : '#e2e8f0'} fillOpacity={t.count > 0 ? 0.85 : 0.4} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

        {/* View toggle */}
      <div className="flex items-center gap-1 mb-4">
        {(['events', 'issues'] as const).map(v => (
          <button
            key={v}
            onClick={() => setView(v)}
            className={`flex items-center gap-1.5 px-4 py-1.5 text-sm font-medium rounded-lg border transition-colors capitalize ${
              view === v
                ? 'bg-slate-800 dark:bg-slate-200 text-white dark:text-slate-900 border-slate-800 dark:border-slate-200'
                : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50'
            }`}
          >
            {v === 'issues' ? `Issues (${grouped.length + pipelineIssues.length})` : `Events (${events.length + pipelineLogs.length})`}
          </button>
        ))}
      </div>

      {/* ── HTTP Events / Issues tabs ──────────────────────────────────── */}
      {/* Toolbar */}
      <div className={`flex flex-col gap-3 mb-4 ${view === 'issues' ? 'hidden' : ''}`}>
        <div className="flex items-center gap-3 flex-wrap">
          <div className="relative flex-1 max-w-sm">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              placeholder="Search errors, job id, node id…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              className="w-full pl-9 pr-4 py-2 text-sm bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 dark:text-slate-200"
            />
          </div>
          {/* Time-range pills */}
          <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 rounded-lg p-1">
            {TIME_RANGES.map(r => (
              <button
                key={r.value}
                onClick={() => setTimeRange(r.value)}
                className={`px-3 py-1 text-xs rounded-md font-medium transition-colors ${
                  timeRange === r.value
                    ? 'bg-white dark:bg-slate-700 text-slate-800 dark:text-slate-100 shadow-sm'
                    : 'text-slate-500 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-200'
                }`}
              >
                {r.label}
              </button>
            ))}
          </div>
          {/* Show resolved toggle */}
          <button
            onClick={() => setShowResolved(v => !v)}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-lg font-medium border transition-colors ${
              showResolved
                ? 'bg-slate-200 dark:bg-slate-700 border-slate-300 dark:border-slate-600 text-slate-700 dark:text-slate-200'
                : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-500 dark:text-slate-400 hover:text-slate-700'
            }`}
          >
            {showResolved ? '✓ Showing resolved' : 'Show resolved'}
          </button>
          {hasActiveFilters && (
            <span className="text-xs text-slate-500 dark:text-slate-400">
              {eventsTotalUnfiltered === 0
                ? 'No history recorded yet'
                : `${eventsTotal} of ${eventsTotalUnfiltered} HTTP events match`}
            </span>
          )}
        </div>
        {/* Typed facets — composable, exact-match filters distinct from the generic search above */}
        <div className="flex items-center gap-2 flex-wrap">
          <FacetSelect
            id="facet-severity"
            label="All severities"
            value={severityFilter}
            onChange={v => setSeverityFilter(v as '' | ErrorSeverity)}
            options={errorFacets.severities}
            optionLabel={s => SEVERITY_LABELS[s as ErrorSeverity] ?? s}
          />
          <FacetSelect
            id="facet-error-type"
            label="All error types"
            value={errorTypeFilter}
            onChange={setErrorTypeFilter}
            options={errorFacets.error_types}
          />
          <FacetSelect
            id="facet-job-id"
            label="All job IDs"
            value={jobIdFilter}
            onChange={setJobIdFilter}
            options={errorFacets.job_ids}
          />
          <FacetSelect
            id="facet-node-id"
            label="All node IDs"
            value={nodeIdFilter}
            onChange={setNodeIdFilter}
            options={pipelineFacets.node_ids}
          />
        </div>
      </div>

      {/* Table */}
      {loading ? (
        <LoadingState message="Loading error events…" />
      ) : error ? (
        <div className="flex items-center gap-3 p-4 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-xl text-amber-700 dark:text-amber-400 text-sm">
          <AlertTriangle size={16} />
          {error}
        </div>
      ) : view === 'issues' ? (
        grouped.length === 0 && pipelineIssues.length === 0 ? (
          <EmptyState
            icon={<Bug size={40} className="text-slate-300" />}
            title="No open issues"
            description="All errors have been resolved, or none have been recorded yet."
          />
        ) : (
          <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-900/50 border-b border-slate-200 dark:border-slate-700">
                  <th className="px-4 py-3">Count</th>
                  <th className="px-4 py-3">Type</th>
                  <th className="px-4 py-3">Route</th>
                  <th className="px-4 py-3">Last seen</th>
                  <th className="px-4 py-3">First seen</th>
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody>
                {pipelineIssues.map((p, i) => (
                  <PipelineIssueRow key={`pi-${i}`} issue={p} />
                ))}
                {grouped.map((g, i) => (
                  <GroupedIssueRow key={i} issue={g} onViewSample={handleViewSample} />
                ))}
              </tbody>
            </table>
          </div>
        )
      ) : filtered.length === 0 && filteredPipeline.length === 0 ? (
        <EmptyState
          icon={<Bug size={40} className="text-slate-300" />}
          title={hasActiveFilters ? 'No matching errors' : 'No errors recorded'}
          description={
            hasActiveFilters
              ? eventsTotalUnfiltered === 0
                ? 'No error events have been recorded yet — nothing to search or filter.'
                : `No events match the current search/facets out of ${eventsTotalUnfiltered} recorded. Try widening the time range or clearing a facet.`
              : 'Any unhandled 5xx or failed pipeline will appear here automatically.'
          }
        />
      ) : (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-900/50 border-b border-slate-200 dark:border-slate-700">
                <th className="px-4 py-3 w-8" />
                <th className="px-4 py-3">Severity</th>
                <th className="px-4 py-3">Code</th>
                <th className="px-4 py-3">Type</th>
                <th className="px-4 py-3">Message</th>
                <th className="px-4 py-3">Target</th>
                <th className="px-4 py-3">When</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {filteredPipeline.map(l => (
                <PipelineRow
                  key={`pl-${l.id}`}
                  log={l}
                  origin="/errors"
                  timeRange={operationalTimeRange}
                  filters={linkFilters}
                />
              ))}
              {filtered.map(e => (
                <ErrorRow
                  key={e.id}
                  event={e}
                  onExpand={setModal}
                  onResolve={handleResolve}
                  origin="/errors"
                  timeRange={operationalTimeRange}
                  filters={linkFilters}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
      {modal && (
        <TracebackModal
          event={modal}
          onClose={() => setModal(null)}
          origin="/errors"
          timeRange={operationalTimeRange}
          filters={linkFilters}
        />
      )}
    </div>
  );
};
