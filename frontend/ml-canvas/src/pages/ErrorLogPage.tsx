import React, { useCallback, useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import {
  AlertTriangle, Bug, RefreshCw, Search, Trash2,
  ChevronDown, ChevronRight, X, Clock, Route, Server, Download,
} from 'lucide-react';
import { monitoringApi, ErrorEvent, GroupedIssue } from '../core/api/monitoring';
import { LoadingState, EmptyState } from '../components/shared';

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

function exportCsv(rows: ErrorEvent[]): void {
  const header = ['id', 'status_code', 'error_type', 'message', 'route', 'job_id', 'created_at'];
  const escape = (v: unknown) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const lines = [header.join(','), ...rows.map(r =>
    [r.id, r.status_code, r.error_type, r.message, r.route, r.job_id ?? '', r.created_at].map(escape).join(',')
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

// ─── Traceback modal ─────────────────────────────────────────────────────────

const TracebackModal: React.FC<{ event: ErrorEvent; onClose: () => void }> = ({ event, onClose }) => (
  <div
    className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
    onClick={onClose}
  >
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
        </div>
        <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
          <X size={18} />
        </button>
      </div>

      {/* meta */}
      <div className="flex flex-wrap gap-4 px-5 py-3 border-b border-slate-700 text-xs text-slate-400">
        <span className="flex items-center gap-1"><Route size={12} />{event.route || '—'}</span>
        <span className="flex items-center gap-1"><Clock size={12} />{new Date(event.created_at).toLocaleString()}</span>
        {event.job_id && <span className="flex items-center gap-1"><Server size={12} />job: {event.job_id}</span>}
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
);

// ─── Row ─────────────────────────────────────────────────────────────────────

const ErrorRow: React.FC<{
  event: ErrorEvent;
  onExpand: (e: ErrorEvent) => void;
  onResolve: (e: ErrorEvent) => void;
}> = ({ event, onExpand, onResolve }) => {
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
          <span className={`text-xs font-mono px-2 py-0.5 rounded font-semibold ${statusColor(event.status_code)}`}>
            {event.status_code}
          </span>
        </td>
        <td className={`px-4 py-3 font-mono text-sm text-slate-700 dark:text-slate-300 max-w-[200px] truncate ${isResolved ? 'line-through' : ''}`}>
          {event.error_type}
        </td>
        <td className="px-4 py-3 text-sm text-slate-600 dark:text-slate-400 max-w-[340px] truncate">
          {event.message}
        </td>
        <td className="px-4 py-3 text-xs text-slate-500 dark:text-slate-400 font-mono max-w-[180px] truncate">
          {event.route || '—'}
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
          <td colSpan={7} className="px-6 py-4">
            <div className="grid grid-cols-2 gap-4 text-xs mb-3 text-slate-500 dark:text-slate-400">
              {event.job_id && <span><strong>Job ID:</strong> {event.job_id}</span>}
              <span><strong>Time:</strong> {new Date(event.created_at).toLocaleString()}</span>
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

// ─── Page ────────────────────────────────────────────────────────────────────

export const ErrorLogPage: React.FC = () => {
  const [events, setEvents] = useState<ErrorEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [timeRange, setTimeRange] = useState<TimeRange>('24h');
  const [showResolved, setShowResolved] = useState(false);
  const [modal, setModal] = useState<ErrorEvent | null>(null);
  const [clearing, setClearing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<{ hour: string; count: number }[]>([]);
  const [view, setView] = useState<'events' | 'issues'>('events');
  const [grouped, setGrouped] = useState<GroupedIssue[]>([]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [data, tl, grp] = await Promise.all([
        monitoringApi.getErrors(500, sinceIso(timeRange), showResolved),
        monitoringApi.getTimeline(24),
        monitoringApi.getGrouped(),
      ]);
      setEvents(data);
      setTimeline(tl);
      setGrouped(grp);
    } catch {
      setError('Could not reach the backend. Is the server running?');
    } finally {
      setLoading(false);
    }
  }, [timeRange, showResolved]);

  useEffect(() => { load(); }, [load]);

  const handleClear = async () => {
    if (!window.confirm(`Delete all ${events.length} error events? This cannot be undone.`)) return;
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
    } catch { /* silent */ }
  };

  const handleResolve = async (ev: ErrorEvent) => {
    const updated = ev.resolved_at
      ? await monitoringApi.unresolveError(ev.id)
      : await monitoringApi.resolveError(ev.id);
    setEvents(prev => prev.map(e => e.id === updated.id ? updated : e));
  };

  const filtered = events.filter(e =>
    !search ||
    e.error_type.toLowerCase().includes(search.toLowerCase()) ||
    e.message.toLowerCase().includes(search.toLowerCase()) ||
    e.route.toLowerCase().includes(search.toLowerCase()) ||
    (e.job_id ?? '').toLowerCase().includes(search.toLowerCase())
  );

  // summary stats
  const total500 = events.filter(e => e.status_code >= 500).length;

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
          {events.length > 0 && (
            <button
              onClick={handleClear}
              disabled={clearing}
              className="flex items-center gap-2 px-3 py-2 text-sm bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/40 transition-colors text-red-600 dark:text-red-400"
            >
              <Trash2 size={14} />
              Clear all
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
      {!loading && events.length > 0 && (
        <div className="grid grid-cols-3 gap-4 mb-6">
          {[
            { label: 'Total events', value: events.length, color: 'text-slate-700 dark:text-slate-200' },
            { label: 'Server errors (5xx)', value: total500, color: 'text-red-600 dark:text-red-400' },
            { label: 'Most recent', value: events[0]?.created_at ? relativeTime(events[0].created_at) : '—', color: 'text-slate-600 dark:text-slate-300' },
          ].map(s => (
            <div key={s.label} className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4">
              <p className="text-xs text-slate-500 dark:text-slate-400 mb-1">{s.label}</p>
              <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Timeline chart */}
      {!loading && timeline.some(t => t.count > 0) && (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 p-4 mb-6">
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-3">Errors per hour — last 24 h</p>
          <ResponsiveContainer width="100%" height={80}>
            <BarChart data={timeline} margin={{ top: 0, right: 0, left: -30, bottom: 0 }}>
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
                {timeline.map((t, i) => (
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
            className={`px-4 py-1.5 text-sm font-medium rounded-lg border transition-colors capitalize ${
              view === v
                ? 'bg-slate-800 dark:bg-slate-200 text-white dark:text-slate-900 border-slate-800 dark:border-slate-200'
                : 'bg-white dark:bg-slate-800 border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 hover:bg-slate-50'
            }`}
          >
            {v === 'issues' ? `Issues (${grouped.length})` : `Events (${events.length})`}
          </button>
        ))}
      </div>

      {/* Toolbar */}
      <div className={`flex items-center gap-3 mb-4 ${view === 'issues' ? 'hidden' : ''}`}>
        <div className="relative flex-1 max-w-sm">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
          <input
            type="text"
            placeholder="Search errors…"
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
        {search && (
          <span className="text-xs text-slate-500 dark:text-slate-400">
            {filtered.length} / {events.length} shown
          </span>
        )}
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
        grouped.length === 0 ? (
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
                {grouped.map((g, i) => (
                  <GroupedIssueRow key={i} issue={g} onViewSample={handleViewSample} />
                ))}
              </tbody>
            </table>
          </div>
        )
      ) : filtered.length === 0 ? (
        <EmptyState
          icon={<Bug size={40} className="text-slate-300" />}
          title={search ? 'No matching errors' : 'No errors recorded'}
          description={search ? 'Try a different search term.' : 'Any unhandled 5xx or failed pipeline will appear here automatically.'}
        />
      ) : (
        <div className="bg-white dark:bg-slate-800 rounded-xl border border-slate-200 dark:border-slate-700 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-900/50 border-b border-slate-200 dark:border-slate-700">
                <th className="px-4 py-3 w-8" />
                <th className="px-4 py-3">Code</th>
                <th className="px-4 py-3">Type</th>
                <th className="px-4 py-3">Message</th>
                <th className="px-4 py-3">Route</th>
                <th className="px-4 py-3">When</th>
                <th className="px-4 py-3" />
              </tr>
            </thead>
            <tbody>
              {filtered.map(e => (
                <ErrorRow key={e.id} event={e} onExpand={setModal} onResolve={handleResolve} />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {modal && <TracebackModal event={modal} onClose={() => setModal(null)} />}
    </div>
  );
};
