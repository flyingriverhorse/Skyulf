import React, { useCallback, useState } from 'react';
import { Bug, X, Clock, Route, Server, Copy, Check } from 'lucide-react';
import type { ErrorEvent } from '../../core/api/monitoring';
import { RecordLink, NodeInspectorLink } from '../../components/shared';
import type { OperationalTimeRange } from '../../core/utils/operationalContext';
import { statusColor, severityBadgeClass, SEVERITY_LABELS } from './errorLogFormatting';

export const CopyDiagnosticId: React.FC<{ id: string | number; label: string }> = ({ id, label }) => {
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
export const ErrorResourceLink: React.FC<{
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

export const TracebackModal: React.FC<{
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
