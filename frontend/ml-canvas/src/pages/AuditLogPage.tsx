/**
 * AuditLogPage — G4 admin view.
 *
 * Read-only audit trail for the canvas pipeline. Walks the existing
 * append-only `PipelineVersion` history server-side and surfaces
 * "who saved what, when, and what changed" for one dataset at a time.
 *
 * No new schema — the backend just diffs successive snapshots stored
 * by /pipeline/save and the explicit "Save as version" affordance.
 * Use cases:
 *   • A model trained yesterday but fails today — find the save that
 *     introduced the regression.
 *   • Trace lineage of a specific node back to the save that added it.
 *   • Lightweight "git log" for the canvas in multi-user setups.
 */
import React from 'react';
import { AlertCircle, RefreshCw, ScrollText } from 'lucide-react';
import { AuditFilters } from './auditLog/AuditFilters';
import { AuditHistory } from './auditLog/AuditHistory';
import { AuditSummary } from './auditLog/AuditSummary';
import { useAuditLog } from './auditLog/useAuditLog';

export const AuditLogPage: React.FC = () => {
    const state = useAuditLog();
    const { load, isLoading, datasetId, data, error, summary, historyScopeText, historyMetaText } = state;
    return (
        <div className="p-6 max-w-7xl mx-auto">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
                        <ScrollText className="w-6 h-6 text-violet-500" />
                        Pipeline Audit Log
                    </h1>
                    <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                        Chronological history of canvas saves. Each entry shows who, when, and
                        which nodes changed vs. the previous save.
                    </p>
                </div>
                <button
                    type="button"
                    onClick={() => void load()}
                    disabled={isLoading || !datasetId}
                    className="inline-flex items-center gap-2 px-3 py-2 text-sm font-medium rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-slate-800 hover:bg-gray-50 dark:hover:bg-slate-700 disabled:opacity-50"
                >
                    <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
                    Refresh
                </button>
            </div>

            <AuditFilters state={state} />
            {data && (
                <p className="mb-4 text-xs text-gray-500 dark:text-gray-400">
                    Actor, action kind and time filters apply across the full history before the page limit.
                </p>
            )}

            {error && (
                <div className="mb-4 flex items-start gap-2 p-3 rounded border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-sm">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>{error}</span>
                </div>
            )}

            <AuditSummary summary={summary} />

            <AuditHistory state={state} />

            {historyScopeText && (
                <p className="mt-3 text-xs text-gray-500 dark:text-gray-400 text-center">
                    {historyScopeText} {historyMetaText}
                </p>
            )}
        </div>
    );
};
