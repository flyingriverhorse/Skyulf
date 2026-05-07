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
import React, { useCallback, useEffect, useMemo, useState } from 'react';
import {
    AlertCircle,
    ChevronDown,
    ChevronRight,
    Clock,
    GitCommit,
    Hash,
    History,
    Minus,
    Pencil,
    Plus,
    RefreshCw,
    ScrollText,
    User as UserIcon,
} from 'lucide-react';
import { useUsableDatasets } from '../core/hooks/useDatasets';
import { pipelineVersionsApi, AuditLogResponse, AuditLogEntry } from '../core/api/pipelineVersions';
import { toast } from '../core/toast';

const LIMIT_OPTIONS: ReadonlyArray<number> = [25, 50, 100, 200];

const formatTimestamp = (iso: string): string => {
    try {
        return new Date(iso).toLocaleString();
    } catch {
        return iso;
    }
};

/** Resolve the dataset_source_id used by the pipelines API. The canvas
 *  Toolbar saves under `Dataset.id` (the value the dataset dropdown
 *  binds to via `option.value={d.id}`), so the audit picker must use
 *  the same id. We coerce to string because some legacy `Dataset` rows
 *  carry a numeric id that breaks `String.prototype.slice` downstream. */
const resolveDatasetSourceId = (d: { id: string | number }): string => String(d.id);

interface DiffPillProps {
    icon: React.ReactNode;
    label: string;
    count: number;
    tone: 'add' | 'remove' | 'modify';
}

const DiffPill: React.FC<DiffPillProps> = ({ icon, label, count, tone }) => {
    if (count === 0) return null;
    const toneClass =
        tone === 'add'
            ? 'bg-emerald-50 text-emerald-700 border-emerald-200 dark:bg-emerald-900/20 dark:text-emerald-300 dark:border-emerald-800'
            : tone === 'remove'
              ? 'bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-900/20 dark:text-rose-300 dark:border-rose-800'
              : 'bg-amber-50 text-amber-700 border-amber-200 dark:bg-amber-900/20 dark:text-amber-300 dark:border-amber-800';
    return (
        <span
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded border text-[11px] font-medium ${toneClass}`}
            title={`${count} ${label}`}
        >
            {icon}
            {count} {label}
        </span>
    );
};

interface NodeListProps {
    title: string;
    items: string[];
    tone: 'add' | 'remove' | 'modify';
}

const NodeList: React.FC<NodeListProps> = ({ title, items, tone }) => {
    if (items.length === 0) return null;
    const toneText =
        tone === 'add'
            ? 'text-emerald-600 dark:text-emerald-400'
            : tone === 'remove'
              ? 'text-rose-600 dark:text-rose-400'
              : 'text-amber-600 dark:text-amber-400';
    return (
        <div>
            <div className={`text-[11px] font-semibold uppercase tracking-wide mb-1 ${toneText}`}>
                {title} ({items.length})
            </div>
            <div className="flex flex-wrap gap-1">
                {items.map(id => (
                    <code
                        key={id}
                        className="text-[11px] px-1.5 py-0.5 rounded bg-gray-100 dark:bg-slate-800 text-gray-700 dark:text-gray-300 font-mono"
                    >
                        {id}
                    </code>
                ))}
            </div>
        </div>
    );
};

interface AuditRowProps {
    entry: AuditLogEntry;
    isFirst: boolean;
}

const AuditRow: React.FC<AuditRowProps> = ({ entry, isFirst }) => {
    const [expanded, setExpanded] = useState(false);
    const { diff } = entry;
    const totalChanged =
        diff.nodes_added.length + diff.nodes_removed.length + diff.nodes_modified.length;
    // The first save has no predecessor, so the backend reports every node
    // as "added". Render it as a plain "Initial" badge instead of pretending
    // it diffed against a prior version.
    const isGenesis = isFirst && diff.nodes_removed.length === 0 && diff.nodes_modified.length === 0;

    return (
        <div className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
            <button
                type="button"
                onClick={() => setExpanded(v => !v)}
                className="w-full flex items-start gap-3 p-3 text-left hover:bg-gray-50 dark:hover:bg-slate-800/50 transition-colors"
            >
                <div className="pt-0.5 text-gray-400">
                    {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                </div>
                <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap items-center gap-2 mb-1">
                        <span className="inline-flex items-center gap-1 text-xs font-mono text-gray-500 dark:text-gray-400">
                            <Hash size={11} />v{entry.version_int}
                        </span>
                        <span className="text-sm font-medium text-gray-800 dark:text-gray-100 truncate">
                            {entry.name}
                        </span>
                        <span
                            className={`text-[10px] px-1.5 py-0.5 rounded uppercase font-semibold ${
                                entry.kind === 'auto'
                                    ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
                                    : 'bg-gray-100 text-gray-700 dark:bg-slate-700 dark:text-gray-300'
                            }`}
                        >
                            {entry.kind}
                        </span>
                        {isGenesis && (
                            <span className="text-[10px] px-1.5 py-0.5 rounded uppercase font-semibold bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300">
                                initial
                            </span>
                        )}
                    </div>
                    <div className="flex flex-wrap items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
                        <span className="inline-flex items-center gap-1">
                            <Clock size={11} />
                            {formatTimestamp(entry.created_at)}
                        </span>
                        <span className="inline-flex items-center gap-1">
                            <UserIcon size={11} />
                            {entry.user_id !== null ? `user #${entry.user_id}` : 'anonymous'}
                        </span>
                        <span className="inline-flex items-center gap-1">
                            <GitCommit size={11} />
                            {entry.node_count} nodes / {entry.edge_count} edges
                        </span>
                    </div>
                    {entry.note && (
                        <div className="mt-1 text-xs text-gray-600 dark:text-gray-300 italic truncate">
                            “{entry.note}”
                        </div>
                    )}
                    {!isGenesis && (
                        <div className="mt-2 flex flex-wrap gap-1.5">
                            <DiffPill
                                icon={<Plus size={10} />}
                                label="added"
                                count={diff.nodes_added.length}
                                tone="add"
                            />
                            <DiffPill
                                icon={<Minus size={10} />}
                                label="removed"
                                count={diff.nodes_removed.length}
                                tone="remove"
                            />
                            <DiffPill
                                icon={<Pencil size={10} />}
                                label="modified"
                                count={diff.nodes_modified.length}
                                tone="modify"
                            />
                            {totalChanged === 0 && (
                                <span className="text-[11px] text-gray-400 italic">
                                    no node-level changes
                                </span>
                            )}
                        </div>
                    )}
                </div>
            </button>
            {expanded && totalChanged > 0 && (
                <div className="px-10 pb-3 pt-1 space-y-2 bg-gray-50/60 dark:bg-slate-900/40">
                    <NodeList title="Added" items={diff.nodes_added} tone="add" />
                    <NodeList title="Removed" items={diff.nodes_removed} tone="remove" />
                    <NodeList title="Modified" items={diff.nodes_modified} tone="modify" />
                </div>
            )}
        </div>
    );
};

export const AuditLogPage: React.FC = () => {
    const { data: datasets, isLoading: datasetsLoading } = useUsableDatasets();
    const [datasetId, setDatasetId] = useState<string>('');
    const [limit, setLimit] = useState<number>(50);
    const [data, setData] = useState<AuditLogResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Auto-pick the first dataset once the list arrives so the page renders
    // something useful on first load instead of an empty picker.
    useEffect(() => {
        if (!datasetId && datasets && datasets.length > 0) {
            const first = datasets[0];
            if (first) setDatasetId(resolveDatasetSourceId(first));
        }
    }, [datasets, datasetId]);

    const load = useCallback(async () => {
        if (!datasetId) return;
        setIsLoading(true);
        setError(null);
        try {
            const resp = await pipelineVersionsApi.audit(datasetId, limit);
            setData(resp);
        } catch (e) {
            const msg = (e as Error).message || 'Failed to load audit trail';
            setError(msg);
            toast.error(msg);
        } finally {
            setIsLoading(false);
        }
    }, [datasetId, limit]);

    useEffect(() => {
        void load();
    }, [load]);

    const summary = useMemo(() => {
        if (!data) return null;
        const entries = data.entries;
        const uniqueUsers = new Set(
            entries.map(e => (e.user_id !== null ? `u${e.user_id}` : 'anon')),
        );
        let added = 0;
        let removed = 0;
        let modified = 0;
        for (const e of entries) {
            added += e.diff.nodes_added.length;
            removed += e.diff.nodes_removed.length;
            modified += e.diff.nodes_modified.length;
        }
        return {
            saves: entries.length,
            users: uniqueUsers.size,
            added,
            removed,
            modified,
        };
    }, [data]);

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

            <div className="flex flex-wrap items-end gap-4 mb-4">
                <div className="flex-1 min-w-[240px]">
                    <label
                        htmlFor="audit-dataset"
                        className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1"
                    >
                        Dataset
                    </label>
                    <select
                        id="audit-dataset"
                        value={datasetId}
                        onChange={e => setDatasetId(e.target.value)}
                        disabled={datasetsLoading || !datasets || datasets.length === 0}
                        className="w-full px-3 py-2 text-sm rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-slate-800 text-gray-900 dark:text-gray-100 disabled:opacity-50"
                    >
                        {!datasets || datasets.length === 0 ? (
                            <option value="">
                                {datasetsLoading ? 'Loading…' : 'No datasets'}
                            </option>
                        ) : (
                            datasets.map(d => {
                                const sid = resolveDatasetSourceId(d);
                                const shortSid = sid.length > 8 ? sid.slice(0, 8) : sid;
                                return (
                                    <option key={sid} value={sid}>
                                        {d.name} ({shortSid})
                                    </option>
                                );
                            })
                        )}
                    </select>
                </div>
                <div>
                    <span className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
                        Limit
                    </span>
                    <div className="inline-flex rounded border border-gray-300 dark:border-gray-600 overflow-hidden">
                        {LIMIT_OPTIONS.map(opt => (
                            <button
                                key={opt}
                                type="button"
                                onClick={() => setLimit(opt)}
                                className={`px-3 py-2 text-xs font-medium ${
                                    limit === opt
                                        ? 'bg-violet-500 text-white'
                                        : 'bg-white dark:bg-slate-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-slate-700'
                                }`}
                            >
                                {opt}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {error && (
                <div className="mb-4 flex items-start gap-2 p-3 rounded border border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 text-sm">
                    <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                    <span>{error}</span>
                </div>
            )}

            {summary && summary.saves > 0 && (
                <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 mb-4">
                    <SummaryStat label="Saves" value={summary.saves} icon={<History size={14} />} />
                    <SummaryStat label="Users" value={summary.users} icon={<UserIcon size={14} />} />
                    <SummaryStat
                        label="Nodes added"
                        value={summary.added}
                        icon={<Plus size={14} />}
                        tone="add"
                    />
                    <SummaryStat
                        label="Nodes removed"
                        value={summary.removed}
                        icon={<Minus size={14} />}
                        tone="remove"
                    />
                    <SummaryStat
                        label="Nodes modified"
                        value={summary.modified}
                        icon={<Pencil size={14} />}
                        tone="modify"
                    />
                </div>
            )}

            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
                {isLoading ? (
                    <div className="p-8 text-center text-sm text-gray-500 dark:text-gray-400">
                        Loading audit trail…
                    </div>
                ) : !datasetId ? (
                    <div className="p-8 text-center text-sm text-gray-500 dark:text-gray-400">
                        Pick a dataset to view its save history.
                    </div>
                ) : !data || data.entries.length === 0 ? (
                    <div className="p-8 text-center text-sm text-gray-500 dark:text-gray-400">
                        No saves recorded for this dataset yet. Saves appear here automatically
                        once you click Save on the canvas.
                    </div>
                ) : (
                    <div>
                        {data.entries.map((entry, idx) => (
                            <AuditRow
                                key={entry.id}
                                entry={entry}
                                // entries is newest-first; the chronologically-first save is
                                // the LAST element in the array.
                                isFirst={idx === data.entries.length - 1}
                            />
                        ))}
                    </div>
                )}
            </div>

            {data && data.entries.length > 0 && data.entries.length < data.total && (
                <p className="mt-3 text-xs text-gray-500 dark:text-gray-400 text-center">
                    Showing {data.entries.length} of {data.total} saves. Increase the limit to see
                    older history.
                </p>
            )}
        </div>
    );
};

interface SummaryStatProps {
    label: string;
    value: number;
    icon: React.ReactNode;
    tone?: 'add' | 'remove' | 'modify';
}

const SummaryStat: React.FC<SummaryStatProps> = ({ label, value, icon, tone }) => {
    const toneText =
        tone === 'add'
            ? 'text-emerald-600 dark:text-emerald-400'
            : tone === 'remove'
              ? 'text-rose-600 dark:text-rose-400'
              : tone === 'modify'
                ? 'text-amber-600 dark:text-amber-400'
                : 'text-gray-700 dark:text-gray-200';
    return (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-3">
            <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-gray-500 dark:text-gray-400 font-medium">
                {icon}
                {label}
            </div>
            <div className={`mt-1 text-xl font-semibold ${toneText}`}>{value}</div>
        </div>
    );
};
