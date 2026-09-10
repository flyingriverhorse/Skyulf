import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Clock, GitCommit, Hash, Minus, Pencil, Plus, User as UserIcon } from 'lucide-react';
import type { AuditLogEntry } from '../../core/api/pipelineVersions';

const formatTimestamp = (iso: string): string => {
    try {
        return new Date(iso).toLocaleString();
    } catch {
        return iso;
    }
};

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

export const AuditRow: React.FC<AuditRowProps> = ({ entry, isFirst }) => {
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
            <AuditRowDetails expanded={expanded} diff={diff} totalChanged={totalChanged} />
        </div>
    );
};


/** Expanded node lists share the row toggle's lifetime. */
const AuditRowDetails: React.FC<{
    expanded: boolean;
    diff: AuditLogEntry['diff'];
    totalChanged: number;
}> = ({ expanded, diff, totalChanged }) => (
    <>
        {expanded && totalChanged > 0 && (
            <div className="px-10 pb-3 pt-1 space-y-2 bg-gray-50/60 dark:bg-slate-900/40">
                <NodeList title="Added" items={diff.nodes_added} tone="add" />
                <NodeList title="Removed" items={diff.nodes_removed} tone="remove" />
                <NodeList title="Modified" items={diff.nodes_modified} tone="modify" />
            </div>
        )}
    </>
);
