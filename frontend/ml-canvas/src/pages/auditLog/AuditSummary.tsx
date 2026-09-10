import React from 'react';
import { History, Minus, Pencil, Plus, User as UserIcon } from 'lucide-react';
import type { AuditLogState } from './useAuditLog';

/** Render the current page's summary when it contains saves. */
export const AuditSummary: React.FC<{ summary: AuditLogState['summary'] }> = ({ summary }) => (
    <>
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
    </>
);

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
