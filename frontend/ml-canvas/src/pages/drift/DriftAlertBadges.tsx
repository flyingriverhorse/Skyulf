import React from 'react';
import { AlertTriangle, AlertOctagon, CheckCircle2 } from 'lucide-react';
import type { DriftAlertSeverity, DriftAlertStatus } from '../../core/api/monitoring';

const SEVERITY_STYLES: Record<DriftAlertSeverity, { label: string; classes: string }> = {
    none: {
        label: 'No drift',
        classes:
            'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 border-green-200 dark:border-green-800',
    },
    warning: {
        label: 'Warning',
        classes:
            'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400 border-amber-200 dark:border-amber-800',
    },
    critical: {
        label: 'Critical',
        classes:
            'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400 border-red-200 dark:border-red-800',
    },
};

/** Pill badge for a drift alert's derived triage severity. */
export const DriftSeverityBadge: React.FC<{ severity: DriftAlertSeverity }> = ({ severity }) => {
    const v = SEVERITY_STYLES[severity];
    const Icon = severity === 'critical' ? AlertOctagon : severity === 'warning' ? AlertTriangle : CheckCircle2;
    return (
        <span
            className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-medium ${v.classes}`}
        >
            <Icon size={12} aria-hidden="true" />
            {v.label}
        </span>
    );
};

const STATUS_STYLES: Record<DriftAlertStatus, { label: string; classes: string }> = {
    new: {
        label: 'New',
        classes:
            'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border-blue-200 dark:border-blue-800',
    },
    acknowledged: {
        label: 'Acknowledged',
        classes:
            'bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400 border-purple-200 dark:border-purple-800',
    },
    resolved: {
        label: 'Resolved',
        classes:
            'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300 border-slate-200 dark:border-slate-700',
    },
    reopened: {
        label: 'Reopened',
        classes:
            'bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400 border-orange-200 dark:border-orange-800',
    },
};

/** Pill badge for a drift alert's disposition status. */
export const DriftStatusBadge: React.FC<{ status: DriftAlertStatus }> = ({ status }) => {
    const v = STATUS_STYLES[status];
    return (
        <span
            className={`inline-flex items-center px-2 py-0.5 rounded-full border text-xs font-medium ${v.classes}`}
        >
            {v.label}
        </span>
    );
};
