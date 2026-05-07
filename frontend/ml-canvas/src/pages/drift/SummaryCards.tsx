import React from 'react';
import { AlertTriangle, CheckCircle, Columns, Gauge, Target } from 'lucide-react';
import type { ColumnDrift, DriftReport } from '../../core/api/monitoring';

interface SummaryCardsProps {
    report: DriftReport;
}

/** Four headline metric cards: total cols, drifted, avg PSI, most drifted. */
export const SummaryCards: React.FC<SummaryCardsProps> = ({ report }) => {
    const allCols: ColumnDrift[] = Object.values(report.column_drifts);
    const totalCols = allCols.length;
    const driftedCount = report.drifted_columns_count;
    const psiValues = allCols
        .map(c => c.metrics.find(m => m.metric === 'psi')?.value)
        .filter((v): v is number => v != null);
    const avgPsi = psiValues.length > 0 ? psiValues.reduce((a, b) => a + b, 0) / psiValues.length : 0;
    const mostDrifted = [...allCols].sort((a, b) => {
        const pa = a.metrics.find(m => m.metric === 'psi')?.value ?? 0;
        const pb = b.metrics.find(m => m.metric === 'psi')?.value ?? 0;
        return pb - pa;
    })[0];

    const driftedPct = totalCols > 0 ? Math.round((driftedCount / totalCols) * 100) : 0;
    const mostDriftedPsi = mostDrifted?.metrics.find(m => m.metric === 'psi')?.value ?? 0;

    return (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-50 dark:bg-slate-900/50 rounded-lg p-4 border dark:border-slate-700">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-slate-400 mb-1">
                    <Columns size={13} /> Total Columns
                </div>
                <div className="text-2xl font-bold tabular-nums">{totalCols}</div>
                <div className="text-[11px] text-gray-400 mt-0.5">
                    Ref: {report.reference_rows.toLocaleString()} rows | Cur: {report.current_rows.toLocaleString()} rows
                </div>
            </div>
            <div
                className={`rounded-lg p-4 border ${
                    driftedCount > 0
                        ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                        : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
                }`}
            >
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-slate-400 mb-1">
                    {driftedCount > 0 ? <AlertTriangle size={13} /> : <CheckCircle size={13} />} Drifted
                </div>
                <div className="text-2xl font-bold tabular-nums">
                    {driftedCount} <span className="text-sm font-normal text-gray-400">/ {totalCols}</span>
                </div>
                <div className="text-[11px] text-gray-400 mt-0.5">{driftedPct}% of features</div>
            </div>
            <div className="bg-gray-50 dark:bg-slate-900/50 rounded-lg p-4 border dark:border-slate-700">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-slate-400 mb-1">
                    <Gauge size={13} /> Avg PSI
                </div>
                <div
                    className={`text-2xl font-bold tabular-nums ${
                        avgPsi > 0.2
                            ? 'text-red-600 dark:text-red-400'
                            : avgPsi > 0.1
                            ? 'text-amber-600 dark:text-amber-400'
                            : ''
                    }`}
                >
                    {avgPsi.toFixed(4)}
                </div>
                <div className="text-[11px] text-gray-400 mt-0.5">
                    {avgPsi < 0.1 ? 'Stable' : avgPsi < 0.2 ? 'Minor drift' : 'Significant drift'}
                </div>
            </div>
            <div className="bg-gray-50 dark:bg-slate-900/50 rounded-lg p-4 border dark:border-slate-700">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-slate-400 mb-1">
                    <Target size={13} /> Most Drifted
                </div>
                <div className="text-lg font-bold truncate" title={mostDrifted?.column}>
                    {mostDrifted?.column ?? '—'}
                </div>
                <div className="text-[11px] text-gray-400 mt-0.5">PSI: {mostDriftedPsi.toFixed(4)}</div>
            </div>
        </div>
    );
};
