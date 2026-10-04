import React from 'react';
import { AlertTriangle, CheckCircle, Columns, Gauge, Target } from 'lucide-react';
import type { ColumnDrift, DriftReport } from '../../core/api/monitoring';

interface SummaryCardsProps {
    report: DriftReport;
}

/** Summarize measured feature verdicts separately from report-wide schema changes. */
function summarizeReport(report: DriftReport) {
    const allCols: ColumnDrift[] = Object.values(report.column_drifts);
    const totalCols = allCols.length;
    // The report-wide count includes schema changes, which are shown separately.
    // Feature percentages must use the same measured columns as the denominator.
    const driftedCount = allCols.filter(c => c.drift_detected).length;
    const insufficientCount = allCols.filter(c =>
        c.evidence?.status === 'insufficient_data' || c.evidence?.status === 'unavailable',
    ).length;
    const psiColumns = allCols.flatMap(column => {
        const psi = column.metrics.find(metric =>
            (metric.metric === 'psi' || metric.metric === 'psi_categorical') && Number.isFinite(metric.value),
        )?.value;
        return psi == null ? [] : [{ column: column.column, psi }];
    });
    const avgPsi = psiColumns.length > 0
        ? psiColumns.reduce((sum, column) => sum + column.psi, 0) / psiColumns.length : null;
    const mostDrifted = psiColumns.sort((a, b) => b.psi - a.psi)[0];

    const driftedPct = totalCols > 0 ? Math.round((driftedCount / totalCols) * 100) : 0;

    return { totalCols, driftedCount, avgPsi, mostDrifted, driftedPct, insufficientCount };
}

/** Keep unavailable PSI distinct from a measured stable distribution. */
function psiInterpretation(value: number | null): string {
    if (value == null) return 'No PSI available';
    if (value < 0.1) return 'Small distribution difference';
    return value < 0.2 ? 'Moderate distribution difference' : 'Large distribution difference';
}

/** Reserve a healthy color for reports without drift or evidence gaps. */
function verdictCardStyle(drifted: number, insufficient: number): string {
    if (drifted > 0) return 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
    if (insufficient > 0) return 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800';
    return 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800';
}

/** Four headline metric cards: total cols, drifted, avg PSI, most drifted. */
export const SummaryCards: React.FC<SummaryCardsProps> = ({ report }) => {
    const { totalCols, driftedCount, avgPsi, mostDrifted, driftedPct, insufficientCount } = summarizeReport(report);
    const attentionCount = driftedCount + insufficientCount;

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
                className={`rounded-lg p-4 border ${verdictCardStyle(driftedCount, insufficientCount)}`}
            >
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-slate-400 mb-1">
                    {attentionCount > 0 ? <AlertTriangle size={13} /> : <CheckCircle size={13} />} Drifted
                </div>
                <div className="text-2xl font-bold tabular-nums">
                    {driftedCount} <span className="text-sm font-normal text-gray-400">/ {totalCols}</span>
                </div>
                <div className="text-[11px] text-gray-400 mt-0.5">{driftedPct}% of features</div>
                {insufficientCount > 0 && <div className="text-[11px] text-amber-600 dark:text-amber-400">{insufficientCount} without sufficient evidence</div>}
            </div>
            <div className="bg-gray-50 dark:bg-slate-900/50 rounded-lg p-4 border dark:border-slate-700">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-slate-400 mb-1">
                    <Gauge size={13} /> Avg PSI
                </div>
                <div
                    className={`text-2xl font-bold tabular-nums ${
                        (avgPsi ?? 0) > 0.2
                            ? 'text-red-600 dark:text-red-400'
                            : (avgPsi ?? 0) > 0.1
                            ? 'text-amber-600 dark:text-amber-400'
                            : ''
                    }`}
                >
                    {avgPsi?.toFixed(4) ?? '—'}
                </div>
                <div className="text-[11px] text-gray-400 mt-0.5">
                    {psiInterpretation(avgPsi)}
                </div>
            </div>
            <div className="bg-gray-50 dark:bg-slate-900/50 rounded-lg p-4 border dark:border-slate-700">
                <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-slate-400 mb-1">
                    <Target size={13} /> Highest PSI
                </div>
                <div className="text-lg font-bold truncate" title={mostDrifted?.column}>
                    {mostDrifted?.column ?? '—'}
                </div>
                <div className="text-[11px] text-gray-400 mt-0.5">PSI: {mostDrifted?.psi.toFixed(4) ?? '—'}</div>
            </div>
        </div>
    );
};
