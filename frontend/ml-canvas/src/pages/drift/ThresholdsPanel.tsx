import React, { useState } from 'react';
import type { DriftReport, DriftThresholds } from '../../core/api/monitoring';
import { DEFAULT_DRIFT_THRESHOLDS, driftThresholdError } from '../../core/utils/driftThresholds';

interface ThresholdsPanelProps {
    thresholds: DriftThresholds;
    onChange: (next: DriftThresholds) => void;
    report?: DriftReport | null;
}

const FIELDS: ReadonlyArray<{ key: keyof DriftThresholds; label: string; metrics: string[] }> = [
    { key: 'psi', label: 'PSI', metrics: ['psi', 'psi_categorical'] },
    { key: 'ks', label: 'KS statistic', metrics: ['ks_statistic'] },
    { key: 'wasserstein', label: 'Wasserstein', metrics: ['wasserstein_distance'] },
    { key: 'kl', label: 'KL Div', metrics: ['kl_divergence'] },
];

/** Show applied values separately from invalid drafts and saved-report fallbacks. */
export const ThresholdsPanel: React.FC<ThresholdsPanelProps> = ({ thresholds, onChange, report }) => {
    const [drafts, setDrafts] = useState<Partial<Record<keyof DriftThresholds, string>>>({});
    const [errors, setErrors] = useState<Partial<Record<keyof DriftThresholds, string | null>>>({});
    const reset = (next: DriftThresholds) => { setDrafts({}); setErrors({}); onChange(next); };
    return <div className="px-4 pb-3 flex flex-wrap items-start gap-4 border-t border-gray-100 dark:border-slate-700 pt-3">
        {FIELDS.map(({ key, label, metrics }) => {
            const draft = drafts[key];
            const error = errors[key];
            const saved = [...new Set(Object.values(report?.column_drifts ?? {}).flatMap(column =>
                column.metrics.filter(metric => metrics.includes(metric.metric)).map(metric => metric.threshold)))];
            const effective = thresholds[key] ?? (saved.length ? saved.join(', ') : DEFAULT_DRIFT_THRESHOLDS[key]);
            const source = thresholds[key] != null ? 'override' : saved.length ? 'saved report' : 'default';
            return <div key={key} className="text-xs text-gray-600 dark:text-slate-400">
                <label className="flex items-center gap-1.5">
                    <span className="font-medium">{label}</span>
                    <input type="number" value={draft ?? thresholds[key] ?? ''} min={0}
                        max={key === 'ks' ? 1 : undefined} step="any" aria-invalid={!!error}
                        onChange={event => {
                            const raw = event.target.value;
                            setDrafts(previous => ({ ...previous, [key]: raw }));
                            const value = raw.trim() ? Number(raw) : undefined;
                            const nextError = event.target.validity.badInput ? 'Threshold must be a finite number.'
                                : value === undefined ? null : driftThresholdError(key, value);
                            setErrors(previous => ({ ...previous, [key]: nextError }));
                            if (!nextError) onChange({ ...thresholds, [key]: value });
                        }}
                        className="w-20 px-2 py-1 border rounded bg-white dark:bg-slate-900 dark:text-white" />
                </label>
                <p className="mt-1">Effective {label}: {effective} ({source})</p>
                {error && <p role="alert" className="max-w-48 text-red-600">{error} Not applied.</p>}
            </div>;
        })}
        <button onClick={() => reset(DEFAULT_DRIFT_THRESHOLDS)} className="text-xs underline">Reset defaults</button>
        <button onClick={() => reset({})} className="text-xs underline">Remove overrides</button>
        <p className="w-full text-xs text-gray-500">Clearing a field restores its saved report threshold, or the default before analysis. Reset defaults applies defaults as overrides. New analyses use defaults for cleared fields.</p>
    </div>;
};
