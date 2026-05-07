import React from 'react';
import type { DriftThresholds } from '../../core/api/monitoring';

interface ThresholdsPanelProps {
    thresholds: DriftThresholds;
    onChange: (next: DriftThresholds) => void;
}

const FIELDS: ReadonlyArray<{
    key: keyof DriftThresholds;
    label: string;
    step: number;
    min: number;
    max: number;
}> = [
    { key: 'psi', label: 'PSI', step: 0.05, min: 0.01, max: 1 },
    { key: 'ks', label: 'KS p-value', step: 0.01, min: 0.001, max: 0.2 },
    { key: 'wasserstein', label: 'Wasserstein', step: 0.05, min: 0.01, max: 1 },
    { key: 'kl', label: 'KL Div', step: 0.05, min: 0.01, max: 1 },
];

const DEFAULTS: DriftThresholds = { psi: 0.2, ks: 0.05, wasserstein: 0.1, kl: 0.1 };

/** Inline panel with one number input per drift metric and a "reset" link. */
export const ThresholdsPanel: React.FC<ThresholdsPanelProps> = ({ thresholds, onChange }) => (
    <div className="px-4 pb-3 flex flex-wrap items-center gap-4 border-t border-gray-100 dark:border-slate-700 pt-3">
        <span className="text-xs font-medium text-gray-500 dark:text-slate-400">Thresholds:</span>
        {FIELDS.map(({ key, label, step, min, max }) => (
            <label key={key} className="flex items-center gap-1.5 text-xs text-gray-600 dark:text-slate-400">
                <span className="font-medium">{label}</span>
                <input
                    type="number"
                    value={thresholds[key] ?? ''}
                    onChange={e =>
                        onChange({ ...thresholds, [key]: parseFloat(e.target.value) || undefined })
                    }
                    step={step}
                    min={min}
                    max={max}
                    className="w-20 px-2 py-1 text-xs border border-gray-200 dark:border-slate-600 rounded bg-white dark:bg-slate-900 dark:text-white focus:outline-none focus:ring-1 focus:ring-blue-500 tabular-nums"
                />
            </label>
        ))}
        <button
            onClick={() => onChange(DEFAULTS)}
            className="text-[11px] text-gray-400 hover:text-gray-600 dark:hover:text-slate-300 transition-colors"
        >
            Reset defaults
        </button>
    </div>
);
