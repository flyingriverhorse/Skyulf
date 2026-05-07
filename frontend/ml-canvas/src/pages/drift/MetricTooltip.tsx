import React from 'react';

/**
 * Inline label with a hover tooltip — used in drift-metric column headers
 * to explain what PSI / KS / Wasserstein / KL Divergence mean.
 */
export const MetricTooltip: React.FC<{
    label: string;
    tooltip: string;
    icon?: React.ReactNode;
}> = ({ label, tooltip, icon }) => (
    <div className="group relative flex items-center gap-1 cursor-help">
        <span className="border-b border-dotted border-gray-400">{label}</span>
        {icon && <span className="text-gray-400">{icon}</span>}
        <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 w-64 p-2 bg-slate-900 text-white text-xs rounded shadow-xl opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all z-50 text-center pointer-events-none">
            {tooltip}
            <div className="absolute bottom-full left-1/2 -translate-x-1/2 -mb-1 border-4 border-transparent border-b-slate-900"></div>
        </div>
    </div>
);
