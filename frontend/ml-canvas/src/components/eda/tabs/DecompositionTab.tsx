import React, { useState } from 'react';
import { DecompositionTree } from '../DecompositionTree';
import { Filter } from '../../../core/api/eda';
import { RefreshCw } from 'lucide-react';

interface DecompositionTabProps {
    datasetId: number;
    columns: string[];
    initialFilters?: Filter[];
}

export const DecompositionTab: React.FC<DecompositionTabProps> = ({ datasetId, columns, initialFilters = [] }) => {
    const [measureCol, setMeasureCol] = useState<string>('count');
    const [measureAgg, setMeasureAgg] = useState<string>('count');
    const [key, setKey] = useState(0); // Force re-render to reset tree

    const handleReset = () => {
        setKey(prev => prev + 1);
    };

    return (
        <div className="space-y-4">
            {/* Controls */}
            <div className="bg-white dark:bg-slate-800 p-4 rounded-lg shadow-sm border border-slate-200 dark:border-slate-700 flex gap-4 items-end">
                <div className="space-y-1">
                    <label className="text-xs font-medium text-slate-500">Analyze</label>
                    <select 
                        value={measureCol} 
                        onChange={(e) => setMeasureCol(e.target.value)}
                        className="block w-[180px] rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border dark:bg-slate-700 dark:border-slate-600 dark:text-white"
                    >
                        <option value="count">Row Count</option>
                        {columns.map(col => (
                            <option key={col} value={col}>{col}</option>
                        ))}
                    </select>
                </div>

                {measureCol !== 'count' && (
                    <div className="space-y-1">
                        <label className="text-xs font-medium text-slate-500">Aggregation</label>
                        <select 
                            value={measureAgg} 
                            onChange={(e) => setMeasureAgg(e.target.value)}
                            className="block w-[120px] rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm p-2 border dark:bg-slate-700 dark:border-slate-600 dark:text-white"
                        >
                            <option value="sum">Sum</option>
                            <option value="mean">Average</option>
                            <option value="min">Min</option>
                            <option value="max">Max</option>
                        </select>
                    </div>
                )}

                <button 
                    onClick={handleReset} 
                    className="p-2 border rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 text-slate-600 dark:text-slate-300"
                    title="Reset Tree"
                >
                    <RefreshCw className="w-4 h-4" />
                </button>
            </div>

            {/* Tree Visualization */}
            <div className="bg-slate-50 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-800 overflow-hidden">
                <DecompositionTree 
                    key={key}
                    datasetId={datasetId}
                    measureCol={measureCol}
                    measureAgg={measureAgg}
                    columns={columns}
                    initialFilters={initialFilters}
                />
            </div>
        </div>
    );
};
