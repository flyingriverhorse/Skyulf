import { Plus, EyeOff, ChevronUp, ChevronDown } from 'lucide-react';
import type { EDASidebarProps } from './types';

type ExclusionProps = Pick<EDASidebarProps, 'columns' | 'excludedCols' | 'excludedDirty' | 'analyzing' | 'onToggleExclude' | 'onApplyExcluded'> & {
    showExclusions: boolean;
    setShowExclusions: (value: boolean) => void;
    isAddingExclusion: boolean;
    setIsAddingExclusion: (value: boolean) => void;
};

export function ExclusionControls({
    columns,
    excludedCols,
    excludedDirty,
    analyzing,
    onToggleExclude,
    onApplyExcluded,
    showExclusions,
    setShowExclusions,
    isAddingExclusion,
    setIsAddingExclusion
}: ExclusionProps) {
    return (
        <div>
            <button
                onClick={() => setShowExclusions(!showExclusions)}
                className="flex items-center justify-between w-full text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2 hover:text-gray-700 dark:hover:text-gray-300"
            >
                <span>Excluded ({excludedCols.length})</span>
                {showExclusions ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>

            {showExclusions && (
                <div className="space-y-2">
                    {excludedCols.map((col, idx) => (
                        <div key={idx} className="flex items-center justify-between bg-gray-50 dark:bg-gray-900/50 border border-gray-200 dark:border-gray-700 rounded px-2 py-1 text-xs">
                            <span className="font-medium text-gray-500 dark:text-gray-400 line-through truncate max-w-[140px]">{col}</span>
                            <button
                                onClick={() => onToggleExclude(col, false)}
                                className="text-gray-500 dark:text-gray-400 hover:text-green-500"
                                title="Include back"
                            >
                                <Plus className="w-3 h-3" />
                            </button>
                        </div>
                    ))}

                    <ExclusionApply onApplyExcluded={onApplyExcluded} excludedDirty={excludedDirty} analyzing={analyzing} />

                    {isAddingExclusion ? (
                        <div className="bg-white dark:bg-gray-900 p-2 rounded border border-gray-200 dark:border-gray-700 space-y-2 shadow-sm">
                            <select
                                aria-label="Column to exclude from analysis"
                                className="w-full text-xs rounded border-gray-300 dark:border-gray-600 dark:bg-gray-800 p-1"
                                onChange={(e) => {
                                    if (e.target.value) {
                                        onToggleExclude(e.target.value, true);
                                        setIsAddingExclusion(false);
                                    }
                                }}
                                defaultValue=""
                            >
                                <option value="" disabled>Select Column</option>
                                {columns.filter(c => !excludedCols.includes(c)).map(col => (
                                    <option key={col} value={col}>{col}</option>
                                ))}
                            </select>
                            <button onClick={() => setIsAddingExclusion(false)} className="w-full text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 text-center">Cancel</button>
                        </div>
                    ) : (
                        <button
                            onClick={() => setIsAddingExclusion(true)}
                            className="w-full flex items-center justify-center px-2 py-1 text-xs border border-dashed border-gray-300 dark:border-gray-600 rounded text-gray-500 dark:text-gray-400 hover:text-red-600 hover:border-red-300 transition-colors"
                        >
                            <EyeOff className="w-3 h-3 mr-1" /> Exclude Column
                        </button>
                    )}
                </div>
            )}
        </div>
    );
}

function ExclusionApply({ onApplyExcluded, excludedDirty, analyzing }: Pick<EDASidebarProps, 'onApplyExcluded' | 'excludedDirty' | 'analyzing'>) {
    return (
        <button
            onClick={onApplyExcluded}
            disabled={!excludedDirty || analyzing}
            className={`w-full flex items-center justify-center px-2 py-1 text-xs rounded border transition-colors ${excludedDirty && !analyzing
                    ? 'bg-blue-600 text-white border-blue-700 hover:bg-blue-700'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-400 border-gray-200 dark:border-gray-700 cursor-not-allowed'
                }`}
            title={excludedDirty ? 'Apply excluded columns to re-run analysis' : 'No pending exclusion changes'}
        >
            Apply changes
        </button>
    );
}
