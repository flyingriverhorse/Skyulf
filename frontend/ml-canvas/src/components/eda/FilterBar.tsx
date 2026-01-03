import React, { useState } from 'react';
import { Filter, X, Plus, EyeOff } from 'lucide-react';
import { InfoTooltip } from '../ui/InfoTooltip';

interface FilterItem {
    column: string;
    operator: string;
    value: any;
}

interface FilterBarProps {
    filters: FilterItem[];
    columns: string[];
    excludedCols: string[];
    onAddFilter: (column: string, value: any, operator: string) => void;
    onRemoveFilter: (index: number) => void;
    onClearFilters: () => void;
    onToggleExclude: (column: string, exclude: boolean) => void;
}

export const FilterBar: React.FC<FilterBarProps> = ({
    filters,
    columns,
    excludedCols,
    onAddFilter,
    onRemoveFilter,
    onClearFilters,
    onToggleExclude
}) => {
    const [showFilterForm, setShowFilterForm] = useState(false);
    const [newFilterCol, setNewFilterCol] = useState('');
    const [newFilterOp, setNewFilterOp] = useState('==');
    const [newFilterVal, setNewFilterVal] = useState('');
    const [showExcludeDropdown, setShowExcludeDropdown] = useState(false);

    const handleAdd = () => {
        if (newFilterCol && newFilterVal) {
            onAddFilter(
                newFilterCol, 
                isNaN(Number(newFilterVal)) ? newFilterVal : Number(newFilterVal), 
                newFilterOp
            );
            setShowFilterForm(false);
            setNewFilterCol('');
            setNewFilterVal('');
        }
    };

    return (
        <div className="mb-6 space-y-4">
            {/* Active Filters Section */}
            <div className="bg-blue-50 dark:bg-blue-900/20 p-3 rounded-md border border-blue-100 dark:border-blue-800">
                <div className="flex flex-wrap items-center gap-2">
                    <div className="flex items-center mr-2">
                        <Filter className="w-4 h-4 text-blue-600 dark:text-blue-400 mr-2" />
                        <span className="text-sm font-medium text-blue-700 dark:text-blue-300">Active Filters</span>
                        <InfoTooltip text="Filters are applied to the raw dataset before profiling. You can add filters manually or by clicking on distribution bars to filter on specific range." />
                    </div>
                    
                    {filters.map((filter, idx) => (
                        <div key={idx} className="flex items-center bg-white dark:bg-gray-800 border border-blue-200 dark:border-blue-700 rounded-full px-3 py-1 text-sm shadow-sm animate-in fade-in zoom-in duration-200">
                            <span className="font-medium text-gray-700 dark:text-gray-300 mr-1">{filter.column}</span>
                            <span className="text-gray-500 mx-1">{filter.operator}</span>
                            <span className="font-mono text-blue-600 dark:text-blue-400">{String(filter.value)}</span>
                            <button 
                                onClick={() => onRemoveFilter(idx)}
                                className="ml-2 text-gray-400 hover:text-red-500"
                            >
                                <X className="w-3 h-3" />
                            </button>
                        </div>
                    ))}

                    {!showFilterForm ? (
                        <button 
                            onClick={() => setShowFilterForm(true)}
                            className="flex items-center px-2 py-1 text-xs font-medium text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 bg-white dark:bg-gray-800 border border-blue-200 dark:border-blue-700 rounded-full hover:bg-blue-50 dark:hover:bg-gray-700 transition-colors"
                        >
                            <Plus className="w-3 h-3 mr-1" />
                            Add Filter
                        </button>
                    ) : (
                        <div className="flex items-center gap-2 bg-white dark:bg-gray-800 p-1 rounded-md border border-blue-200 dark:border-blue-700 animate-in slide-in-from-left-2 duration-200">
                            <select 
                                value={newFilterCol}
                                onChange={(e) => setNewFilterCol(e.target.value)}
                                className="text-xs border-none bg-transparent focus:ring-0 py-1 pl-2 pr-6"
                            >
                                <option value="" disabled>Column</option>
                                {columns.map(col => (
                                    <option key={col} value={col}>{col}</option>
                                ))}
                            </select>
                            <select 
                                value={newFilterOp}
                                onChange={(e) => setNewFilterOp(e.target.value)}
                                className="text-xs border-none bg-gray-50 dark:bg-gray-900 rounded focus:ring-0 py-1 px-2"
                            >
                                <option value="==">==</option>
                                <option value="!=">!=</option>
                                <option value=">">&gt;</option>
                                <option value="<">&lt;</option>
                                <option value=">=">&gt;=</option>
                                <option value="<=">&lt;=</option>
                            </select>
                            <input 
                                type="text" 
                                value={newFilterVal}
                                onChange={(e) => setNewFilterVal(e.target.value)}
                                placeholder="Value"
                                className="text-xs border-none bg-gray-50 dark:bg-gray-900 rounded focus:ring-0 py-1 px-2 w-20"
                                onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
                            />
                            <button 
                                onClick={handleAdd}
                                className="p-1 text-green-600 hover:bg-green-50 rounded"
                            >
                                <Plus className="w-3 h-3" />
                            </button>
                            <button 
                                onClick={() => setShowFilterForm(false)}
                                className="p-1 text-gray-400 hover:text-gray-600"
                            >
                                <X className="w-3 h-3" />
                            </button>
                        </div>
                    )}

                    {filters.length > 0 && (
                        <button 
                            onClick={onClearFilters}
                            className="ml-auto text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 underline"
                        >
                            Clear All
                        </button>
                    )}
                </div>
            </div>

            {/* Excluded Columns Section */}
            <div className="bg-gray-50 dark:bg-gray-900/50 p-3 rounded-md border border-gray-200 dark:border-gray-800">
                <div className="flex flex-wrap items-center gap-2">
                    <div className="flex items-center mr-2">
                        <EyeOff className="w-4 h-4 text-gray-500 dark:text-gray-400 mr-2" />
                        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Excluded Columns</span>
                        <InfoTooltip text="These columns are completely removed from the analysis. Click 'Add Exclusion' to hide more columns, or click the '+' on an excluded column to bring it back." />
                    </div>

                    {excludedCols.map((col, idx) => (
                        <div key={idx} className="flex items-center bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-full px-3 py-1 text-sm shadow-sm opacity-75 hover:opacity-100 transition-opacity">
                            <span className="font-medium text-gray-500 dark:text-gray-400 mr-1 line-through decoration-gray-400">{col}</span>
                            <button 
                                onClick={() => onToggleExclude(col, false)}
                                className="ml-2 text-gray-400 hover:text-green-500"
                                title="Include back in analysis"
                            >
                                <Plus className="w-3 h-3" />
                            </button>
                        </div>
                    ))}

                    <div className="relative">
                        <button 
                            onClick={() => setShowExcludeDropdown(!showExcludeDropdown)}
                            className="flex items-center px-2 py-1 text-xs font-medium text-gray-600 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-full hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                        >
                            <EyeOff className="w-3 h-3 mr-1" />
                            Exclude Column
                        </button>
                        
                        {showExcludeDropdown && (
                            <div className="absolute top-full left-0 mt-1 w-48 max-h-60 overflow-y-auto bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg z-50">
                                {columns.filter(c => !excludedCols.includes(c)).map(col => (
                                    <button
                                        key={col}
                                        className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                                        onClick={() => {
                                            onToggleExclude(col, true);
                                            setShowExcludeDropdown(false);
                                        }}
                                    >
                                        {col}
                                    </button>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};
