import { Plus, X, ChevronUp, ChevronDown } from 'lucide-react';
import { FormField } from '../../ui/FormField';
import type { EDASidebarProps, FilterItem } from './types';
import type { useFilterForm } from './useFilterForm';

type Form = ReturnType<typeof useFilterForm>;
type FilterProps = Pick<EDASidebarProps, 'filtersDraft' | 'filtersApplied' | 'filtersDirty' | 'columns' | 'analyzing' | 'onRemoveFilter' | 'onResetFilters' | 'onApplyFilters'> & {
    showFilters: boolean;
    setShowFilters: (value: boolean) => void;
    form: Form;
};
const filterSignature = (filter: FilterItem): string =>
    `${filter.column}\u001f${filter.operator}\u001f${JSON.stringify(filter.value)}`;

const areFilterListsEqual = (left: FilterItem[], right: FilterItem[]): boolean => {
    if (left.length !== right.length) return false;
    return left.every((filter, index) => filterSignature(filter) === filterSignature(right[index]!));
};

export function FilterControls({
    filtersDraft,
    filtersApplied,
    filtersDirty,
    columns,
    analyzing,
    onRemoveFilter,
    onResetFilters,
    onApplyFilters,
    showFilters,
    setShowFilters,
    form
}: FilterProps) {
    const draftFiltersDirty = filtersDirty || !areFilterListsEqual(filtersDraft, filtersApplied);
    return (
        <div>
            <div className="flex items-center justify-between w-full mb-2">
                <button
                    onClick={() => setShowFilters(!showFilters)}
                    className="flex items-center text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider hover:text-gray-700 dark:hover:text-gray-300"
                >
                    <span>Draft Filters ({filtersDraft.length})</span>
                    {showFilters ? <ChevronUp className="w-3 h-3 ml-1" /> : <ChevronDown className="w-3 h-3 ml-1" />}
                </button>
                {filtersDirty && (
                    <span className="text-[10px] font-medium rounded-full bg-amber-100 text-amber-800 px-2 py-0.5">
                        Pending
                    </span>
                )}
            </div>

            {showFilters && (
                <div className="space-y-2">
                    <p className="text-[10px] text-gray-500 dark:text-gray-400">
                        Report uses {filtersApplied.length} applied filter{filtersApplied.length === 1 ? '' : 's'} until you apply draft changes.
                    </p>

                    <DraftFilterList filtersDraft={filtersDraft} filtersDirty={filtersDirty} analyzing={analyzing} onRemoveFilter={onRemoveFilter} />

                    <FilterEditor form={form} columns={columns} />

                    <FilterActions disabled={!draftFiltersDirty || analyzing} onResetFilters={onResetFilters} onApplyFilters={onApplyFilters} />
                </div>
            )}
        </div>
    );
}

function DraftFilterList({
    filtersDraft,
    filtersDirty,
    analyzing,
    onRemoveFilter
}: Pick<EDASidebarProps, 'filtersDraft' | 'filtersDirty' | 'analyzing' | 'onRemoveFilter'>) {
    return <>
        {filtersDraft.length > 0 ? (
            filtersDraft.map((filter, idx) => (
                <div
                    key={idx}
                    className={`flex items-center justify-between rounded px-2 py-1 text-xs border ${filtersDirty
                            ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-100 dark:border-blue-800'
                            : 'bg-gray-50 dark:bg-gray-900/50 border-gray-200 dark:border-gray-700'
                        }`}
                >
                    <div className="truncate max-w-[140px]">
                        <span className="font-medium text-blue-700 dark:text-blue-300">{filter.column}</span>
                        <span className="mx-1 text-gray-500 dark:text-gray-400">{filter.operator}</span>
                        <span className="text-gray-600 dark:text-gray-400">{String(filter.value)}</span>
                    </div>
                    <button
                        onClick={() => onRemoveFilter(idx)}
                        className="text-gray-500 dark:text-gray-400 hover:text-red-500"
                        aria-label={`Remove filter ${idx + 1}`}
                        disabled={analyzing}
                    >
                        <X className="w-3 h-3" />
                    </button>
                </div>
            ))
        ) : (
            <p className="text-xs text-gray-500 dark:text-gray-400">No draft filters yet.</p>
        )}
    </>;
}

function FilterEditor({ form, columns }: { form: Form; columns: string[]; }) {
    const {
        isAddingFilter,
        setIsAddingFilter,
        newFilterCol,
        setNewFilterCol,
        newFilterOp,
        setNewFilterOp,
        newFilterVal,
        setNewFilterVal,
        filterValidationAttempted,
        setFilterValidationAttempted,
        columnError,
        operatorError,
        valueError,
        handleAddFilterSubmit
     } = form;
    return <>
        {isAddingFilter ? (
            <div className="bg-white dark:bg-gray-900 p-2 rounded border border-gray-200 dark:border-gray-700 space-y-2 shadow-sm">
                <FormField
                    label="Filter column"
                    hideLabel
                    required
                    {...(filterValidationAttempted ? { error: columnError } : {})}
                >
                    {(field) => (
                        <select
                            {...field}
                            value={newFilterCol}
                            onChange={(e) => setNewFilterCol(e.target.value)}
                            className="w-full text-xs rounded border-gray-300 dark:border-gray-600 dark:bg-gray-800 p-1"
                        >
                            <option value="" disabled>Column</option>
                            {columns.map((col) => <option key={col} value={col}>{col}</option>)}
                        </select>
                    )}
                </FormField>
                <div className="flex gap-1">
                    <FormField
                        label="Filter operator"
                        hideLabel
                        required
                        className="w-1/3"
                        {...(filterValidationAttempted ? { error: operatorError } : {})}
                    >
                        {(field) => (
                            <select
                                {...field}
                                value={newFilterOp}
                                onChange={(e) => setNewFilterOp(e.target.value)}
                                className="w-full text-xs rounded border-gray-300 dark:border-gray-600 dark:bg-gray-800 p-1"
                            >
                                <option value="==">==</option>
                                <option value="!=">!=</option>
                                <option value=">">&gt;</option>
                                <option value="<">&lt;</option>
                                <option value=">=">&gt;=</option>
                                <option value="<=">&lt;=</option>
                            </select>
                        )}
                    </FormField>
                    <FormField
                        label="Filter value"
                        hideLabel
                        required
                        className="w-2/3"
                        {...(filterValidationAttempted ? { error: valueError } : {})}
                    >
                        {(field) => (
                            <input
                                {...field}
                                type="text"
                                value={newFilterVal}
                                onChange={(e) => setNewFilterVal(e.target.value)}
                                placeholder="Value"
                                className="w-full text-xs rounded border-gray-300 dark:border-gray-600 dark:bg-gray-800 p-1"
                                onKeyDown={(e) => e.key === 'Enter' && handleAddFilterSubmit()}
                            />
                        )}
                    </FormField>
                </div>
                <div className="flex justify-end gap-2">
                    <button
                        onClick={() => {
                            setIsAddingFilter(false);
                            setFilterValidationAttempted(false);
                            setNewFilterCol('');
                            setNewFilterOp('==');
                            setNewFilterVal('');
                        }}
                        className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleAddFilterSubmit}
                        className="text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700"
                    >
                        Save draft
                    </button>
                </div>
            </div>
        ) : (
            <button
                onClick={() => setIsAddingFilter(true)}
                className="w-full flex items-center justify-center px-2 py-1 text-xs border border-dashed border-gray-300 dark:border-gray-600 rounded text-gray-500 dark:text-gray-400 hover:text-blue-600 hover:border-blue-300 transition-colors"
            >
                <Plus className="w-3 h-3 mr-1" /> Add Filter
            </button>
        )}
    </>;
}

function FilterActions({
    disabled,
    onResetFilters,
    onApplyFilters
}: Pick<EDASidebarProps, 'onResetFilters' | 'onApplyFilters'> & { disabled: boolean; }) {
    return (
        <div className="flex gap-2 pt-1">
            <button
                onClick={onResetFilters}
                disabled={disabled}
                className={`flex-1 text-xs px-2 py-1 rounded border transition-colors ${disabled
                        ? 'bg-gray-100 dark:bg-gray-800 text-gray-400 border-gray-200 dark:border-gray-700 cursor-not-allowed'
                        : 'bg-white dark:bg-gray-900 text-gray-600 border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-800'
                    }`}
            >
                Reset filters
            </button>
            <button
                onClick={onApplyFilters}
                disabled={disabled}
                className={`flex-1 text-xs px-2 py-1 rounded border transition-colors ${!disabled
                        ? 'bg-blue-600 text-white border-blue-700 hover:bg-blue-700'
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-400 border-gray-200 dark:border-gray-700 cursor-not-allowed'
                    }`}
            >
                Apply filters
            </button>
        </div>
    );
}
