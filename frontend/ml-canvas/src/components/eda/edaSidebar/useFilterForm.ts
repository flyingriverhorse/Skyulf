import { useState } from 'react';
import type { EDASidebarProps } from './types';

function filterValueError(newFilterOp: string, hasFilterValue: boolean, isComparisonOperator: boolean, isNumericFilterValue: boolean): string {
    const currentFilterValueError = !hasFilterValue
        ? 'Enter a filter value'
        : isComparisonOperator && !isNumericFilterValue
            ? `${newFilterOp === '>' ? 'Greater than' : newFilterOp === '<' ? 'Less than' : newFilterOp === '>=' ? 'Greater than or equal to' : 'Less than or equal to'} needs a numeric value`
            : '';

    return currentFilterValueError;
}

function validateFilter(newFilterCol: string, newFilterOp: string, newFilterVal: string) {
    const isComparisonOperator = ['>', '<', '>=', '<='].includes(newFilterOp);
    const hasFilterColumn = newFilterCol.trim().length > 0;
    const hasFilterOperator = newFilterOp.trim().length > 0;
    const hasFilterValue = newFilterVal.trim().length > 0;
    const isNumericFilterValue = Number.isFinite(Number(newFilterVal));

    const currentFilterValueError = filterValueError(newFilterOp, hasFilterValue, isComparisonOperator, isNumericFilterValue);
    return { hasFilterColumn, hasFilterOperator, hasFilterValue, isComparisonOperator, currentFilterValueError };
}

export function useFilterForm(onAddFilter: EDASidebarProps['onAddFilter']) {
    // Called unconditionally by EDASidebar so hiding controls retains the draft.
    const [isAddingFilter, setIsAddingFilter] = useState(false);
    const [newFilterCol, setNewFilterCol] = useState('');
    const [newFilterOp, setNewFilterOp] = useState('==');
    const [newFilterVal, setNewFilterVal] = useState('');
    const [filterValidationAttempted, setFilterValidationAttempted] = useState(false);

    const { hasFilterColumn, hasFilterOperator, hasFilterValue, isComparisonOperator, currentFilterValueError } = validateFilter(newFilterCol, newFilterOp, newFilterVal);
    const columnError = filterValidationAttempted && !hasFilterColumn ? 'Choose a filter column' : '';
    const operatorError = filterValidationAttempted && !hasFilterOperator ? 'Choose a filter operator' : '';
    const valueError = filterValidationAttempted && currentFilterValueError ? currentFilterValueError : '';

    const handleAddFilterSubmit = () => {
        setFilterValidationAttempted(true);
        if (!hasFilterColumn || !hasFilterOperator || !hasFilterValue || currentFilterValueError) return;

        onAddFilter(
            newFilterCol,
            isComparisonOperator ? Number(newFilterVal) : Number.isNaN(Number(newFilterVal)) ? newFilterVal : Number(newFilterVal),
            newFilterOp
        );
        setIsAddingFilter(false);
        setFilterValidationAttempted(false);
        setNewFilterCol('');
        setNewFilterOp('==');
        setNewFilterVal('');
    };

    return {
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
    };
}
