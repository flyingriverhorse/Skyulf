import React, { useEffect, useState } from 'react';

export interface HyperparameterInputProps {
    value: unknown;
    type: string;
    onChange: (value: unknown) => void;
    step?: number | undefined;
    min?: number | undefined;
    max?: number | undefined;
}

/**
 * Single-value hyperparameter input used by BasicTrainingSettings.
 * Accepts numeric, boolean, or string entries (case-insensitive "None" maps to null).
 * Commits on blur to avoid thrashing onChange while typing.
 */
export const HyperparameterInput: React.FC<HyperparameterInputProps> = ({
    value, type, onChange, step, min, max,
}) => {
    const [localValue, setLocalValue] = useState<string>('');

    useEffect(() => {
        setLocalValue(value === null ? 'None' : value?.toString() ?? '');
    }, [value]);

    const handleBlur = () => {
        const trimmed = localValue.trim();

        if (trimmed.toLowerCase() === 'none') {
            onChange(null);
            return;
        }

        if (type === 'number') {
            if (trimmed === '') return;
            const num = Number(trimmed);
            if (!isNaN(num)) {
                onChange(num);
            } else {
                // Revert if invalid
                setLocalValue(value === null ? 'None' : value?.toString() ?? '');
            }
        } else if (type === 'boolean') {
            if (trimmed.toLowerCase() === 'true') onChange(true);
            else if (trimmed.toLowerCase() === 'false') onChange(false);
            else setLocalValue(value?.toString() ?? '');
        } else {
            onChange(trimmed);
        }
    };

    return (
        <input
            type="text"
            value={localValue}
            onChange={(e) => { setLocalValue(e.target.value); }}
            onBlur={handleBlur}
            placeholder={value === null ? 'None' : ''}
            className="w-full border border-gray-300 dark:border-gray-600 rounded-lg p-2 text-sm bg-white dark:bg-gray-800 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
            {...(type === 'number' ? { step, min, max } : {})}
        />
    );
};
